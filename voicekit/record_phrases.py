#!/usr/bin/env python3
"""
Начитка своим голосом тех фраз, которые синтез озвучил хуже всего.

Берёт out/report.csv (его пишет voice_tool.py), выбирает фразы с оценкой ниже порога —
от худшей к лучшей, не больше --max штук, — и по одной показывает их для записи
(ударная гласная — заглавной буквой, из stress.txt). Запись подставляется вместо синтеза:
  out/clips/<ключ>.mp3, out/clips.json, out/report.csv (engine = voice), out/review.html,
  исходник — out/voice/<ключ>.wav.
Начитанные фразы voice_tool.py больше не переозвучивает. Можно выйти и продолжить потом:
уже записанные пропускаются.

Порядок: текст на экране -> писк -> запись уже идёт, читайте -> Enter: сохранить и следующая.
Вместо простого Enter: «п» + Enter — перечитать эту фразу, «в» + Enter — выйти (эта не сохраняется).

Не запускайте одновременно с voice_tool.py на ту же папку — обе программы пишут в одни файлы.

Примеры:
  python record_phrases.py                         # худшие фразы: ниже 0.9, не больше 21
  python record_phrases.py --below 0.95 --max 10
  python record_phrases.py --also "ладья" --also "Ничья"   # добавить фразы по кусочку текста
  python record_phrases.py --redo "ладья"          # перезаписать уже начитанную фразу
  python record_phrases.py --dry                   # только показать список
"""
import argparse, base64, csv, json, re, threading, time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from record_voice import REC_SR, MAX_SEC, analyze_and_trim, db
from voice_tool import apply_stress, load_stress, postprocess, to_mp3, write_review

FIELDS = ["key", "text", "score", "heard", "engine"]


def beep(freq=880, sec=0.15, sr=REC_SR):
    t = np.arange(int(sr * sec)) / sr
    tone = 0.3 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))
    sd.play(tone.astype(np.float32), sr); sd.wait()


def record_until_enter(device):
    """Пишет с микрофона до Enter. Возвращает (звук, что набрано перед Enter)."""
    blocks, typed, stop = [], [""], threading.Event()

    def cb(indata, frames, t, status):
        blocks.append(indata[:, 0].copy())

    def wait_enter():
        typed[0] = input().strip().lower(); stop.set()

    with sd.InputStream(samplerate=REC_SR, channels=1, dtype="float32", device=device, callback=cb):
        # микрофон включаем ДО писка: иначе его запуск (доли секунды) съедает начало фразы
        time.sleep(0.3)
        beep()
        time.sleep(0.05)                    # хвост писка в комнате
        start = len(blocks)
        threading.Thread(target=wait_enter, daemon=True).start()
        t0 = time.time()
        while not stop.is_set() and time.time() - t0 < MAX_SEC:
            level = float(np.max(np.abs(blocks[-1]))) if blocks else 0.0
            bar = "█" * int(min(level, 1.0) * 30)
            print(f"\r  ● {time.time() - t0:4.1f} с  |{bar:<30}|  Enter — дальше ", end="", flush=True)
            time.sleep(0.05)
    print()
    if not stop.is_set():                   # вышло время — ждём Enter, чтобы не сбить следующую фразу
        print(f"  (прошло {MAX_SEC} с, запись остановлена — нажмите Enter)")
        stop.wait()
    blocks = blocks[start:]                  # всё, что было до конца писка, выбрасываем
    return (np.concatenate(blocks) if blocks else np.zeros(0, np.float32)), typed[0]


def show_stress(text, stress):
    """«лад+ья» -> «ладьЯ»: так ударение видно в любом окне терминала."""
    return re.sub(r"\+([А-Яа-яЁё])", lambda m: m.group(1).upper(), apply_stress(text, stress))


def load_report(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["score"] = float(r["score"])
    return rows


def save_all(out_dir, rows, clips):
    rows = sorted(rows, key=lambda r: r["score"])
    with open(out_dir / "report.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    (out_dir / "clips.json").write_text(json.dumps(clips, ensure_ascii=False), encoding="utf-8")
    write_review(out_dir, rows)


def pick(rows, args):
    def has(r, parts):
        return any(p.lower() in r["text"].lower() for p in parts)
    if args.redo:
        return [r for r in rows if has(r, args.redo)]
    extra = [r for r in rows if args.also and has(r, args.also) and r["engine"] != "voice"]
    weak = [r for r in sorted(rows, key=lambda r: r["score"])
            if r["score"] < args.below and r["engine"] != "voice" and r not in extra]
    return (extra + weak)[: args.max]


def main():
    ap = argparse.ArgumentParser(description="Начитка своим голосом фраз, которые синтез озвучил плохо")
    ap.add_argument("--out", default="out", help="папка результатов voice_tool.py")
    ap.add_argument("--below", type=float, default=0.9, help="порог оценки: берём фразы ниже него")
    ap.add_argument("--max", type=int, default=21, help="не больше стольких фраз за раз")
    ap.add_argument("--also", action="append", help="добавить фразы, содержащие этот текст (можно несколько раз)")
    ap.add_argument("--redo", action="append", help="перезаписать уже начитанные фразы с этим текстом")
    ap.add_argument("--stress", default="stress.txt")
    ap.add_argument("--device", type=int, help="номер микрофона (python record_voice.py --list)")
    ap.add_argument("--dry", action="store_true", help="только показать, какие фразы попали в список")
    args = ap.parse_args()

    out_dir = Path(args.out)
    report = out_dir / "report.csv"
    if not report.exists():
        raise SystemExit(f"Нет {report} — сначала озвучьте фразы: python voice_tool.py ...")
    rows = load_report(report)
    cj = out_dir / "clips.json"
    clips = json.loads(cj.read_text(encoding="utf-8")) if cj.exists() else {}
    stress = load_stress(args.stress)

    todo = pick(rows, args)
    done = sum(r["engine"] == "voice" for r in rows)
    print(f"\nУже начитано: {done}. К записи сейчас: {len(todo)} (порог {args.below}, не больше {args.max}).\n")
    for i, r in enumerate(todo, 1):
        print(f"  {i:2}. {r['score']:.2f}  {r['text']}")
    if not todo or args.dry:
        return

    dev_name = sd.query_devices(args.device if args.device is not None else sd.default.device[0])["name"]
    print(f"\nМикрофон: {dev_name}")
    print("Читайте спокойно и по-доброму, так же, как образец. Ударная гласная — ЗАГЛАВНАЯ.")
    print("Текст -> писк -> читайте -> Enter: сохранить и следующая.")
    print("Вместо Enter: «п» + Enter — перечитать фразу, «в» + Enter — выйти.\n")
    input("Enter — начать…")

    (out_dir / "voice").mkdir(exist_ok=True)
    (out_dir / "clips").mkdir(exist_ok=True)
    for i, r in enumerate(todo, 1):
        print("─" * 70)
        print(f"[{i}/{len(todo)}]\n")
        print(f"   «{show_stress(r['text'], stress)}»\n")
        while True:
            time.sleep(1.7)                  # + 0.3 с на включение микрофона = 2 с до писка
            raw, ans = record_until_enter(args.device)
            if ans in ("в", "q"):
                print("Выход. Эта фраза не сохранена, предыдущие — на месте.")
                return
            if ans in ("п", "p", "r"):
                print("Ещё раз.")
                continue
            y, peak, noise = analyze_and_trim(raw)
            if y is None:
                print("Голос не слышен — ещё раз. (Если так повторяется, проверьте микрофон.)")
                continue
            notes = []
            if peak >= 0.99:
                notes.append("перегруз")
            if peak < 0.1:
                notes.append("очень тихо")
            if db(noise) > -45:
                notes.append("много шума")
            # та же обработка, что у синтеза: обрезка тишины, 24 кГц, паузы по краям, громкость
            y2, sr2 = postprocess(y, REC_SR, 1.0, 1.0)
            k = r["key"]
            sf.write(out_dir / "voice" / f"{k}.wav", y, REC_SR, subtype="PCM_16")
            mp3 = to_mp3(y2, sr2)
            (out_dir / "clips" / f"{k}.mp3").write_bytes(mp3)
            clips[k] = "data:audio/mpeg;base64," + base64.b64encode(mp3).decode()
            r.update(score=1.0, heard="", engine="voice")
            save_all(out_dir, rows, clips)       # сохраняем сразу: выход в любой момент ничего не теряет
            print(f"Сохранено ({len(y) / REC_SR:.1f} с)" + (" — внимание: " + ", ".join(notes) if notes else "") + "\n")
            break

    print("\nВсе фразы из списка пройдены. Послушать все записи: " + str(out_dir / "review.html"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОтменено. Всё, что сохранено до этого, на месте.")
