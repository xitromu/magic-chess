#!/usr/bin/env python3
"""
Запись образца голоса для F5-TTS (--ref-audio / --ref-text).

Показывает текст, записывает с микрофона (Enter — старт, Enter — стоп),
обрезает тишину, проверяет громкость и шум, проигрывает запись и сохраняет:
  образец.wav — моно, 24 кГц, 16 бит
  образец.txt — точный текст образца

Примеры:
  python record_voice.py
  python record_voice.py --text "Свой текст." --out образец2.wav
  python record_voice.py --list            # показать микрофоны
  python record_voice.py --device 1        # записывать с микрофона №1
"""
import argparse, sys, threading, time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

DEFAULT_TEXT = ("Лизочка, ходи, пожалуйста, конём или королём. "
                "Ладьёй пока не ходи, подожди немножко. Молодец, у тебя всё получится!")
REC_SR, OUT_SR, MAX_SEC = 48000, 24000, 20


def db(x):
    return 20 * np.log10(max(float(x), 1e-9))


def record(device):
    blocks, stop = [], threading.Event()

    def cb(indata, frames, t, status):
        blocks.append(indata[:, 0].copy())

    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    t0 = time.time()
    with sd.InputStream(samplerate=REC_SR, channels=1, dtype="float32", device=device, callback=cb):
        while not stop.is_set() and time.time() - t0 < MAX_SEC:
            level = float(np.max(np.abs(blocks[-1]))) if blocks else 0.0
            bar = "█" * int(min(level, 1.0) * 30)
            print(f"\r  ● запись {time.time() - t0:4.1f} с  |{bar:<30}|  Enter — стоп ", end="", flush=True)
            time.sleep(0.05)
    print()
    return np.concatenate(blocks) if blocks else np.zeros(0, np.float32)


def analyze_and_trim(y):
    frame = int(REC_SR * 0.02)
    n = len(y) // frame
    rms = np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1))
    noise = np.percentile(rms, 10)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    loud = np.where(rms > max(noise * 4, peak * 0.03))[0]
    if len(loud) == 0:
        return None, peak, noise
    pad = int(0.15 * REC_SR)
    a = max(loud[0] * frame - pad, 0)
    b = min((loud[-1] + 1) * frame + pad, len(y))
    return y[a:b], peak, noise


def main():
    ap = argparse.ArgumentParser(description="Запись образца голоса для F5-TTS")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="что читать (станет --ref-text)")
    ap.add_argument("--out", default="образец.wav")
    ap.add_argument("--device", type=int, help="номер микрофона (см. --list)")
    ap.add_argument("--list", action="store_true", help="показать звуковые устройства")
    args = ap.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    dev_name = sd.query_devices(args.device if args.device is not None else sd.default.device[0])["name"]
    print(f"\nМикрофон: {dev_name}")
    print("Советы: тихая комната, без эха (лучше среди мебели и штор), 15–20 см от микрофона,")
    print("        говорите спокойно и по-доброму, как будто объясняете ребёнку.\n")
    print("Прочитайте ТОЧНО этот текст (6–10 секунд):\n")
    print(f"   «{args.text}»\n")

    while True:
        input("Нажмите Enter, чтобы начать запись…")
        for i in (3, 2, 1):
            print(f"  {i}…", end=" ", flush=True)
            time.sleep(0.7)
        print("говорите!")
        raw = record(args.device)

        y, peak, noise = analyze_and_trim(raw)
        if y is None:
            print("Голос не слышен. Проверьте микрофон (python record_voice.py --list).\n")
            continue
        dur = len(y) / REC_SR
        print(f"\nДлительность: {dur:.1f} с | пик: {db(peak):.0f} дБ | фоновый шум: {db(noise):.0f} дБ")
        problems = []
        if dur < 5:
            problems.append("слишком коротко — нужно 6–10 секунд")
        if dur > 12:
            problems.append("длинновато — лучше уложиться в 10 секунд")
        if peak >= 0.99:
            problems.append("перегруз — говорите тише или отодвиньтесь от микрофона")
        if peak < 0.1:
            problems.append("очень тихо — говорите громче или ближе к микрофону")
        if db(noise) > -45:
            problems.append("много фонового шума — найдите место потише")
        print("Замечания: " + "; ".join(problems) if problems else "Технически запись хорошая.")

        y = y / (np.max(np.abs(y)) or 1.0) * 0.89
        out = resample_poly(y, OUT_SR, REC_SR).astype(np.float32)
        print("Слушаем…")
        sd.play(out, OUT_SR)
        sd.wait()

        ans = input("\nСохранить? Enter — да, «п» — перезаписать, «в» — выйти: ").strip().lower()
        if ans in ("в", "q"):
            return
        if ans in ("п", "p", "r"):
            print()
            continue
        wav = Path(args.out)
        sf.write(wav, out, OUT_SR, subtype="PCM_16")
        wav.with_suffix(".txt").write_text(args.text, encoding="utf-8")
        print(f"\nСохранено: {wav.resolve()}  и  {wav.with_suffix('.txt').name}")
        print("Для озвучки (после установки F5: setup_windows.bat f5):")
        print(f'  python voice_tool.py --from-html Волшебные-шахматы.html --engine f5 '
              f'--ref-audio {wav.name} --ref-text "{args.text}"')
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОтменено.")
