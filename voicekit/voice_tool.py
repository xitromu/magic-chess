#!/usr/bin/env python3
"""
Озвучка фраз для «Волшебных шахмат» (и будущих версий) на своей видеокарте.

Что делает:
  1. Берёт фразы из .txt (по одной на строку) или прямо из html-файла игры.
  2. Озвучивает каждую несколькими дублями (движок Silero или F5-TTS).
  3. Проверяет каждый дубль распознаванием речи (Whisper) и оставляет лучший.
  4. Делает голос «мультяшнее» (выше и чуть медленнее), чистит тишину, выравнивает громкость.
  5. Сохраняет: out/clips/<ключ>.mp3, out/clips.json (формат игры), out/report.csv, out/review.html.
  6. По желанию вшивает озвучку прямо в html-файл игры (--embed-html).

Примеры:
  python voice_tool.py --phrases Фразы-Искры-1.txt Фразы-Искры-2.txt
  python voice_tool.py --from-html Волшебные-шахматы.html --embed-html Волшебные-шахматы.html
  python voice_tool.py --from-html game.html --engine f5 --ref-audio ref.wav --ref-text "Текст из ref.wav"
  python voice_tool.py --phrases a.txt --only "конь" --takes 8          # переозвучить фразы со словом «конь»
  python voice_tool.py --phrases a.txt --redo-below 0.95                # переозвучить только слабые
"""
import argparse, base64, csv, html, json, os, re, sys, time, difflib
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- ключ фразы (как в игре)
def hkey(s: str) -> str:
    """Тот же хеш, что в JS игры: djb2-вариант по UTF-16 code units."""
    x = 5381
    b = s.encode("utf-16-le")
    for i in range(0, len(b), 2):
        cu = b[i] | (b[i + 1] << 8)
        x = ((x * 33) ^ cu) & 0xFFFFFFFF
    return format(x, "x")

# ---------------------------------------------------------------- фразы
def phrases_from_txt(paths):
    out = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and line not in out:
                out.append(line)
    return out

def phrases_from_html(path):
    """Все произносимые строки игры: кириллица в одинарных кавычках, заканчивается на . ! ?"""
    src = Path(path).read_text(encoding="utf-8")
    # вырезаем уже вшитую озвучку, чтобы не сканировать мегабайты base64
    src = re.sub(r"window\.DINO_CLIPS=\{.*?\};</script>", "</script>", src, flags=re.S)
    out = []
    for l in re.findall(r"'([^'\n]*[А-Яа-яЁё][^'\n]*[.!?])'", src):
        if len(l) >= 4 and "<" not in l and "$" not in l and l not in out:
            out.append(l)
    return out

# ---------------------------------------------------------------- подготовка текста
def load_stress(path):
    d = {}
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                a, b = [x.strip() for x in line.split("=", 1)]
                if a and b:
                    d[a.lower()] = b
    return d

def apply_stress(text, d):
    def rep(m):
        w = m.group(0)
        v = d.get(w.lower())
        if not v:
            return w
        return v[0].upper() + v[1:] if w[0].isupper() else v
    return re.sub(r"[А-Яа-яЁё]+", rep, text)

def clean_for_tts(text):
    t = text.replace("«", "").replace("»", "")
    t = re.sub(r"\bГ\b", "гэ", t)          # «буквой Г» -> «буквой гэ»
    t = t.replace(" — ", ", ").replace("—", ",")
    return t

# ---------------------------------------------------------------- движки озвучки
class SileroEngine:
    """Silero TTS: стабильные ударения и «ё», быстро, добрые голоса (baya, kseniya, xenia, aidar, eugene).
    Лицензия моделей Silero — некоммерческая (CC BY-NC-SA): для семьи и клуба подходит."""
    def __init__(self, speaker, device):
        import torch
        self.torch = torch
        self.speaker = speaker
        self.model = None
        for mid in ("v5_ru", "v4_ru"):
            try:
                self.model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models", model="silero_tts",
                                               language="ru", speaker=mid, trust_repo=True)
                print(f"[silero] модель {mid}")
                break
            except Exception as e:
                print(f"[silero] {mid} не загрузилась: {e}")
        if self.model is None:
            sys.exit("Не удалось загрузить Silero. Проверьте интернет (нужен доступ к github.com и models.silero.ai).")
        self.model.to(device)
        self.sr = 48000

    def variants(self, text, stress):
        base = clean_for_tts(text)
        v = [base]
        s = apply_stress(base, stress)
        if s != base:
            v.insert(0, s)                  # сначала вариант с ударениями из словаря
        return v

    def synth(self, text, take):
        # Silero детерминирован: разные «дубли» = разные варианты текста (см. variants)
        audio = self.model.apply_tts(text=text, speaker=self.speaker, sample_rate=self.sr,
                                     put_accent=True, put_yo=True)
        return audio.detach().cpu().numpy().astype(np.float32), self.sr


class F5Engine:
    """F5-TTS с русским чекпойнтом: самое живое звучание, голос копируется с вашего образца ref.wav (5–12 с).
    Ударения расставляет RUAccent (+ ваш stress.txt)."""
    def __init__(self, args, device):
        from huggingface_hub import snapshot_download
        repo_dir = Path(snapshot_download(args.f5_repo))
        ckpt = args.f5_ckpt or self._find(repo_dir, "*.safetensors") or self._find(repo_dir, "*.pt")
        vocab = args.f5_vocab or self._find(repo_dir, "vocab.txt")
        if not ckpt or not vocab:
            sys.exit(f"В {repo_dir} не нашёлся чекпойнт или vocab.txt — укажите --f5-ckpt и --f5-vocab вручную.")
        print(f"[f5] чекпойнт: {ckpt}\n[f5] словарь: {vocab}")
        from f5_tts.api import F5TTS
        try:
            self.tts = F5TTS(model=args.f5_model, ckpt_file=str(ckpt), vocab_file=str(vocab), device=device)
        except TypeError:   # старые версии f5-tts
            self.tts = F5TTS(model_type="F5-TTS", ckpt_file=str(ckpt), vocab_file=str(vocab), device=device)
        if not args.ref_audio or not args.ref_text:
            sys.exit("Для F5 нужны --ref-audio (образец голоса) и --ref-text (что в нём сказано).")
        self.ref_audio, self.ref_text = args.ref_audio, args.ref_text
        self.speed = args.f5_speed
        self.acc = None
        try:
            from ruaccent import RUAccent
            self.acc = RUAccent()
            for size in ("turbo3.1", "turbo3", "turbo2", "turbo"):
                try:
                    self.acc.load(omograph_model_size=size, use_dictionary=True); print(f"[ruaccent] {size}"); break
                except Exception:
                    continue
        except Exception as e:
            print("[ruaccent] не загрузился, ударения только из stress.txt:", e)
            self.acc = None

    @staticmethod
    def _find(root, pattern):
        c = sorted(root.rglob(pattern), key=lambda p: (("v1" not in str(p).lower()), -p.stat().st_mtime))
        return c[0] if c else None

    def variants(self, text, stress):
        t = clean_for_tts(text)
        if self.acc:
            try:
                t = self.acc.process_all(t)
            except Exception:
                pass
        # словарь поверх RUAccent: сначала убираем его «+» в этих словах
        def fix(m):
            w = m.group(0); plain = w.replace("+", "")
            v = stress.get(plain.lower())
            if not v:
                return w
            return v[0].upper() + v[1:] if plain[0].isupper() else v
        t = re.sub(r"[А-Яа-яЁё+]+", fix, t)
        return [t]

    def synth(self, text, take):
        kw = dict(ref_file=self.ref_audio, ref_text=self.ref_text, gen_text=text, speed=self.speed, remove_silence=True)
        try:
            wav, sr, _ = self.tts.infer(seed=1000 + take * 7919, **kw)
        except TypeError:
            import torch; torch.manual_seed(1000 + take * 7919)
            wav, sr, _ = self.tts.infer(**kw)
        return np.asarray(wav, dtype=np.float32), sr

# ---------------------------------------------------------------- проверка распознаванием
class Checker:
    def __init__(self, device, model_name):
        import torch
        from transformers import pipeline
        self.pipe = pipeline("automatic-speech-recognition", model=model_name,
                             torch_dtype=torch.float16 if device == "cuda" else torch.float32, device=device)
        print(f"[asr] {model_name}")

    def __call__(self, y, sr):
        import librosa
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
        r = self.pipe({"raw": y16, "sampling_rate": 16000},
                      generate_kwargs={"language": "russian", "task": "transcribe"})
        return r["text"].strip()

def norm_words(s):
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^а-я\s]", " ", s)
    words = s.split()
    return [re.sub(r"зь$", "сь", w) for w in words]   # ферзь ≈ «ферсь» — это правильное произношение

def score(expected, heard, keywords):
    e, g = norm_words(expected), norm_words(heard)
    r = difflib.SequenceMatcher(None, "".join(e), "".join(g)).ratio()
    for w in e:
        if w in keywords and w not in g:
            r -= 0.25      # важное шахматное слово не расслышано — сильный штраф
    return r

KEYWORDS = {"конь", "коня", "король", "короля", "королю", "ферсь", "ферзя", "ладья", "ладью", "ладьей",
            "слон", "пешка", "пешку", "рокировку", "рокировка", "шах", "мат", "вилка"}

# ---------------------------------------------------------------- обработка звука
def postprocess(y, sr, pitch, tempo, out_sr=24000):
    import librosa
    y, _ = librosa.effects.trim(y, top_db=38)
    if pitch and abs(pitch - 1) > 1e-3:
        # как «asetrate»: выше и быстрее, затем возвращаем темп
        y = librosa.resample(y, orig_sr=sr, target_sr=int(sr / pitch))
        y = librosa.effects.time_stretch(y, rate=1 / pitch)
    if tempo and abs(tempo - 1) > 1e-3:
        y = librosa.effects.time_stretch(y, rate=tempo)
    y = librosa.resample(y, orig_sr=sr, target_sr=out_sr)
    pad = np.zeros(int(out_sr * 0.08), dtype=np.float32)
    y = np.concatenate([pad, y, pad])
    peak = float(np.max(np.abs(y))) or 1.0
    y = y / peak * 0.89
    return y.astype(np.float32), out_sr

def to_mp3(y, sr, kbps=48):
    import lameenc
    enc = lameenc.Encoder()
    enc.set_bit_rate(kbps); enc.set_in_sample_rate(sr); enc.set_channels(1); enc.set_quality(2)
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    return enc.encode(pcm) + enc.flush()

# ---------------------------------------------------------------- отчёты
def write_review(out_dir, rows):
    items = "\n".join(
        f"<tr class='{'bad' if r['score'] < 0.9 else ''}'><td>{html.escape(r['text'])}</td>"
        f"<td>{r['score']:.2f}</td><td>{html.escape(r['heard'])}</td>"
        f"<td><audio controls preload='none' src='clips/{r['key']}.mp3'></audio></td></tr>" for r in rows)
    (out_dir / "review.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Проверка озвучки</title>
<style>body{{font:15px system-ui;margin:16px}}td{{padding:6px;border-bottom:1px solid #ddd;vertical-align:middle}}
tr.bad td{{background:#fff1f1}}audio{{height:32px}}</style>
<h2>Проверка озвучки ({len(rows)} фраз)</h2><p>Красным — фразы с низкой оценкой: послушайте их первыми.</p>
<table><tr><th>Фраза</th><th>Оценка</th><th>Что услышал распознаватель</th><th></th></tr>{items}</table>""", encoding="utf-8")

def embed_into_html(html_path, clips):
    p = Path(html_path)
    s = p.read_text(encoding="utf-8")
    new = "window.DINO_CLIPS=" + json.dumps(clips, ensure_ascii=False) + ";</script>"
    if "window.DINO_CLIPS=" in s:
        s2 = re.sub(r"window\.DINO_CLIPS=\{.*?\};</script>", lambda m: new, s, count=1, flags=re.S)
    else:
        sys.exit("В html нет блока window.DINO_CLIPS — это не офлайн-версия игры.")
    out = p.with_name(p.stem + "-новый-голос" + p.suffix)
    out.write_text(s2, encoding="utf-8")
    return out

# ---------------------------------------------------------------- главный цикл
def main():
    ap = argparse.ArgumentParser(description="Озвучка фраз игры с проверкой распознаванием речи")
    src = ap.add_argument_group("откуда брать фразы")
    src.add_argument("--phrases", nargs="*", default=[], help="txt-файлы, одна фраза на строку")
    src.add_argument("--from-html", help="html-файл игры: фразы будут извлечены автоматически")
    ap.add_argument("--engine", choices=["silero", "f5"], default="silero")
    ap.add_argument("--speaker", default="xenia", help="голос Silero: xenia, kseniya, baya, aidar, eugene")
    ap.add_argument("--takes", type=int, default=5, help="сколько дублей пробовать на фразу (для F5)")
    ap.add_argument("--good", type=float, default=0.97, help="оценка, после которой дальше не пробуем")
    ap.add_argument("--pitch", type=float, default=1.12, help="выше голос: 1.0 — как есть, 1.15 — мультяшнее")
    ap.add_argument("--tempo", type=float, default=0.95, help="темп речи: меньше 1 — медленнее")
    ap.add_argument("--stress", default="stress.txt", help="словарь ударений")
    ap.add_argument("--out", default="out")
    ap.add_argument("--clips-in", help="существующий clips.json игры: звуки игры (sfx_*) и готовые фразы сохранятся")
    ap.add_argument("--only", help="переозвучить только фразы, содержащие эту подстроку")
    ap.add_argument("--redo-below", type=float, help="переозвучить только фразы с прошлой оценкой ниже этой")
    ap.add_argument("--no-check", action="store_true", help="не проверять распознаванием (быстрее)")
    ap.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--embed-html", help="офлайн html игры: сохранить копию с новой озвучкой")
    f5 = ap.add_argument_group("F5-TTS")
    f5.add_argument("--ref-audio", help="образец голоса, wav 5–12 секунд, чистый, без музыки")
    f5.add_argument("--ref-text", help="точный текст образца")
    f5.add_argument("--f5-repo", default="Misha24-10/F5-TTS_RUSSIAN", help="репозиторий Hugging Face с русским чекпойнтом")
    f5.add_argument("--f5-model", default="F5TTS_v1_Base")
    f5.add_argument("--f5-ckpt"); f5.add_argument("--f5-vocab")
    f5.add_argument("--f5-speed", type=float, default=0.95)
    args = ap.parse_args()

    phrases = phrases_from_txt(args.phrases)
    if args.from_html:
        for p in phrases_from_html(args.from_html):
            if p not in phrases:
                phrases.append(p)
    if not phrases:
        sys.exit("Нет фраз: укажите --phrases или --from-html")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " — будет медленно"))

    out_dir = Path(args.out); (out_dir / "clips").mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.csv"
    prev = {}
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["score"] = float(r["score"]); prev[r["key"]] = r

    clips = {}
    if args.clips_in:
        clips = json.loads(Path(args.clips_in).read_text(encoding="utf-8"))
    cj = out_dir / "clips.json"
    if cj.exists():
        clips.update(json.loads(cj.read_text(encoding="utf-8")))

    def needs(text):
        k = hkey(text)
        if args.only:
            return args.only.lower() in text.lower()
        if args.redo_below is not None:
            return k not in prev or prev[k]["score"] < args.redo_below
        return not (out_dir / "clips" / f"{k}.mp3").exists()

    todo = [t for t in phrases if needs(t)]
    print(f"Фраз всего: {len(phrases)}, озвучить сейчас: {len(todo)}")

    stress = load_stress(args.stress)
    engine = SileroEngine(args.speaker, device) if args.engine == "silero" else F5Engine(args, device)
    checker = None if args.no_check else Checker(device, args.asr_model)

    from tqdm import tqdm
    t0 = time.time()
    for text in tqdm(todo, unit="фраза"):
        k = hkey(text)
        best = (-9.0, None, None, "")
        variants = engine.variants(text, stress)
        tries = []
        if args.engine == "silero":
            tries = [(v, 0) for v in variants]
        else:
            tries = [(variants[0], i) for i in range(args.takes)]
        for v, take in tries:
            try:
                y, sr = engine.synth(v, take)
            except Exception as e:
                print(f"\n  ошибка синтеза «{text[:40]}…»: {e}"); continue
            if checker:
                heard = checker(y, sr); sc = score(text, heard, KEYWORDS)
            else:
                heard, sc = "", 1.0
            if sc > best[0]:
                best = (sc, y, sr, heard)
            if sc >= args.good:
                break
        if best[1] is None:
            continue
        y, sr = postprocess(best[1], best[2], args.pitch, args.tempo)
        mp3 = to_mp3(y, sr)
        (out_dir / "clips" / f"{k}.mp3").write_bytes(mp3)
        clips[k] = "data:audio/mpeg;base64," + base64.b64encode(mp3).decode()
        prev[k] = {"key": k, "text": text, "score": round(best[0], 3), "heard": best[3], "engine": args.engine}

    # сохраняем результаты
    rows = sorted([prev[hkey(t)] for t in phrases if hkey(t) in prev], key=lambda r: r["score"])
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "text", "score", "heard", "engine"]); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ["key", "text", "score", "heard", "engine"]})
    cj.write_text(json.dumps(clips, ensure_ascii=False), encoding="utf-8")
    write_review(out_dir, rows)

    weak = [r for r in rows if r["score"] < 0.9]
    print(f"\nГотово за {time.time() - t0:.0f} с. Фраз с хорошей оценкой: {len(rows) - len(weak)} из {len(rows)}.")
    for r in weak[:15]:
        print(f"  {r['score']:.2f}  {r['text']}  →  {r['heard']}")
    print(f"Файлы: {cj}  (озвучка для игры), {out_dir / 'review.html'} (прослушать), {report_path}")
    if args.embed_html:
        p = embed_into_html(args.embed_html, clips)
        print(f"Офлайн-версия игры с новым голосом: {p}")

if __name__ == "__main__":
    main()
