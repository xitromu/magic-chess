#!/usr/bin/env python3
"""
Сборка офлайн-версии «Волшебных шахмат» в один файл.

Берёт код игры (game/index.html) и озвучку (game/clips.json),
вшивает озвучку в html и сохраняет dist/Волшебные-шахматы.html —
его можно открыть двойным щелчком или отправить на телефон.

Запуск:  python build.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "game" / "index.html"
CLIPS = ROOT / "game" / "clips.json"
OUT = ROOT / "dist" / "Волшебные-шахматы.html"
SLOT = "window.DINO_CLIPS={};</script>"


def main():
    html = SRC.read_bytes().decode("utf-8")
    clips = CLIPS.read_bytes().decode("utf-8").strip()
    if html.count(SLOT) != 1:
        sys.exit(f"В {SRC.name} должно быть ровно одно место для озвучки: {SLOT}")
    if not clips.startswith("{"):
        sys.exit(f"{CLIPS.name} не похож на озвучку (ожидается JSON-объект)")
    out = html.replace(SLOT, "window.DINO_CLIPS=" + clips + ";</script>")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(out.encode("utf-8"))
    print(f"Готово: {OUT}  ({len(out.encode('utf-8')) / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
