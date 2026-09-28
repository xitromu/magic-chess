#!/usr/bin/env bash
# Установка окружения для озвучки. Запуск: bash setup_linux.sh      (только Silero)
#                                         bash setup_linux.sh f5   (Silero + F5-TTS)
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
if [ "$1" = "f5" ]; then pip install f5-tts; fi
# RTX 50xx (Blackwell) нужен PyTorch со сборкой CUDA 12.8 — ставим последним
pip install --upgrade --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch;print('torch',torch.__version__,'| CUDA:',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU не найден')"
echo "Готово. Дальше: source .venv/bin/activate && python voice_tool.py --help"
