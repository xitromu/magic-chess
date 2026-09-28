#!/usr/bin/env bash
# Установка окружения для озвучки. Запуск: bash setup_linux.sh      (только Silero)
#                                         bash setup_linux.sh f5   (Silero + F5-TTS)
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
# Сначала PyTorch с CUDA 12.8 (нужен для RTX 50xx), версия закреплена в constraints.txt
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -c constraints.txt
if [ "$1" = "f5" ]; then pip install f5-tts -c constraints.txt; fi
# torchcodec тянет f5-tts, но не использует; с torch 2.8 он не грузится и ломает проверку распознаванием
pip uninstall -y torchcodec
python -c "import torch;print('torch',torch.__version__,'| CUDA:',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU не найден')"
echo "Готово. Дальше: source .venv/bin/activate && python voice_tool.py --help"
