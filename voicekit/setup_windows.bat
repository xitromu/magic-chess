@echo off
chcp 65001 >nul
REM Установка окружения для озвучки. Запуск: setup_windows.bat        (только Silero)
REM                                         setup_windows.bat f5     (Silero + F5-TTS)
cd /d %~dp0
where py >nul 2>nul && (py -3.11 -m venv .venv) || (python -m venv .venv)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip wheel
REM Сначала PyTorch с CUDA 12.8 (нужен для RTX 50xx), версия закреплена в constraints.txt
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -c constraints.txt
if /I "%1"=="f5" pip install f5-tts -c constraints.txt
REM torchcodec тянет f5-tts, но не использует; с torch 2.8 он не грузится и ломает проверку распознаванием
pip uninstall -y torchcodec
python -c "import torch;print('torch',torch.__version__,'| CUDA:',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU не найден')"
echo.
echo Готово. Дальше:  .venv\Scripts\activate  и  python voice_tool.py --help
pause
