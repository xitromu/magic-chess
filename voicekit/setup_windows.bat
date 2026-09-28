@echo off
chcp 65001 >nul
REM Установка окружения для озвучки. Запуск: setup_windows.bat        (только Silero)
REM                                         setup_windows.bat f5     (Silero + F5-TTS)
cd /d %~dp0
where py >nul 2>nul && (py -3.11 -m venv .venv) || (python -m venv .venv)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
if /I "%1"=="f5" pip install f5-tts
REM RTX 50xx (Blackwell) нужен PyTorch со сборкой CUDA 12.8 — ставим последним, чтобы ничто его не перезаписало
pip install --upgrade --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch;print('torch',torch.__version__,'| CUDA:',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU не найден')"
echo.
echo Готово. Дальше:  .venv\Scripts\activate  и  python voice_tool.py --help
pause
