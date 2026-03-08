@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERRO] Ambiente virtual nao encontrado em .venv\Scripts\python.exe
  echo Execute primeiro: python -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install faster-whisper websockets numpy
  pause
  exit /b 1
)

".venv\Scripts\python.exe" "voxly_launcher_gui.py"
endlocal
