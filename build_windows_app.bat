@echo off
setlocal
cd /d "%~dp0"

echo =========================================
echo      Construindo Voxly Standalone 
echo =========================================

echo [1/4] Removendo pastas de build antigas, VENV e criando nova...
rmdir /s /q .venv
rmdir /s /q build
rmdir /s /q dist
rmdir /s /q Voxly_App_Standalone
python -m venv .venv
call .venv\Scripts\activate

echo [2/4] Instalando Dependencias Pesadas...
pip install pyinstaller faster-whisper websockets numpy sounddevice soundfile ctranslate2 transformers huggingface-hub piper-tts customtkinter darkdetect

echo [3/4] Compilando IA Python (PyInstaller)...
:: Faster Whisper e CTranslate requerem explicit collectors para nao quebrar no .exe
pyinstaller --noconfirm --onedir --collect-all faster_whisper --collect-all ctranslate2 --collect-all transformers --collect-data piper --name "transcriber" "bridge\transcriber.py"
pyinstaller --noconfirm --onedir --noconsole --collect-all customtkinter --name "Voxly_Launcher" "voxly_launcher_gui.py"

echo [4/4] Compilando Core Audio Rust em Release...
cargo build --release

echo [!] Montando pasta de distribuicao final...
rmdir /s /q Voxly_App_Standalone
mkdir Voxly_App_Standalone
mkdir Voxly_App_Standalone\bridge
mkdir Voxly_App_Standalone\data
mkdir Voxly_App_Standalone\models
mkdir Voxly_App_Standalone\tools

xcopy /E /I /y dist\Voxly_Launcher\* Voxly_App_Standalone\
xcopy /E /I /y dist\transcriber\* Voxly_App_Standalone\bridge\
copy target\release\Voxly.exe Voxly_App_Standalone\
xcopy /E /I /y models\* Voxly_App_Standalone\models\
xcopy /E /I /y tools\* Voxly_App_Standalone\tools\
copy voxly_config.json Voxly_App_Standalone\
xcopy /E /I /y data\* Voxly_App_Standalone\data\

echo =========================================
echo SUCESSO! Pasta 'Voxly_App_Standalone' gerada.
echo Agora voce pode zipar e enviar essa pasta, sem o python instalado na maquina dele!
echo Basta rodar o Voxly_Launcher.exe la dentro!
echo =========================================
:: pause
endlocal
