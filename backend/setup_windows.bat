@echo off
echo ============================================
echo   SignSpeak AI - Windows Setup
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found on PATH.
    echo Install Python 3.10 or 3.11 ^(64-bit^) from https://www.python.org/downloads/
    echo IMPORTANT: check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

echo Creating virtual environment (venv)...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies (this can take a few minutes)...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: pip install failed. Common fixes:
    echo   1. Make sure you installed 64-bit Python 3.10 or 3.11
    echo   2. Install "Microsoft Visual C++ Redistributable" if TensorFlow fails
    echo   3. Re-run this script after fixing the above
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Setup complete!
echo   Next steps:
echo     1. python collect_data.py   (record your gestures)
echo     2. python train_model.py    (train the AI model)
echo     3. start_server.bat         (run the app)
echo ============================================
pause
