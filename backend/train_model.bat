@echo off
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Virtual environment not found. Run setup_windows.bat first.
    pause
    exit /b 1
)
python train_model.py
pause
