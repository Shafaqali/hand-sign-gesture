@echo off
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Virtual environment not found. Run setup_windows.bat first.
    pause
    exit /b 1
)
echo Starting SignSpeak AI backend...
echo Open http://127.0.0.1:5000 in Chrome once it says "Running on".
echo Press CTRL+C to stop the server.
echo.
python app.py
pause
