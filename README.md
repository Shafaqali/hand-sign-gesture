# SignSpeak AI

Real-time hand sign gesture recognition web app with a browser frontend and a Python Flask backend.

## Tech Stack

- Frontend: HTML, CSS, JavaScript, MediaPipe Hands
- Backend: Python, Flask
- AI/ML: TensorFlow, scikit-learn, NumPy
- Computer Vision: OpenCV, MediaPipe
- Voice: Browser Web Speech API and gTTS backend endpoint
- Deployment: GitHub + Render

## How This Project Works

The website UI is built in:

```text
frontend/index.html
```

The Python backend is built in:

```text
backend/app.py
```

The frontend opens the camera in the browser, tracks hand landmarks, and can detect gestures in two ways:

- Browser mode: works directly in the frontend using built-in rules.
- Server AI mode: sends hand landmarks to Flask, where TensorFlow predicts the trained sign.

Important: there is no universal trained sign-language model included by default. For Server AI mode, you collect your own gesture data and train your own model.

## Project Structure

```text
signspeak-ai/
  backend/
    app.py                  Flask server and API routes
    requirements.txt         Python dependencies
    setup_windows.bat        Creates venv and installs dependencies
    start_server.bat         Starts Flask server locally
    collect_data.py          Webcam data collection script
    collect_data.bat         Runs collect_data.py on Windows
    train_model.py           Trains TensorFlow gesture model
    train_model.bat          Runs train_model.py on Windows
    utils.py                 Landmark normalization helpers
    data/.gitkeep            Training data folder placeholder
    model/.gitkeep           Trained model folder placeholder
  frontend/
    index.html               Main website UI
    sign.html                Extra frontend page
    newsign.html             Extra frontend page
  .python-version            Python version for Render
  requirements.txt           Root Render compatibility requirements file
  render.yaml                Render deployment config
  README.md
```

## Local Setup in VS Code

Open the project folder in VS Code:

```text
C:\Users\kashaf\Downloads\signspeak-ai\signspeak-ai
```

Open VS Code terminal and go to the backend folder:

```powershell
cd backend
```

Run setup first:

```powershell
.\setup_windows.bat
```

This creates `backend/venv` and installs all Python packages from `backend/requirements.txt`.

Start the local server:

```powershell
.\start_server.bat
```

Then open this URL in Chrome or Edge:

```text
http://127.0.0.1:5000
```

Allow camera permission in the browser.

## Local Run Workflow

Use this when you only want to run the website locally:

```powershell
cd C:\Users\kashaf\Downloads\signspeak-ai\signspeak-ai\backend
.\setup_windows.bat
.\start_server.bat
```

After server starts:

```text
http://127.0.0.1:5000
```

The browser will load `frontend/index.html`, but it is served through Python Flask.

## Training Workflow

Use this when you want to train your own signs for Server AI mode.

Go to backend:

```powershell
cd C:\Users\kashaf\Downloads\signspeak-ai\signspeak-ai\backend
```

Collect gesture data:

```powershell
.\collect_data.bat
```

During data collection:

- A webcam window opens.
- Press number keys shown on screen to select a gesture.
- Press `c` to start or stop recording samples.
- Press `q` to save and quit.
- Try to collect 150-300 samples per gesture.

Train the model:

```powershell
.\train_model.bat
```

After training, these files are created:

```text
backend/model/gesture_model.h5
backend/model/labels.json
```

Start the server again:

```powershell
.\start_server.bat
```

Open:

```text
http://127.0.0.1:5000
```

Then select `Server AI` mode in the frontend.

## Deploying to GitHub

The project is connected to:

```text
https://github.com/Shafaqali/hand-sign-gesture.git
```

Normal push workflow:

```powershell
cd C:\Users\kashaf\Downloads\signspeak-ai\signspeak-ai
git status
git add .
git commit -m "Update project"
git push
```

Note: `backend/venv`, cache files, local `.env` files, generated training data, and generated model files are ignored by `.gitignore`.

If you want to upload the trained model to GitHub, force-add it:

```powershell
git add -f backend/model/gesture_model.h5 backend/model/labels.json
git commit -m "Add trained gesture model"
git push
```

If `gesture_model.h5` is larger than GitHub's file limit, use Git LFS or host the model separately.

## Render Deployment

Render can deploy this project from GitHub.

Recommended Render settings:

```text
Runtime: Python
Branch: main
Root Directory: leave blank
Build Command: pip install -r requirements.txt
Start Command: gunicorn --chdir backend app:app
```

This repo also includes `render.yaml`, so Render Blueprint can detect the service config automatically.

## Render Training Note

Do not train the model on Render.

Training needs webcam access, and Render servers cannot access your laptop camera. Train locally in VS Code, generate the model files, then push the trained model files if you want Server AI mode to work online.

## API Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Serves `frontend/index.html` |
| `/api/health` | GET | Checks backend and model status |
| `/api/predict` | POST | Predicts gesture from hand landmarks |
| `/api/speak` | POST | Converts text to MP3 using gTTS |

## Common Problems

### Render says `requirements.txt` not found

Use:

```text
Build Command: pip install -r requirements.txt
```

This root file points to `backend/requirements.txt`.

### Render uses Python 3.14

This project includes:

```text
.python-version
```

Current pinned version:

```text
3.11.11
```

### No trained model found

Run locally:

```powershell
cd backend
.\collect_data.bat
.\train_model.bat
```

Then start again:

```powershell
.\start_server.bat
```

### Camera does not open

- Use Chrome or Edge.
- Allow camera permission.
- Close Zoom, Teams, or any other app using the camera.
- Reload the page.

### TensorFlow install fails

Use 64-bit Python 3.10 or 3.11. Python 3.12+ can cause compatibility issues with TensorFlow/MediaPipe on some systems.

## Quick Commands

Local setup:

```powershell
cd backend
.\setup_windows.bat
```

Local run:

```powershell
.\start_server.bat
```

Collect data:

```powershell
.\collect_data.bat
```

Train model:

```powershell
.\train_model.bat
```

Push changes:

```powershell
git add .
git commit -m "Update project"
git push
```
