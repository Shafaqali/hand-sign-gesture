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

## Complete Working Flow

This section explains what happens from opening the website to getting text/voice output.

### 1. User opens the website

When you open:

```text
http://127.0.0.1:5000
```

Flask runs from:

```text
backend/app.py
```

and serves the frontend file:

```text
frontend/index.html
```

So the visible website is HTML/CSS/JavaScript, but it is loaded through the Python Flask server.

### 2. Browser asks for camera permission

The frontend uses the browser camera. When the user clicks the camera/start button, Chrome or Edge asks for camera permission.

If permission is allowed, the video stream appears inside the website.

### 3. MediaPipe tracks the hand

The frontend loads MediaPipe Hands in `frontend/index.html`.

MediaPipe looks at each camera frame and detects hand landmark points. For one hand, MediaPipe gives 21 points:

```text
landmark 0  = wrist
landmark 4  = thumb tip
landmark 8  = index finger tip
landmark 12 = middle finger tip
landmark 16 = ring finger tip
landmark 20 = pinky finger tip
```

Each point has:

```text
x, y, z
```

These points describe the hand shape and finger positions.

### 4. Browser mode detects gestures without backend AI

In Browser mode, the frontend uses JavaScript rules inside `frontend/index.html`.

Example idea:

- If thumb/index/middle/ring/pinky are open or closed in a certain pattern, classify a sign.
- If two hands are visible, compare distance and position between both hands.
- For some motion signs like wave/J/Z, track fingertip movement over a short time.

This mode works immediately because it does not need a trained TensorFlow model.

## Alphabet Fingerspelling Mode

Alphabet mode builds words from confirmed letter gestures. When you sign letters with a short pause between words, the frontend formats the final text into readable words:

```text
T H E  -> The
H O M E -> Home
H E L L O  W O R L D -> Hello World
```

The browser rule model recognizes these alphabet gestures:

| Letter | Gesture shape |
| --- | --- |
| A | Closed fist, thumb resting on the side of the hand. |
| B | Four fingers straight up together, thumb folded across the palm. |
| C | Curved open hand shaped like the letter C. |
| D | Index finger straight up, other fingers folded, thumb near the folded middle finger. |
| E | Compact folded fingers with the thumb across the fingertips. Best-effort browser rule. |
| F | Thumb and index fingertip touching, other three fingers up. |
| G | Index finger and thumb held roughly parallel, pointing sideways. Best-effort browser rule. |
| H | Index and middle fingers together and pointing sideways. Best-effort browser rule. |
| I | Pinky finger up, other fingers folded. |
| J | Start from I and trace a J motion with the pinky. |
| K | Index and middle fingers up/spread, thumb held between them. Best-effort browser rule. |
| L | Index finger up and thumb out, making an L shape. |
| M | Closed fist with thumb tucked under three fingers. Best-effort browser rule. |
| N | Closed fist with thumb tucked under two fingers. Best-effort browser rule. |
| O | Fingers and thumb curved into an O loop. |
| P | K handshape angled downward. Best-effort browser rule. |
| Q | G handshape angled downward. Best-effort browser rule. |
| R | Index and middle fingers crossed or very close together. Best-effort browser rule. |
| S | Closed fist with thumb wrapped across the front of the fingers. |
| T | Closed fist with thumb tucked between index and middle finger. Best-effort browser rule. |
| U | Index and middle fingers straight up together. |
| V | Index and middle fingers straight up and spread apart. |
| W | Index, middle, and ring fingers straight up. |
| X | Index finger bent like a hook, other fingers folded. |
| Y | Thumb and pinky out, middle fingers folded. |
| Z | Point with index finger and trace a Z motion. |

Letters like G, K, M, N, P, Q, and R are harder for a normal 2D webcam because they rely on wrist angle and thumb/finger overlap. The browser includes best-effort rules for them, but Server AI mode with your own trained TensorFlow samples will be more accurate.

### 5. Server AI mode sends landmarks to Flask

In Server AI mode, the frontend does not directly decide the final sign. Instead, it sends the hand landmarks to Flask.

The frontend calls:

```text
POST /api/predict
```

with data like:

```json
{
  "landmarks": [
    [x, y, z],
    [x, y, z],
    "... 21 points total"
  ]
}
```

The backend receives this request in:

```text
backend/app.py
```

### 6. Backend normalizes the landmarks

Raw camera landmarks can change if:

- hand is closer or farther from camera
- hand is slightly moved left/right
- user has bigger/smaller hands

So backend uses:

```text
backend/utils.py
```

to normalize the 21 hand points. Normalization makes the data more consistent before prediction.

The same normalization is used during:

- data collection
- model training
- live prediction

This is important because the model should see data in the same format during training and real use.

### 7. TensorFlow model predicts the sign

After normalization, Flask loads the trained model from:

```text
backend/model/gesture_model.h5
```

and labels from:

```text
backend/model/labels.json
```

Then TensorFlow predicts the most likely sign and returns:

```json
{
  "label": "hello",
  "confidence": 0.93
}
```

The frontend receives this result and shows the detected word on the website.

### 8. Text and voice output

After a gesture is detected, the frontend updates:

- current detected sign
- transcript/conversation text
- history
- confidence/statistics

For voice output, the app can use:

- Browser Web Speech API for live speech in the frontend
- `/api/speak` backend endpoint using gTTS for MP3 audio

The browser speech method is usually faster for live use.

## Training Flow Explained

Training is only needed for Server AI mode.

### Step A: Choose gestures

The gesture list is inside:

```text
backend/collect_data.py
```

Look for:

```python
GESTURES = [
    "come here", "hello", "thanks", "yes", "no", "please",
    "sorry", "help", "love", "stop", "ok",
]
```

You can edit this list to train your own signs.

### Step B: Collect hand landmark samples

Run:

```powershell
.\collect_data.bat
```

This opens a webcam window. When you record a gesture, the script:

1. reads camera frames
2. detects hand landmarks using MediaPipe
3. normalizes the landmarks
4. saves the label and features into:

```text
backend/data/landmarks.csv
```

Each row in the CSV means:

```text
gesture label + normalized hand landmark numbers
```

### Step C: Train TensorFlow model

Run:

```powershell
.\train_model.bat
```

This script reads:

```text
backend/data/landmarks.csv
```

Then it trains a small TensorFlow classifier and saves:

```text
backend/model/gesture_model.h5
backend/model/labels.json
```

### Step D: Use trained model in website

Run:

```powershell
.\start_server.bat
```

Open:

```text
http://127.0.0.1:5000
```

Select Server AI mode. Now the website sends live landmarks to the backend, and the backend predicts using your trained model.

## Local vs Render Flow

### Local VS Code flow

Local machine is used for:

- opening camera
- collecting training data
- training model
- testing the app

Commands:

```powershell
cd backend
.\setup_windows.bat
.\start_server.bat
```

For training:

```powershell
.\collect_data.bat
.\train_model.bat
```

### Render flow

Render is used for:

- hosting the Flask backend
- serving `frontend/index.html`
- running `/api/health`, `/api/predict`, and `/api/speak`

Render is not used for:

- collecting webcam training data
- training from your laptop camera

Reason: Render server cannot access your laptop webcam. Training must be done locally, then trained model files can be pushed to GitHub if needed.

## Full Request Flow Diagram

```text
User opens Render/local URL
        |
        v
Flask backend serves frontend/index.html
        |
        v
Browser opens camera after permission
        |
        v
MediaPipe detects 21 hand landmarks
        |
        v
Frontend mode selected?
        |
        +-- Browser mode
        |       |
        |       v
        |   JavaScript rules classify gesture
        |
        +-- Server AI mode
                |
                v
            POST /api/predict
                |
                v
            Flask normalizes landmarks
                |
                v
            TensorFlow model predicts label
                |
                v
            Backend returns label + confidence
        |
        v
Frontend shows text, updates history, speaks output
```

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
