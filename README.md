# SignSpeak AI — Real-Time Sign Language to Voice & Text Converter

Full-stack project matching your synopsis tech stack:
**Frontend:** HTML/CSS/JS (MediaPipe Hands) · **Backend:** Python + Flask ·
**AI:** TensorFlow · **Computer Vision:** OpenCV + MediaPipe Hands ·
**Speech:** Web Speech API (live) + gTTS (backend endpoint)

## How it actually works (read this first)

There is no universal pretrained "sign language" AI model — every real
SignSpeak-style system (including yours, per the synopsis) works by
**collecting hand-landmark samples for the signs you choose, then training a
small model on them.** This project gives you the full pipeline:

```
collect_data.py  →  train_model.py  →  app.py (Flask) serves the trained model
                                          ↑
                            frontend (browser) tracks your hand with
                            MediaPipe Hands and streams landmarks to it
```

Two detection engines are built into the frontend (toggle in the UI):
- **Browser (rules)** — works immediately, zero setup, geometric rule-based
  classifier already in the page (your uploaded design, untouched).
- **Server AI (TensorFlow)** — real ML model you train yourself, served by
  the Flask backend. This is the one that matches your document's stack.

---

## Windows Setup (step by step, copy-paste ready)

### 0. Prerequisites
- Install **Python 3.10 or 3.11 (64-bit)** from python.org.
  During install, tick **"Add python.exe to PATH"**.
  (TensorFlow/MediaPipe do not reliably support Python 3.12+ on Windows yet —
  using 3.10/3.11 avoids install errors.)
- Use **Google Chrome** or **Edge** (best WebRTC/camera support).

### 1. Open the project folder
Extract the zip, then open **Command Prompt** inside the `backend` folder
(Shift + Right-click → "Open PowerShell/Command window here").

### 2. One-click setup
Double-click **`setup_windows.bat`** (or run it from the terminal).
It creates a virtual environment and installs everything from
`requirements.txt`.

### 3. Collect your training data
Double-click **`collect_data.bat`**.
- A webcam window opens.
- Press number keys `0-9` to select which gesture you're about to sign
  (list shown on screen; edit `GESTURES` inside `collect_data.py` to use
  your own words).
- Press `c` to start/stop recording samples for the selected gesture.
  Hold the sign steady, move your hand slightly (angle/distance) between
  captures.
- Aim for **150–300 samples per gesture**.
- Press `q` to save and quit.

### 4. Train the model
Double-click **`train_model.bat`**.
Trains a TensorFlow model on your captured data and saves it to
`backend/model/gesture_model.h5`. Takes 1-3 minutes on a normal laptop.

### 5. Run the app
Double-click **`start_server.bat`**.
Open **http://127.0.0.1:5000** in Chrome.
- Click **Enable camera**, allow permission.
- In the viewfinder controls, set **Engine → Server AI (TensorFlow)**.
- Perform the signs you trained — text appears in the transcript panel and
  is spoken aloud automatically.

---

## Project structure

```
signspeak-ai/
├── backend/
│   ├── app.py              Flask server: /api/predict, /api/speak, /api/health
│   ├── utils.py             Landmark normalization (shared by all scripts)
│   ├── collect_data.py      Webcam tool to build your training dataset
│   ├── train_model.py       Trains the TensorFlow classifier
│   ├── requirements.txt
│   ├── setup_windows.bat    One-click environment setup
│   ├── start_server.bat     Run the app
│   ├── collect_data.bat / train_model.bat
│   ├── data/landmarks.csv   (created after step 3)
│   └── model/gesture_model.h5, labels.json   (created after step 4)
└── frontend/
    └── index.html           Your original UI + Server-AI engine toggle
```

## API Reference

| Endpoint        | Method | Body                                | Returns                              |
|------------------|--------|--------------------------------------|----------------------------------------|
| `/api/health`    | GET    | —                                    | `{status, model_loaded, gestures}`     |
| `/api/predict`   | POST   | `{"landmarks":[[x,y,z]×21]}`         | `{label, confidence, all_scores}`      |
| `/api/speak`     | POST   | `{"text":"hello there"}`             | MP3 audio (gTTS, needs internet)       |

## Troubleshooting (common Windows errors)

- **`pip install` fails on tensorflow** → You're likely on 32-bit Python or
  Python 3.12+. Reinstall 64-bit Python 3.10/3.11.
- **`DLL load failed while importing _pywrap_tensorflow_internal`** → Install
  "Microsoft Visual C++ Redistributable x64" from Microsoft's site, restart,
  retry.
- **Camera doesn't open in `collect_data.py`** → Close Zoom/Teams/Browser
  tabs using the camera; only one app can use it at a time.
- **Browser says camera blocked** → Click the padlock icon in Chrome's
  address bar → Site settings → allow Camera → reload the page.
- **"No trained model found yet"** on the Server AI engine → run
  `collect_data.bat` then `train_model.bat` before `start_server.bat`.
- **`ModuleNotFoundError`** → You ran `python app.py` without activating the
  virtual environment. Use the provided `.bat` files, they activate it for
  you automatically.
- **Low accuracy / wrong predictions** → Collect more samples per gesture
  (200+), keep your hand fully in frame, and vary angle/distance while
  recording so the model generalizes.

## Notes
- `/api/speak` (gTTS) needs internet; the frontend already speaks using the
  browser's **Web Speech API**, which works fully offline — so voice output
  never breaks even without internet.
- To add more gestures later: edit the `GESTURES` list in `collect_data.py`,
  collect samples for the new word, re-run `train_model.py` — no code
  changes needed elsewhere.
