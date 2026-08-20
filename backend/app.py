"""
app.py
------------------------------------------------------------------
Flask backend for SignSpeak AI.

Architecture (important to understand):
  - Hand tracking (MediaPipe Hands) runs in the BROWSER (JavaScript),
    because that gives real-time webcam access with zero network lag.
  - The browser sends the 21 (x,y,z) landmark points to this backend.
  - This backend normalizes them (utils.py, same math used in training)
    and runs them through the TensorFlow model trained by
    train_model.py to get the predicted word/letter.
  - Optional: /api/speak turns any text into an MP3 using gTTS (needs
    internet) so you also satisfy the "gTTS" part of the tech stack.
    The frontend's Web Speech API voice output keeps working even
    with zero internet, as a live fallback.

Endpoints:
  GET  /                  -> serves the frontend (index.html)
  GET  /api/health        -> {"status": "ok", "model_loaded": bool}
  POST /api/predict       -> body: {"landmarks": [[x,y,z], ... x21]}
                              returns: {"label": "hello", "confidence": 0.93}
  POST /api/speak         -> body: {"text": "hello there"}
                              returns: audio/mpeg (mp3 file)
------------------------------------------------------------------
"""

import os
import io
import json

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

from utils import normalize_landmarks

BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "model", "gesture_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.json")
METADATA_PATH = os.path.join(BASE_DIR, "model", "training_metadata.json")
MIN_CONFIDENCE = float(os.environ.get("PREDICT_MIN_CONFIDENCE", "0.55"))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ---- Lazy-load TensorFlow model (only if it exists) ----
_model = None
_labels = None


def load_model_if_available():
    global _model, _labels
    if _model is not None:
        return True
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        return False
    # Imported here so the server still starts even if tensorflow isn't
    # installed correctly, with a clear error only when prediction is used.
    import tensorflow as tf
    _model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        _labels = json.load(f)
    print(f"[SignSpeak AI] Loaded model with {len(_labels)} gestures: {_labels}")
    return True


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    loaded = load_model_if_available()
    metadata = {}
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH) as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    return jsonify({
        "status": "ok",
        "model_loaded": loaded,
        "gestures": _labels if loaded else [],
        "min_confidence": MIN_CONFIDENCE,
        "training_metadata": metadata,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if not load_model_if_available():
        return jsonify({
            "error": "No trained model found yet. Run collect_data.py then "
                      "train_model.py in the backend folder first."
        }), 400

    data = request.get_json(silent=True)
    if not data or "landmarks" not in data:
        return jsonify({"error": "Request body must be {'landmarks': [[x,y,z], ...]}"}), 400

    landmarks = data["landmarks"]
    if len(landmarks) != 21:
        return jsonify({"error": f"Expected 21 landmarks, got {len(landmarks)}"}), 400

    try:
        features = normalize_landmarks(landmarks).reshape(1, -1)
        preds = _model.predict(features, verbose=0)[0]
        best_idx = int(preds.argmax())
        confidence = float(preds[best_idx])
        return jsonify({
            "label": _labels[best_idx],
            "confidence": confidence,
            "matched": confidence >= MIN_CONFIDENCE,
            "all_scores": {
                _labels[i]: float(p) for i, p in enumerate(preds)
            },
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/speak", methods=["POST"])
def speak():
    data = request.get_json(silent=True)
    if not data or "text" not in data or not data["text"].strip():
        return jsonify({"error": "Request body must be {'text': '...'}"}), 400

    text = data["text"].strip()

    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        buf.seek(0)
        return send_file(buf, mimetype="audio/mpeg", download_name="speech.mp3")
    except Exception as e:
        # gTTS needs internet access. Frontend already has a Web Speech API
        # fallback for offline voice output, so this failing is non-fatal.
        return jsonify({"error": f"gTTS failed (needs internet): {str(e)}"}), 500


if __name__ == "__main__":
    load_model_if_available()
    port = int(os.environ.get("PORT", 5000))
    print("\n  SignSpeak AI backend running:")
    print(f"  -> http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
