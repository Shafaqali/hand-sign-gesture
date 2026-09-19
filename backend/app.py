"""
app.py
------------------------------------------------------------------
Flask backend for SignSpeak AI.

Architecture (important to understand):
  - Hand tracking (MediaPipe Hands) runs in the BROWSER (JavaScript),
    because that gives real-time webcam access with zero network lag.
  - The browser sends one or two hands of 21 (x,y,z) landmark points to this backend.
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
  POST /api/predict       -> body: {"hands": [[[x,y,z], ... x21], ... up to 2]}
                              returns: {"label": "hello", "confidence": 0.93}
  POST /api/speak         -> body: {"text": "hello there"}
                              returns: audio/mpeg (mp3 file)
------------------------------------------------------------------
"""

import os
import io
import json

import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

from utils import normalize_multi_hand_landmarks, pad_or_trim_features

BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "model", "gesture_model.h5")
NPZ_MODEL_PATH = os.path.join(BASE_DIR, "model", "gesture_model.npz")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.json")
METADATA_PATH = os.path.join(BASE_DIR, "model", "training_metadata.json")
MIN_CONFIDENCE = float(os.environ.get("PREDICT_MIN_CONFIDENCE", "0.55"))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ---- Lazy-load TensorFlow model (only if it exists) ----
_model = None
_npz_model = None
_labels = None
_model_error = None


def model_files_status():
    model_exists = os.path.exists(MODEL_PATH)
    npz_model_exists = os.path.exists(NPZ_MODEL_PATH)
    labels_exists = os.path.exists(LABELS_PATH)
    return {
        "model_file_exists": model_exists,
        "npz_model_file_exists": npz_model_exists,
        "labels_file_exists": labels_exists,
        "model_ready": (model_exists or npz_model_exists) and labels_exists,
    }


def labels_from_disk_or_metadata():
    if _labels is not None:
        return _labels
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH) as f:
                return json.load(f).get("labels", [])
        except Exception:
            pass
    return []


def load_model_if_available():
    global _model, _npz_model, _labels, _model_error
    if _model is not None or _npz_model is not None:
        return True
    if not ((os.path.exists(NPZ_MODEL_PATH) or os.path.exists(MODEL_PATH)) and os.path.exists(LABELS_PATH)):
        _model_error = "Model file or labels file is missing on the server."
        return False
    try:
        with open(LABELS_PATH) as f:
            _labels = json.load(f)
        if os.path.exists(NPZ_MODEL_PATH):
            with np.load(NPZ_MODEL_PATH) as data:
                _npz_model = {key: data[key].astype(np.float32) for key in data.files}
            _model_error = None
            print(f"[SignSpeak AI] Loaded NumPy model with {len(_labels)} gestures: {_labels}")
            return True

        # Imported here so the server still starts even if tensorflow isn't
        # installed correctly, with a clear error only when prediction is used.
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        _model_error = None
        print(f"[SignSpeak AI] Loaded TensorFlow model with {len(_labels)} gestures: {_labels}")
        return True
    except Exception as exc:
        _model = None
        _npz_model = None
        _labels = None
        _model_error = str(exc)
        print(f"[SignSpeak AI] Model load failed: {_model_error}")
        return False


def predict_with_numpy_model(features):
    x = np.asarray(features, dtype=np.float32).reshape(1, -1)

    def dense(idx, values):
        return values @ _npz_model[f"dense_{idx}_kernel"] + _npz_model[f"dense_{idx}_bias"]

    def batch_norm(idx, values):
        gamma = _npz_model[f"bn_{idx}_gamma"]
        beta = _npz_model[f"bn_{idx}_beta"]
        mean = _npz_model[f"bn_{idx}_mean"]
        var = _npz_model[f"bn_{idx}_var"]
        epsilon = float(_npz_model[f"bn_{idx}_epsilon"])
        return gamma * (values - mean) / np.sqrt(var + epsilon) + beta

    x = np.maximum(dense(0, x), 0)
    x = batch_norm(0, x)
    x = np.maximum(dense(1, x), 0)
    x = batch_norm(1, x)
    x = np.maximum(dense(2, x), 0)
    logits = dense(3, x)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / np.sum(exp, axis=1, keepdims=True))[0]


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_file(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/health")
def health():
    metadata = {}
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH) as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    file_status = model_files_status()
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None or _npz_model is not None,
        "numpy_model_loaded": _npz_model is not None,
        "model_ready": file_status["model_ready"],
        "model_file_exists": file_status["model_file_exists"],
        "npz_model_file_exists": file_status["npz_model_file_exists"],
        "labels_file_exists": file_status["labels_file_exists"],
        "gestures": labels_from_disk_or_metadata(),
        "model_error": _model_error,
        "min_confidence": MIN_CONFIDENCE,
        "training_metadata": metadata,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if not load_model_if_available():
        return jsonify({
            "error": "No trained model found yet. Run collect_data.py then "
                      "train_model.py in the backend folder first.",
            "model_error": _model_error,
        }), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must include 'hands' or 'landmarks'."}), 400

    hands = data.get("hands")
    if hands is None:
        landmarks = data.get("landmarks")
        if not landmarks:
            return jsonify({"error": "Request body must include 'hands' or 'landmarks'."}), 400
        if len(landmarks) == 21 and isinstance(landmarks[0], list) and len(landmarks[0]) == 3:
            hands = [landmarks]
        else:
            hands = landmarks

    if not isinstance(hands, list) or len(hands) == 0:
        return jsonify({"error": "Expected at least one detected hand."}), 400
    if len(hands) > 2:
        hands = hands[:2]
    for hand in hands:
        if not isinstance(hand, list) or len(hand) != 21:
            return jsonify({"error": "Each hand must contain exactly 21 landmarks."}), 400

    try:
        features = normalize_multi_hand_landmarks(hands)
        expected_shape = _npz_model["dense_0_kernel"].shape[0] if _npz_model is not None else _model.input_shape[-1]
        if expected_shape is not None and expected_shape != len(features):
            features = pad_or_trim_features(features, int(expected_shape))
        if _npz_model is not None:
            preds = predict_with_numpy_model(features)
        else:
            preds = _model.predict(features.reshape(1, -1), verbose=0)[0]
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
