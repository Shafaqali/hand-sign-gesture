"""
utils.py
Shared helper functions used by collect_data.py, train_model.py and app.py.
Keeping this logic in ONE place guarantees that data collected, the model
trained, and the predictions made at inference time all use the exact same
math -- this is the #1 cause of "my model works in training but not live"
bugs, so we centralize it here.
"""

import numpy as np

NUM_LANDMARKS = 21          # MediaPipe Hands always returns 21 points per hand
NUM_FEATURES = NUM_LANDMARKS * 3   # x, y, z per point = 63 features


def normalize_landmarks(landmark_list):
    """
    Takes a list of 21 (x, y, z) tuples (MediaPipe raw, already 0-1 normalized
    to image size) and converts them into a translation- and scale-invariant
    feature vector of length 63.

    Why this matters: raw MediaPipe coordinates depend on WHERE in the frame
    your hand is and HOW CLOSE it is to the camera. Two identical gestures
    performed in different spots would look totally different to a model
    trained on raw coordinates. We fix this by:
      1. Making the wrist (landmark 0) the origin  -> translation invariant
      2. Dividing by the max distance from wrist    -> scale invariant
    """
    pts = np.array(landmark_list, dtype=np.float32).reshape(NUM_LANDMARKS, 3)

    wrist = pts[0].copy()
    pts -= wrist  # translation invariance

    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist < 1e-6:
        max_dist = 1e-6
    pts /= max_dist  # scale invariance

    return pts.flatten().astype(np.float32)  # shape (63,)


def landmarks_from_mediapipe(hand_landmarks):
    """Convert a MediaPipe `hand_landmarks.landmark` object into a plain list
    of (x, y, z) tuples, ready for normalize_landmarks()."""
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
