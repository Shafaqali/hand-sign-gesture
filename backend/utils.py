"""
Shared landmark helpers for data collection, training, and prediction.

The server model now supports one-hand and two-hand signs. Every sample is
converted to a fixed 126-number vector:
  hand 1: 21 landmarks * x/y/z = 63
  hand 2: 21 landmarks * x/y/z = 63

If only one hand is visible, the second half is zero padded. If two hands are
visible, they are sorted left-to-right by wrist x position so collection and
live prediction use the same ordering.
"""

import numpy as np

NUM_LANDMARKS = 21
HAND_FEATURES = NUM_LANDMARKS * 3
MAX_HANDS = 2
NUM_FEATURES = HAND_FEATURES * MAX_HANDS


def normalize_landmarks(landmark_list):
    """
    Normalize one MediaPipe hand into 63 translation/scale invariant features.
    This function is kept for backward compatibility with old 1-hand code.
    """
    pts = np.array(landmark_list, dtype=np.float32).reshape(NUM_LANDMARKS, 3)

    wrist = pts[0].copy()
    pts -= wrist

    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist < 1e-6:
        max_dist = 1e-6
    pts /= max_dist

    return pts.flatten().astype(np.float32)


def normalize_multi_hand_landmarks(hands_landmarks, max_hands=MAX_HANDS):
    """
    Normalize 0-2 hands into one fixed-length feature vector.

    Input shape:
      [
        [(x,y,z), ... 21 points],
        [(x,y,z), ... 21 points],
      ]
    """
    hands = list(hands_landmarks or [])[:max_hands]
    hands = sorted(hands, key=lambda hand: float(hand[0][0]) if hand else 0.0)

    features = []
    for hand in hands:
        features.extend(normalize_landmarks(hand).tolist())

    missing = max_hands - len(hands)
    if missing > 0:
        features.extend([0.0] * HAND_FEATURES * missing)

    return np.array(features, dtype=np.float32)


def pad_or_trim_features(values, size=NUM_FEATURES):
    """Make older 63-feature rows and newer 126-feature rows train together."""
    arr = np.array(values, dtype=np.float32).flatten()
    if len(arr) >= size:
        return arr[:size].astype(np.float32)
    return np.pad(arr, (0, size - len(arr)), mode="constant").astype(np.float32)


def landmarks_from_mediapipe(hand_landmarks):
    """Convert one MediaPipe hand object into plain (x, y, z) tuples."""
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]


def multi_hand_landmarks_from_mediapipe(multi_hand_landmarks):
    """Convert a MediaPipe multi-hand result list into plain nested tuples."""
    return [landmarks_from_mediapipe(hand) for hand in (multi_hand_landmarks or [])]
