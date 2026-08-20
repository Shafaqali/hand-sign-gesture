"""
collect_data.py
------------------------------------------------------------------
Records your own webcam hand landmarks for the Server AI model.

Flow:
  1. Select a gesture with the shown keyboard key.
  2. Press c to start/stop capturing samples for that gesture.
  3. Collect 150-300 varied samples per gesture.
  4. Press q to save/quit.
  5. Run train_model.py.

The browser rule engine can guess a few signs without training, but the
server TensorFlow model must learn from real labeled samples. This script
keeps the labels clean and consistent so the trained model works properly.
------------------------------------------------------------------
"""

import csv
import os
import time

import cv2
import mediapipe as mp

from gestures import GESTURES, canonical_label, validate_gestures
from utils import (
    NUM_FEATURES,
    multi_hand_landmarks_from_mediapipe,
    normalize_multi_hand_landmarks,
    pad_or_trim_features,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
SAMPLES_PER_SECOND = 8


def build_key_map(gestures):
    keys = (
        [str(i) for i in range(10)]
        + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        + list("!@#$%^&*()-_=+[]{};,./?<>")
    )
    if len(gestures) > len(keys):
        raise ValueError(f"Too many gestures ({len(gestures)}) - max supported is {len(keys)}.")
    return {keys[i]: i for i in range(len(gestures))}


def ensure_csv():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"f{i}" for i in range(NUM_FEATURES)]
            writer.writerow(header)
        return

    with open(CSV_PATH, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])
        return

    current_features = max(0, len(rows[0]) - 1)
    if current_features == NUM_FEATURES:
        return

    backup_path = CSV_PATH.replace(".csv", f"_backup_{current_features}_features.csv")
    os.replace(CSV_PATH, backup_path)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])
        for row in rows[1:]:
            if not row:
                continue
            label = canonical_label(row[0])
            if label not in GESTURES:
                continue
            features = pad_or_trim_features(row[1:], NUM_FEATURES)
            writer.writerow([label] + features.tolist())
    print(f"Migrated CSV from {current_features} to {NUM_FEATURES} features.")
    print(f"Backup saved: {backup_path}")


def load_existing_counts():
    counts = {gesture: 0 for gesture in GESTURES}
    if not os.path.exists(CSV_PATH):
        return counts

    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            label = canonical_label(row[0])
            if label in counts:
                counts[label] += 1
    return counts


def draw_hud(frame, key_to_index, selected_idx, capturing, counts):
    selected = GESTURES[selected_idx]
    cv2.putText(
        frame,
        f"Selected: {selected}  (count: {counts[selected]})",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 180, 255),
        2,
    )

    state_txt = "CAPTURING..." if capturing else "paused (press c)"
    state_color = (0, 0, 255) if capturing else (210, 210, 210)
    cv2.putText(frame, state_txt, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
    cv2.putText(
        frame,
        "Select sign key | c capture | q quit",
        (10, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    start_y = 106
    for key_char, idx in key_to_index.items():
        gesture = GESTURES[idx]
        col = idx % 3
        row = idx // 3
        x = 10 + col * 210
        y = start_y + row * 20
        color = (0, 220, 255) if idx == selected_idx else (235, 235, 235)
        cv2.putText(
            frame,
            f"[{key_char}] {gesture} ({counts[gesture]})",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )


def main():
    validate_gestures()
    key_to_index = build_key_map(GESTURES)
    ensure_csv()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.55,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Close other apps using the camera and try again.")
        return

    selected_idx = 0
    capturing = False
    counts = load_existing_counts()
    last_capture_at = 0.0

    print("Server AI data collection")
    print("-------------------------")
    print("Gestures:")
    print("  " + " | ".join(f"{key}:{GESTURES[idx]}" for key, idx in key_to_index.items()))
    print("Controls: key selects gesture | c toggles capture | q saves and quits")
    print(f"Capture rate: {SAMPLES_PER_SECOND} samples/sec")
    print("Target: 150-300 samples per gesture, with small angle/distance/one-hand/two-hand variations.")
    print("Two-hand signs are supported. Keep both hands visible while capturing those signs.")

    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_lm in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                now = time.time()
                if capturing and now - last_capture_at >= 1 / SAMPLES_PER_SECOND:
                    hands_coords = multi_hand_landmarks_from_mediapipe(result.multi_hand_landmarks)
                    features = normalize_multi_hand_landmarks(hands_coords)
                    label = GESTURES[selected_idx]
                    writer.writerow([label] + features.tolist())
                    counts[label] += 1
                    last_capture_at = now

            draw_hud(frame, key_to_index, selected_idx, capturing, counts)
            cv2.imshow("MR Voxa - Server AI Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            key_char = chr(key) if 0 <= key < 256 else ""

            if key_char == "q":
                break
            if key_char == "c":
                capturing = not capturing
            elif key_char in key_to_index:
                selected_idx = key_to_index[key_char]
                capturing = False
    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("\nSaved. Sample counts:")
        for gesture in GESTURES:
            print(f"  {gesture}: {counts[gesture]}")
        print(f"\nData file: {CSV_PATH}")
        print("Next step: python train_model.py")


if __name__ == "__main__":
    main()
