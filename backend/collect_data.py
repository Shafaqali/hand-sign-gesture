"""
collect_data.py
------------------------------------------------------------------
Builds your training dataset by recording YOUR OWN hand landmarks
for each sign, straight from the webcam. This is the real-world
approach used for custom sign-language recognition: there is no
universal pretrained ISL/ASL landmark-classifier model, so the
system learns from data you provide (exactly what "Proposed
Solution" in your synopsis describes).

HOW TO USE
------------------------------------------------------------------
1. Edit the GESTURES list below to match the words/letters you want
   to recognize (defaults to a starter set of common ISL/ASL words).
2. Run:  python collect_data.py
3. A camera window opens. Press the NUMBER key shown on screen for
   the gesture you are about to perform, hold the pose steady, and
   the tool auto-captures ~5 samples per second while you hold the
   key... actually simpler: press 'c' to start/stop capturing for
   the CURRENTLY SELECTED gesture (selected with number keys).
4. Repeat for every gesture. Aim for 150-300 samples per gesture,
   captured with slightly varied hand angle/distance for a model
   that generalizes.
5. Press 'q' to quit and save. Data is appended to
   backend/data/landmarks.csv
6. Run train_model.py next.
------------------------------------------------------------------
"""

import csv
import os
import cv2
import mediapipe as mp

from utils import normalize_landmarks, landmarks_from_mediapipe

# ---- EDIT THIS LIST to the signs you want to train ----
GESTURES = [
    "come here", "hello", "thanks", "yes", "no", "please",
    "sorry", "help", "love", "stop", "ok",
]
# ---------------------------------------------------------

# Assigns a selection key to each gesture: '0'-'9' first, then 'a','b','c'...
# for any gesture beyond the 10th. This lets GESTURES have more than 10
# entries (fixes the old 0-9-only limit, which silently left extra
# gestures with no way to select them).
def build_key_map(gestures):
    keys = [str(i) for i in range(10)] + [chr(c) for c in range(ord('a'), ord('z') + 1)]
    if len(gestures) > len(keys):
        raise ValueError(f"Too many gestures ({len(gestures)}) — max supported is {len(keys)}.")
    return {keys[i]: i for i in range(len(gestures))}

KEY_TO_INDEX = build_key_map(GESTURES)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")


def ensure_csv():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"f{i}" for i in range(63)]
            writer.writerow(header)


def main():
    ensure_csv()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Close other apps using the "
              "camera (Zoom/Teams/Browser) and try again.")
        return

    selected_idx = 0
    capturing = False
    counts = {g: 0 for g in GESTURES}

    # Load existing counts so re-running the tool doesn't lose progress info
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1

    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)

    key_list_str = " ".join(f"[{k}]" for k in KEY_TO_INDEX.keys())
    print(f"Controls: {key_list_str} select gesture | [c] toggle capture | [q] quit & save")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                if capturing:
                    coords = landmarks_from_mediapipe(hand_lm)
                    features = normalize_landmarks(coords)
                    label = GESTURES[selected_idx]
                    writer.writerow([label] + features.tolist())
                    counts[label] += 1

            # ---- HUD ----
            y = 24
            cv2.putText(frame, f"Selected: {GESTURES[selected_idx]}  "
                                f"(count: {counts[GESTURES[selected_idx]]})",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
            y += 28
            state_txt = "CAPTURING..." if capturing else "paused (press c)"
            color = (0, 0, 255) if capturing else (200, 200, 200)
            cv2.putText(frame, state_txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
            y += 26
            for key_char, idx in KEY_TO_INDEX.items():
                g = GESTURES[idx]
                cv2.putText(frame, f"[{key_char}] {g} ({counts[g]})", (10, y + idx * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow("SignSpeak AI - Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF
            key_char = chr(key) if 0 <= key < 256 else ''

            if key_char == 'q':
                break
            elif key_char == 'c':
                capturing = not capturing
            elif key_char in KEY_TO_INDEX:
                selected_idx = KEY_TO_INDEX[key_char]
                capturing = False
    finally:
        csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("\nSaved. Sample counts:")
        for g, c in counts.items():
            print(f"  {g}: {c}")
        print(f"\nData file: {CSV_PATH}")
        print("Next step: python train_model.py")


if __name__ == "__main__":
    main()