"""
train_model.py
------------------------------------------------------------------
Trains the Server AI TensorFlow model from backend/data/landmarks.csv.

Run after collect_data.py has produced enough samples. For reliable live use,
aim for 150-300 samples per gesture with varied distance, angle, and lighting.

Outputs:
  model/gesture_model.h5
  model/labels.json
  model/training_metadata.json
------------------------------------------------------------------
"""

import json
import os
import csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

from gestures import GESTURES, canonical_label, validate_gestures
from utils import NUM_FEATURES, pad_or_trim_features

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "data", "landmarks.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
METADATA_PATH = os.path.join(MODEL_DIR, "training_metadata.json")
MIN_TOTAL_SAMPLES = 20
MIN_SAMPLES_PER_CLASS = 2
RECOMMENDED_SAMPLES_PER_CLASS = 150


def load_clean_dataset():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found. Run collect_data.py first.")

    rows = []
    before = 0
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            before += 1
            if len(row) < 2:
                continue
            label = canonical_label(row[0])
            if label not in GESTURES:
                continue
            try:
                features = pad_or_trim_features(row[1:], NUM_FEATURES)
            except ValueError:
                continue
            rows.append((label, features))

    if not rows:
        return pd.DataFrame(columns=["label"] + [f"f{i}" for i in range(NUM_FEATURES)])

    labels = [label for label, _features in rows]
    feature_rows = np.vstack([features for _label, features in rows])
    feature_df = pd.DataFrame(feature_rows, columns=[f"f{i}" for i in range(NUM_FEATURES)])
    df = pd.concat([pd.DataFrame({"label": labels}), feature_df], axis=1)
    after = len(rows)

    if after < before:
        print(f"Dropped {before - after} invalid/old-label rows ({before} -> {after}).")
    return df


def build_model(input_dim, num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.35),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    validate_gestures()

    try:
        df = load_clean_dataset()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return

    if len(df) < MIN_TOTAL_SAMPLES:
        print("ERROR: Not enough samples yet. Collect more data before training.")
        return

    label_counts = df["label"].value_counts().sort_index()
    print("Samples per gesture:")
    print(label_counts.to_string())

    low_classes = label_counts[label_counts < MIN_SAMPLES_PER_CLASS]
    if not low_classes.empty:
        print("\nERROR: These gestures need at least 2 samples for validation split:")
        print(low_classes.to_string())
        return

    weak_classes = label_counts[label_counts < RECOMMENDED_SAMPLES_PER_CLASS]
    if not weak_classes.empty:
        print("\nWARNING: For proper live accuracy, collect more samples for:")
        print(weak_classes.to_string())
        print(f"Recommended: {RECOMMENDED_SAMPLES_PER_CLASS}+ samples per gesture.\n")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_dict = {int(i): float(w) for i, w in enumerate(class_weights_arr)}

    model = build_model(input_dim=X.shape[1], num_classes=num_classes)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=22,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=220,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc * 100:.1f}%")

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    report = classification_report(
        y_val,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0,
        output_dict=True,
    )
    print("\nPer-class report:")
    print(classification_report(y_val, y_pred, target_names=encoder.classes_, zero_division=0))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    labels = list(encoder.classes_)
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": CSV_PATH,
        "input_features": int(X.shape[1]),
        "max_hands": 2,
        "labels": labels,
        "sample_counts": {label: int(label_counts[label]) for label in labels},
        "validation_accuracy": float(val_acc),
        "validation_loss": float(val_loss),
        "epochs_ran": len(history.history.get("loss", [])),
        "recommended_samples_per_class": RECOMMENDED_SAMPLES_PER_CLASS,
        "classification_report": report,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model    -> {MODEL_PATH}")
    print(f"Saved labels   -> {LABELS_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")
    print("\nNow run: python app.py, open the frontend, and select Engine: Server AI.")


if __name__ == "__main__":
    main()
