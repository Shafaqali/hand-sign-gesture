"""
train_model.py
------------------------------------------------------------------
Trains a small dense neural network (TensorFlow/Keras) to classify
normalized hand-landmark vectors (63 floats) into gesture labels.

Run AFTER collect_data.py has produced backend/data/landmarks.csv
with a good number of samples per gesture (150+ recommended).

Usage:
    python train_model.py

Outputs:
    model/gesture_model.h5   - the trained Keras model
    model/labels.json        - index -> label mapping used by app.py
------------------------------------------------------------------
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "data", "landmarks.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Run collect_data.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    if len(df) < 20:
        print("ERROR: Not enough samples yet. Collect more data "
              "(aim for 150+ per gesture) before training.")
        return

    label_counts = df["label"].value_counts()
    print("Samples per gesture:")
    print(label_counts.to_string())

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = models.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=15, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=16,
        callbacks=[early_stop],
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc*100:.1f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    with open(LABELS_PATH, "w") as f:
        json.dump(list(encoder.classes_), f, indent=2)

    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved labels -> {LABELS_PATH}")
    print("\nNow run: python app.py")


if __name__ == "__main__":
    main()
