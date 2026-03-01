"""
Training script for a TensorFlow/Keras CNN on the FER2013 dataset.
Assumes a fer2013.csv file is available locally.
This script filters to five classes and saves a Keras model and labels.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import class_weight

from .tf_emotion_model import build_cnn, DEFAULT_LABELS


# Mapping from FER2013 integer labels to target labels
# 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
FER_TO_TARGET = {
    0: "anger",
    2: "fear",
    3: "happiness",
    4: "sadness",
    6: "neutral",
}


def load_fer2013(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    df = pd.read_csv(csv_path)
    # Filter to Training and PublicTest splits
    train_df = df[df["Usage"] == "Training"]
    val_df = df[df["Usage"].isin(["PublicTest", "PrivateTest"])]

    def _process(split_df: pd.DataFrame):
        xs, ys = [], []
        for _, row in split_df.iterrows():
            lbl = row["emotion"]
            if lbl not in FER_TO_TARGET:
                continue
            target = FER_TO_TARGET[lbl]
            pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
            if pixels.size != 48 * 48:
                continue
            img = pixels.reshape(48, 48).astype("float32") / 255.0
            xs.append(np.expand_dims(img, axis=-1))
            ys.append(target)
        return np.stack(xs, axis=0), np.array(ys)

    x_train, y_train = _process(train_df)
    x_val, y_val = _process(val_df)

    label_set = ["neutral", "fear", "anger", "sadness", "happiness"]
    lbl_to_idx = {l: i for i in label_set}

    y_train_idx = np.array([lbl_to_idx[y] for y in y_train])
    y_val_idx = np.array([lbl_to_idx[y] for y in y_val])

    y_train_oh = tf.keras.utils.to_categorical(y_train_idx, num_classes=len(label_set))
    y_val_oh = tf.keras.utils.to_categorical(y_val_idx, num_classes=len(label_set))

    return x_train, y_train_oh, x_val, y_val_oh, lbl_to_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to fer2013.csv")
    parser.add_argument("--outdir", required=True, help="Directory to save model and labels")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, lbl_to_idx = load_fer2013(Path(args.csv))

    model = build_cnn(input_shape=(48, 48, 1), num_classes=y_train.shape[1])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Compute class weights to handle imbalance
    y_train_idx = np.argmax(y_train, axis=1)
    cw = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train_idx), y=y_train_idx
    )
    class_weights = {i: float(w) for i, w in enumerate(cw)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    model_path = outdir / "emotion_model.keras"
    model.save(model_path)
    labels_path = outdir / "labels.json"
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(["neutral", "fear", "anger", "sadness", "happiness"], f, ensure_ascii=False, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved labels to {labels_path}")


if __name__ == "__main__":
    main()

