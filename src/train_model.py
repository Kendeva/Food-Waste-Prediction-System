from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "data_penjualan.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "food_waste_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
IMAGE_DIR = BASE_DIR / "images"
TREE_IMAGE_PATH = IMAGE_DIR / "decision_tree.png"

FEATURES = ["stok", "terjual", "hari_ke", "cuaca", "hari_besar", "sisa_persen"]
TARGET = "label"


def train_and_save_model() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Add 'data_penjualan.csv' inside the data/ folder."
        )

    data = pd.read_csv(DATASET_PATH)
    required_columns = FEATURES + [TARGET]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)
    accuracy = float(model.score(X_test, y_test))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "features": FEATURES,
        "classes": {0: "Safe", 1: "Potential Waste"},
        "accuracy": accuracy,
        "dataset_rows": int(len(data)),
    }
    joblib.dump(bundle, MODEL_PATH)

    metadata = {
        "model": "Decision Tree Classifier",
        "criterion": "entropy",
        "max_depth": 4,
        "accuracy": accuracy,
        "dataset_rows": int(len(data)),
        "features": FEATURES,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plt.figure(figsize=(13, 8))
    tree.plot_tree(
        model,
        feature_names=FEATURES,
        class_names=["Safe", "Potential Waste"],
        filled=True,
        rounded=True,
    )
    plt.tight_layout()
    plt.savefig(TREE_IMAGE_PATH, dpi=180, bbox_inches="tight")
    plt.close()

    return metadata


if __name__ == "__main__":
    result = train_and_save_model()
    print("Model training completed.")
    print(f"Accuracy: {result['accuracy']:.2%}")
    print(f"Saved to: {MODEL_PATH}")
