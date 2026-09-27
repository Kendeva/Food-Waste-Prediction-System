from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "food_waste_model.joblib"


def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run 'python src/train_model.py' first."
        )
    return joblib.load(MODEL_PATH)


def build_recommendation(prediction: int, remaining_percentage: float) -> str:
    if prediction == 1:
        if remaining_percentage >= 60:
            return (
                "A large share of the stock remains. Consider reviewing the next "
                "restock quantity and using the current inventory first."
            )
        return (
            "The model identifies potential waste. Review sales progress before "
            "adding more stock and consider actions to move the remaining inventory."
        )

    if remaining_percentage <= 15:
        return (
            "Remaining stock is low. The current condition is classified as safe, "
            "but monitor availability before the next restock."
        )
    return (
        "The current condition is classified as safe. Continue monitoring sales "
        "and remaining inventory before the next restock."
    )


def predict_food_waste(
    *,
    stock: int,
    sold: int,
    day_index: int,
    weather_code: int,
    special_day: bool,
) -> dict:
    if stock <= 0:
        raise ValueError("Stock must be greater than 0.")
    if sold < 0:
        raise ValueError("Items sold cannot be negative.")
    if sold > stock:
        raise ValueError("Items sold cannot be greater than the available stock.")

    bundle = load_model_bundle()
    model = bundle["model"]
    features = bundle["features"]

    remaining = stock - sold
    remaining_percentage = (remaining / stock) * 100

    model_input = pd.DataFrame(
        [
            {
                "stok": stock,
                "terjual": sold,
                "hari_ke": day_index,
                "cuaca": weather_code,
                "hari_besar": int(special_day),
                "sisa_persen": round(remaining_percentage, 2),
            }
        ],
        columns=features,
    )

    prediction = int(model.predict(model_input)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        class_positions = {int(value): index for index, value in enumerate(model.classes_)}
        confidence = float(probabilities[class_positions[prediction]])

    return {
        "prediction": prediction,
        "status": "Potential Waste" if prediction == 1 else "Safe",
        "risk_label": "HIGH RISK" if prediction == 1 else "LOW RISK",
        "stock": stock,
        "sold": sold,
        "remaining": remaining,
        "remaining_percentage": remaining_percentage,
        "confidence": confidence,
        "recommendation": build_recommendation(prediction, remaining_percentage),
    }
