from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "food_waste_model.joblib"


def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run 'python src/train_model.py' first."
        )

    bundle = joblib.load(MODEL_PATH)

    if "model" not in bundle or "features" not in bundle:
        raise ValueError("The saved model file is incomplete. Train the model again.")

    return bundle


def build_recommendation(prediction, remaining_percentage):
    if prediction == 1:
        if remaining_percentage >= 60:
            return (
                "A large amount of stock remains. Review the next restock quantity "
                "and prioritize the current inventory."
            )

        return (
            "The model detects potential waste. Review current sales before adding "
            "more stock and consider actions to reduce the remaining inventory."
        )

    if remaining_percentage <= 15:
        return (
            "Remaining stock is low. The condition is classified as safe, but keep "
            "monitoring availability before the next restock."
        )

    return (
        "The condition is classified as safe. Continue monitoring sales and remaining "
        "inventory before the next restock."
    )


def predict_food_waste(stock, sold, day_index, weather_code, special_day):
    if stock <= 0:
        raise ValueError("Stock must be greater than 0.")
    if sold < 0:
        raise ValueError("Items sold cannot be negative.")
    if sold > stock:
        raise ValueError("Items sold cannot be greater than available stock.")
    if day_index < 1 or day_index > 30:
        raise ValueError("Day index must be between 1 and 30.")
    if weather_code not in [0, 1]:
        raise ValueError("Weather code must be 0 or 1.")

    remaining = stock - sold
    remaining_percentage = (remaining / stock) * 100

    bundle = load_model_bundle()
    model = bundle["model"]
    features = bundle["features"]

    input_data = pd.DataFrame(
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

    prediction = int(model.predict(input_data)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = float(probabilities[class_index])

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
