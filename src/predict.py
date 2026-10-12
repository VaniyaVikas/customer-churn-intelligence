
from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_PATH


def load_model():
    """Load and return the trained churn prediction pipeline."""

    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            "Run `python -m src.train` first."
        )

    return joblib.load(model_path)


def predict_customer(customer_data: dict) -> dict:
    """Predict churn for one customer."""

    model = load_model()
    customer_df = pd.DataFrame([customer_data])

    prediction = int(model.predict(customer_df)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(customer_df)[0]
        probability = round(float(probabilities[1]), 4)

    return {
        "churn_prediction": prediction,
        "churn_result": (
            "Will Churn" if prediction == 1 else "Will Not Churn"
        ),
        "churn_probability": probability,
    }


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 53.85,
        "TotalCharges": 108.15,
    }

    result = predict_customer(sample_customer)

    print("Prediction Result:")
    for key, value in result.items():
        print(f"{key}: {value}")
