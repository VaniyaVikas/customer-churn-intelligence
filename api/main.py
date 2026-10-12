
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import CustomerInput, PredictionResponse
from src.config import MODEL_PATH
from src.predict import load_model
from src.retention import recommend_retention


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained ML model when the API starts."""

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Trained model not found. "
            "Run `python -m src.train` first."
        )

    app.state.model = load_model()
    yield


app = FastAPI(
    title="Customer Churn Intelligence API",
    description=(
        "Predict customer churn risk and provide "
        "customer retention recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["General"])
def home():
    """API welcome endpoint."""

    return {
        "message": "Customer Churn Intelligence API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["General"])
def health():
    """Check API and model status."""

    model = getattr(app.state, "model", None)

    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(customer: CustomerInput):
    """Predict churn probability and suggest retention actions."""

    try:
        model = getattr(app.state, "model", None)

        if model is None:
            raise HTTPException(
                status_code=503,
                detail="ML model is not loaded.",
            )

        customer_data = customer.model_dump()
        customer_df = pd.DataFrame([customer_data])

        probability = float(
            model.predict_proba(customer_df)[0, 1]
        )

        prediction = int(probability >= 0.5)

        retention = recommend_retention(
            customer_data,
            probability,
        )

        return PredictionResponse(
            churn_prediction=prediction,
            churn_label="Yes" if prediction == 1 else "No",
            churn_probability=round(probability, 4),
            risk_level=retention["risk_level"],
            recommendations=retention["recommendations"],
        )

    except HTTPException:
        raise

    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to process customer data: {exc}",
        ) from exc
