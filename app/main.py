"""
FastAPI Backend for House Price Prediction MLOps.
Provides real-time model inference, health/readiness probes,
model metadata inspection, and static web interface serving.
"""

import os
import sys
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("house_price_api")

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "metrics.json")
ALLOWED_CORS = os.getenv("CORS_ORIGINS", "*").split(",")


def load_model(path: str = MODEL_PATH):
    """Safely load the persistent scikit-learn/XGBoost inference pipeline."""
    if not os.path.exists(path):
        logger.warning(f"Model artifact not found at {path}")
        return None
    try:
        model = joblib.load(path)
        logger.info(f"Successfully loaded inference pipeline from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model artifact from {path}: {e}")
        return None


def load_metrics(path: str = METRICS_PATH) -> Optional[Dict[str, Any]]:
    """Load latest evaluation metrics if available."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed reading metrics artifact from {path}: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for state initialization."""
    logger.info("Initializing House Price Prediction API lifespan...")
    app.state.model = load_model(MODEL_PATH)
    app.state.metrics = load_metrics(METRICS_PATH)
    app.state.model_path = MODEL_PATH
    yield
    logger.info("Shutting down House Price Prediction API...")


app = FastAPI(
    title="HOMESTEAD House Price Valuation API",
    description="Production MLOps API providing machine-learning property valuations and metadata.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_CORS if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request and Response Schemas ---

class PropertyPredictionRequest(BaseModel):
    sqft: float = Field(
        ...,
        gt=100.0,
        lt=25000.0,
        description="Property interior living area in square feet. Must be between 100 and 25,000 sqft."
    )
    bedrooms: int = Field(
        ...,
        ge=1,
        le=20,
        description="Number of bedrooms. Must be between 1 and 20."
    )
    bathrooms: float = Field(
        ...,
        ge=0.5,
        le=15.0,
        description="Number of bathrooms. Must be between 0.5 and 15."
    )
    neighborhood: str = Field(
        ...,
        min_length=2,
        max_length=80,
        description="Property neighborhood or district name."
    )

    @field_validator("neighborhood")
    @classmethod
    def validate_neighborhood(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Neighborhood name cannot be empty or purely whitespace.")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "example": {
                "sqft": 2450.0,
                "bedrooms": 4,
                "bathrooms": 2.5,
                "neighborhood": "Oakridge"
            }
        }
    }


class PropertyPredictionResponse(BaseModel):
    predicted_price: float
    formatted_price: str
    currency: str
    property_details: Dict[str, Any]
    summary: str
    model_version: str
    timestamp: str
    disclaimer: str


# --- Endpoints ---

@app.get("/", summary="API Status")
def api_status():
    """Returns application status, version, and documentation pointers."""
    model_loaded = getattr(app.state, "model", None) is not None
    return {
        "status": "online",
        "service": "HOMESTEAD Property Valuation MLOps API",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "docs_url": "/docs",
        "ui_url": "/ui/",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health", summary="Application and Model Readiness")
def health_check():
    """
    Readiness and health probe.
    Fails with HTTP 503 if model artifact is not loaded.
    """
    model = getattr(app.state, "model", None)
    if model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "ready": False,
                "detail": f"Model artifact is not loaded. Checked path: {app.state.model_path}. "
                          f"Execute training pipeline ('python src/train.py') to generate model.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    return {
        "status": "healthy",
        "ready": True,
        "model_path": app.state.model_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PropertyPredictionResponse, summary="Predict House Price")
def predict_house_price(payload: PropertyPredictionRequest):
    """
    Accepts validated property details and returns an XGBoost model prediction.
    Uses the trained scikit-learn + XGBoost inference pipeline directly.
    """
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Please ensure model.joblib exists and service is initialized."
        )

    # Format input into single-row DataFrame expected by the Pipeline preprocessor
    input_data = pd.DataFrame([{
        "sqft": payload.sqft,
        "bedrooms": payload.bedrooms,
        "bathrooms": payload.bathrooms,
        "neighborhood": payload.neighborhood
    }])

    try:
        raw_prediction = float(model.predict(input_data)[0])
        # Ensure floor on non-negative real estate valuation
        predicted_price = max(raw_prediction, 50000.0)
    except Exception as e:
        logger.error(f"Inference error during model prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}"
        )

    formatted_price = f"${predicted_price:,.2f}"
    summary = (
        f"Valuation estimate for a {payload.sqft:,.0f} sqft, "
        f"{payload.bedrooms} bed, {payload.bathrooms} bath residence in {payload.neighborhood}."
    )

    metrics = getattr(app.state, "metrics", None)
    model_version = metrics.get("model_type", "XGBRegressor-v1") if metrics else "XGBRegressor-v1"

    return PropertyPredictionResponse(
        predicted_price=round(predicted_price, 2),
        formatted_price=formatted_price,
        currency="USD",
        property_details={
            "sqft": payload.sqft,
            "bedrooms": payload.bedrooms,
            "bathrooms": payload.bathrooms,
            "neighborhood": payload.neighborhood,
        },
        summary=summary,
        model_version=model_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        disclaimer="Estimates provided by HOMESTEAD are statistical models and do not constitute formal appraisals or appraisal guarantees."
    )


@app.get("/model-info", summary="Model Metadata and Evaluation Metrics")
def model_info():
    """
    Returns actual available model metadata and metrics recorded during training.
    If information is unavailable, displays 'Not available' per requirements.
    """
    model_loaded = getattr(app.state, "model", None) is not None
    metrics = load_metrics(app.state.metrics_path if hasattr(app.state, "metrics_path") else METRICS_PATH)

    if not metrics and not model_loaded:
        return {
            "model_name": "Not available",
            "model_version": "Not available",
            "training_timestamp": "Not available",
            "rmse": "Not available",
            "mae": "Not available",
            "r2_score": "Not available",
            "dataset_version": "Not available",
            "experiment_parameters": "Not available",
            "training_status": "No model artifact found. Pipeline must be run.",
            "model_available": False,
        }

    return {
        "model_name": metrics.get("model_type", "XGBRegressor") if metrics else "XGBRegressor",
        "model_version": "1.0.0",
        "training_timestamp": metrics.get("timestamp", "Not available") if metrics else "Not available",
        "rmse": metrics.get("rmse", "Not available") if metrics else "Not available",
        "mae": metrics.get("mae", "Not available") if metrics else "Not available",
        "r2_score": metrics.get("r2_score", "Not available") if metrics else "Not available",
        "train_samples": metrics.get("train_samples", "Not available") if metrics else "Not available",
        "test_samples": metrics.get("test_samples", "Not available") if metrics else "Not available",
        "dataset_version": "DVC tracked data/raw/housing_data.csv",
        "experiment_parameters": metrics.get("hyperparameters", "Not available") if metrics else "Not available",
        "features_expected": metrics.get("features", ["sqft", "bedrooms", "bathrooms", "neighborhood"]) if metrics else ["sqft", "bedrooms", "bathrooms", "neighborhood"],
        "training_status": metrics.get("status", "Active") if metrics else "Model loaded",
        "model_available": model_loaded,
    }


# Mount static frontend directory at /ui
FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
