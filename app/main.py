"""
FastAPI Backend for HOMESTEAD House Price Prediction in Karnataka, India.
Provides real-time model inference in Indian Rupees (INR - ₹),
health/readiness probes, model metadata inspection, and static web interface serving.
"""

import os
import sys
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

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
from pydantic import BaseModel, Field, model_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("house_price_api")

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "metrics.json")
ALLOWED_CORS = os.getenv("CORS_ORIGINS", "*").split(",")

KARNATAKA_CITIES = [
    "Bengaluru",
    "Mysuru",
    "Mangaluru",
    "Hubballi-Dharwad",
    "Belagavi",
    "Kalaburagi",
    "Davanagere",
    "Shivamogga",
    "Tumakuru",
    "Udupi",
    "Hassan",
    "Ballari",
]


def format_inr(amount: float) -> str:
    """
    Format numeric value into Indian Rupee representation (e.g. ₹85,00,000).
    Uses the Indian numbering system (Lakhs, Crores).
    """
    amount_int = int(round(amount))
    if amount_int < 0:
        return f"-{format_inr(-amount_int)}"
    s = str(amount_int)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    remaining = s[:-3]
    groups = []
    while len(remaining) > 2:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        groups.insert(0, remaining)
    return f"₹{','.join(groups)},{last3}"


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
        with open(path, "r", encoding="utf-8") as f:
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
    app.state.metrics_path = METRICS_PATH
    yield
    logger.info("Shutting down House Price Prediction API...")


app = FastAPI(
    title="HOMESTEAD House Price Valuation API",
    description="Production MLOps API providing machine-learning property valuations in Karnataka, India.",
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
    location: Optional[str] = Field(
        default=None,
        description="Property location or city in Karnataka."
    )
    neighborhood: Optional[str] = Field(
        default=None,
        description="Legacy alias for location to maintain backwards compatibility."
    )

    @model_validator(mode="after")
    def validate_location(self):
        loc = self.location or self.neighborhood
        if not loc or not loc.strip():
            raise ValueError("Location name cannot be empty or purely whitespace.")
        self.location = loc.strip()
        self.neighborhood = self.location
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "sqft": 1500.0,
                "bedrooms": 3,
                "bathrooms": 2.0,
                "location": "Bengaluru"
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
    price_per_sqft: Optional[float] = None
    formatted_price_per_sqft: Optional[str] = None
    disclaimer: str


# --- Endpoints ---

@app.get("/", summary="API Status")
def api_status():
    """Returns application status, version, and documentation pointers."""
    model_loaded = getattr(app.state, "model", None) is not None
    return {
        "status": "online",
        "service": "HOMESTEAD Property Valuation MLOps API (Karnataka, India)",
        "version": "1.0.0",
        "currency": "INR",
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
    model_path = getattr(app.state, "model_path", MODEL_PATH)
    if model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "ready": False,
                "detail": f"Model artifact is not loaded. Checked path: {model_path}. "
                          f"Execute training pipeline ('python src/train.py') to generate model.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    return {
        "status": "healthy",
        "ready": True,
        "model_path": model_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/locations", summary="Supported Karnataka Locations")
def get_locations():
    """Returns list of supported Karnataka cities for property valuation."""
    return {
        "state": "Karnataka",
        "country": "India",
        "locations": KARNATAKA_CITIES,
    }


@app.post("/predict", response_model=PropertyPredictionResponse, summary="Predict House Price")
def predict_house_price(payload: PropertyPredictionRequest):
    """
    Accepts validated property details and returns an XGBoost model prediction in INR (₹).
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
        "location": payload.location
    }])

    try:
        raw_prediction = float(model.predict(input_data)[0])
        # Ensure floor on realistic minimum real estate valuation in INR (₹5 Lakhs)
        predicted_price = max(raw_prediction, 500000.0)
    except Exception as e:
        logger.error(f"Inference error during model prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}"
        )

    price_per_sqft = round(predicted_price / payload.sqft, 2)
    formatted_price = format_inr(predicted_price)
    formatted_price_per_sqft = f"{format_inr(price_per_sqft)}/sq ft"

    summary = (
        f"Valuation estimate for a {payload.sqft:,.0f} sq ft, "
        f"{payload.bedrooms} bed, {payload.bathrooms:g} bath residence in {payload.location}."
    )

    metrics = getattr(app.state, "metrics", None)
    model_version = metrics.get("model_type", "XGBRegressor-v1") if metrics else "XGBRegressor-v1"

    return PropertyPredictionResponse(
        predicted_price=round(predicted_price, 2),
        formatted_price=formatted_price,
        currency="INR",
        property_details={
            "sqft": payload.sqft,
            "bedrooms": payload.bedrooms,
            "bathrooms": payload.bathrooms,
            "location": payload.location,
            "neighborhood": payload.location,
            "price_per_sqft": price_per_sqft,
            "formatted_price_per_sqft": formatted_price_per_sqft,
        },
        summary=summary,
        model_version=model_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        price_per_sqft=price_per_sqft,
        formatted_price_per_sqft=formatted_price_per_sqft,
        disclaimer="Demonstration estimate based on statistical modeling, not a verified market valuation."
    )


@app.get("/model-info", summary="Model Metadata and Evaluation Metrics")
def model_info():
    """
    Returns actual available model metadata and metrics recorded during training in INR.
    If information is unavailable, displays 'Not available' per requirements.
    """
    model_loaded = getattr(app.state, "model", None) is not None
    metrics_path = getattr(app.state, "metrics_path", METRICS_PATH)
    metrics = load_metrics(metrics_path)

    if not metrics and not model_loaded:
        return {
            "model_name": "Not available",
            "model_version": "Not available",
            "training_timestamp": "Not available",
            "currency": "INR",
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
        "currency": "INR",
        "region": "Karnataka, India",
        "training_timestamp": metrics.get("timestamp", "Not available") if metrics else "Not available",
        "rmse": metrics.get("rmse", "Not available") if metrics else "Not available",
        "formatted_rmse": format_inr(metrics["rmse"]) if metrics and isinstance(metrics.get("rmse"), (int, float)) else "Not available",
        "mae": metrics.get("mae", "Not available") if metrics else "Not available",
        "formatted_mae": format_inr(metrics["mae"]) if metrics and isinstance(metrics.get("mae"), (int, float)) else "Not available",
        "r2_score": metrics.get("r2_score", "Not available") if metrics else "Not available",
        "train_samples": metrics.get("train_samples", "Not available") if metrics else "Not available",
        "test_samples": metrics.get("test_samples", "Not available") if metrics else "Not available",
        "dataset_version": "DVC tracked data/raw/housing_data.csv (Karnataka, INR)",
        "experiment_parameters": metrics.get("hyperparameters", "Not available") if metrics else "Not available",
        "features_expected": metrics.get("features", ["sqft", "bedrooms", "bathrooms", "location"]) if metrics else ["sqft", "bedrooms", "bathrooms", "location"],
        "supported_locations": KARNATAKA_CITIES,
        "training_status": metrics.get("status", "Active") if metrics else "Model loaded",
        "model_available": model_loaded,
    }


# Mount static frontend directory at /ui
FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
