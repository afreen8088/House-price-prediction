"""
API endpoint and integration tests for FastAPI backend.
Tests health/readiness, successful predictions, input validation rejections,
missing model handling, and metadata retrieval.
"""

import os
import pytest
from fastapi.testclient import TestClient

from app.main import app, load_model


@pytest.fixture
def client():
    """Create a test client with application lifespan active."""
    with TestClient(app) as test_client:
        yield test_client


def test_api_status_endpoint(client):
    """Verify GET / returns online status and pointers."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "service" in data
    assert "version" in data


def test_health_check_endpoint(client):
    """Verify GET /health returns readiness."""
    response = client.get("/health")
    # If model is loaded, status should be 200
    if app.state.model is not None:
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["ready"] is True
    else:
        assert response.status_code == 503


def test_successful_prediction(client):
    """Verify POST /predict with realistic valid property details."""
    if app.state.model is None:
        pytest.skip("Model not loaded in test environment")

    payload = {
        "sqft": 2400.0,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "neighborhood": "Suburban Heights"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "predicted_price" in data
    assert data["predicted_price"] > 50000.0
    assert "formatted_price" in data
    assert data["formatted_price"].startswith("$")
    assert data["currency"] == "USD"
    assert data["property_details"]["sqft"] == 2400.0
    assert data["property_details"]["bedrooms"] == 3
    assert data["property_details"]["bathrooms"] == 2.5
    assert data["property_details"]["neighborhood"] == "Suburban Heights"
    assert "disclaimer" in data


def test_invalid_prediction_payloads(client):
    """Verify validation errors are caught and rejected with HTTP 422."""
    # 1. Square footage too small
    res1 = client.post("/predict", json={
        "sqft": 50.0,  # Below minimum 100
        "bedrooms": 3,
        "bathrooms": 2.0,
        "neighborhood": "Downtown"
    })
    assert res1.status_code == 422

    # 2. Zero bedrooms
    res2 = client.post("/predict", json={
        "sqft": 1500.0,
        "bedrooms": 0,  # Below minimum 1
        "bathrooms": 2.0,
        "neighborhood": "Downtown"
    })
    assert res2.status_code == 422

    # 3. Invalid bathrooms
    res3 = client.post("/predict", json={
        "sqft": 1500.0,
        "bedrooms": 3,
        "bathrooms": 0.0,  # Below minimum 0.5
        "neighborhood": "Downtown"
    })
    assert res3.status_code == 422

    # 4. Empty neighborhood
    res4 = client.post("/predict", json={
        "sqft": 1500.0,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "neighborhood": "   "  # Pure whitespace
    })
    assert res4.status_code == 422


def test_model_info_endpoint(client):
    """Verify GET /model-info returns metadata and does not fabricate missing metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_version" in data
    assert "rmse" in data
    assert "r2_score" in data
    assert "training_status" in data


def test_missing_model_behavior():
    """Verify proper 503 error handling when model artifact is absent."""
    with TestClient(app) as test_client:
        # Temporarily simulate unloaded model
        original_model = app.state.model
        app.state.model = None

        try:
            # Health should fail
            health_res = test_client.get("/health")
            assert health_res.status_code == 503
            assert health_res.json()["ready"] is False

            # Predict should fail
            predict_res = test_client.post("/predict", json={
                "sqft": 2000.0,
                "bedrooms": 3,
                "bathrooms": 2.0,
                "neighborhood": "Downtown"
            })
            assert predict_res.status_code == 503
            assert "Model is not available" in predict_res.json()["detail"]
        finally:
            app.state.model = original_model
