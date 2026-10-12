"""
API endpoint and integration tests for FastAPI backend in Karnataka, India.
Tests health/readiness, successful predictions in INR (₹), city differential pricing,
input validation rejections, missing model handling, and metadata retrieval.
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
    assert data["currency"] == "INR"


def test_health_check_endpoint(client):
    """Verify GET /health returns readiness."""
    response = client.get("/health")
    if app.state.model is not None:
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["ready"] is True
    else:
        assert response.status_code == 503


def test_locations_endpoint(client):
    """Verify GET /locations returns supported Karnataka cities."""
    response = client.get("/locations")
    assert response.status_code == 200
    data = response.json()
    assert "locations" in data
    assert "Bengaluru" in data["locations"]
    assert "Mysuru" in data["locations"]
    assert len(data["locations"]) == 12


def test_successful_prediction_inr(client):
    """Verify POST /predict with realistic valid Karnataka property details in INR."""
    if app.state.model is None:
        pytest.skip("Model not loaded in test environment")

    payload = {
        "sqft": 1500.0,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "location": "Bengaluru"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "predicted_price" in data
    assert data["predicted_price"] > 1000000.0  # Above 10 Lakhs in INR
    assert "formatted_price" in data
    assert data["formatted_price"].startswith("₹")
    assert data["currency"] == "INR"
    assert data["property_details"]["sqft"] == 1500.0
    assert data["property_details"]["bedrooms"] == 3
    assert data["property_details"]["bathrooms"] == 2.0
    assert data["property_details"]["location"] == "Bengaluru"
    assert "price_per_sqft" in data
    assert data["price_per_sqft"] > 0
    assert "formatted_price_per_sqft" in data
    assert data["formatted_price_per_sqft"].startswith("₹")
    assert "/sq ft" in data["formatted_price_per_sqft"]
    assert "disclaimer" in data


def test_prediction_neighborhood_alias_backward_compatibility(client):
    """Verify POST /predict accepts legacy 'neighborhood' payload and maps it to location."""
    if app.state.model is None:
        pytest.skip("Model not loaded in test environment")

    payload = {
        "sqft": 1500.0,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "neighborhood": "Mysuru"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["property_details"]["location"] == "Mysuru"
    assert data["currency"] == "INR"
    assert data["formatted_price"].startswith("₹")


def test_city_differential_prediction(client):
    """
    Verify that the selected city directly influences the estimated house price.
    A property with the same area and rooms in Bengaluru must receive a different estimate than Kalaburagi.
    """
    if app.state.model is None:
        pytest.skip("Model not loaded in test environment")

    base = {"sqft": 1500.0, "bedrooms": 3, "bathrooms": 2.0}
    res_bengaluru = client.post("/predict", json={**base, "location": "Bengaluru"}).json()
    res_kalaburagi = client.post("/predict", json={**base, "location": "Kalaburagi"}).json()

    assert res_bengaluru["predicted_price"] != res_kalaburagi["predicted_price"]
    # Due to market differences, metropolitan Bengaluru price should be higher than regional Kalaburagi
    assert res_bengaluru["predicted_price"] > res_kalaburagi["predicted_price"]


def test_invalid_prediction_payloads(client):
    """Verify validation errors are caught and rejected with HTTP 422."""
    # 1. Square footage too small
    res1 = client.post("/predict", json={
        "sqft": 50.0,  # Below minimum 100
        "bedrooms": 3,
        "bathrooms": 2.0,
        "location": "Bengaluru"
    })
    assert res1.status_code == 422

    # 2. Zero bedrooms
    res2 = client.post("/predict", json={
        "sqft": 1500.0,
        "bedrooms": 0,  # Below minimum 1
        "bathrooms": 2.0,
        "location": "Bengaluru"
    })
    assert res2.status_code == 422

    # 3. Invalid bathrooms
    res3 = client.post("/predict", json={
        "sqft": 1500.0,
        "bedrooms": 3,
        "bathrooms": 0.0,  # Below minimum 0.5
        "location": "Bengaluru"
    })
    assert res3.status_code == 422

    # 4. Empty location and neighborhood
    res4 = client.post("/predict", json={
        "sqft": 1500.0,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "location": "   "  # Pure whitespace
    })
    assert res4.status_code == 422


def test_model_info_endpoint(client):
    """Verify GET /model-info returns metadata in INR and does not fabricate missing metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_version" in data
    assert "rmse" in data
    assert "r2_score" in data
    assert data["currency"] == "INR"
    assert "supported_locations" in data
    assert "Bengaluru" in data["supported_locations"]


def test_missing_model_behavior():
    """Verify proper 503 error handling when model artifact is absent."""
    with TestClient(app) as test_client:
        original_model = app.state.model
        app.state.model = None

        try:
            health_res = test_client.get("/health")
            assert health_res.status_code == 503
            assert health_res.json()["ready"] is False

            predict_res = test_client.post("/predict", json={
                "sqft": 1500.0,
                "bedrooms": 3,
                "bathrooms": 2.0,
                "location": "Bengaluru"
            })
            assert predict_res.status_code == 503
            assert "Model is not available" in predict_res.json()["detail"]
        finally:
            app.state.model = original_model
