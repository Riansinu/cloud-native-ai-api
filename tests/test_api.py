"""Tests for the Cloud-Native AI Prediction API."""

import pytest


# ------------------------------------------------------------------
# GET /
# ------------------------------------------------------------------
class TestRoot:
    def test_root_returns_api_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Cloud-Native AI Prediction API"
        assert "version" in data
        assert "description" in data
        assert data["docs_url"] == "/docs"


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------
class TestHealth:
    def test_health_returns_healthy(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


# ------------------------------------------------------------------
# GET /model-info
# ------------------------------------------------------------------
class TestModelInfo:
    def test_model_info_returns_expected_fields(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "features" in data
        assert isinstance(data["features"], list)
        assert len(data["features"]) == 8
        assert "description" in data


# ------------------------------------------------------------------
# POST /predict
# ------------------------------------------------------------------
class TestPredict:
    VALID_INPUT = {
        "MedInc": 3.5,
        "HouseAge": 25.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1500.0,
        "AveOccup": 3.0,
        "Latitude": 37.5,
        "Longitude": -122.0,
    }

    def test_predict_returns_valid_response(self, client):
        response = client.post("/predict", json=self.VALID_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_price" in data
        assert isinstance(data["predicted_price"], float)
        assert "model_name" in data
        assert "model_version" in data

    def test_predict_price_is_reasonable(self, client):
        """Prediction should be within a plausible range for California housing."""
        response = client.post("/predict", json=self.VALID_INPUT)
        price = response.json()["predicted_price"]
        # California housing target is in hundreds of thousands;
        # values typically range from ~0.15 to ~5.0.
        assert 0.0 < price < 10.0


# ------------------------------------------------------------------
# POST /predict — invalid inputs
# ------------------------------------------------------------------
class TestPredictValidation:
    def test_missing_field_returns_422(self, client):
        incomplete = {"MedInc": 3.5, "HouseAge": 25.0}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_negative_income_returns_422(self, client):
        bad = {
            "MedInc": -1.0,
            "HouseAge": 25.0,
            "AveRooms": 5.0,
            "AveBedrms": 1.0,
            "Population": 1500.0,
            "AveOccup": 3.0,
            "Latitude": 37.5,
            "Longitude": -122.0,
        }
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_latitude_out_of_range_returns_422(self, client):
        bad = {
            "MedInc": 3.5,
            "HouseAge": 25.0,
            "AveRooms": 5.0,
            "AveBedrms": 1.0,
            "Population": 1500.0,
            "AveOccup": 3.0,
            "Latitude": 99.0,
            "Longitude": -122.0,
        }
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_string_instead_of_number_returns_422(self, client):
        bad = {
            "MedInc": "not_a_number",
            "HouseAge": 25.0,
            "AveRooms": 5.0,
            "AveBedrms": 1.0,
            "Population": 1500.0,
            "AveOccup": 3.0,
            "Latitude": 37.5,
            "Longitude": -122.0,
        }
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
