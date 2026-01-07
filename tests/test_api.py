"""
Tests for FastAPI endpoints.
"""

import pytest
import numpy as np
import cv2
import io
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create test client fixture with proper startup event handling."""
    from src.api import app, model_manager

    # Manually load models before creating client
    model_manager.load_models()

    # Create client
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_image():
    """Create a sample test image."""
    # Create a simple face-like image
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200

    # Add face-like features
    cv2.circle(img, (112, 112), 80, (180, 160, 140), -1)  # Face
    cv2.circle(img, (85, 95), 10, (50, 50, 50), -1)  # Left eye
    cv2.circle(img, (139, 95), 10, (50, 50, 50), -1)  # Right eye

    # Encode as bytes
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()  # Return bytes directly for reuse


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns valid response."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert 'status' in data
        assert 'model_loaded' in data
        assert 'device' in data
        assert 'version' in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert 'message' in data
        assert 'docs' in data


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_predict_tabular(self, client):
        """Test tabular-only prediction endpoint."""
        clinical_data = {
            "age": 5.0,
            "height_cm": 85.0,
            "weight_kg": 12.0,
            "small_jaw": 1,
            "prominent_eyes": 1,
            "thin_skin": 1,
            "hair_loss": 0,
            "lmna_mut": 0
        }

        response = client.post("/predict/tabular", json=clinical_data)

        assert response.status_code == 200
        data = response.json()

        assert 'risk_score' in data
        assert 'risk_class' in data
        assert 'recommendation' in data
        assert 'confidence' in data

        # Risk score should be between 0 and 1
        assert 0 <= data['risk_score'] <= 1

        # Risk class should be valid
        assert data['risk_class'] in ['Low', 'Moderate', 'High']

    def test_predict_tabular_validation(self, client):
        """Test input validation for tabular endpoint."""
        # Invalid age (negative)
        invalid_data = {
            "age": -1.0,
            "height_cm": 85.0,
            "weight_kg": 12.0
        }

        response = client.post("/predict/tabular", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_predict_multimodal(self, client, sample_image):
        """Test multimodal prediction endpoint."""
        files = {
            "image": ("test.png", io.BytesIO(sample_image), "image/png")
        }
        data = {
            "age": 5.0,
            "height_cm": 85.0,
            "weight_kg": 12.0,
            "small_jaw": 1,
            "prominent_eyes": 1,
            "thin_skin": 1,
            "hair_loss": 0,
            "lmna_mut": 0
        }

        response = client.post("/predict", files=files, data=data)

        assert response.status_code == 200
        result = response.json()

        assert 'risk_score' in result
        assert 'progression_class' in result
        assert 'progression_probs' in result

    def test_qml_comparison(self, client):
        """Test QML comparison endpoint."""
        clinical_data = {
            "age": 5.0,
            "height_cm": 85.0,
            "weight_kg": 12.0,
            "small_jaw": 1,
            "prominent_eyes": 1,
            "thin_skin": 1,
            "hair_loss": 0,
            "lmna_mut": 0
        }

        response = client.post("/predict/qml", json=clinical_data)

        assert response.status_code == 200
        data = response.json()

        assert 'classical_prediction' in data
        assert 'comparison_summary' in data


class TestExplanationEndpoint:
    """Tests for explanation endpoint."""

    def test_explain(self, client):
        """Test explanation endpoint."""
        clinical_data = {
            "age": 5.0,
            "height_cm": 85.0,
            "weight_kg": 12.0,
            "small_jaw": 1,
            "prominent_eyes": 1,
            "thin_skin": 1,
            "hair_loss": 0,
            "lmna_mut": 0
        }

        response = client.post("/explain", json=clinical_data)

        assert response.status_code == 200
        data = response.json()

        assert 'feature_importance' in data
        assert 'top_contributing_features' in data


class TestGrowthCurveEndpoint:
    """Tests for growth curve endpoint."""

    def test_growth_curve_normal(self, client):
        """Test growth curve for normal trajectory."""
        response = client.get("/growth-curve/5.0?is_hgps=false")

        assert response.status_code == 200
        data = response.json()

        assert 'ages' in data
        assert 'heights' in data
        assert 'weights' in data
        assert data['current_age'] == 5.0
        assert data['is_hgps_trajectory'] == False

    def test_growth_curve_hgps(self, client):
        """Test growth curve for HGPS trajectory."""
        response = client.get("/growth-curve/5.0?is_hgps=true")

        assert response.status_code == 200
        data = response.json()

        assert data['is_hgps_trajectory'] == True


class TestPredictionConsistency:
    """Tests for prediction consistency."""

    def test_high_risk_features(self, client):
        """Test that high-risk features increase risk score."""
        # Low risk patient
        low_risk = {
            "age": 5.0,
            "height_cm": 110.0,  # Normal height
            "weight_kg": 18.0,   # Normal weight
            "small_jaw": 0,
            "prominent_eyes": 0,
            "thin_skin": 0,
            "hair_loss": 0,
            "lmna_mut": 0
        }

        # High risk patient
        high_risk = {
            "age": 5.0,
            "height_cm": 75.0,   # Very short
            "weight_kg": 10.0,   # Underweight
            "small_jaw": 1,
            "prominent_eyes": 1,
            "thin_skin": 1,
            "hair_loss": 1,
            "lmna_mut": 1
        }

        low_response = client.post("/predict/tabular", json=low_risk)
        high_response = client.post("/predict/tabular", json=high_risk)

        low_score = low_response.json()['risk_score']
        high_score = high_response.json()['risk_score']

        # High risk features should increase risk score
        assert high_score > low_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
