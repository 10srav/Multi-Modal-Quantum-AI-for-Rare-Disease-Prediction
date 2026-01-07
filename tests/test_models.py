"""
Tests for ML models.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import FaceCNN, TabularMLP, LateFusionClassifier, MultimodalFusionModel
from src.models import ClassicalTabularModels, get_class_weights


class TestFaceCNN:
    """Tests for Face CNN model."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return FaceCNN(
            embedding_dim=256,
            pretrained=False,  # Faster for testing
            freeze_backbone=False
        )

    def test_forward_shape(self, model):
        """Test forward pass output shapes."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        output = model(x)

        assert 'risk_logits' in output
        assert 'progression_logits' in output
        assert output['risk_logits'].shape == (batch_size, 2)
        assert output['progression_logits'].shape == (batch_size, 3)

    def test_embedding_extraction(self, model):
        """Test embedding extraction."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        output = model(x, return_embedding=True)

        assert 'embedding' in output
        assert output['embedding'].shape == (batch_size, 256)

    def test_predict_proba(self, model):
        """Test probability prediction."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)

        probs = model.predict_proba(x)

        assert probs['risk_probs'].shape == (batch_size, 2)
        assert probs['progression_probs'].shape == (batch_size, 3)

        # Probabilities should sum to 1
        risk_sums = probs['risk_probs'].sum(dim=1)
        prog_sums = probs['progression_probs'].sum(dim=1)

        torch.testing.assert_close(risk_sums, torch.ones(batch_size), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(prog_sums, torch.ones(batch_size), rtol=1e-5, atol=1e-5)


class TestTabularMLP:
    """Tests for Tabular MLP model."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return TabularMLP(
            input_dim=11,
            hidden_dims=[64, 128, 64],
            embedding_dim=64
        )

    def test_forward_shape(self, model):
        """Test forward pass output shapes."""
        batch_size = 8
        x = torch.randn(batch_size, 11)

        output = model(x)

        assert output['risk_logits'].shape == (batch_size, 2)
        assert output['progression_logits'].shape == (batch_size, 3)

    def test_embedding_extraction(self, model):
        """Test embedding extraction."""
        batch_size = 8
        x = torch.randn(batch_size, 11)

        output = model(x, return_embedding=True)

        assert output['embedding'].shape == (batch_size, 64)

    def test_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        x = torch.randn(4, 11, requires_grad=True)
        output = model(x)

        loss = output['risk_logits'].sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestFusionModel:
    """Tests for multimodal fusion models."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return LateFusionClassifier(
            face_embedding_dim=128,  # Smaller for testing
            tabular_embedding_dim=32,
            tabular_input_dim=11,
            pretrained_face=False
        )

    def test_forward_shape(self, model):
        """Test forward pass with both modalities."""
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        tabular = torch.randn(batch_size, 11)

        output = model(images, tabular)

        assert output['risk_logits'].shape == (batch_size, 2)
        assert output['progression_logits'].shape == (batch_size, 3)

    def test_embeddings(self, model):
        """Test embedding extraction from fusion model."""
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        tabular = torch.randn(batch_size, 11)

        output = model(images, tabular, return_embeddings=True)

        assert 'face_embedding' in output
        assert 'tabular_embedding' in output
        assert 'fused_embedding' in output

    @pytest.mark.parametrize("fusion_type", ["concat", "gated", "attention"])
    def test_fusion_types(self, fusion_type):
        """Test different fusion strategies."""
        model = MultimodalFusionModel(
            face_embedding_dim=64,
            tabular_embedding_dim=32,
            tabular_input_dim=11,
            fusion_type=fusion_type,
            pretrained_face=False
        )

        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        tabular = torch.randn(batch_size, 11)

        output = model(images, tabular)

        assert output['risk_logits'].shape == (batch_size, 2)


class TestClassicalModels:
    """Tests for classical sklearn models."""

    @pytest.fixture
    def data(self):
        """Create synthetic data fixture."""
        np.random.seed(42)
        n_samples = 200
        n_features = 11

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Split
        train_idx = int(0.7 * n_samples)
        val_idx = int(0.85 * n_samples)

        return {
            'X_train': X[:train_idx],
            'y_train': y[:train_idx],
            'X_val': X[train_idx:val_idx],
            'y_val': y[train_idx:val_idx],
            'X_test': X[val_idx:],
            'y_test': y[val_idx:]
        }

    def test_fit_and_predict(self, data):
        """Test model fitting and prediction."""
        models = ClassicalTabularModels(calibrate=False)
        models.fit(
            data['X_train'], data['y_train'],
            model_names=['svm_rbf', 'random_forest']
        )

        predictions = models.predict(data['X_test'], 'random_forest')

        assert len(predictions) == len(data['y_test'])
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, data):
        """Test probability predictions."""
        models = ClassicalTabularModels(calibrate=False)
        models.fit(data['X_train'], data['y_train'], model_names=['random_forest'])

        proba = models.predict_proba(data['X_test'], 'random_forest')

        assert proba.shape == (len(data['y_test']), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_evaluate(self, data):
        """Test model evaluation."""
        models = ClassicalTabularModels(calibrate=False)
        models.fit(data['X_train'], data['y_train'], model_names=['random_forest'])

        results = models.evaluate(data['X_test'], data['y_test'])

        assert 'random_forest' in results
        assert 'accuracy' in results['random_forest']
        assert 'f1' in results['random_forest']
        assert 'auc' in results['random_forest']

        # Metrics should be in valid range
        for metric in ['accuracy', 'f1', 'auc']:
            assert 0 <= results['random_forest'][metric] <= 1

    def test_calibration(self, data):
        """Test model calibration."""
        models = ClassicalTabularModels(calibrate=True)
        models.fit(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            model_names=['random_forest']
        )

        # Model should be in calibrated models
        assert 'random_forest' in models.calibrated_models

    def test_feature_importance(self, data):
        """Test feature importance extraction."""
        models = ClassicalTabularModels(calibrate=False)
        models.fit(data['X_train'], data['y_train'], model_names=['random_forest'])

        importance = models.get_feature_importance('random_forest')

        assert importance is not None
        assert len(importance) == data['X_train'].shape[1]


class TestClassWeights:
    """Tests for class weight calculation."""

    def test_balanced_weights(self):
        """Test weights for balanced classes."""
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        weights = get_class_weights(labels)

        assert len(weights) == 2
        assert torch.allclose(weights[0], weights[1])

    def test_imbalanced_weights(self):
        """Test weights for imbalanced classes."""
        labels = np.array([0] * 90 + [1] * 10)  # 90:10 imbalance
        weights = get_class_weights(labels)

        # Minority class should have higher weight
        assert weights[1] > weights[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
