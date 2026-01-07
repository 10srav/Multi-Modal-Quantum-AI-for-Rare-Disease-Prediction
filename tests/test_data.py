"""
Tests for data module.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    generate_hgps_tabular_data,
    generate_synthetic_face_image,
    TabularPreprocessor,
    FacePreprocessor,
    create_data_splits,
    get_qml_features,
    HGPSMultimodalDataset
)


class TestDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_tabular_data_shape(self):
        """Test that tabular data has correct shape."""
        n_hgps, n_controls = 10, 40
        df = generate_hgps_tabular_data(n_hgps=n_hgps, n_controls=n_controls)

        assert len(df) == n_hgps + n_controls
        assert 'risk_label' in df.columns
        assert 'progression_label' in df.columns

    def test_generate_tabular_data_labels(self):
        """Test that labels are correctly distributed."""
        n_hgps, n_controls = 10, 40
        df = generate_hgps_tabular_data(n_hgps=n_hgps, n_controls=n_controls)

        assert (df['risk_label'] == 1).sum() == n_hgps
        assert (df['risk_label'] == 0).sum() == n_controls

    def test_generate_tabular_data_features(self):
        """Test that all expected features are present."""
        df = generate_hgps_tabular_data(n_hgps=5, n_controls=10)

        expected_features = [
            'patient_id', 'age', 'height_cm', 'weight_kg', 'bmi',
            'height_z_score', 'weight_z_score', 'small_jaw',
            'prominent_eyes', 'thin_skin', 'hair_loss', 'lmna_mut'
        ]

        for feature in expected_features:
            assert feature in df.columns, f"Missing feature: {feature}"

    def test_hgps_characteristics(self):
        """Test that HGPS samples have expected characteristics."""
        df = generate_hgps_tabular_data(n_hgps=50, n_controls=50)

        hgps = df[df['risk_label'] == 1]
        controls = df[df['risk_label'] == 0]

        # HGPS should have lower z-scores on average
        assert hgps['height_z_score'].mean() < controls['height_z_score'].mean()
        assert hgps['weight_z_score'].mean() < controls['weight_z_score'].mean()

        # HGPS should have higher phenotype feature frequency
        assert hgps['small_jaw'].mean() > controls['small_jaw'].mean()
        assert hgps['lmna_mut'].mean() > controls['lmna_mut'].mean()

    def test_reproducibility(self):
        """Test that data generation is reproducible with seed."""
        df1 = generate_hgps_tabular_data(n_hgps=10, n_controls=20, random_state=42)
        df2 = generate_hgps_tabular_data(n_hgps=10, n_controls=20, random_state=42)

        pd.testing.assert_frame_equal(df1, df2)


class TestSyntheticImages:
    """Tests for synthetic face image generation."""

    def test_image_shape(self):
        """Test that generated images have correct shape."""
        image = generate_synthetic_face_image(is_hgps=True, image_size=(224, 224))

        assert image.shape == (224, 224, 3)
        assert image.dtype == np.uint8

    def test_image_values(self):
        """Test that image values are in valid range."""
        image = generate_synthetic_face_image(is_hgps=False)

        assert image.min() >= 0
        assert image.max() <= 255

    def test_hgps_vs_control_different(self):
        """Test that HGPS and control images differ."""
        hgps_img = generate_synthetic_face_image(is_hgps=True, random_state=42)
        control_img = generate_synthetic_face_image(is_hgps=False, random_state=42)

        # Images should not be identical
        assert not np.array_equal(hgps_img, control_img)


class TestPreprocessors:
    """Tests for data preprocessors."""

    def test_tabular_preprocessor_fit_transform(self):
        """Test tabular preprocessor fit and transform."""
        df = generate_hgps_tabular_data(n_hgps=10, n_controls=40)
        preprocessor = TabularPreprocessor()

        features = preprocessor.fit_transform(df)

        assert features.shape[0] == len(df)
        assert features.shape[1] == 11  # Number of feature columns
        assert features.dtype == np.float32

    def test_tabular_preprocessor_consistency(self):
        """Test that preprocessor produces consistent transformations."""
        df = generate_hgps_tabular_data(n_hgps=10, n_controls=40)
        preprocessor = TabularPreprocessor()

        features1 = preprocessor.fit_transform(df)
        features2 = preprocessor.transform(df)

        np.testing.assert_array_almost_equal(features1, features2)

    def test_face_preprocessor_detect(self):
        """Test face preprocessor detection and alignment."""
        preprocessor = FacePreprocessor(target_size=(224, 224))

        # Create a simple test image
        test_image = np.ones((300, 300, 3), dtype=np.uint8) * 128

        result = preprocessor.detect_and_align(test_image)

        assert result.shape == (224, 224, 3)


class TestDataSplits:
    """Tests for data splitting functionality."""

    def test_split_sizes(self):
        """Test that splits have correct proportions."""
        df = generate_hgps_tabular_data(n_hgps=20, n_controls=80)
        train, val, test = create_data_splits(df, test_size=0.15, val_size=0.15)

        total = len(train) + len(val) + len(test)
        assert total == len(df)

        # Check approximate proportions (within 5%)
        assert abs(len(test) / len(df) - 0.15) < 0.05
        assert abs(len(val) / len(df) - 0.15) < 0.05

    def test_stratification(self):
        """Test that splits maintain class distribution."""
        df = generate_hgps_tabular_data(n_hgps=20, n_controls=80)
        train, val, test = create_data_splits(df)

        original_ratio = (df['risk_label'] == 1).mean()

        for split, name in [(train, 'train'), (val, 'val'), (test, 'test')]:
            split_ratio = (split['risk_label'] == 1).mean()
            # Ratio should be within 10% of original
            assert abs(split_ratio - original_ratio) < 0.1, f"{name} split not stratified"


class TestQMLFeatures:
    """Tests for QML feature extraction."""

    def test_qml_features_shape(self):
        """Test QML feature extraction shape."""
        df = generate_hgps_tabular_data(n_hgps=10, n_controls=40)
        preprocessor = TabularPreprocessor()
        preprocessor.fit(df)

        features, risk_labels, prog_labels = get_qml_features(df, preprocessor)

        assert features.shape[0] == len(df)
        assert features.shape[1] == 6  # Default 6 QML features
        assert len(risk_labels) == len(df)
        assert len(prog_labels) == len(df)

    def test_qml_features_normalized(self):
        """Test that QML features are normalized to [0, 1]."""
        df = generate_hgps_tabular_data(n_hgps=10, n_controls=40)
        preprocessor = TabularPreprocessor()
        preprocessor.fit(df)

        features, _, _ = get_qml_features(df, preprocessor)

        # Allow small floating point tolerance
        assert features.min() >= -1e-6
        assert features.max() <= 1.0 + 1e-6


class TestDataset:
    """Tests for PyTorch dataset classes."""

    def test_multimodal_dataset_length(self):
        """Test dataset length matches data."""
        df = generate_hgps_tabular_data(n_hgps=5, n_controls=15)
        preprocessor = TabularPreprocessor()
        features = preprocessor.fit_transform(df)

        # Add dummy image paths
        df['image_path'] = 'dummy.png'

        dataset = HGPSMultimodalDataset(df, features)

        assert len(dataset) == len(df)

    def test_multimodal_dataset_item(self):
        """Test dataset returns correct item structure."""
        df = generate_hgps_tabular_data(n_hgps=5, n_controls=15)
        preprocessor = TabularPreprocessor()
        features = preprocessor.fit_transform(df)

        df['image_path'] = 'dummy.png'

        dataset = HGPSMultimodalDataset(df, features)
        item = dataset[0]

        assert 'image' in item
        assert 'tabular' in item
        assert 'risk_label' in item
        assert 'progression_label' in item
        assert 'patient_id' in item

        assert isinstance(item['image'], torch.Tensor)
        assert isinstance(item['tabular'], torch.Tensor)
        assert item['image'].shape == (3, 224, 224)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
