"""
Feature extraction modules for HGPS AI System.
"""

from .face_cnn import FaceCNN, FaceFeatureExtractor
from .tabular_mlp import TabularMLP, TabularFeatureExtractor
from .fusion import MultimodalFusionModel, LateFusionClassifier

__all__ = [
    'FaceCNN',
    'FaceFeatureExtractor',
    'TabularMLP',
    'TabularFeatureExtractor',
    'MultimodalFusionModel',
    'LateFusionClassifier'
]
