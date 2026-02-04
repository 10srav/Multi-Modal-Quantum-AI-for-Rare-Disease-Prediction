"""
Quantum Machine Learning Module for HGPS Detection

Implements:
- Quantum Support Vector Machine (QSVM)
- Variational Quantum Classifier (QNN/VQC)
- Quantum feature maps and ansatzes
- Fusion-Quantum bridge for multi-modal embeddings
"""

from .qsvm import QuantumSVM, train_qsvm, evaluate_qsvm
from .qnn import QuantumNeuralNetwork, train_qnn, evaluate_qnn
from .quantum_features import (
    create_feature_map,
    create_ansatz,
    ZZFeatureMap,
    RealAmplitudes
)
from .fusion_quantum import FusionQuantumBridge, create_fusion_quantum_bridge

__all__ = [
    'QuantumSVM',
    'QuantumNeuralNetwork',
    'train_qsvm',
    'train_qnn',
    'evaluate_qsvm',
    'evaluate_qnn',
    'create_feature_map',
    'create_ansatz',
    'ZZFeatureMap',
    'RealAmplitudes',
    'FusionQuantumBridge',
    'create_fusion_quantum_bridge'
]
