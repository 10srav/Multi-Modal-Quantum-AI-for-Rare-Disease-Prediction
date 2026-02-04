"""
Fusion-Quantum Bridge Module for HGPS Detection

Bridges multi-modal fusion model embeddings to quantum ML models,
allowing QSVM and QNN to use rich fused representations instead of raw features.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class FusionQuantumBridge:
    """
    Bridges fusion model embeddings to quantum ML.

    Extracts fused embeddings from the multi-modal fusion model,
    reduces dimensionality via PCA, and scales for quantum encoding.

    This allows quantum models (QSVM, QNN) to operate on rich
    multi-modal representations rather than raw tabular features.
    """

    def __init__(
        self,
        fusion_model: Any,
        device: Any,
        n_quantum_features: int = 6,
        random_state: int = 42
    ):
        """
        Initialize the fusion-quantum bridge.

        Args:
            fusion_model: Trained multi-modal fusion model
            device: PyTorch device
            n_quantum_features: Number of features for quantum models (default 6)
            random_state: Random seed for reproducibility
        """
        self.fusion_model = fusion_model
        self.device = device
        self.n_quantum_features = n_quantum_features
        self.random_state = random_state

        # Initialize PCA and scaler (will be fit later)
        self.pca = PCA(n_components=n_quantum_features, random_state=random_state)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.is_fitted = False
        self.embedding_dim = None

        logger.info(
            f"FusionQuantumBridge initialized: n_quantum_features={n_quantum_features}"
        )

    def extract_embedding(
        self,
        image_tensor: 'torch.Tensor',
        tabular_tensor: 'torch.Tensor'
    ) -> np.ndarray:
        """
        Extract fused embedding from fusion model.

        Args:
            image_tensor: Face image tensor (B, 3, H, W)
            tabular_tensor: Tabular features tensor (B, 11)

        Returns:
            Fused embedding as numpy array (B, embedding_dim)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for embedding extraction")

        # Set model to evaluation mode (PyTorch's model.eval(), not Python eval())
        self.fusion_model.train(False)
        with torch.no_grad():
            output = self.fusion_model(
                image_tensor.to(self.device),
                tabular_tensor.to(self.device),
                return_embeddings=True
            )
            # Get the fused embedding (64-dim from LateFusionClassifier)
            embedding = output['fused_embedding'].cpu().numpy()

        return embedding

    def extract_embeddings_batch(
        self,
        images: 'torch.Tensor',
        tabular: np.ndarray,
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of samples.

        Args:
            images: Face images tensor (N, 3, H, W)
            tabular: Tabular features array (N, 11)
            batch_size: Batch size for processing

        Returns:
            Embeddings array (N, embedding_dim)
        """
        import torch

        n_samples = len(tabular)
        all_embeddings = []

        # Set model to evaluation mode
        self.fusion_model.train(False)
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)

                batch_images = images[i:end_idx].to(self.device)
                batch_tabular = torch.from_numpy(
                    tabular[i:end_idx]
                ).float().to(self.device)

                output = self.fusion_model(
                    batch_images,
                    batch_tabular,
                    return_embeddings=True
                )

                all_embeddings.append(output['fused_embedding'].cpu().numpy())

        return np.vstack(all_embeddings)

    def fit(self, embeddings: np.ndarray) -> 'FusionQuantumBridge':
        """
        Fit PCA and scaler on training embeddings.

        Args:
            embeddings: Training embeddings (N, embedding_dim)

        Returns:
            self
        """
        self.embedding_dim = embeddings.shape[1]

        # Fit PCA to reduce dimensionality
        n_components = min(self.n_quantum_features, embeddings.shape[1], embeddings.shape[0])
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        reduced = self.pca.fit_transform(embeddings)

        # Fit scaler to normalize to [0, 1]
        self.scaler.fit(reduced)

        self.is_fitted = True

        explained_var = sum(self.pca.explained_variance_ratio_) * 100
        logger.info(
            f"FusionQuantumBridge fitted: {self.embedding_dim}D -> {n_components}D "
            f"(explained variance: {explained_var:.1f}%)"
        )

        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings to quantum-ready features.

        Args:
            embeddings: Fused embeddings (N, embedding_dim)

        Returns:
            Quantum-ready features (N, n_quantum_features), scaled to [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("FusionQuantumBridge must be fitted before transform")

        reduced = self.pca.transform(embeddings)
        scaled = self.scaler.transform(reduced)

        # Clip to [0, 1] to handle edge cases
        return np.clip(scaled, 0, 1)

    def get_quantum_features(
        self,
        image_tensor: 'torch.Tensor',
        tabular_tensor: 'torch.Tensor'
    ) -> np.ndarray:
        """
        One-step extraction: extract embedding, project via PCA, scale to [0, 1].

        This is the main method for inference - takes raw inputs and
        produces quantum-ready features.

        Args:
            image_tensor: Face image tensor (B, 3, H, W)
            tabular_tensor: Tabular features tensor (B, 11)

        Returns:
            Quantum features (B, n_quantum_features), scaled to [0, 1]
        """
        embedding = self.extract_embedding(image_tensor, tabular_tensor)
        return self.transform(embedding)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance information from PCA.

        Returns:
            Dictionary with PCA components and explained variance
        """
        if not self.is_fitted:
            raise ValueError("FusionQuantumBridge must be fitted first")

        return {
            'components': self.pca.components_,
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'total_explained_variance': sum(self.pca.explained_variance_ratio_)
        }


def create_fusion_quantum_bridge(
    fusion_model: Any,
    device: Any,
    train_images: 'torch.Tensor',
    train_tabular: np.ndarray,
    n_quantum_features: int = 6,
    batch_size: int = 16
) -> FusionQuantumBridge:
    """
    Convenience function to create and fit a FusionQuantumBridge.

    Args:
        fusion_model: Trained fusion model
        device: PyTorch device
        train_images: Training face images
        train_tabular: Training tabular features
        n_quantum_features: Number of quantum features
        batch_size: Batch size for embedding extraction

    Returns:
        Fitted FusionQuantumBridge
    """
    bridge = FusionQuantumBridge(
        fusion_model=fusion_model,
        device=device,
        n_quantum_features=n_quantum_features
    )

    # Extract training embeddings
    logger.info("Extracting training embeddings for quantum bridge...")
    train_embeddings = bridge.extract_embeddings_batch(
        train_images, train_tabular, batch_size=batch_size
    )

    # Fit the bridge
    bridge.fit(train_embeddings)

    return bridge


if __name__ == "__main__":
    print("Testing FusionQuantumBridge...")

    # Create mock embeddings for testing
    np.random.seed(42)
    n_samples = 100
    embedding_dim = 64  # Typical fusion embedding dimension

    mock_embeddings = np.random.randn(n_samples, embedding_dim)

    # Create a simple mock "fusion model" for testing
    class MockFusionModel:
        def train(self, mode=True):
            pass

        def __call__(self, image, tabular, return_embeddings=False):
            import torch
            batch_size = tabular.shape[0]
            output = {
                'risk_logits': torch.randn(batch_size, 2),
                'progression_logits': torch.randn(batch_size, 3)
            }
            if return_embeddings:
                output['fused_embedding'] = torch.randn(batch_size, 64)
            return output

    # Test bridge fitting
    bridge = FusionQuantumBridge(
        fusion_model=MockFusionModel(),
        device='cpu',
        n_quantum_features=6
    )

    # Fit on mock embeddings
    bridge.fit(mock_embeddings)

    # Transform
    quantum_features = bridge.transform(mock_embeddings[:10])

    print(f"Input embeddings shape: {mock_embeddings[:10].shape}")
    print(f"Output quantum features shape: {quantum_features.shape}")
    print(f"Feature range: [{quantum_features.min():.3f}, {quantum_features.max():.3f}]")

    # Check feature importance
    importance = bridge.get_feature_importance()
    print(f"Total explained variance: {importance['total_explained_variance']:.2%}")

    print("\nFusionQuantumBridge test complete!")
