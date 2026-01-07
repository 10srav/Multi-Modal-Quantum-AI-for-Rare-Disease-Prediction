"""
Tabular MLP Module for HGPS Detection

Multi-layer perceptron for processing clinical/tabular features
including age, growth metrics, and phenotypic indicators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class TabularFeatureExtractor(nn.Module):
    """
    MLP backbone for extracting features from tabular clinical data.

    Processes normalized clinical features (age, height, weight, phenotype flags)
    and produces a learned embedding representation.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dims: List[int] = [64, 128, 64],
        embedding_dim: int = 64,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Number of input tabular features
            hidden_dims: List of hidden layer dimensions
            embedding_dim: Final embedding dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(embedding_dim))

        self.network = nn.Sequential(*layers)

        logger.info(
            f"TabularFeatureExtractor: input_dim={input_dim}, "
            f"hidden={hidden_dims}, embedding_dim={embedding_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from tabular features.

        Args:
            x: Input tensor (B, input_dim)

        Returns:
            Embedding tensor (B, embedding_dim)
        """
        return self.network(x)


class TabularMLP(nn.Module):
    """
    Complete Tabular MLP model with classification heads.

    Combines TabularFeatureExtractor with classification heads for
    HGPS risk and progression prediction.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dims: List[int] = [64, 128, 64],
        embedding_dim: int = 64,
        num_risk_classes: int = 2,
        num_progression_classes: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
            embedding_dim: Embedding dimension
            num_risk_classes: Risk classification classes
            num_progression_classes: Progression classification classes
            dropout: Dropout rate
            use_batch_norm: Use batch normalization
        """
        super().__init__()

        self.feature_extractor = TabularFeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

        # Risk classification head
        self.risk_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_risk_classes)
        )

        # Progression classification head
        self.progression_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_progression_classes)
        )

        self.num_risk_classes = num_risk_classes
        self.num_progression_classes = num_progression_classes

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the tabular MLP.

        Args:
            x: Input tensor (B, input_dim)
            return_embedding: If True, also return embedding

        Returns:
            Dictionary with logits and optionally embedding
        """
        embedding = self.feature_extractor(x)

        risk_logits = self.risk_head(embedding)
        progression_logits = self.progression_head(embedding)

        output = {
            'risk_logits': risk_logits,
            'progression_logits': progression_logits
        }

        if return_embedding:
            output['embedding'] = embedding

        return output

    def predict_proba(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get probability predictions."""
        output = self.forward(x)
        return {
            'risk_probs': F.softmax(output['risk_logits'], dim=1),
            'progression_probs': F.softmax(output['progression_logits'], dim=1)
        }

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding without classification."""
        return self.feature_extractor(x)


class ResidualBlock(nn.Module):
    """Residual block for deeper tabular networks."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class DeepTabularMLP(nn.Module):
    """
    Deeper tabular model with residual connections.

    For experiments requiring more model capacity.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_residual_blocks: int = 3,
        embedding_dim: int = 64,
        num_risk_classes: int = 2,
        num_progression_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_residual_blocks)
        ])

        # Embedding projection
        self.embedding_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # Classification heads
        self.risk_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_risk_classes)
        )

        self.progression_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_progression_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through deep tabular model."""
        h = self.input_proj(x)

        for block in self.residual_blocks:
            h = block(h)

        embedding = self.embedding_proj(h)

        risk_logits = self.risk_head(embedding)
        progression_logits = self.progression_head(embedding)

        output = {
            'risk_logits': risk_logits,
            'progression_logits': progression_logits
        }

        if return_embedding:
            output['embedding'] = embedding

        return output

    def predict_proba(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get probability predictions."""
        output = self.forward(x)
        return {
            'risk_probs': F.softmax(output['risk_logits'], dim=1),
            'progression_probs': F.softmax(output['progression_logits'], dim=1)
        }


class AttentionTabularMLP(nn.Module):
    """
    Tabular MLP with self-attention for feature interactions.

    Learns which features to attend to for better predictions.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        num_heads: int = 4,
        embedding_dim: int = 64,
        num_risk_classes: int = 2,
        num_progression_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        # Feature embedding (treat each feature as a token)
        self.feature_embed = nn.Linear(1, hidden_dim)

        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim * hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # Classification heads
        self.risk_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_risk_classes)
        )

        self.progression_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_progression_classes)
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with attention.

        Args:
            x: Input (B, input_dim)
            return_embedding: Return embedding
            return_attention: Return attention weights for interpretability

        Returns:
            Output dictionary
        """
        B = x.size(0)

        # Reshape: (B, input_dim) -> (B, input_dim, 1) -> (B, input_dim, hidden_dim)
        x = x.unsqueeze(-1)
        tokens = self.feature_embed(x)  # (B, input_dim, hidden_dim)

        # Self-attention
        attn_output, attn_weights = self.attention(
            tokens, tokens, tokens,
            need_weights=True
        )

        # Residual + norm
        tokens = self.layer_norm(tokens + attn_output)

        # Flatten and project
        flat = tokens.view(B, -1)  # (B, input_dim * hidden_dim)
        embedding = self.output_proj(flat)

        # Classify
        risk_logits = self.risk_head(embedding)
        progression_logits = self.progression_head(embedding)

        output = {
            'risk_logits': risk_logits,
            'progression_logits': progression_logits
        }

        if return_embedding:
            output['embedding'] = embedding

        if return_attention:
            output['attention_weights'] = attn_weights

        return output

    def predict_proba(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get probability predictions."""
        output = self.forward(x)
        return {
            'risk_probs': F.softmax(output['risk_logits'], dim=1),
            'progression_probs': F.softmax(output['progression_logits'], dim=1)
        }


def get_tabular_model(
    model_type: str = 'standard',
    input_dim: int = 11,
    config: Optional[Dict] = None,
    device: str = 'auto'
) -> Tuple[nn.Module, torch.device]:
    """
    Factory function to create a tabular model.

    Args:
        model_type: 'standard', 'deep', or 'attention'
        input_dim: Number of input features
        config: Model configuration
        device: Device to use

    Returns:
        Tuple of (model, device)
    """
    # Determine device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    if config is None:
        config = {}

    config['input_dim'] = input_dim

    if model_type == 'standard':
        model = TabularMLP(**config)
    elif model_type == 'deep':
        model = DeepTabularMLP(**config)
    elif model_type == 'attention':
        model = AttentionTabularMLP(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    logger.info(f"Created {model_type} tabular model on {device}")

    return model, device


if __name__ == "__main__":
    # Test tabular models
    print("Testing Tabular MLP models...")

    batch_size = 16
    input_dim = 11

    for model_type in ['standard', 'deep', 'attention']:
        print(f"\nTesting {model_type} model:")

        model, device = get_tabular_model(model_type=model_type, input_dim=input_dim)

        # Dummy input
        x = torch.randn(batch_size, input_dim).to(device)

        # Forward pass
        output = model(x, return_embedding=True)

        print(f"  Risk logits: {output['risk_logits'].shape}")
        print(f"  Progression logits: {output['progression_logits'].shape}")
        print(f"  Embedding: {output['embedding'].shape}")

        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {params:,}")

    print("\nAll tabular models tested successfully!")
