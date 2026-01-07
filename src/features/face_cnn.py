"""
Face CNN Module for HGPS Detection

Uses pre-trained ResNet18 with fine-tuning for facial feature extraction
and HGPS risk classification from facial images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FaceFeatureExtractor(nn.Module):
    """
    CNN backbone for extracting facial features using pretrained ResNet18.

    The model extracts a 512-dimensional embedding from facial images,
    which captures visual features relevant to HGPS detection.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            embedding_dim: Dimension of output face embedding
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze convolutional layers initially
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Get the number of features from backbone
        backbone_features = self.backbone.fc.in_features  # 512 for ResNet18

        # Remove original FC layer
        self.backbone.fc = nn.Identity()

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

        # Add custom head for face embedding
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        logger.info(f"FaceFeatureExtractor initialized with embedding_dim={embedding_dim}")

    def _freeze_backbone(self):
        """Freeze backbone convolutional layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_backbone(self, unfreeze_from: str = 'layer3'):
        """
        Unfreeze backbone layers from a certain point for fine-tuning.

        Args:
            unfreeze_from: Layer name to start unfreezing from
                         Options: 'layer1', 'layer2', 'layer3', 'layer4', 'all'
        """
        if unfreeze_from == 'all':
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            unfreeze = False
            for name, child in self.backbone.named_children():
                if name == unfreeze_from:
                    unfreeze = True
                if unfreeze:
                    for param in child.parameters():
                        param.requires_grad = True

        logger.info(f"Backbone unfrozen from {unfreeze_from}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract face embedding from image.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Face embedding of shape (B, embedding_dim)
        """
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        return embedding


class FaceCNN(nn.Module):
    """
    Complete Face CNN model with classification heads.

    Combines FaceFeatureExtractor with classification heads for:
    - Binary HGPS risk prediction
    - Progression speed classification (3 classes)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_risk_classes: int = 2,
        num_progression_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            embedding_dim: Size of face embedding
            num_risk_classes: Number of risk classes (default 2: HGPS/control)
            num_progression_classes: Number of progression classes (default 3: slow/mod/rapid)
            pretrained: Use pretrained backbone
            freeze_backbone: Freeze backbone initially
            dropout: Dropout rate
        """
        super().__init__()

        self.feature_extractor = FaceFeatureExtractor(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )

        # Risk classification head
        self.risk_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_risk_classes)
        )

        # Progression classification head
        self.progression_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_progression_classes)
        )

        self.num_risk_classes = num_risk_classes
        self.num_progression_classes = num_progression_classes

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the face CNN.

        Args:
            x: Input tensor (B, 3, H, W)
            return_embedding: If True, also return face embedding

        Returns:
            Dictionary with 'risk_logits', 'progression_logits', and optionally 'embedding'
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
        """
        Get probability predictions.

        Args:
            x: Input tensor

        Returns:
            Dictionary with 'risk_probs' and 'progression_probs'
        """
        output = self.forward(x)
        return {
            'risk_probs': F.softmax(output['risk_logits'], dim=1),
            'progression_probs': F.softmax(output['progression_logits'], dim=1)
        }

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embedding without classification."""
        return self.feature_extractor(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    This loss focuses training on hard examples and down-weights
    easy examples, which is crucial for imbalanced HGPS detection.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Class weights tensor
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits (B, C)
            targets: Ground truth labels (B,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def get_face_model(
    config: Optional[Dict] = None,
    device: str = 'auto'
) -> Tuple[FaceCNN, torch.device]:
    """
    Factory function to create a FaceCNN model.

    Args:
        config: Model configuration dictionary
        device: Device to use ('auto', 'cuda', 'cpu', 'mps')

    Returns:
        Tuple of (model, device)
    """
    if config is None:
        config = {
            'embedding_dim': 256,
            'num_risk_classes': 2,
            'num_progression_classes': 3,
            'pretrained': True,
            'freeze_backbone': True,
            'dropout': 0.3
        }

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

    model = FaceCNN(**config)
    model = model.to(device)

    logger.info(f"FaceCNN created on device: {device}")

    return model, device


if __name__ == "__main__":
    # Test the face CNN
    print("Testing FaceCNN...")

    model, device = get_face_model()
    print(f"Model device: {device}")

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    # Forward pass
    output = model(dummy_input, return_embedding=True)

    print(f"Risk logits shape: {output['risk_logits'].shape}")
    print(f"Progression logits shape: {output['progression_logits'].shape}")
    print(f"Embedding shape: {output['embedding'].shape}")

    # Test probability output
    probs = model.predict_proba(dummy_input)
    print(f"Risk probs shape: {probs['risk_probs'].shape}")
    print(f"Risk probs sum: {probs['risk_probs'].sum(dim=1)}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nFaceCNN test passed!")
