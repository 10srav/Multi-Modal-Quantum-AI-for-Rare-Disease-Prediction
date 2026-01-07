"""
Multimodal Fusion Module for HGPS Detection

Combines facial image features with tabular clinical data
using various fusion strategies for improved prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import logging

from .face_cnn import FaceFeatureExtractor
from .tabular_mlp import TabularFeatureExtractor

logger = logging.getLogger(__name__)


class LateFusionClassifier(nn.Module):
    """
    Late fusion model that concatenates embeddings from face and tabular encoders.

    This is the simplest and most robust fusion strategy - extract embeddings
    from each modality independently, then concatenate and classify.
    """

    def __init__(
        self,
        face_embedding_dim: int = 256,
        tabular_embedding_dim: int = 64,
        tabular_input_dim: int = 11,
        num_risk_classes: int = 2,
        num_progression_classes: int = 3,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.3,
        pretrained_face: bool = True,
        freeze_face_backbone: bool = True
    ):
        """
        Args:
            face_embedding_dim: Dimension of face embeddings
            tabular_embedding_dim: Dimension of tabular embeddings
            tabular_input_dim: Number of tabular input features
            num_risk_classes: Risk output classes
            num_progression_classes: Progression output classes
            fusion_hidden_dim: Hidden dimension for fusion layers
            dropout: Dropout rate
            pretrained_face: Use pretrained face backbone
            freeze_face_backbone: Freeze face backbone initially
        """
        super().__init__()

        # Face encoder
        self.face_encoder = FaceFeatureExtractor(
            embedding_dim=face_embedding_dim,
            pretrained=pretrained_face,
            freeze_backbone=freeze_face_backbone,
            dropout=dropout
        )

        # Tabular encoder
        self.tabular_encoder = TabularFeatureExtractor(
            input_dim=tabular_input_dim,
            hidden_dims=[64, 128, 64],
            embedding_dim=tabular_embedding_dim,
            dropout=dropout
        )

        # Combined dimension
        combined_dim = face_embedding_dim + tabular_embedding_dim

        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.BatchNorm1d(fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        fusion_out_dim = fusion_hidden_dim // 2

        # Classification heads
        self.risk_head = nn.Linear(fusion_out_dim, num_risk_classes)
        self.progression_head = nn.Linear(fusion_out_dim, num_progression_classes)

        self.face_embedding_dim = face_embedding_dim
        self.tabular_embedding_dim = tabular_embedding_dim

        logger.info(
            f"LateFusionClassifier: face_emb={face_embedding_dim}, "
            f"tab_emb={tabular_embedding_dim}, fusion_hidden={fusion_hidden_dim}"
        )

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the fusion model.

        Args:
            image: Face image tensor (B, 3, H, W)
            tabular: Tabular features (B, tabular_input_dim)
            return_embeddings: Return individual modality embeddings

        Returns:
            Dictionary with predictions and optionally embeddings
        """
        # Extract embeddings
        face_emb = self.face_encoder(image)
        tabular_emb = self.tabular_encoder(tabular)

        # Concatenate
        combined = torch.cat([face_emb, tabular_emb], dim=1)

        # Fusion
        fusion_out = self.fusion_network(combined)

        # Classify
        risk_logits = self.risk_head(fusion_out)
        progression_logits = self.progression_head(fusion_out)

        output = {
            'risk_logits': risk_logits,
            'progression_logits': progression_logits
        }

        if return_embeddings:
            output['face_embedding'] = face_emb
            output['tabular_embedding'] = tabular_emb
            output['fused_embedding'] = fusion_out

        return output

    def predict_proba(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get probability predictions."""
        output = self.forward(image, tabular)
        return {
            'risk_probs': F.softmax(output['risk_logits'], dim=1),
            'progression_probs': F.softmax(output['progression_logits'], dim=1)
        }

    def unfreeze_face_backbone(self, from_layer: str = 'layer3'):
        """Unfreeze face backbone for fine-tuning."""
        self.face_encoder.unfreeze_backbone(from_layer)


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism that learns to weight modalities.

    Uses learned gates to determine how much each modality
    should contribute to the final representation.
    """

    def __init__(self, face_dim: int, tabular_dim: int, output_dim: int):
        super().__init__()

        self.face_gate = nn.Sequential(
            nn.Linear(face_dim + tabular_dim, output_dim),
            nn.Sigmoid()
        )

        self.tabular_gate = nn.Sequential(
            nn.Linear(face_dim + tabular_dim, output_dim),
            nn.Sigmoid()
        )

        self.face_proj = nn.Linear(face_dim, output_dim)
        self.tabular_proj = nn.Linear(tabular_dim, output_dim)

    def forward(
        self,
        face_emb: torch.Tensor,
        tabular_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply gated fusion.

        Returns:
            Tuple of (fused representation, gate weights dict)
        """
        combined = torch.cat([face_emb, tabular_emb], dim=1)

        # Compute gates
        g_face = self.face_gate(combined)
        g_tabular = self.tabular_gate(combined)

        # Project and gate
        face_proj = self.face_proj(face_emb)
        tabular_proj = self.tabular_proj(tabular_emb)

        fused = g_face * face_proj + g_tabular * tabular_proj

        gate_info = {
            'face_gate': g_face.mean(dim=1),
            'tabular_gate': g_tabular.mean(dim=1)
        }

        return fused, gate_info


class AttentionFusion(nn.Module):
    """
    Cross-attention fusion between modalities.

    Allows each modality to attend to the other for richer
    multimodal representations.
    """

    def __init__(
        self,
        face_dim: int,
        tabular_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Project to common dimension
        self.face_proj = nn.Linear(face_dim, output_dim)
        self.tabular_proj = nn.Linear(tabular_dim, output_dim)

        # Cross attention: face attends to tabular
        self.face_to_tabular_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross attention: tabular attends to face
        self.tabular_to_face_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        face_emb: torch.Tensor,
        tabular_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply cross-attention fusion."""
        # Project
        face_proj = self.face_proj(face_emb).unsqueeze(1)  # (B, 1, D)
        tabular_proj = self.tabular_proj(tabular_emb).unsqueeze(1)

        # Cross attention
        face_attended, attn_f2t = self.face_to_tabular_attn(
            face_proj, tabular_proj, tabular_proj
        )
        tabular_attended, attn_t2f = self.tabular_to_face_attn(
            tabular_proj, face_proj, face_proj
        )

        # Squeeze and concatenate
        face_attended = face_attended.squeeze(1)
        tabular_attended = tabular_attended.squeeze(1)

        combined = torch.cat([face_attended, tabular_attended], dim=1)
        fused = self.output_proj(combined)

        attn_info = {
            'face_to_tabular': attn_f2t.squeeze(1),
            'tabular_to_face': attn_t2f.squeeze(1)
        }

        return fused, attn_info


class MultimodalFusionModel(nn.Module):
    """
    Flexible multimodal fusion model with multiple fusion strategies.

    Supports:
    - 'concat': Simple concatenation (late fusion)
    - 'gated': Learned gating mechanism
    - 'attention': Cross-attention fusion
    """

    def __init__(
        self,
        face_embedding_dim: int = 256,
        tabular_embedding_dim: int = 64,
        tabular_input_dim: int = 11,
        fusion_type: str = 'concat',
        fusion_dim: int = 128,
        num_risk_classes: int = 2,
        num_progression_classes: int = 3,
        dropout: float = 0.3,
        pretrained_face: bool = True,
        freeze_face_backbone: bool = True
    ):
        """
        Args:
            face_embedding_dim: Face embedding dimension
            tabular_embedding_dim: Tabular embedding dimension
            tabular_input_dim: Number of tabular features
            fusion_type: 'concat', 'gated', or 'attention'
            fusion_dim: Dimension after fusion
            num_risk_classes: Risk classes
            num_progression_classes: Progression classes
            dropout: Dropout rate
            pretrained_face: Use pretrained face backbone
            freeze_face_backbone: Freeze backbone initially
        """
        super().__init__()

        self.fusion_type = fusion_type

        # Modality encoders
        self.face_encoder = FaceFeatureExtractor(
            embedding_dim=face_embedding_dim,
            pretrained=pretrained_face,
            freeze_backbone=freeze_face_backbone,
            dropout=dropout
        )

        self.tabular_encoder = TabularFeatureExtractor(
            input_dim=tabular_input_dim,
            hidden_dims=[64, 128, 64],
            embedding_dim=tabular_embedding_dim,
            dropout=dropout
        )

        # Fusion module
        if fusion_type == 'concat':
            self.fusion = None
            classifier_input_dim = face_embedding_dim + tabular_embedding_dim
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(
                face_embedding_dim, tabular_embedding_dim, fusion_dim
            )
            classifier_input_dim = fusion_dim
        elif fusion_type == 'attention':
            self.fusion = AttentionFusion(
                face_embedding_dim, tabular_embedding_dim, fusion_dim,
                num_heads=4, dropout=dropout
            )
            classifier_input_dim = fusion_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Shared classifier backbone
        self.classifier_backbone = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Classification heads
        self.risk_head = nn.Linear(64, num_risk_classes)
        self.progression_head = nn.Linear(64, num_progression_classes)

        logger.info(f"MultimodalFusionModel created with fusion_type={fusion_type}")

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            image: Face images (B, 3, H, W)
            tabular: Tabular features (B, input_dim)
            return_all: Return all intermediate representations

        Returns:
            Predictions dictionary
        """
        # Encode modalities
        face_emb = self.face_encoder(image)
        tabular_emb = self.tabular_encoder(tabular)

        # Fuse
        fusion_info = {}
        if self.fusion_type == 'concat':
            fused = torch.cat([face_emb, tabular_emb], dim=1)
        else:
            fused, fusion_info = self.fusion(face_emb, tabular_emb)

        # Classify
        features = self.classifier_backbone(fused)
        risk_logits = self.risk_head(features)
        progression_logits = self.progression_head(features)

        output = {
            'risk_logits': risk_logits,
            'progression_logits': progression_logits
        }

        if return_all:
            output['face_embedding'] = face_emb
            output['tabular_embedding'] = tabular_emb
            output['fused_representation'] = fused
            output['fusion_info'] = fusion_info

        return output

    def predict_proba(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get probability predictions."""
        output = self.forward(image, tabular)
        return {
            'risk_probs': F.softmax(output['risk_logits'], dim=1),
            'progression_probs': F.softmax(output['progression_logits'], dim=1)
        }

    def unfreeze_face_backbone(self, from_layer: str = 'layer3'):
        """Unfreeze face backbone for fine-tuning."""
        self.face_encoder.unfreeze_backbone(from_layer)


def get_fusion_model(
    fusion_type: str = 'concat',
    tabular_input_dim: int = 11,
    config: Optional[Dict] = None,
    device: str = 'auto'
) -> Tuple[nn.Module, torch.device]:
    """
    Factory function to create a fusion model.

    Args:
        fusion_type: Type of fusion ('concat', 'gated', 'attention')
        tabular_input_dim: Number of tabular features
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
        config = {
            'face_embedding_dim': 256,
            'tabular_embedding_dim': 64,
            'fusion_dim': 128,
            'dropout': 0.3,
            'pretrained_face': True,
            'freeze_face_backbone': True
        }

    config['tabular_input_dim'] = tabular_input_dim
    config['fusion_type'] = fusion_type

    model = MultimodalFusionModel(**config)
    model = model.to(device)

    logger.info(f"Created {fusion_type} fusion model on {device}")

    return model, device


if __name__ == "__main__":
    # Test fusion models
    print("Testing Multimodal Fusion models...")

    batch_size = 8
    tabular_dim = 11

    for fusion_type in ['concat', 'gated', 'attention']:
        print(f"\nTesting {fusion_type} fusion:")

        model, device = get_fusion_model(
            fusion_type=fusion_type,
            tabular_input_dim=tabular_dim
        )

        # Dummy inputs
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        tabular = torch.randn(batch_size, tabular_dim).to(device)

        # Forward pass
        output = model(images, tabular, return_all=True)

        print(f"  Risk logits: {output['risk_logits'].shape}")
        print(f"  Progression logits: {output['progression_logits'].shape}")
        print(f"  Face embedding: {output['face_embedding'].shape}")
        print(f"  Tabular embedding: {output['tabular_embedding'].shape}")
        print(f"  Fused representation: {output['fused_representation'].shape}")

        # Test probabilities
        probs = model.predict_proba(images, tabular)
        print(f"  Risk probs sum: {probs['risk_probs'].sum(dim=1)}")

        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {params:,}")
        print(f"  Trainable params: {trainable:,}")

    print("\nAll fusion models tested successfully!")
