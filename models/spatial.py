"""
Spatial branch for DeepVision-Forensics.

Extracts deep spatial and visual representations using an ImageNet-pretrained
EfficientNet-B3 backbone.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class SpatialBranch(nn.Module):
    """
    Pretrained EfficientNet-B3 spatial feature extractor.
    
    Accepts unnormalized [0, 1] RGB images, applies ImageNet normalization,
    and extracts a 1536-dimensional spatial embedding via Global Average Pooling.
    """
    def __init__(
        self,
        pretrained: bool = True,
        apply_imagenet_norm: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.apply_imagenet_norm = apply_imagenet_norm

        # ImageNet normalization statistics
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b3(weights=weights)

        # Retain feature extractor (convolutional stages)
        self.features = backbone.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Output spatial embedding dimension for EfficientNet-B3 is 1536
        self.embedding_dim = 1536

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """Returns the final convolutional block for Grad-CAM explainability."""
        return self.features[-1]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Applies ImageNet mean and std normalization to [0, 1] RGB tensors."""
        if not self.apply_imagenet_norm:
            return x
        return (x - self.imagenet_mean) / self.imagenet_std

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial backbone up to Global Average Pooling.
        
        Args:
            x: RGB tensor of shape (B, 3, H, W) in range [0, 1].
            
        Returns:
            Spatial feature embedding of shape (B, 1536).
        """
        x_norm = self.normalize(x)
        feat_map = self.features(x_norm)  # (B, 1536, H', W')
        pooled = self.gap(feat_map)       # (B, 1536, 1, 1)
        emb = self.flatten(pooled)        # (B, 1536)
        emb = self.dropout(emb)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class SpatialClassifier(nn.Module):
    """
    E1 Baseline: Spatial-Only Classifier.
    
    Combines the SpatialBranch with the standard classification head.
    """
    def __init__(
        self,
        pretrained: bool = True,
        apply_imagenet_norm: bool = True,
        dropout_p1: float = 0.4,
        dropout_p2: float = 0.2,
    ) -> None:
        super().__init__()
        self.spatial_branch = SpatialBranch(
            pretrained=pretrained,
            apply_imagenet_norm=apply_imagenet_norm,
        )
        in_dim = self.spatial_branch.embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB tensor of shape (B, 3, H, W) in range [0, 1].
        Returns:
            Single unscaled binary logit of shape (B, 1).
        """
        emb = self.spatial_branch(x)
        logit = self.classifier(emb)
        return logit
