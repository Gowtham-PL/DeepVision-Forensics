"""
Dual-branch Spatial + Frequency Fusion Model for DeepVision-Forensics.

Combines the 1536-D EfficientNet-B3 spatial embedding with the 256-D
Fourier frequency embedding via direct concatenation into a 1792-D
fused representation, followed by a multi-layer classification head.
"""

from typing import Dict, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn

from models.spatial import SpatialBranch, SpatialClassifier
from models.frequency import FrequencyBranch, FrequencyClassifier


class DeepVisionFusionModel(nn.Module):
    """
    E3 Primary Architecture: Dual-Branch Spatial + Frequency Fusion.
    
    Accepts unnormalized [0, 1] RGB image tensors, routes them through:
    1. Spatial Branch: ImageNet-normalized -> Pretrained EfficientNet-B3 -> 1536-D
    2. Frequency Branch: Unnormalized -> 2D FFT Log-Mag -> 4-Block CNN -> 256-D
    
    Concatenates to 1792-D and outputs a single binary logit.
    """
    def __init__(
        self,
        spatial_pretrained: bool = True,
        freq_norm_strategy: Literal["minmax", "standardize", "instance_norm", "none"] = "minmax",
        freq_embedding_dim: int = 256,
        dropout_p1: float = 0.4,
        dropout_p2: float = 0.2,
    ) -> None:
        super().__init__()
        self.spatial_branch = SpatialBranch(
            pretrained=spatial_pretrained,
            apply_imagenet_norm=True,
        )
        self.frequency_branch = FrequencyBranch(
            norm_strategy=freq_norm_strategy,
            embedding_dim=freq_embedding_dim,
        )

        self.spatial_dim = self.spatial_branch.embedding_dim   # 1536
        self.freq_dim = self.frequency_branch.embedding_dim     # 256
        self.fused_dim = self.spatial_dim + self.freq_dim       # 1792

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p2),
            nn.Linear(128, 1),
        )

    def get_spatial_gradcam_layer(self) -> nn.Module:
        """Returns the final convolutional block of the spatial backbone for Grad-CAM."""
        return self.spatial_branch.get_target_layer_for_gradcam()

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Unnormalized RGB image tensor of shape (B, 3, H, W) in range [0, 1].
            return_features: If True, returns dictionary with intermediate representations.
            
        Returns:
            If return_features is False:
                Single unscaled binary logit tensor of shape (B, 1).
            If return_features is True:
                Dictionary containing 'logit', 'spatial_embedding', 'frequency_embedding',
                'fused_embedding', and 'fft_spectrum'.
        """
        # 1. Spatial branch (applies internal ImageNet normalization)
        e_spatial = self.spatial_branch(x)  # (B, 1536)

        # 2. Frequency branch (receives unnormalized [0, 1] RGB tensor)
        if return_features:
            e_freq, fft_spec = self.frequency_branch(x, return_spectrum=True)  # (B, 256), (B, 1, H, W)
        else:
            e_freq = self.frequency_branch(x, return_spectrum=False)           # (B, 256)
            fft_spec = None

        # 3. Direct feature concatenation
        e_fused = torch.cat([e_spatial, e_freq], dim=1)  # (B, 1792)

        # 4. Classification
        logit = self.classifier(e_fused)  # (B, 1)

        if return_features:
            return {
                "logit": logit,
                "spatial_embedding": e_spatial,
                "frequency_embedding": e_freq,
                "fused_embedding": e_fused,
                "fft_spectrum": fft_spec,
            }

        return logit


def build_model(
    experiment: Literal["E1", "E2", "E3"] = "E3",
    pretrained: bool = True,
    freq_norm_strategy: Literal["minmax", "standardize", "instance_norm", "none"] = "minmax",
    freq_embedding_dim: int = 256,
) -> nn.Module:
    """
    Factory function for building baseline and primary experimental models.
    
    Args:
        experiment: 'E1' (Spatial-only), 'E2' (Frequency-only), or 'E3' (Dual-branch Fusion).
        pretrained: Whether to load ImageNet pretrained weights for EfficientNet-B3.
        freq_norm_strategy: Frequency log-magnitude normalization method.
        freq_embedding_dim: Dimensionality of the frequency embedding (default 256).
        
    Returns:
        Instantiated PyTorch model.
    """
    if experiment == "E1":
        return SpatialClassifier(
            pretrained=pretrained,
            apply_imagenet_norm=True,
        )
    elif experiment == "E2":
        return FrequencyClassifier(
            norm_strategy=freq_norm_strategy,
            embedding_dim=freq_embedding_dim,
        )
    elif experiment == "E3":
        return DeepVisionFusionModel(
            spatial_pretrained=pretrained,
            freq_norm_strategy=freq_norm_strategy,
            freq_embedding_dim=freq_embedding_dim,
        )
    else:
        raise ValueError(f"Unknown experiment configuration: {experiment}. Expected 'E1', 'E2', or 'E3'.")
