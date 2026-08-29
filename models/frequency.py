"""
Frequency branch for DeepVision-Forensics.

Extracts spectral representations via 2D Fast Fourier Transform (FFT)
log-magnitude spectrum and encodes high-frequency periodic artifacts using
a lightweight 4-block CNN.
"""

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn


class FFTTransform(nn.Module):
    """
    2D Fast Fourier Transform (FFT) pipeline for spectral artifact extraction.
    
    Accepts unnormalized [0, 1] RGB tensors, computes 1-channel luminance,
    calculates centered 2D FFT log-magnitude spectrum, and applies
    configurable normalization.
    """
    def __init__(
        self,
        norm_strategy: Literal["minmax", "standardize", "instance_norm", "none"] = "minmax",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.norm_strategy = norm_strategy
        self.eps = eps

        # ITU-R BT.601-2 luma transform coefficients: Y = 0.299R + 0.587G + 0.114B
        self.register_buffer(
            "rgb_weights",
            torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        )

        if norm_strategy == "instance_norm":
            self.inst_norm = nn.InstanceNorm2d(1, affine=False)
        else:
            self.inst_norm = None

    def rgb_to_luminance(self, x: torch.Tensor) -> torch.Tensor:
        """Converts RGB tensor (B, 3, H, W) in [0, 1] to luminance (B, 1, H, W)."""
        if x.size(1) == 1:
            return x
        return (x * self.rgb_weights).sum(dim=1, keepdim=True)

    def normalize_spectrum(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Applies configurable normalization to the log-magnitude spectrum.
        
        Args:
            spec: Tensor of shape (B, 1, H, W).
        Returns:
            Normalized spectrum of shape (B, 1, H, W).
        """
        if self.norm_strategy == "minmax":
            # Per-sample min-max scaling to [0, 1]
            B = spec.size(0)
            spec_flat = spec.view(B, -1)
            min_val = spec_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            max_val = spec_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            return (spec - min_val) / (max_val - min_val + self.eps)

        elif self.norm_strategy == "standardize":
            # Per-sample zero-mean unit-variance
            B = spec.size(0)
            spec_flat = spec.view(B, -1)
            mean = spec_flat.mean(dim=1, keepdim=True).view(B, 1, 1, 1)
            std = spec_flat.std(dim=1, keepdim=True).view(B, 1, 1, 1)
            return (spec - mean) / (std + self.eps)

        elif self.norm_strategy == "instance_norm":
            if self.inst_norm is not None:
                return self.inst_norm(spec)
            return spec

        elif self.norm_strategy == "none":
            return spec

        else:
            raise ValueError(f"Unknown frequency normalization strategy: {self.norm_strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Unnormalized RGB image tensor of shape (B, 3, H, W) in range [0, 1].
            
        Returns:
            Centered, normalized 2D FFT log-magnitude spectrum of shape (B, 1, H, W).
        """
        # 1. Luminance conversion
        y = self.rgb_to_luminance(x)

        # 2. 2D Fast Fourier Transform
        fft_complex = torch.fft.fft2(y, dim=(-2, -1))

        # 3. Magnitude spectrum
        mag = torch.abs(fft_complex)

        # 4. Center DC component (zero-frequency) to image center
        mag_shifted = torch.fft.fftshift(mag, dim=(-2, -1))

        # 5. Logarithmic dynamic range compression: S(u,v) = log(1 + |F(u,v)|)
        log_mag = torch.log1p(mag_shifted)

        # 6. Configurable normalization
        norm_spec = self.normalize_spectrum(log_mag)

        return norm_spec


class FrequencyEncoder(nn.Module):
    """
    Lightweight 4-block CNN encoder designed specifically for 2D frequency spectra.
    
    Processes the centered log-magnitude spectrum and produces a 256-dimensional
    frequency embedding.
    """
    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # Block 1: (B, 1, 224, 224) -> (B, 32, 112, 112)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
        )

        # Block 2: (B, 32, 112, 112) -> (B, 64, 56, 56)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
        )

        # Block 3: (B, 64, 56, 56) -> (B, 128, 28, 28)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
        )

        # Block 4: (B, 128, 28, 28) -> (B, 256, 14, 14)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(256, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: 2D frequency spectrum of shape (B, 1, H, W).
            
        Returns:
            Frequency embedding of shape (B, embedding_dim).
        """
        x = self.block1(spec)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        pooled = self.gap(x)
        flat = self.flatten(pooled)
        proj = self.projection(flat)
        emb = self.layer_norm(proj)
        return emb


class FrequencyBranch(nn.Module):
    """
    Unified Frequency Branch combining FFTTransform + FrequencyEncoder.
    
    Accepts unnormalized [0, 1] RGB tensor and returns a 256-D frequency embedding.
    """
    def __init__(
        self,
        norm_strategy: Literal["minmax", "standardize", "instance_norm", "none"] = "minmax",
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transform = FFTTransform(norm_strategy=norm_strategy)
        self.encoder = FrequencyEncoder(in_channels=1, embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, return_spectrum: bool = False):
        """
        Args:
            x: Unnormalized RGB image tensor of shape (B, 3, H, W) in range [0, 1].
            return_spectrum: If True, returns (embedding, normalized_spectrum).
        """
        spec = self.transform(x)
        emb = self.encoder(spec)
        if return_spectrum:
            return emb, spec
        return emb


class FrequencyClassifier(nn.Module):
    """
    E2 Baseline: Frequency-Only Classifier.
    
    Combines FrequencyBranch with classification head.
    """
    def __init__(
        self,
        norm_strategy: Literal["minmax", "standardize", "instance_norm", "none"] = "minmax",
        embedding_dim: int = 256,
        dropout_p1: float = 0.4,
        dropout_p2: float = 0.2,
    ) -> None:
        super().__init__()
        self.frequency_branch = FrequencyBranch(
            norm_strategy=norm_strategy,
            embedding_dim=embedding_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Unnormalized RGB image tensor of shape (B, 3, H, W) in range [0, 1].
        Returns:
            Single unscaled binary logit of shape (B, 1).
        """
        emb = self.frequency_branch(x)
        logit = self.classifier(emb)
        return logit
