import math
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

import sys
PROJECT_ROOT = Path("d:/DeepVision-Forensics")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fusion import build_model, DeepVisionFusionModel

def create_radial_frequency_mask(
    height: int = 224,
    width: int = 224,
    low_cutoff: float = 0.10,
    high_cutoff: float = 0.50
) -> torch.Tensor:
    """
    Creates a deterministic 2D radial mid-frequency mask for centered (fftshifted) 2D FFT.
    
    Coordinates are normalized such that:
      - Center (DC component) has radius r = 0.0
      - Axes edges have radius r = 1.0 (at distance H/2 or W/2)
      - Corners have radius r = sqrt(2) ~ 1.414
      
    Mask is inclusive: 1.0 where low_cutoff <= r <= high_cutoff, else 0.0.
    """
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    
    y = torch.arange(height, dtype=torch.float32).unsqueeze(1) # (H, 1)
    x = torch.arange(width, dtype=torch.float32).unsqueeze(0)  # (1, W)
    
    # Normalized radial distance
    norm_y = (y - cy) / (height / 2.0)
    norm_x = (x - cx) / (width / 2.0)
    r = torch.sqrt(norm_y**2 + norm_x**2) # (H, W)
    
    mask = ((r >= low_cutoff) & (r <= high_cutoff)).to(dtype=torch.float32)
    return mask.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

class ResidualCNN(nn.Module):
    """
    Lightweight CNN for extracting forensic features from reconstruction residuals.
    Input: (B * V, 5, 224, 224)
      - Channels 0..2: RGB residual
      - Channel 3: Grayscale residual
      - Channel 4: Masked mid-frequency residual magnitude
    Output: 256-D embedding
    """
    def __init__(self, in_channels: int = 5, out_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # -> 56x56
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) # -> 28x28
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ) # -> 14x14
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        ) # -> 7x7
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.global_pool(h)
        return torch.flatten(h, 1) # (B * V, 256)

class E25ReconstructionDetector(nn.Module):
    """
    E25: FIRE-Inspired Reconstruction-Residual Multi-View Forensic Detector.
    
    Integrates:
      1. EfficientNet-B3 spatial branch (1536-D) initialized from E6-C
      2. Standardized FFT frequency branch (256-D) initialized from E6-C
      3. Frozen Stable Diffusion VAE (AutoencoderKL) reconstruction residual branch
      4. Deterministic radial mid-frequency mask FFT on residual
      5. Lightweight 5-channel Residual CNN (256-D)
      6. Projections: 1536->256, 256->128, 256->128 => 512-D per view
      7. Multi-view self-attention over 5 views (1 global + 4 local)
      8. Explicit preservation of global view representation
      9. Fusion MLP -> 256-D forensic representation -> Final classifier
    """
    def __init__(
        self,
        e6c_checkpoint_path: Optional[Union[str, Path]] = None,
        vae_model_id: str = "stabilityai/sd-vae-ft-mse",
        num_views: int = 5,
        low_cutoff: float = 0.10,
        high_cutoff: float = 0.50
    ):
        super().__init__()
        self.num_views = num_views
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        
        # 1. Base E6-C Dual-Branch Model (Spatial + Frequency)
        self.base_model: DeepVisionFusionModel = build_model(
            experiment="E3",
            pretrained=False,
            freq_norm_strategy="standardize",
            freq_embedding_dim=256
        )
        
        # 2. Pretrained VAE (FROZEN)
        print(f"[*] Loading pretrained VAE from {vae_model_id}...")
        self.vae = AutoencoderKL.from_pretrained(vae_model_id)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
            
        # 3. Deterministic radial mid-frequency mask buffer
        mask = create_radial_frequency_mask(224, 224, low_cutoff=low_cutoff, high_cutoff=high_cutoff)
        self.register_buffer("mid_freq_mask", mask)
        
        # 4. Residual CNN
        self.residual_cnn = ResidualCNN(in_channels=5, out_dim=256)
        
        # 5. Domain Projections (per view)
        # Spatial: 1536 -> 256
        self.spatial_proj = nn.Sequential(
            nn.Linear(1536, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        # Frequency: 256 -> 128
        self.freq_proj = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        # Residual: 256 -> 128
        self.res_proj = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # Per-view representation dim = 256 + 128 + 128 = 512
        self.view_dim = 512
        
        # 6. Multi-View Attention
        self.view_attention = nn.Sequential(
            nn.Linear(self.view_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # Zero-initialize the final attention projection so initial weights are uniform (1/5)
        nn.init.zeros_(self.view_attention[2].weight)
        nn.init.zeros_(self.view_attention[2].bias)
        
        # 7. Global View Preservation & Fusion MLP
        # Concatenate explicit global view (512) + attention-aggregated views (512) = 1024
        self.fusion_mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
        
        # 8. AI/Real Classifier
        self.classifier = nn.Linear(256, 1)
        
        # Load E6-C weights if provided
        if e6c_checkpoint_path is not None:
            self.load_e6c_weights(e6c_checkpoint_path)

    def load_e6c_weights(self, checkpoint_path: Union[str, Path]):
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"E6-C checkpoint not found at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        
        # Extract spatial and frequency branch weights
        spatial_sd = {}
        freq_sd = {}
        for k, v in state_dict.items():
            if k.startswith("base_model.spatial_branch."):
                spatial_sd[k[len("base_model.spatial_branch."):]] = v
            elif k.startswith("base_model.frequency_branch."):
                freq_sd[k[len("base_model.frequency_branch."):]] = v
                
        if spatial_sd:
            msg_s = self.base_model.spatial_branch.load_state_dict(spatial_sd, strict=False)
            print(f"[*] Loaded E6-C spatial branch weights ({len(spatial_sd)} tensors, missing={len(msg_s.missing_keys)})")
        if freq_sd:
            msg_f = self.base_model.frequency_branch.load_state_dict(freq_sd, strict=False)
            print(f"[*] Loaded E6-C frequency branch weights ({len(freq_sd)} tensors, missing={len(msg_f.missing_keys)})")

    def compute_residual_features(
        self,
        x_flat: torch.Tensor # (B * V, 3, 224, 224) in [0, 1]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Reconstructs images with frozen VAE, computes spatial and mid-frequency residuals,
        and returns the 5-channel residual input along with diagnostic scalar statistics.
        """
        # 1. Scale to VAE input range [-1, 1]
        x_vae = x_flat * 2.0 - 1.0
        
        # 2. VAE Reconstruction under no_grad (VAE is completely frozen)
        with torch.no_grad():
            posterior = self.vae.encode(x_vae).latent_dist
            latent = posterior.mean # Deterministic mean
            recon_vae = self.vae.decode(latent).sample
            recon_01 = (recon_vae / 2.0 + 0.5).clamp(0.0, 1.0)
            
            # RGB residual
            residual_rgb = torch.abs(x_flat - recon_01) # (B*V, 3, 224, 224)
            
            # Grayscale residual (luminance weights)
            residual_gray = (
                0.2989 * residual_rgb[:, 0:1, :, :] +
                0.5870 * residual_rgb[:, 1:2, :, :] +
                0.1140 * residual_rgb[:, 2:3, :, :]
            ) # (B*V, 1, 224, 224)
            
            # Frequency representation of residual
            fft_res = torch.fft.fft2(residual_gray, norm="ortho")
            fft_res_shift = torch.fft.fftshift(fft_res, dim=(-2, -1))
            fft_mag = torch.log1p(torch.abs(fft_res_shift)) # (B*V, 1, 224, 224)
            
            # Radial mid-frequency masked magnitude
            mid_freq_mag = fft_mag * self.mid_freq_mask # (B*V, 1, 224, 224)
            
            # Standardize mid_freq_mag per image instance for numerical stability in CNN
            mid_freq_norm = (mid_freq_mag - mid_freq_mag.mean(dim=(-2, -1), keepdim=True)) / (
                mid_freq_mag.std(dim=(-2, -1), keepdim=True) + 1e-6
            )
            
            # Diagnostic scalar statistics
            res_mae = residual_rgb.mean(dim=(1, 2, 3)) # (B*V,)
            res_std = residual_rgb.std(dim=(1, 2, 3))  # (B*V,)
            total_energy = torch.sum(residual_rgb**2, dim=(1, 2, 3)) # (B*V,)
            mid_freq_energy = torch.sum(mid_freq_mag**2, dim=(1, 2, 3)) # (B*V,)
            
            diagnostics = {
                "residual_mae": res_mae,
                "residual_std": res_std,
                "total_energy": total_energy,
                "mid_freq_energy": mid_freq_energy,
                "raw_recon": recon_01,
                "raw_residual": residual_rgb,
                "mid_freq_mag": mid_freq_mag
            }

        # 3. Stack 5-channel residual input: [RGB (3), Grayscale (1), MidFreq (1)]
        residual_input = torch.cat([residual_rgb, residual_gray, mid_freq_norm], dim=1) # (B*V, 5, 224, 224)
        return residual_input, diagnostics

    def forward(
        self,
        x_views: torch.Tensor, # (B, V, 3, 224, 224), range [0, 1]
        cached_residuals: Optional[torch.Tensor] = None, # (B, V, 5, 224, 224)
        ablation_mode: Optional[str] = None, # None, 'no_residual', 'residual_only'
        return_view_weights: bool = False,
        return_diagnostics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        B, V, C, H, W = x_views.shape
        x_flat = x_views.view(B * V, C, H, W)
        
        # 1. Spatial branch (EfficientNet-B3 -> 1536-D)
        e_spatial_raw = self.base_model.spatial_branch(x_flat) # (B*V, 1536)
        
        # 2. Frequency branch (Standardized FFT -> 256-D)
        e_freq_raw = self.base_model.frequency_branch(x_flat) # (B*V, 256)
        
        # 3. Reconstruction-Residual branch (VAE -> 5-channel residual -> CNN -> 256-D)
        if cached_residuals is not None:
            residual_input = cached_residuals.view(B * V, 5, 224, 224).to(device=x_flat.device, dtype=x_flat.dtype)
            diagnostics = None
        else:
            residual_input, diagnostics = self.compute_residual_features(x_flat)
        e_res_raw = self.residual_cnn(residual_input) # (B*V, 256)
        
        # 4. Projections to unified per-view embedding
        e_sp_proj = self.spatial_proj(e_spatial_raw) # (B*V, 256)
        e_fr_proj = self.freq_proj(e_freq_raw)       # (B*V, 128)
        e_re_proj = self.res_proj(e_res_raw)        # (B*V, 128)
        
        # Apply Ablation Zeroing if requested
        if ablation_mode == "no_residual":
            e_re_proj = torch.zeros_like(e_re_proj)
        elif ablation_mode == "residual_only":
            e_sp_proj = torch.zeros_like(e_sp_proj)
            e_fr_proj = torch.zeros_like(e_fr_proj)
            
        # Concatenate: 256 + 128 + 128 = 512-D per view
        e_view_flat = torch.cat([e_sp_proj, e_fr_proj, e_re_proj], dim=1) # (B*V, 512)
        e_views = e_view_flat.view(B, V, self.view_dim) # (B, V, 512)
        
        # 5. Multi-View Attention
        attn_scores = self.view_attention(e_views) # (B, V, 1)
        attn_weights = torch.softmax(attn_scores, dim=1) # (B, V, 1)
        
        # Aggregated view representation (weighted combination across all 5 views)
        e_attn = torch.sum(attn_weights * e_views, dim=1) # (B, 512)
        
        # Explicit preservation of Global View (view 0)
        e_global = e_views[:, 0, :] # (B, 512)
        
        # 6. Combined Multi-View Forensic Embedding (1024-D)
        e_fused = torch.cat([e_global, e_attn], dim=1) # (B, 1024)
        
        # 7. Fusion MLP -> 256-D Forensic Representation
        forensic_rep = self.fusion_mlp(e_fused) # (B, 256)
        
        # 8. Classification Logit
        logits = self.classifier(forensic_rep) # (B, 1)
        
        outputs = [logits]
        if return_view_weights:
            outputs.append(attn_weights.squeeze(-1))
        if return_diagnostics:
            # Reshape diagnostics back to (B, V)
            for k in ["residual_mae", "residual_std", "total_energy", "mid_freq_energy"]:
                diagnostics[k] = diagnostics[k].view(B, V)
            outputs.append(diagnostics)
            
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
