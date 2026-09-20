"""
E7 Learned Local/Global Feature-Level Fusion Architecture.

Combines pre-classifier representations from E6-C dual-branch backbone:
- Global spatial representation (1536-D) + global frequency representation (256-D)
- Four local spatial representations (1536-D) + four local frequency representations (256-D)
- Cross-attention between global query and local keys/values
- Peak local feature retention via elementwise maximum pooling
- Preserved E6-C learned-attention pooled context
- Final classification MLP
"""

import sys
from pathlib import Path
from typing import Dict, Union, Tuple, Optional
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e6_multiscale_inference.e6b_multiview_model import MultiViewE5Model


class E7LearnedFusionHead(nn.Module):
    """
    Compact trainable fusion head operating directly on spatial and frequency features
    extracted across 5 views (1 global + 4 corner crops).
    """
    def __init__(
        self,
        spatial_in_dim: int = 1536,
        freq_in_dim: int = 256,
        spatial_proj_dim: int = 256,
        freq_proj_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.spatial_proj_dim = spatial_proj_dim
        self.freq_proj_dim = freq_proj_dim
        self.token_dim = spatial_proj_dim + freq_proj_dim  # 384-D

        # Shared projection layers across views
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_in_dim, spatial_proj_dim),
            nn.LayerNorm(spatial_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_in_dim, freq_proj_dim),
            nn.LayerNorm(freq_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Cross-attention: Global query attends over local crops
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.cross_norm = nn.LayerNorm(self.token_dim)

        # Projection for original E6-C pooled embedding (1792 -> 256)
        self.e6c_proj = nn.Sequential(
            nn.Linear(1792, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Final fused vector: Global (384) + Cross-attended Local (384) + Max Local (384) + E6-C Proj (256) = 1408-D
        fused_in_dim = self.token_dim * 3 + 256  # 1408

        self.classifier = nn.Sequential(
            nn.Linear(fused_in_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, 1),
        )

    def forward(
        self,
        e_spatial_views: torch.Tensor,   # (B, 5, 1536)
        e_freq_views: torch.Tensor,      # (B, 5, 256)
        e_fused_views: torch.Tensor,     # (B, 5, 1792)
        view_attn_weights: torch.Tensor, # (B, 5, 1)
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        B, V, _ = e_spatial_views.shape

        # 1. Project spatial and frequency features for all views
        # Flatten (B*5) for shared linear projections
        spatial_flat = e_spatial_views.view(B * V, -1)
        freq_flat = e_freq_views.view(B * V, -1)

        h_spatial = self.spatial_proj(spatial_flat).view(B, V, self.spatial_proj_dim)  # (B, 5, 256)
        h_freq = self.freq_proj(freq_flat).view(B, V, self.freq_proj_dim)              # (B, 5, 128)

        # Combined view tokens: (B, 5, 384)
        h_views = torch.cat([h_spatial, h_freq], dim=-1)

        # Separate Global (View 0) and Local (Views 1..4)
        h_global = h_views[:, 0]                     # (B, 384)
        h_local = h_views[:, 1:]                     # (B, 4, 384)

        # 2. Cross-Attention: Global queries the 4 local crops
        query = h_global.unsqueeze(1)                # (B, 1, 384)
        attn_out, cross_weights = self.cross_attn(
            query=query, key=h_local, value=h_local
        )                                            # (B, 1, 384), (B, 1, 4)
        h_cross = self.cross_norm(attn_out.squeeze(1))  # (B, 384)

        # 3. Peak Local Representation (Elementwise max across local crops)
        h_max_local = torch.max(h_local, dim=1).values  # (B, 384)

        # 4. E6-C Pooled Context
        e6c_pooled = torch.sum(view_attn_weights * e_fused_views, dim=1)  # (B, 1792)
        h_e6c = self.e6c_proj(e6c_pooled)                                # (B, 256)

        # 5. Feature Fusion
        fused = torch.cat([h_global, h_cross, h_max_local, h_e6c], dim=-1)  # (B, 1408)

        # 6. Final Logit
        logits = self.classifier(fused)  # (B, 1)

        if return_diagnostics:
            diag = {
                "cross_attn_weights": cross_weights.squeeze(1),  # (B, 4)
                "view_attn_weights": view_attn_weights.squeeze(-1),  # (B, 5)
                "fused_embedding": fused,
            }
            return logits, diag

        return logits


class E7LearnedFusionModel(nn.Module):
    """
    End-to-End E7 Model wrapping the frozen/fine-tuned E6-C dual-branch backbone
    with the trainable E7LearnedFusionHead.
    """
    def __init__(
        self,
        e6c_checkpoint_path: Optional[Union[str, Path]] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.multiview_e5 = MultiViewE5Model(
            checkpoint_path=None,
            num_views=5,
            freq_norm_strategy="standardize",
            freq_embedding_dim=256,
        )

        if e6c_checkpoint_path is not None:
            ckpt_path = Path(e6c_checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.multiview_e5.load_state_dict(state_dict)
            print(f"[*] E7Model: Successfully loaded E6-C checkpoint from {ckpt_path}")

        if freeze_backbone:
            for param in self.multiview_e5.parameters():
                param.requires_grad = False
            self.multiview_e5.eval()
            print("[*] E7Model: Backbone weights FROZEN (requires_grad=False).")

        self.freeze_backbone = freeze_backbone
        self.fusion_head = E7LearnedFusionHead()

    def set_backbone_trainable(self, trainable: bool = True):
        """Toggles backbone parameter gradient requirements."""
        self.freeze_backbone = not trainable
        for param in self.multiview_e5.parameters():
            param.requires_grad = trainable
        if not trainable:
            self.multiview_e5.eval()
        else:
            self.multiview_e5.train()

    def forward(
        self,
        x_views: torch.Tensor,  # (B, 5, 3, 224, 224)
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        B, V, C, H, W = x_views.shape
        x_flat = x_views.view(B * V, C, H, W)

        # Context manager for frozen backbone
        with torch.set_grad_enabled(not self.freeze_backbone):
            feats = self.multiview_e5.base_model(x_flat, return_features=True)
            e_spatial = feats["spatial_embedding"].view(B, V, 1536)
            e_freq = feats["frequency_embedding"].view(B, V, 256)
            e_fused = feats["fused_embedding"].view(B, V, 1792)

            # View attention weights from base model
            attn_scores = self.multiview_e5.view_attention(e_fused)  # (B, 5, 1)
            view_attn_weights = torch.softmax(attn_scores, dim=1)    # (B, 5, 1)

        return self.fusion_head(
            e_spatial_views=e_spatial,
            e_freq_views=e_freq,
            e_fused_views=e_fused,
            view_attn_weights=view_attn_weights,
            return_diagnostics=return_diagnostics,
        )
