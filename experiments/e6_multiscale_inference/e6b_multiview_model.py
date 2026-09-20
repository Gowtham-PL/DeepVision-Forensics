import sys
from pathlib import Path
from typing import Dict, Union, Tuple, Optional
import torch
import torch.nn as nn

PROJECT_ROOT = Path("d:/DeepVision-Forensics")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fusion import build_model, DeepVisionFusionModel

class MultiViewE5Model(nn.Module):
    """
    E6-B Multi-View Training Model.
    Wraps DeepVision-E5 (Spatial + Frequency Dual-Branch) with a Learnable Self-Attention
    View Aggregation Head over 5 spatial views (1 global + 4 corner crops).
    
    Initialization:
    Loads pre-trained E5 weights (`best_model.pt`). Attention logits are initialized to zero
    so that view weights start as uniform average (0.2 per view), exactly matching the best E6-A
    Mean Aggregation baseline at initial step.
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        num_views: int = 5,
        freq_norm_strategy: str = "standardize",
        freq_embedding_dim: int = 256,
        dropout_p1: float = 0.4,
        dropout_p2: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        
        # Base E5 dual-domain fusion model
        self.base_model: DeepVisionFusionModel = build_model(
            experiment="E3",
            pretrained=False,
            freq_norm_strategy=freq_norm_strategy,
            freq_embedding_dim=freq_embedding_dim,
        )
        
        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt.get("model_state_dict", ckpt)
                self.base_model.load_state_dict(state_dict)
                print(f"[*] MultiViewE5Model: Loaded E5 base weights from {ckpt_path}")
            else:
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        self.fused_dim = self.base_model.fused_dim # 1792
        
        # Learnable view attention module (maps 1792-D fused embedding -> 1-D view attention score)
        self.view_attention = nn.Sequential(
            nn.Linear(self.fused_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # Initialize final linear layer of attention to zero so initial attention is uniform (1/num_views)
        nn.init.zeros_(self.view_attention[2].weight)
        nn.init.zeros_(self.view_attention[2].bias)
        
        # Shared E5 classification head (reused directly from base_model)
        self.classifier = self.base_model.classifier

    def forward(
        self,
        x_views: torch.Tensor, # Shape: (B, V, 3, H, W), where V = num_views (e.g. 5)
        return_view_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass over multi-view tensor.
        
        Args:
            x_views: Multi-view image tensor of shape (B, V, 3, 224, 224), range [0, 1].
            return_view_weights: If True, returns (logits, attn_weights).
            
        Returns:
            Logits of shape (B, 1), and optionally view attention weights of shape (B, V, 1).
        """
        B, V, C, H, W = x_views.shape
        # Flatten B and V for parallel GPU backbone execution
        x_flat = x_views.view(B * V, C, H, W)
        
        # Extract dual-branch features
        feats = self.base_model(x_flat, return_features=True)
        e_fused = feats["fused_embedding"] # (B * V, 1792)
        
        # Reshape back to (B, V, 1792)
        e_fused_views = e_fused.view(B, V, self.fused_dim)
        
        # Compute view attention weights
        attn_scores = self.view_attention(e_fused_views) # (B, V, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, V, 1), sums to 1 across views
        
        # Learnable Weighted Aggregation of view embeddings
        pooled_embedding = torch.sum(attn_weights * e_fused_views, dim=1) # (B, 1792)
        
        # Final classification
        logits = self.classifier(pooled_embedding) # (B, 1)
        
        if return_view_weights:
            return logits, attn_weights.squeeze(-1)
            
        return logits
