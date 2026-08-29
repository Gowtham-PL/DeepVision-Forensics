"""
Explainability module for DeepVision-Forensics.

Provides Gradient-weighted Class Activation Mapping (Grad-CAM) for the
spatial branch and spectral analysis tools for the frequency branch.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for the EfficientNet-B3 spatial backbone.
    
    Hooks into the final convolutional feature maps, computes activation weights
    via backpropagated gradients, and generates a visual spatial localization map.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "get_spatial_gradcam_layer"):
                self.target_layer = model.get_spatial_gradcam_layer()
            elif hasattr(model, "spatial_branch"):
                self.target_layer = model.spatial_branch.get_target_layer_for_gradcam()
            else:
                raise ValueError("Could not automatically determine target layer for Grad-CAM.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Removes registered PyTorch hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        self.remove_hooks()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Generates a 2D normalized Grad-CAM heatmap for a single image tensor.
        
        Args:
            input_tensor: Tensor of shape (1, 3, H, W) in range [0, 1].
            eps: Epsilon to prevent division by zero during normalization.
            
        Returns:
            2D numpy array of shape (H, W) with values in range [0.0, 1.0].
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        H, W = input_tensor.shape[-2:]

        # Zero gradients
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, dict):
            logit = output["logit"]
        else:
            logit = output

        # Backward pass on the single binary logit
        logit.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations during Grad-CAM backward pass.")

        # Channel-wise Global Average Pooling of gradients: alpha_k
        alpha = torch.mean(self.gradients, dim=(-2, -1), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)  # (1, 1, H', W')

        # Apply ReLU to keep only features that have a positive influence
        cam = F.relu(cam)

        # Bilinearly interpolate to original input resolution (H, W)
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min = cam_np.min()
        cam_max = cam_np.max()
        cam_normalized = (cam_np - cam_min) / (cam_max - cam_min + eps)

        return cam_normalized


def overlay_heatmap(
    image_rgb_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap onto an RGB image.
    
    Args:
        image_rgb_np: RGB image array in range [0, 255] (uint8) or [0, 1] (float).
        heatmap: 2D array of shape (H, W) with values in range [0.0, 1.0].
        alpha: Blending weight for heatmap (0.0 = only image, 1.0 = only heatmap).
        colormap: OpenCV colormap constant.
        
    Returns:
        RGB uint8 image with blended heatmap overlay.
    """
    if image_rgb_np.dtype != np.uint8:
        if image_rgb_np.max() <= 1.0:
            image_rgb_np = (image_rgb_np * 255).astype(np.uint8)
        else:
            image_rgb_np = image_rgb_np.astype(np.uint8)

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_rgb_np, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def compute_log_magnitude_spectrum(
    image_tensor: torch.Tensor,
    norm_strategy: str = "minmax",
) -> np.ndarray:
    """
    Computes the centered 2D FFT log-magnitude spectrum for visualization.
    
    Args:
        image_tensor: Tensor of shape (1, 3, H, W) or (3, H, W) in [0, 1].
        norm_strategy: Normalization method ('minmax', 'standardize', 'none').
        
    Returns:
        2D numpy array of shape (H, W) representing the frequency spectrum.
    """
    from models.frequency import FFTTransform

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    transform = FFTTransform(norm_strategy=norm_strategy)
    with torch.no_grad():
        spec = transform(image_tensor)

    spec_np = spec.squeeze().cpu().numpy()
    return spec_np
