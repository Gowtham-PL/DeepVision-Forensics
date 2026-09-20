"""
Smoke test for E7 Learned Fusion Model.
Verifies:
1. E6-C checkpoint presence and integrity
2. E5 manifest presence
3. CUDA availability and device info
4. Trainable vs frozen parameter counts
5. Tensor shapes through all intermediate projection and cross-attention stages
6. 3-step synthetic training loop with loss decrease under AMP
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e7_learned_fusion.models.e7_fusion_model import E7LearnedFusionModel

CHECKPOINT_PATH = PROJECT_ROOT / "experiments/e6_multiscale_training/e6c_finetune_benchmark/e6c_checkpoint_epoch2.pt"
MANIFEST_PATH = PROJECT_ROOT / "data/e5_external/manifests/e5_manifest.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_smoke_test():
    print("=" * 70)
    print("E7 LEARNED FUSION: SYSTEM SAFETY & ARCHITECTURAL SMOKE TEST")
    print("=" * 70)

    # 1. Verification of Checkpoint & Manifest
    assert CHECKPOINT_PATH.exists(), f"E6-C Checkpoint not found at: {CHECKPOINT_PATH}"
    print(f"[1/6] Checkpoint verified: {CHECKPOINT_PATH.name} ({CHECKPOINT_PATH.stat().st_size / (1024**2):.1f} MB)")

    assert MANIFEST_PATH.exists(), f"E5 Manifest not found at: {MANIFEST_PATH}"
    print(f"[2/6] Manifest verified: {MANIFEST_PATH.name}")

    # 2. CUDA verification
    print(f"[3/6] Compute Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"      GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"      Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 3. Model Instantiation & Parameter Verification
    model = E7LearnedFusionModel(
        e6c_checkpoint_path=CHECKPOINT_PATH,
        freeze_backbone=True,
    )
    model.to(DEVICE)

    n_backbone_total = sum(p.numel() for p in model.multiview_e5.parameters())
    n_backbone_trainable = sum(p.numel() for p in model.multiview_e5.parameters() if p.requires_grad)
    n_head_trainable = sum(p.numel() for p in model.fusion_head.parameters() if p.requires_grad)
    n_total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert n_backbone_trainable == 0, f"Backbone should have 0 trainable params, got {n_backbone_trainable}"
    assert n_head_trainable > 0, "Fusion head must have trainable parameters"

    print(f"[4/6] Parameter Counts:")
    print(f"      Backbone Frozen Parameters:    {n_backbone_total:,}")
    print(f"      Backbone Trainable Parameters: {n_backbone_trainable:,}")
    print(f"      Fusion Head Trainable Params:  {n_head_trainable:,}")
    print(f"      Total Trainable Parameters:    {n_total_trainable:,}")

    # 4. Shape and Forward Pass Verification
    dummy_input = torch.rand((2, 5, 3, 224, 224), dtype=torch.float32, device=DEVICE)
    logits, diag = model(dummy_input, return_diagnostics=True)

    assert logits.shape == (2, 1), f"Expected logits shape (2, 1), got {logits.shape}"
    assert diag["cross_attn_weights"].shape == (2, 4), f"Expected cross_attn (2, 4), got {diag['cross_attn_weights'].shape}"
    assert diag["view_attn_weights"].shape == (2, 5), f"Expected view_attn (2, 5), got {diag['view_attn_weights'].shape}"
    assert diag["fused_embedding"].shape == (2, 1408), f"Expected fused_emb (2, 1408), got {diag['fused_embedding'].shape}"

    print(f"[5/6] Forward Pass Tensor Shapes:")
    print(f"      Input views shape:        {dummy_input.shape}")
    print(f"      Fused embedding shape:    {diag['fused_embedding'].shape}")
    print(f"      Cross-attention shape:    {diag['cross_attn_weights'].shape}")
    print(f"      Output logits shape:      {logits.shape}")

    # 5. Optimization & Gradient Flow Verification
    optimizer = torch.optim.AdamW(model.fusion_head.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    targets = torch.tensor([[1.0], [0.0]], dtype=torch.float32, device=DEVICE)

    losses = []
    print(f"[6/6] Gradient Flow & Loss Reduction (5 test steps with AMP, lr=1e-4):")
    for step in range(5):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            out = model(dummy_input)
            loss = criterion(out, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        print(f"      Step {step+1}: Loss = {loss.item():.6f}")

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
    print(f"      Loss successfully decreased: {losses[0]:.6f} -> {losses[-1]:.6f}")

    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS PASSED SUCCESSFULLY! E7 IS READY FOR TRAINING.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
