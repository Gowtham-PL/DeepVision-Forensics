import os
import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("d:/DeepVision-Forensics")
PYTHON_EXE = PROJECT_ROOT / ".venv/Scripts/python.exe"

def main():
    print("=" * 70)
    print("E25 END-TO-END PIPELINE: CACHING -> TRAINING -> ABLATION -> BENCHMARKS")
    print("=" * 70)

    # 1. Dev Cache
    dev_cache_dir = PROJECT_ROOT / "experiments/e25_reconstruction_residual/cache/dev"
    n_dev = len(list(dev_cache_dir.glob("*.pt"))) if dev_cache_dir.exists() else 0
    print(f"[*] Dev cache currently has {n_dev}/1477 files.")
    if n_dev < 1477:
        print("[*] Generating / finishing Dev cache...")
        cmd_dev = [str(PYTHON_EXE), "experiments/e25_reconstruction_residual/scripts/generate_e25_cache.py", "--split", "dev"]
        subprocess.run(cmd_dev, cwd=str(PROJECT_ROOT), check=True)

    # 2. Train Cache
    train_cache_dir = PROJECT_ROOT / "experiments/e25_reconstruction_residual/cache/train"
    n_train = len(list(train_cache_dir.glob("*.pt"))) if train_cache_dir.exists() else 0
    print(f"[*] Train cache currently has {n_train}/6279 files.")
    if n_train < 6279:
        print("[*] Generating / finishing Train cache...")
        cmd_train = [str(PYTHON_EXE), "experiments/e25_reconstruction_residual/scripts/generate_e25_cache.py", "--split", "train"]
        subprocess.run(cmd_train, cwd=str(PROJECT_ROOT), check=True)

    # 3. Training & Evaluation
    print("\n[*] Launching E25 Training, Ablations, External Benchmarks, and Hard Cases...")
    cmd_train_model = [str(PYTHON_EXE), "experiments/e25_reconstruction_residual/scripts/run_e25_training.py"]
    subprocess.run(cmd_train_model, cwd=str(PROJECT_ROOT), check=True)

    print("\n" + "=" * 70)
    print("E25 PIPELINE EXECUTION COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()
