from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms
import time
import csv
from pathlib import Path

def gen_5_crops(img):
    w, h = img.size
    w60, h60 = max(1, int(round(w * 0.6))), max(1, int(round(h * 0.6)))
    return [
        img.resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((0, 0, w60, h60)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((w - w60, 0, w, h60)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((0, h - h60, w60, h)).resize((224, 224), Image.Resampling.BILINEAR),
        img.crop((w - w60, h - h60, w, h)).resize((224, 224), Image.Resampling.BILINEAR),
    ]

class TestDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.tt = transforms.ToTensor()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fp = Path("data/e20_training") / self.rows[idx]["filepath"]
        with Image.open(fp) as img:
            rgb = img.convert("RGB")
        crops = gen_5_crops(rgb)
        stacked = torch.stack([self.tt(c) for c in crops], dim=0)
        lbl = 1.0 if self.rows[idx]["label"] == "AI" else 0.0
        return stacked, torch.tensor(lbl, dtype=torch.float32)

if __name__ == "__main__":
    rows = list(csv.DictReader(open("data/e20_training/manifests/e20_dev_manifest.csv", encoding="utf-8")))[:50]
    ds = TestDataset(rows)
    loader = DataLoader(ds, batch_size=4, num_workers=4, pin_memory=True)
    t0 = time.time()
    for b_x, b_y in loader:
        pass
    print(f"Loaded {len(rows)} samples with num_workers=4 in {time.time() - t0:.2f}s")
