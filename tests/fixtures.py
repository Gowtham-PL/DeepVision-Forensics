import os
from pathlib import Path
from PIL import Image
import random
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = BASE_DIR / "tests" / "test_data"

def create_fixtures():
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
        
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    real_dir = TEST_DATA_DIR / "real"
    ai_dir = TEST_DATA_DIR / "ai_generated" / "midjourney"
    
    real_dir.mkdir(parents=True, exist_ok=True)
    ai_dir.mkdir(parents=True, exist_ok=True)

    # Generate 50 real images and 50 ai images
    for i in range(50):
        img = Image.new('RGB', (256, 256), color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        img.save(real_dir / f"real_{i}.jpg")

        img_ai = Image.new('RGB', (256, 256), color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        img_ai.save(ai_dir / f"ai_{i}.jpg")
    
    # Generate an invalid image
    with open(real_dir / "invalid.jpg", "w") as f:
        f.write("This is not an image.")
        
    # Generate a duplicate
    shutil.copy(real_dir / "real_0.jpg", real_dir / "real_0_duplicate.jpg")
        
    print(f"Fixtures created in {TEST_DATA_DIR}")

if __name__ == "__main__":
    create_fixtures()
