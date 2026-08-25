import csv
from collections import Counter
from ml.data import config

def run_audit():
    if not config.MANIFEST_PATH.exists():
        print("Manifest not found. Run prepare.py first.")
        return

    total = 0
    real = 0
    ai = 0
    splits = Counter()
    formats = Counter()
    sources = Counter()
    generators = Counter()
    dimensions = Counter()

    with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if int(row["label"]) == config.LABEL_REAL:
                real += 1
            else:
                ai += 1
            
            splits[row["split"]] += 1
            formats[row["format"]] += 1
            sources[row["source"]] += 1
            generators[row["generator"]] += 1
            dimensions[f"{row['width']}x{row['height']}"] += 1

    print("=== DATASET AUDIT ===")
    print(f"Total Images: {total}")
    print(f"Real Images: {real}")
    print(f"AI-Generated Images: {ai}")
    print("\nSplits:")
    for k, v in splits.items():
        print(f"  {k}: {v}")
    
    print("\nFormats:")
    for k, v in formats.items():
        print(f"  {k}: {v}")
        
    print("\nDimensions:")
    # Print top 5 dimensions
    for k, v in dimensions.most_common(5):
        print(f"  {k}: {v}")
        
    print("\nSources:")
    for k, v in sources.items():
        print(f"  {k}: {v}")
        
    print("\nGenerators:")
    for k, v in generators.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    run_audit()
