import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import load_config
from datasets.aptos_dataset import build_aptos_dataloaders
import torch
from collections import Counter


def main():
    print("=" * 60)
    print("  Testing APTOS 2019 Dataset Loader")
    print("=" * 60)

    # Load configuration
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    print(f"Loading config from: {config_path}")
    config = load_config(str(config_path))

    # Build Dataloaders
    print("Building dataloaders...")
    loaders = build_aptos_dataloaders(config)

    for phase in ["train", "val", "test"]:
        loader = loaders[phase]
        dataset = loader.dataset
        print(f"\nPhase: {phase.upper()}")
        print(f"  Dataset size: {len(dataset)}")
        
       
        labels = dataset.df["diagnosis"].tolist()
        dist = Counter(labels)
        dist_sorted = sorted(dist.items())
        print("  Class distribution:")
        for grade, count in dist_sorted:
            percentage = (count / len(dataset)) * 100
            print(f"    Grade {grade}: {count:>4d} ({percentage:.2f}%)")

    #single batch
    train_loader = loaders["train"]
    print("\nFetching one batch from training loader...")
    try:
        images, labels, meta = next(iter(train_loader))
        print("Success!")
        print(f"  Images batch shape: {images.shape}")
        print(f"  Images dtype:       {images.dtype}")
        print(f"  Images range:       min={images.min().item():.4f}, max={images.max().item():.4f}")
        print(f"  Labels batch shape: {labels.shape}")
        print(f"  Labels:             {labels.tolist()}")
        print(f"  Sample metadata:    {[{k: v[0] for k, v in meta.items()}]}")
    except Exception as e:
        print(f"Error fetching training batch: {e}")
        import traceback
        traceback.print_exc()

    val_loader = loaders["val"]
    print("\nFetching one batch from validation loader...")
    try:
        images, labels, meta = next(iter(val_loader))
        print("Success!")
        print(f"  Images batch shape: {images.shape}")
        print(f"  Images dtype:       {images.dtype}")
        print(f"  Images range:       min={images.min().item():.4f}, max={images.max().item():.4f}")
        print(f"  Labels batch shape: {labels.shape}")
        print(f"  Labels:             {labels.tolist()}")
    except Exception as e:
        print(f"Error fetching validation batch: {e}")

    print("\n" + "=" * 60)
    print("  Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
