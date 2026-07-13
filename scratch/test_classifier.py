import sys
from pathlib import Path

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from utils.helpers import load_config
from models.hybrid_classifier import build_classifier, HybridDRClassifier


def main():
    print("=" * 60)
    print("  Testing Hybrid DR Classifier Architecture")
    print("=" * 60)

    # configuration
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    print(f"Loading config from: {config_path}")
    config = load_config(str(config_path))

    # 2. Build Classifier Model
    print("Instantiating model...")
    model = build_classifier(config)
    print(f"Model Type: {type(model).__name__}")
    
    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.classifier_head.parameters())
    
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Backbone parameters:   {backbone_params:,}")
    print(f"  Class head parameters: {head_params:,}")

    # 3. Create dummy inputs (Batch size = 4)
    batch_size = 4
    img_size = config["preprocessing"].get("image_size", 512)
    
    dummy_images = torch.randn(batch_size, 3, img_size, img_size)
    dummy_lesion_counts = torch.randint(0, 100, (batch_size, 4)).float()
    
    print("\nDummy inputs created:")
    print(f"  Images shape:        {dummy_images.shape}")
    print(f"  Lesion counts shape: {dummy_lesion_counts.shape}")
    print(f"  Lesion counts batch:\n{dummy_lesion_counts}")

    # 4. Dry-run forward pass on CPU
    print("\nRunning forward pass on CPU...")
    model.eval()
    with torch.no_grad():
        logits = model(dummy_images, dummy_lesion_counts)
    
    print("Forward pass completed successfully!")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Output logits:\n{logits}")

    # 5. Dry-run forward pass on GPU (if CUDA available)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nCUDA is available. Moving model and tensors to {device}...")
        try:
            model.to(device)
            dummy_images_gpu = dummy_images.to(device)
            dummy_lesion_counts_gpu = dummy_lesion_counts.to(device)
            
            with torch.no_grad():
                gpu_logits = model(dummy_images_gpu, dummy_lesion_counts_gpu)
            
            print("GPU Forward pass completed successfully!")
            print(f"  Output GPU logits shape: {gpu_logits.shape}")
            print(f"  Device:                  {gpu_logits.device}")
        except Exception as e:
            print("\nWARNING: GPU execution failed.")
            print(f"  Error: {e}")
            print("  This is likely due to compatibility issues between the installed PyTorch version and the RTX 5050 Laptop GPU (compute capability sm_120).")
    else:
        print("\nCUDA is not available, skipping GPU check.")


if __name__ == "__main__":
    main()
