#!/usr/bin/env python
"""
Test Time Augmentation (TTA) - predict with multiple augmentations and average
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def tta_predict(model, test_loader, device, num_augmentations=5):
    """
    Perform TTA: predict num_augmentations times with different augmentations
    and average the probabilities

    Args:
        model: trained model
        test_loader: test dataloader with augmentation enabled
        device: cuda or cpu
        num_augmentations: number of augmented predictions to average

    Returns:
        averaged_probs: averaged probability predictions
    """
    model.eval()

    print(f"Running TTA with {num_augmentations} augmentations...")

    all_logits = []

    with torch.no_grad():
        for aug_idx in range(num_augmentations):
            print(f"  Augmentation {aug_idx+1}/{num_augmentations}")

            logits_list = []
            for imgs, _, _ in test_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                logits_list.append(logits.cpu().numpy())

            all_logits.append(np.concatenate(logits_list, axis=0))

    # Average logits
    averaged_logits = np.mean(all_logits, axis=0)
    probs = torch.softmax(torch.from_numpy(averaged_logits), dim=1).numpy()

    print(f"✅ TTA predictions generated: {probs.shape}")

    return probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTA prediction")
    parser.add_argument("--config", type=str, required=True, help="Config path")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--num-aug", type=int, default=5, help="Number of augmentations")
    parser.add_argument("-o", "--output", type=str, help="Output CSV path")

    args = parser.parse_args()

    print("TTA mode would require integration with src/predict.py")
    print(f"Use: python src/predict.py --tta {args.num_aug} instead")
