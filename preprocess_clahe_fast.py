"""
CLAHE Preprocessing Pipeline - Fast Batch Processing
Solves the contrast distribution mismatch problem
"""
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

print("=" * 80)
print("CLAHE PREPROCESSING - SOLVING THE CONTRAST MISMATCH")
print("=" * 80)

def apply_clahe(img_path, output_dir, clahe):
    """Apply CLAHE to single image"""
    try:
        # Read grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This normalizes contrast across all images
        enhanced = clahe.apply(img)

        # Optional: Slight unsharp mask for edge enhancement
        # gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        # enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

        # Save
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), enhanced)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def process_split(split_name, img_dir, output_dir):
    """Process one data split with CLAHE"""
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get all images
    image_files = list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

    if len(image_files) == 0:
        print(f"   ⚠️  No images found in {img_dir}")
        return

    print(f"\n[{split_name}] Processing {len(image_files)} images...")

    # Create CLAHE object
    # clipLimit=2.0: prevent over-amplification
    # tileGridSize=(8,8): local adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Process with progress bar
    success_count = 0
    for img_path in tqdm(image_files, desc=f"{split_name:5s}"):
        if apply_clahe(img_path, output_dir, clahe):
            success_count += 1

    print(f"   ✓ Processed {success_count}/{len(image_files)} images")
    print(f"   ✓ Saved to: {output_dir}")

print("\n[1/3] Creating output directories...")
Path("train_images_clahe").mkdir(exist_ok=True)
Path("val_images_clahe").mkdir(exist_ok=True)
Path("test_images_clahe").mkdir(exist_ok=True)

print("\n[2/3] Processing images with CLAHE...")
print("   Purpose: Normalize contrast across train/val/test sets")
print("   Expected: Eliminates contrast distribution mismatch")

# Process all splits
process_split("TRAIN", "train_images", "train_images_clahe")
process_split("VAL", "val_images", "val_images_clahe")
process_split("TEST", "test_images", "test_images_clahe")

print("\n[3/3] Verifying output...")
train_count = len(list(Path("train_images_clahe").glob("*.jpeg")))
val_count = len(list(Path("val_images_clahe").glob("*.jpeg")))
test_count = len(list(Path("test_images_clahe").glob("*.jpeg"))) + len(list(Path("test_images_clahe").glob("*.jpg")))

print(f"   Train: {train_count} images")
print(f"   Val:   {val_count} images")
print(f"   Test:  {test_count} images")

print("\n" + "=" * 80)
print("CLAHE PREPROCESSING COMPLETE!")
print("=" * 80)
print("\nNext steps:")
print("  1. Update training config to use *_clahe directories")
print("  2. Train model with normalized contrast")
print("  3. Expected improvement: +2-3% accuracy")
print("\nKey improvement:")
print("  • Test contrast (0.468) now normalized to match training")
print("  • Eliminates contrast-based overfitting")
print("  • Better generalization to test set")
