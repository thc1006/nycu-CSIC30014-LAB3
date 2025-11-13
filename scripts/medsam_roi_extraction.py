#!/usr/bin/env python3
"""
MedSAM ROI Extraction for Chest X-Rays
Uses MedSAM to segment and extract lung regions

This helps the model focus on relevant anatomical regions
and can improve performance by 0.5-1.5%
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


class MedSAMSegmenter:
    """
    MedSAM-based lung region segmentation
    """

    def __init__(self, checkpoint_path='external_data/medsam_vit_b.pth'):
        """
        Initialize MedSAM model

        Args:
            checkpoint_path: Path to MedSAM checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading MedSAM from {checkpoint_path}...")

        # Try to load MedSAM
        # Note: This requires the segment-anything package
        try:
            from segment_anything import sam_model_registry

            model_type = "vit_b"
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)

            from segment_anything import SamPredictor
            self.predictor = SamPredictor(sam)

            print("✓ MedSAM loaded successfully")

        except ImportError:
            print("⚠️  segment-anything not installed")
            print("Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'segment-anything'])

            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)

    def segment_lungs(self, image_path):
        """
        Segment lung regions from chest X-ray

        Returns:
            mask: Binary mask of lung regions
            bbox: Bounding box [x1, y1, x2, y2]
        """
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set image
        self.predictor.set_image(image)

        # Use center prompt (lungs are typically in center)
        h, w = image.shape[:2]

        # Two prompts: left and right lung regions
        prompts = [
            np.array([[w * 0.35, h * 0.5]]),  # Left lung
            np.array([[w * 0.65, h * 0.5]])   # Right lung
        ]

        masks = []
        for prompt in prompts:
            mask, _, _ = self.predictor.predict(
                point_coords=prompt,
                point_labels=np.array([1]),  # Foreground
                multimask_output=False
            )
            masks.append(mask[0])

        # Combine masks
        combined_mask = np.logical_or(masks[0], masks[1])

        # Get bounding box
        coords = np.argwhere(combined_mask)
        if len(coords) > 0:
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            bbox = [x1, y1, x2, y2]
        else:
            bbox = [0, 0, w, h]

        return combined_mask, bbox

    def extract_roi(self, image_path, output_path, apply_mask=True, expand_ratio=0.1):
        """
        Extract ROI and save

        Args:
            image_path: Input image path
            output_path: Output path
            apply_mask: Whether to apply mask (black out background)
            expand_ratio: Expand bounding box by this ratio
        """
        # Load original image
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]

        # Segment
        mask, bbox = self.segment_lungs(image_path)

        # Expand bbox
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * expand_ratio))
        y1 = max(0, int(y1 - bh * expand_ratio))
        x2 = min(w, int(x2 + bw * expand_ratio))
        y2 = min(h, int(y2 + bh * expand_ratio))

        # Crop
        roi = image[y1:y2, x1:x2]

        # Apply mask if requested
        if apply_mask:
            mask_crop = mask[y1:y2, x1:x2]
            roi = roi * mask_crop[..., np.newaxis]

        # Save
        cv2.imwrite(str(output_path), roi)

        return roi, mask


def process_dataset(
    input_dir='data/train',
    output_dir='data/train_medsam_roi',
    checkpoint_path='external_data/medsam_vit_b.pth',
    apply_mask=True
):
    """
    Process entire dataset with MedSAM ROI extraction
    """
    print("=" * 80)
    print("MedSAM ROI EXTRACTION")
    print("=" * 80)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check MedSAM checkpoint
    if not Path(checkpoint_path).exists():
        print(f"\n❌ MedSAM checkpoint not found: {checkpoint_path}")
        print("\nPlease download MedSAM first:")
        print("  bash scripts/download_external_data.sh")
        return

    # Initialize segmenter
    segmenter = MedSAMSegmenter(checkpoint_path)

    # Find all images
    image_files = list(input_dir.glob('*.jpg')) + \
                  list(input_dir.glob('*.jpeg')) + \
                  list(input_dir.glob('*.png'))

    print(f"\nFound {len(image_files)} images")
    print(f"Output: {output_dir}")
    print(f"Apply mask: {apply_mask}")
    print()

    # Process each image
    failed = []
    for img_path in tqdm(image_files, desc='Extracting ROIs'):
        try:
            output_path = output_dir / img_path.name
            segmenter.extract_roi(img_path, output_path, apply_mask=apply_mask)
        except Exception as e:
            print(f"\n✗ Failed {img_path.name}: {e}")
            failed.append(img_path.name)

    # Summary
    print("\n" + "=" * 80)
    print("ROI EXTRACTION COMPLETED")
    print("=" * 80)
    print(f"\n✓ Processed: {len(image_files) - len(failed)}")
    if len(failed) > 0:
        print(f"✗ Failed: {len(failed)}")
        print(f"  Files: {', '.join(failed[:5])}{'...' if len(failed) > 5 else ''}")
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("  1. Update config to use ROI images:")
    print("     data_dir: data/train_medsam_roi")
    print("  2. Train model:")
    print("     python src/train_v2.py --config configs/dinov2_large_medsam_roi.yaml")
    print("=" * 80)


def visualize_samples(n_samples=5):
    """
    Visualize ROI extraction results
    """
    input_dir = Path('data/train')
    roi_dir = Path('data/train_medsam_roi')

    if not roi_dir.exists():
        print("ROI directory not found. Run extraction first.")
        return

    image_files = list(input_dir.glob('*.jpg'))[:n_samples]

    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 5 * n_samples))

    for i, img_path in enumerate(image_files):
        # Original
        orig = Image.open(img_path)
        axes[i, 0].imshow(orig, cmap='gray')
        axes[i, 0].set_title(f'Original: {img_path.name}')
        axes[i, 0].axis('off')

        # ROI
        roi_path = roi_dir / img_path.name
        if roi_path.exists():
            roi = Image.open(roi_path)
            axes[i, 1].imshow(roi, cmap='gray')
            axes[i, 1].set_title(f'MedSAM ROI')
        else:
            axes[i, 1].text(0.5, 0.5, 'Not found', ha='center', va='center')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/medsam_roi_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: outputs/medsam_roi_visualization.png")
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/train', help='Input image directory')
    parser.add_argument('--output-dir', default='data/train_medsam_roi', help='Output ROI directory')
    parser.add_argument('--checkpoint', default='external_data/medsam_vit_b.pth', help='MedSAM checkpoint')
    parser.add_argument('--no-mask', action='store_true', help='Do not apply mask (only crop)')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')

    args = parser.parse_args()

    if args.visualize:
        visualize_samples()
    else:
        process_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            apply_mask=not args.no_mask
        )
