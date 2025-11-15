#!/usr/bin/env python3
"""
Generate predictions from all 11 champion models
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import timm
import argparse
from tqdm import tqdm

class TestDataset(Dataset):
    def __init__(self, test_csv, images_dir='test_images', img_size=384):
        self.df = pd.read_csv(test_csv)
        self.images_dir = images_dir
        self.img_size = img_size

        self.transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['new_filename'])
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        return image, row['new_filename']

def build_model(model_name, num_classes=4):
    """Build model matching training configuration"""
    if model_name == 'convnext_large':
        model = timm.create_model('convnext_large', pretrained=False, num_classes=num_classes)
    elif model_name == 'vit_large':
        model = timm.create_model('vit_large_patch16_384', pretrained=False,
                                   num_classes=num_classes, img_size=384)
    elif model_name == 'maxvit_large':
        model = timm.create_model('maxvit_large_tf_384', pretrained=False, num_classes=num_classes)
    elif model_name == 'beit_large':
        model = timm.create_model('beit_large_patch16_384', pretrained=False, num_classes=num_classes)
    elif model_name == 'coatnet':
        model = timm.create_model('coatnet_3_rw_224', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove _orig_mod prefix if exists (from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    return model

def predict_with_tta(model, loader, device, num_tta=5):
    """Generate predictions with Test-Time Augmentation"""
    model.eval()

    all_probs = []
    filenames = []

    with torch.no_grad():
        for images, fnames in tqdm(loader, desc="Predicting"):
            images = images.to(device)

            # Original prediction
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            # TTA: horizontal flip
            if num_tta > 1:
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip = torch.softmax(logits_flip, dim=1)
                probs = (probs + probs_flip) / 2

            all_probs.append(probs.cpu().numpy())
            filenames.extend(fnames)

    all_probs = np.concatenate(all_probs, axis=0)
    return filenames, all_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--tta', type=int, default=5, help='Number of TTA augmentations')
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    print("="*80)
    print(f"Generating predictions: {args.model} Fold {args.fold}")
    print("="*80)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Image size: {args.img_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  TTA: {args.tta}")
    print(f"  Output: {args.output}")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    test_dataset = TestDataset('data/test_data_sample.csv', img_size=args.img_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Build and load model
    print(f"\nLoading model...")
    model = build_model(args.model, num_classes=4)
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    print(f"Generating predictions with TTA={args.tta}...")
    filenames, probs = predict_with_tta(model, test_loader, device, num_tta=args.tta)

    # Save predictions
    output_df = pd.DataFrame({
        'new_filename': filenames,
        'normal': probs[:, 0],
        'bacteria': probs[:, 1],
        'virus': probs[:, 2],
        'COVID-19': probs[:, 3],
    })

    output_df.to_csv(args.output, index=False)
    print(f"\n[DONE] Saved predictions to {args.output}")
    print(f"  Predictions shape: {probs.shape}")
    print(f"  Confidence mean: {np.max(probs, axis=1).mean():.4f}")

if __name__ == '__main__':
    main()
