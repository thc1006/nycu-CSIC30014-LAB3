#!/usr/bin/env python3
"""
🔮 DINOv2 5-Fold Ensemble Prediction
Generate test predictions from all 5 folds and ensemble them
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
from tqdm import tqdm
import timm

class TestDataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=518):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['new_filename'])
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), row['new_filename']

def main():
    print("=" * 70)
    print("🔮 DINOv2 5-Fold Ensemble Prediction")
    print("=" * 70)

    # Configuration
    OUTPUT_DIR = 'outputs/dinov2_breakthrough'
    IMG_SIZE = 518
    BATCH_SIZE = 32
    NUM_FOLDS = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✅ Device: {device}")

    # Load test dataset
    test_dataset = TestDataset('data/test_data_sample.csv', 'data/test_images', img_size=IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"✅ Test dataset: {len(test_dataset)} samples\n")

    all_preds = []
    filenames = None

    for fold in range(NUM_FOLDS):
        model_path = f'{OUTPUT_DIR}/fold{fold}/best.pt'

        if not os.path.exists(model_path):
            print(f"⚠️ Fold {fold} model not found, skipping")
            continue

        print(f"📊 Fold {fold} predicting...")

        # Create model
        model = timm.create_model('vit_base_patch14_dinov2', pretrained=False, num_classes=4)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        fold_probs = []
        fold_filenames = []

        with torch.no_grad():
            for images, fnames in tqdm(test_loader, desc=f'Fold {fold}', leave=False):
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                fold_probs.append(probs.cpu().numpy())
                fold_filenames.extend(fnames)

        fold_probs = np.concatenate(fold_probs, axis=0)
        all_preds.append(fold_probs)

        if filenames is None:
            filenames = fold_filenames

        print(f"   ✅ Complete\n")

        # Free memory
        del model
        torch.cuda.empty_cache()

    if len(all_preds) == 0:
        print("❌ No fold models found!")
        return 1

    # Ensemble predictions
    print(f"🔮 Ensembling {len(all_preds)} models...")
    avg_probs = np.mean(all_preds, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)

    # Create submission
    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
    submission_df = pd.DataFrame({
        'new_filename': filenames,
        'label': [class_names[p] for p in final_preds]
    })

    submission_path = f'data/submission_dinov2_5fold.csv'
    submission_df.to_csv(submission_path, index=False)

    print(f"\n✅ Submission saved: {submission_path}")
    print(f"\n📊 Prediction distribution:")
    print(submission_df['label'].value_counts())
    print("\n" + "=" * 70)

    return 0

if __name__ == '__main__':
    sys.exit(main())
