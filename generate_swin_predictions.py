#!/usr/bin/env python3
"""
生成 Swin-Large 5-Fold 測試集預測並創建最終提交
"""
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import timm
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class TestDataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=384):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.img_size = img_size

        self.transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['new_filename']
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        return img, row['new_filename']

def predict_fold(fold, device='cuda'):
    print(f"\n🔮 Fold {fold} 預測中...")

    # Load model
    model = timm.create_model('swin_large_patch4_window12_384', pretrained=False, num_classes=4)
    checkpoint = torch.load(f'outputs/swin_large_ultimate/fold{fold}/best.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  ✅ 模型加載完成 (Val F1: {checkpoint['f1']:.2f}%)")

    # Load test data
    test_dataset = TestDataset('data/test_data.csv', 'data/test_images', img_size=384)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Predict
    all_probs = []
    all_filenames = []

    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc=f"  Fold {fold}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_filenames.extend(filenames)

    all_probs = np.vstack(all_probs)

    print(f"  📊 預測完成: {len(all_filenames)} 樣本")
    return all_probs, all_filenames

def main():
    print("="*70)
    print("🚀 Swin-Large 5-Fold 測試集預測")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Device: {device}")

    # Predict all folds
    fold_probs = []

    for fold in range(5):
        probs, filenames = predict_fold(fold, device)
        fold_probs.append(probs)

    # Average predictions
    print("\n📊 集成 5-Fold 預測...")
    avg_probs = np.mean(fold_probs, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)

    # Map to class names
    class_map = {0: 'normal', 1: 'bacteria', 2: 'virus', 3: 'COVID-19'}
    pred_labels = [class_map[p] for p in final_preds]

    # Create submission
    submission = pd.DataFrame({
        'filename': filenames,
        'label': pred_labels
    })

    # Save
    output_path = 'data/submission_swin_large_5fold.csv'
    submission.to_csv(output_path, index=False)

    print(f"\n✅ 提交文件已保存: {output_path}")

    # Show distribution
    print("\n📈 預測分布:")
    dist = submission['label'].value_counts()
    for label, count in dist.items():
        pct = count / len(submission) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    print("\n" + "="*70)
    print("🎯 準備提交至 Kaggle!")
    print("="*70)

    return output_path

if __name__ == '__main__':
    output_path = main()
