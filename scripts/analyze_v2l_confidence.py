#!/usr/bin/env python3
"""
分析 V2-L 模型的預測置信度分布
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_breakthrough import build_model

class TestDataset(Dataset):
    def __init__(self, csv_path, images_dir, img_size=384):
        self.df = pd.read_csv(csv_path)
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

@torch.no_grad()
def analyze_confidence(model_paths, model_arch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = TestDataset('data/test_data.csv', 'test_images', 384)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    print(f"測試集樣本數: {len(test_dataset)}")
    print(f"模型數量: {len(model_paths)}")

    all_probs = []

    for i, model_path in enumerate(model_paths[:1]):  # 只用第一個模型快速分析
        print(f"\n載入模型: {model_path}")

        model = build_model(model_arch, num_classes=4, dropout=0.3, img_size=384)
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        fold_probs = []
        for images, filenames in tqdm(test_loader, desc="預測"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            fold_probs.append(probs)

        fold_probs = np.concatenate(fold_probs, axis=0)
        all_probs.append(fold_probs)

        del model
        torch.cuda.empty_cache()

    # 分析置信度分布
    avg_probs = np.mean(all_probs, axis=0)
    max_probs = np.max(avg_probs, axis=1)

    print("\n" + "="*60)
    print("置信度分布分析:")
    print("="*60)

    thresholds = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    for threshold in thresholds:
        count = np.sum(max_probs >= threshold)
        percentage = 100 * count / len(max_probs)
        print(f"≥ {threshold}: {count:4d} 樣本 ({percentage:5.1f}%)")

    print(f"\n最高置信度: {np.max(max_probs):.4f}")
    print(f"平均置信度: {np.mean(max_probs):.4f}")
    print(f"中位數置信度: {np.median(max_probs):.4f}")
    print(f"最低置信度: {np.min(max_probs):.4f}")

if __name__ == '__main__':
    model_dir = Path('outputs/breakthrough_20251113_004854/layer1/efficientnet_v2_l')
    model_paths = sorted(model_dir.glob('fold*/best.pt'))

    print(f"找到 {len(model_paths)} 個 V2-L 模型\n")

    analyze_confidence(
        model_paths=[str(p) for p in model_paths],
        model_arch='efficientnet_v2_l'
    )
