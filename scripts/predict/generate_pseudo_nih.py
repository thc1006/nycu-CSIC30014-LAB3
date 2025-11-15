#!/usr/bin/env python3
"""
路徑 A: 使用 NIH Stage 2 模型生成偽標籤
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import from train_breakthrough
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
def generate_pseudo_labels(model_paths, model_arch, test_csv, test_images, output_csv, threshold=0.95, img_size=384):
    """生成偽標籤"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加載測試數據
    test_dataset = TestDataset(test_csv, test_images, img_size)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    print(f"測試集樣本數: {len(test_dataset)}")
    print(f"模型架構: {model_arch}")
    print(f"模型數量: {len(model_paths)}")

    # 集成預測
    all_probs = []

    for i, model_path in enumerate(model_paths):
        print(f"\n[{i+1}/{len(model_paths)}] 載入模型: {model_path}")

        # 構建模型
        model = build_model(model_arch, num_classes=4, dropout=0.3, img_size=img_size)

        # 載入權重
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        # 預測
        fold_probs = []
        for images, filenames in tqdm(test_loader, desc=f"預測 {i+1}/{len(model_paths)}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            fold_probs.append(probs)

        fold_probs = np.concatenate(fold_probs, axis=0)
        all_probs.append(fold_probs)

        # 釋放記憶體
        del model
        torch.cuda.empty_cache()

    # 平均所有模型的預測
    avg_probs = np.mean(all_probs, axis=0)

    # 獲取最高置信度的預測
    max_probs = np.max(avg_probs, axis=1)
    pred_labels = np.argmax(avg_probs, axis=1)

    # 篩選高置信度樣本
    high_conf_mask = max_probs >= threshold
    high_conf_count = np.sum(high_conf_mask)

    print(f"\n高置信度 (≥{threshold}) 樣本數: {high_conf_count} / {len(test_dataset)} ({100*high_conf_count/len(test_dataset):.1f}%)")

    # 創建偽標籤 DataFrame
    test_df = pd.read_csv(test_csv)
    pseudo_df = test_df[high_conf_mask].copy()

    # 添加偽標籤（one-hot encoding）
    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
    for i, class_name in enumerate(class_names):
        pseudo_df[class_name] = (pred_labels[high_conf_mask] == i).astype(int)

    # 添加置信度
    pseudo_df['confidence'] = max_probs[high_conf_mask]

    # 添加 source 標記（用於追蹤）
    pseudo_df['source'] = 'pseudo'
    pseudo_df['source_dir'] = 'test_images'

    # 保存
    pseudo_df.to_csv(output_csv, index=False)
    print(f"✅ 偽標籤已保存: {output_csv}")

    # 統計
    print("\n偽標籤分布:")
    for i, class_name in enumerate(class_names):
        count = np.sum(pred_labels[high_conf_mask] == i)
        print(f"  {class_name}: {count} ({100*count/high_conf_count:.1f}%)")

    return pseudo_df

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='模型目錄')
    parser.add_argument('--model_arch', required=True, help='模型架構')
    parser.add_argument('--output', required=True, help='輸出路徑')
    parser.add_argument('--threshold', type=float, default=0.95, help='置信度閾值')
    args = parser.parse_args()

    # 查找所有 fold 模型
    model_dir = Path(args.model_dir)
    model_paths = sorted(model_dir.glob('fold*_best.pt'))

    if len(model_paths) == 0:
        print(f"❌ 在 {model_dir} 中找不到模型！")
        sys.exit(1)

    print(f"找到 {len(model_paths)} 個模型")

    # 生成偽標籤
    generate_pseudo_labels(
        model_paths=[str(p) for p in model_paths],
        model_arch=args.model_arch,
        test_csv='data/test_data.csv',
        test_images='test_images',
        output_csv=args.output,
        threshold=args.threshold,
        img_size=384
    )
