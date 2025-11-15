#!/usr/bin/env python3
"""
生成所有基礎模型的測試集預測（用於超級集成）
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
def generate_model_predictions(model_path, model_arch, img_size=384):
    """生成單個模型的測試預測"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    test_dataset = TestDataset('data/test_data.csv', 'test_images', img_size)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                            num_workers=8, pin_memory=True)

    # Build model
    model = build_model(model_arch, num_classes=4, dropout=0.3, img_size=img_size)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Predict
    all_probs = []
    for images, filenames in tqdm(test_loader, desc=f"Predicting {Path(model_path).parent.name}"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)

    del model
    torch.cuda.empty_cache()

    return all_probs

def main():
    print("="*80)
    print("生成所有基礎模型的測試預測")
    print("="*80)

    # Model configurations
    configs = [
        {
            'name': 'efficientnet_v2_l',
            'arch': 'efficientnet_v2_l',
            'base_dir': 'outputs/breakthrough_20251113_004854/layer1/efficientnet_v2_l',
            'img_size': 384,
            'folds': 5
        },
        {
            'name': 'swin_large',
            'arch': 'swin_large_patch4_window12_384',
            'base_dir': 'outputs/breakthrough_20251113_004854/layer1/swin_large',
            'img_size': 384,
            'folds': 5
        }
    ]

    test_df = pd.read_csv('data/test_data.csv')
    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

    for config in configs:
        print(f"\n{'='*80}")
        print(f"處理: {config['name']}")
        print(f"{'='*80}")

        for fold in range(config['folds']):
            model_path = Path(config['base_dir']) / f"fold{fold}" / "best.pt"

            if not model_path.exists():
                print(f"⚠️ 模型不存在: {model_path}")
                continue

            print(f"\n[{fold+1}/{config['folds']}] 載入模型: {model_path}")

            # Generate predictions
            probs = generate_model_predictions(
                str(model_path),
                config['arch'],
                config['img_size']
            )

            # Save as probability CSV
            output_df = test_df[['new_filename']].copy()
            for i, class_name in enumerate(class_names):
                output_df[class_name] = probs[:, i]

            output_path = f"data/test_predictions/{config['name']}_fold{fold}.csv"
            os.makedirs('data/test_predictions', exist_ok=True)
            output_df.to_csv(output_path, index=False)

            print(f"✅ 已保存: {output_path}")
            print(f"   平均置信度: {np.max(probs, axis=1).mean():.4f}")

    print(f"\n{'='*80}")
    print("✅ 所有預測已生成！")
    print(f"{'='*80}")

    # Count total predictions
    pred_files = list(Path('data/test_predictions').glob('*.csv'))
    print(f"\n總共生成: {len(pred_files)} 個測試預測文件")

if __name__ == '__main__':
    main()
