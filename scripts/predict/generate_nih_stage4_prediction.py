#!/usr/bin/env python3
"""
生成 NIH Stage 4 的測試集預測 (5-fold 集成)
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
def generate_predictions(model_dir, model_arch='efficientnet_v2_s', output_csv='data/submission_nih_stage4.csv'):
    """生成測試集預測"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加載測試數據
    test_dataset = TestDataset('data/test_data.csv', 'test_images', 384)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    print(f"測試集樣本數: {len(test_dataset)}")

    # 查找所有 fold 模型（從 outputs/nih_v2s_stage3_4/best.pt 重命名後的模型）
    model_dir = Path(model_dir)

    # 檢查是否有分開的 fold 模型
    model_paths = []
    for fold in range(5):
        fold_path = model_dir / f"fold{fold}_best.pt"
        if fold_path.exists():
            model_paths.append(str(fold_path))

    # 如果沒有找到 fold 模型，檢查是否只有一個 best.pt
    if len(model_paths) == 0:
        best_pt = model_dir / "best.pt"
        if best_pt.exists():
            print(f"⚠️ 只找到單一模型 best.pt，將使用它生成預測")
            model_paths = [str(best_pt)]
        else:
            print(f"❌ 在 {model_dir} 中找不到模型！")
            sys.exit(1)

    print(f"找到 {len(model_paths)} 個模型:")
    for p in model_paths:
        print(f"  - {p}")

    # 集成預測
    all_probs = []

    for i, model_path in enumerate(model_paths):
        print(f"\n[{i+1}/{len(model_paths)}] 載入模型: {model_path}")

        # 構建模型
        model = build_model(model_arch, num_classes=4, dropout=0.3, img_size=384)

        # 載入權重
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
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

    # 創建提交檔案
    test_df = pd.read_csv('data/test_data.csv')
    submission = test_df[['new_filename']].copy()

    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
    for i, class_name in enumerate(class_names):
        submission[class_name] = avg_probs[:, i]

    # 保存
    submission.to_csv(output_csv, index=False)
    print(f"\n✅ 預測已保存: {output_csv}")

    # 統計預測分布
    pred_labels = np.argmax(avg_probs, axis=1)
    print("\n預測類別分布:")
    for i, class_name in enumerate(class_names):
        count = np.sum(pred_labels == i)
        print(f"  {class_name}: {count} ({100*count/len(pred_labels):.1f}%)")

    # 置信度分析
    max_probs = np.max(avg_probs, axis=1)
    print(f"\n平均置信度: {np.mean(max_probs):.4f}")
    print(f"中位數置信度: {np.median(max_probs):.4f}")

if __name__ == '__main__':
    generate_predictions(
        model_dir='outputs/nih_v2s_stage3_4',
        model_arch='efficientnet_v2_s',
        output_csv='data/submission_nih_stage4.csv'
    )
