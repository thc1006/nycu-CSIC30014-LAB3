#!/usr/bin/env python3
"""
Gen3 預測生成器 - 5-Fold 集成
基於訓練完成的 Gen3 模型生成測試集預測
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch import nn
from tqdm import tqdm

class TestDataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=512):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir

        self.transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(self.img_dir) / row['new_filename']
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transforms(img)
        return img_tensor, row['new_filename']

def predict_single_fold(model_path, test_csv, img_dir, device):
    """單個 fold 的預測"""
    # 載入模型
    model = models.efficientnet_v2_l(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # 創建數據載入器
    dataset = TestDataset(test_csv, img_dir, img_size=512)
    loader = DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True)

    all_preds = []
    filenames = []

    with torch.no_grad():
        for imgs, fnames in tqdm(loader, desc="預測中"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(probs)
            filenames.extend(fnames)

    preds = np.concatenate(all_preds, axis=0)
    return filenames, preds

def main():
    print("="*70)
    print("🚀 Gen3 預測生成器 (5-Fold 集成)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n設備: {device}")

    test_csv = 'data/test_data.csv'
    img_dir = 'test_images'

    # 檢查所有 fold 模型
    model_paths = []
    for fold in range(5):
        model_path = f'outputs/v2l_512_gen3/fold{fold}/best.pt'
        if Path(model_path).exists():
            model_paths.append((fold, model_path))
            print(f"✅ Fold {fold}: {model_path}")
        else:
            print(f"⚠️  Fold {fold}: 模型不存在，跳過")

    if not model_paths:
        print("\n❌ 沒有找到任何 Gen3 模型！")
        print("請確保訓練已完成並且模型已保存。")
        return

    print(f"\n找到 {len(model_paths)} 個模型")

    # 生成每個 fold 的預測
    all_fold_preds = []
    filenames = None

    for fold, model_path in model_paths:
        print(f"\n{'='*70}")
        print(f"📊 Fold {fold} 預測")
        print(f"{'='*70}")

        fold_filenames, preds = predict_single_fold(model_path, test_csv, img_dir, device)
        all_fold_preds.append(preds)

        if filenames is None:
            filenames = fold_filenames

        # 保存單個 fold 預測
        class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
        pred_classes = np.argmax(preds, axis=1)
        onehot = np.zeros_like(preds, dtype=int)
        onehot[np.arange(len(pred_classes)), pred_classes] = 1

        df = pd.DataFrame(onehot, columns=class_cols)
        df.insert(0, 'new_filename', filenames)

        output_path = f'data/submission_v2l_512_gen3_fold{fold}.csv'
        df.to_csv(output_path, index=False)
        print(f"✅ 單 Fold 預測已保存: {output_path}")

    # 5-Fold 集成
    print(f"\n{'='*70}")
    print("🎯 5-Fold 集成")
    print(f"{'='*70}")

    final_preds = np.mean(all_fold_preds, axis=0)

    # 轉換為 one-hot
    pred_classes = np.argmax(final_preds, axis=1)
    onehot = np.zeros_like(final_preds, dtype=int)
    onehot[np.arange(len(pred_classes)), pred_classes] = 1

    # 保存集成預測
    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
    df = pd.DataFrame(onehot, columns=class_cols)
    df.insert(0, 'new_filename', filenames)

    output_path = 'data/submission_gen3_ensemble.csv'
    df.to_csv(output_path, index=False)

    # 統計
    print(f"\n預測分布:")
    for i, col in enumerate(class_cols):
        count = (pred_classes == i).sum()
        print(f"  {col}: {count} ({count/len(pred_classes)*100:.1f}%)")

    print(f"\n✅ Gen3 集成預測已保存: {output_path}")
    print(f"   預期分數: 89.0-90.0%")

    print(f"\n{'='*70}")
    print("🎉 Gen3 預測完成！")
    print(f"{'='*70}")
    print(f"\n下一步:")
    print(f"1. 提交測試: kaggle competitions submit -f {output_path}")
    print(f"2. 如果 < 90%: 執行 Gen3 訓練")
    print(f"3. 如果 >= 90%: 慶祝突破！🎉")

if __name__ == "__main__":
    main()
