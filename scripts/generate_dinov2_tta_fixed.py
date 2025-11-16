#!/usr/bin/env python3
"""
🔮 DINOv2 TTA 10-Crop (Fixed版本)
自适应resize，避免crop尺寸错误
"""

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

class TTADatasetFixed(Dataset):
    def __init__(self, csv_path, img_dir, img_size=518):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_size = img_size
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(self.img_dir) / row['new_filename']
        img = Image.open(img_path).convert('RGB')

        # 先resize到足够大的尺寸 (至少crop_size)
        w, h = img.size
        if w < self.img_size or h < self.img_size:
            # Resize使最小边 >= crop_size
            scale = max(self.img_size / w, self.img_size / h) * 1.1  # 1.1倍确保够大
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)

        crops = []

        # 原图 5-crop
        five_crop = T.FiveCrop(self.img_size)
        for crop in five_crop(img):
            crops.append(self.normalize(T.ToTensor()(crop)))

        # 水平翻转 5-crop
        img_flip = T.functional.hflip(img)
        for crop in five_crop(img_flip):
            crops.append(self.normalize(T.ToTensor()(crop)))

        return torch.stack(crops), row['new_filename']

def main():
    print("=" * 70)
    print("🔮 DINOv2 TTA 10-Crop Prediction (Fixed)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # 加载数据
    test_dataset = TTADatasetFixed('data/test_data_sample.csv', 'data/test_images', img_size=518)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,
                             num_workers=2, pin_memory=True)  # 减少workers避免OOM

    print(f"✅ Test dataset: {len(test_dataset)} samples\n")

    # 加载5个fold模型并进行TTA预测
    all_fold_preds = []

    for fold in range(5):
        model_path = f'outputs/dinov2_breakthrough/fold{fold}/best.pt'

        if not Path(model_path).exists():
            print(f"⚠️ Fold {fold} model not found, skipping")
            continue

        print(f"📊 Fold {fold} TTA predicting...")

        # 加载模型
        model = timm.create_model('vit_base_patch14_dinov2', pretrained=False, num_classes=4)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        fold_probs = []
        filenames_list = []

        with torch.no_grad():
            for crops_batch, fnames in tqdm(test_loader, desc=f'Fold {fold} TTA'):
                # crops_batch: [batch_size, 10, 3, 518, 518]
                batch_size = crops_batch.size(0)
                n_crops = crops_batch.size(1)

                # Reshape: [batch_size * 10, 3, 518, 518]
                crops = crops_batch.view(-1, 3, 518, 518).to(device)

                # 预测
                outputs = model(crops)
                probs = torch.softmax(outputs, dim=1)

                # Reshape back: [batch_size, 10, 4]
                probs = probs.view(batch_size, n_crops, 4)

                # 平均10个crop
                avg_probs = probs.mean(dim=1)  # [batch_size, 4]

                fold_probs.append(avg_probs.cpu().numpy())
                if fold == 0:  # 只需要收集一次文件名
                    filenames_list.extend(fnames)

        fold_probs = np.concatenate(fold_probs, axis=0)
        all_fold_preds.append(fold_probs)

        print(f"   ✅ Fold {fold} TTA complete\n")

        del model
        torch.cuda.empty_cache()

    # 保存filenames
    if len(filenames_list) == 0:
        filenames_list = pd.read_csv('data/test_data_sample.csv')['new_filename'].values

    # 集成5个fold
    print(f"🔮 Ensembling {len(all_fold_preds)} folds...")
    avg_probs = np.mean(all_fold_preds, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)

    # 创建提交
    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

    # One-hot格式
    submission_df = pd.DataFrame({
        'new_filename': filenames_list[:len(final_preds)]
    })

    for i, cls in enumerate(class_names):
        submission_df[cls] = (final_preds == i).astype(int)

    submission_path = 'data/submission_dinov2_tta_10crop.csv'
    submission_df.to_csv(submission_path, index=False)

    print(f"\n✅ TTA Submission saved: {submission_path}")
    print(f"\n📊 Prediction distribution:")
    for i, cls in enumerate(class_names):
        count = (final_preds == i).sum()
        print(f"  {cls}: {count} ({count/len(final_preds)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("✅ DINOv2 TTA Complete!")
    print("=" * 70)

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
