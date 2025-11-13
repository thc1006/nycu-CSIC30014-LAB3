#!/usr/bin/env python3
"""
創建 K-Fold 數據分割
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def create_kfold_splits(train_csv='data/train.csv', val_csv='data/val.csv', n_folds=5, output_dir='data/kfold_splits'):
    """創建 K-Fold 數據分割"""

    print(f"Creating {n_folds}-Fold CV splits...")

    # 讀取數據
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # 合併 train + val
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    # 添加 source_dir 欄位（指向原始圖片目錄）
    combined_df['source_dir'] = 'data/train'

    print(f"Total samples: {len(combined_df)}")

    # 獲取標籤
    label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
    labels = combined_df[label_cols].values.argmax(axis=1)

    print(f"Class distribution:")
    for i, col in enumerate(label_cols):
        count = (labels == i).sum()
        print(f"  {col}: {count} ({100*count/len(labels):.1f}%)")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 創建輸出目錄
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成每個 fold 的 train/val split
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(combined_df, labels)):
        fold_train_df = combined_df.iloc[train_indices].copy()
        fold_val_df = combined_df.iloc[val_indices].copy()

        # 保存
        train_path = output_dir / f'fold{fold_idx}_train.csv'
        val_path = output_dir / f'fold{fold_idx}_val.csv'

        fold_train_df.to_csv(train_path, index=False)
        fold_val_df.to_csv(val_path, index=False)

        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(fold_train_df)} samples -> {train_path}")
        print(f"  Val:   {len(fold_val_df)} samples -> {val_path}")

        # Val 分布
        val_labels = fold_val_df[label_cols].values.argmax(axis=1)
        for i, col in enumerate(label_cols):
            count = (val_labels == i).sum()
            print(f"    {col}: {count}")

    print(f"\n✅ {n_folds}-Fold splits created in {output_dir}")

if __name__ == '__main__':
    create_kfold_splits()
