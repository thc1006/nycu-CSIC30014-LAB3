#!/usr/bin/env python3
"""
合併訓練數據和偽標籤
"""

import pandas as pd
import os

def merge_datasets(fold, kfold_dir='data/kfold_splits', pseudo_csv='data/pseudo_labels_nih/high_conf.csv', output_dir='data'):
    """合併指定 fold 的訓練數據和偽標籤"""

    # 讀取原始訓練數據
    train_csv = os.path.join(kfold_dir, f'fold{fold}_train.csv')
    train_df = pd.read_csv(train_csv)
    print(f"Fold {fold} 原始訓練集: {len(train_df)} 樣本")

    # 讀取偽標籤
    pseudo_df = pd.read_csv(pseudo_csv)
    print(f"高置信度偽標籤: {len(pseudo_df)} 樣本")

    # 合併
    merged_df = pd.concat([train_df, pseudo_df], ignore_index=True)
    print(f"合併後: {len(merged_df)} 樣本")

    # 統計
    print("\n類別分布:")
    for class_name in ['normal', 'bacteria', 'virus', 'COVID-19']:
        count = merged_df[class_name].sum()
        print(f"  {class_name}: {count}")

    # 保存
    output_csv = os.path.join(output_dir, f'fold{fold}_train_with_pseudo.csv')
    merged_df.to_csv(output_csv, index=False)
    print(f"\n✅ 已保存: {output_csv}")

    return merged_df

if __name__ == '__main__':
    # 為所有 5 個 fold 創建合併數據
    for fold in range(5):
        print(f"\n{'='*60}")
        print(f"處理 Fold {fold}")
        print('='*60)
        merge_datasets(fold)
