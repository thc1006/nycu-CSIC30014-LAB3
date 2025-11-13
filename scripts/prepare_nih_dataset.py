#!/usr/bin/env python3
"""
準備 NIH ChestX-ray14 數據集用於預訓練
將多標籤轉換為 multi-hot encoding
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 14種疾病標籤（NIH ChestX-ray14）
DISEASE_LABELS = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia'
]

def parse_finding_labels(finding_str):
    """
    將 Finding Labels 字符串轉換為 multi-hot vector

    Examples:
        'No Finding' -> [0, 0, 0, ..., 0]
        'Pneumonia' -> [0, 0, 0, 0, 0, 0, 1, 0, ...]
        'Pneumonia|Effusion' -> [0, 0, 1, 0, 0, 0, 1, 0, ...]
    """
    labels = np.zeros(len(DISEASE_LABELS), dtype=np.float32)

    if finding_str == 'No Finding':
        return labels

    findings = finding_str.split('|')
    for finding in findings:
        if finding in DISEASE_LABELS:
            idx = DISEASE_LABELS.index(finding)
            labels[idx] = 1.0

    return labels

def main():
    print("=" * 80)
    print("準備 NIH ChestX-ray14 數據集用於多標籤預訓練")
    print("=" * 80)

    project_root = Path(__file__).parent.parent
    nih_dir = project_root / 'data' / 'external' / 'nih_chestxray14'

    # 讀取數據
    print("\n讀取 Data_Entry_2017.csv...")
    df = pd.read_csv(nih_dir / 'Data_Entry_2017.csv')
    print(f"總樣本數: {len(df)}")

    # 讀取訓練/驗證/測試分割
    train_val_list = pd.read_csv(nih_dir / 'train_val_list_NIH.txt', header=None, names=['filename'])
    test_list = pd.read_csv(nih_dir / 'test_list_NIH.txt', header=None, names=['filename'])

    print(f"訓練+驗證: {len(train_val_list)}")
    print(f"測試: {len(test_list)}")

    # 創建 multi-hot 標籤
    print("\n轉換為 multi-hot encoding...")
    labels_array = []
    for finding in df['Finding Labels']:
        labels_array.append(parse_finding_labels(finding))

    labels_array = np.array(labels_array)

    # 添加到 DataFrame
    for i, disease in enumerate(DISEASE_LABELS):
        df[disease] = labels_array[:, i]

    # 統計
    print("\n疾病統計:")
    for disease in DISEASE_LABELS:
        count = df[disease].sum()
        print(f"  {disease:25} {int(count):6} ({count/len(df)*100:5.2f}%)")

    # 分割訓練/驗證集（90/10）
    train_val_df = df[df['Image Index'].isin(train_val_list['filename'])]
    test_df = df[df['Image Index'].isin(test_list['filename'])]

    # 從訓練+驗證中分出10%作為驗證
    val_size = int(len(train_val_df) * 0.1)
    train_val_df = train_val_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = train_val_df.iloc[val_size:]
    val_df = train_val_df.iloc[:val_size]

    print(f"\n最終分割:")
    print(f"  訓練: {len(train_df)}")
    print(f"  驗證: {len(val_df)}")
    print(f"  測試: {len(test_df)}")

    # 添加影像路徑
    images_dir = 'data/external/nih_chestxray14/images-224/images-224'
    train_df['image_path'] = images_dir
    val_df['image_path'] = images_dir
    test_df['image_path'] = images_dir

    # 保存
    output_dir = nih_dir / 'processed'
    output_dir.mkdir(exist_ok=True)

    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)

    print(f"\n✓ 保存至: {output_dir}/")
    print("  - train.csv")
    print("  - val.csv")
    print("  - test.csv")

    print("\n" + "=" * 80)
    print("數據準備完成！")
    print("下一步: python3 scripts/train_pretrain_nih.py")
    print("=" * 80)

if __name__ == '__main__':
    main()
