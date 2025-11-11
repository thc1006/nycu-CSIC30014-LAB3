#!/usr/bin/env python3
"""快速修復 CSV - 只保留實際存在的影像"""
import pandas as pd
from pathlib import Path

print("修復 CSV 檔案...")

# 訓練集
train_csv = pd.read_csv('data/train_data.csv')
train_dir = Path('train_images')
valid_train = []

for idx, row in train_csv.iterrows():
    filename = row['new_filename']
    path = train_dir / filename
    if path.exists():
        valid_train.append(row)

train_df = pd.DataFrame(valid_train)
train_df.to_csv('data/train_data.csv', index=False)
print(f"✅ 訓練集: {len(train_csv)} -> {len(train_df)} (-{len(train_csv)-len(train_df)})")

# 驗證集
val_csv = pd.read_csv('data/val_data.csv')
val_dir = Path('val_images')
valid_val = []

for idx, row in val_csv.iterrows():
    filename = row['new_filename']
    path = val_dir / filename
    if path.exists():
        valid_val.append(row)

val_df = pd.DataFrame(valid_val)
val_df.to_csv('data/val_data.csv', index=False)
print(f"✅ 驗證集: {len(val_csv)} -> {len(val_df)} (-{len(val_csv)-len(val_df)})")

# 測試集
test_csv = pd.read_csv('data/test_data.csv')
test_dir = Path('test_images')
valid_test = []

for idx, row in test_csv.iterrows():
    filename = row['new_filename']
    path = test_dir / filename
    if path.exists():
        valid_test.append(row)

test_df = pd.DataFrame(valid_test)
test_df.to_csv('data/test_data.csv', index=False)
print(f"✅ 測試集: {len(test_csv)} -> {len(test_df)} (-{len(test_csv)-len(test_df)})")

print("\n✅ CSV 修復完成！")
