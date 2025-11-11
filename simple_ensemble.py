#!/usr/bin/env python3
"""
Simple Ensemble - 融合現有最佳提交文件
快速融合現有的高品質預測，無需重新生成
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("SIMPLE ENSEMBLE - 融合現有最佳提交")
print("=" * 80)
print()

# 選擇最佳的現有提交文件
# 根據已知表現和多樣性選擇
submissions = [
    {
        'file': 'data/submission_improved.csv',
        'name': 'Improved Breakthrough (83.90%)',
        'weight': 0.35,  # 目前最佳，權重最高
        'format': 'onehot'
    },
    {
        'file': 'data/submission_efficientnet_tta.csv',
        'name': 'EfficientNet TTA (83.82%)',
        'weight': 0.25,
        'format': 'prob'  # 包含概率
    },
    {
        'file': 'data/submission_convnext_tta_prob.csv',
        'name': 'ConvNeXt TTA prob',
        'weight': 0.25,
        'format': 'prob'
    },
    {
        'file': 'data/submission_breakthrough.csv',
        'name': 'Breakthrough',
        'weight': 0.15,
        'format': 'onehot'
    }
]

class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

# 載入並處理所有提交
dfs = []
weights = []
successful = []

for sub in submissions:
    file_path = Path(sub['file'])
    if not file_path.exists():
        print(f"✗ 跳過 {sub['name']} (文件不存在)")
        continue

    df = pd.read_csv(file_path)

    # 檢查是否包含必要的列
    if not all(col in df.columns for col in class_cols):
        print(f"✗ 跳過 {sub['name']} (缺少類別列)")
        continue

    # 對於 one-hot 格式，轉換為偽概率
    # 如果所有值都是 0 或 1，認為是 one-hot
    probs = df[class_cols].values
    is_onehot = np.all(np.isin(probs, [0.0, 1.0]))

    if is_onehot or sub['format'] == 'onehot':
        # 將 one-hot 轉換為 soft labels (0.95/0.05)
        probs_soft = np.where(probs == 1.0, 0.95, 0.05/3)
        df[class_cols] = probs_soft
        print(f"✓ 載入 {sub['name']} (權重: {sub['weight']:.2f}) [one-hot → soft]")
    else:
        print(f"✓ 載入 {sub['name']} (權重: {sub['weight']:.2f}) [probability]")

    dfs.append(df)
    weights.append(sub['weight'])
    successful.append(sub['name'])

if len(dfs) == 0:
    print()
    print("錯誤：沒有可用的提交文件！")
    exit(1)

# 標準化權重
weights = np.array(weights)
weights = weights / weights.sum()

print()
print(f"總共載入 {len(dfs)} 個提交")
print(f"標準化權重: {weights}")
print()

# 加權平均
print("進行加權融合...")
ensemble_probs = np.zeros((len(dfs[0]), 4))

for i, (df, w, name) in enumerate(zip(dfs, weights, successful)):
    probs = df[class_cols].values
    ensemble_probs += w * probs
    print(f"  [{i+1}/{len(dfs)}] {name:40s} 權重: {w:.3f}")

# 確保概率總和為 1
ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

# 創建 one-hot 提交（競賽要求）
predicted_idx = ensemble_probs.argmax(axis=1)
onehot = np.zeros_like(ensemble_probs)
onehot[np.arange(len(ensemble_probs)), predicted_idx] = 1.0

# 創建最終提交文件
final_submission = pd.DataFrame({
    'new_filename': dfs[0]['new_filename'],
    'normal': onehot[:, 0],
    'bacteria': onehot[:, 1],
    'virus': onehot[:, 2],
    'COVID-19': onehot[:, 3]
})

# 保存
output_path = 'data/submission_ultimate_final.csv'
final_submission.to_csv(output_path, index=False)

print()
print("=" * 80)
print("預測分布:")
print("=" * 80)
for i, cls in enumerate(class_cols):
    count = (predicted_idx == i).sum()
    print(f"  {cls:12s}: {count:4d} ({count/len(predicted_idx)*100:.1f}%)")

print()
print("平均預測置信度 (fusion後):")
confidence = ensemble_probs.max(axis=1).mean()
print(f"  {confidence:.4f}")

print()
print("=" * 80)
print("✅ 簡單融合完成！")
print("=" * 80)
print()
print(f"融合模型數: {len(dfs)}")
print(f"輸出文件: {output_path}")
print()
print("下一步: 提交到 Kaggle")
print()
