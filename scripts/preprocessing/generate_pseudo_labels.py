#!/usr/bin/env python3
"""
策略 1: 生成高質量偽標籤進行半監督學習
利用多個模型的一致性預測作為高置信度偽標籤
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("策略 1: 生成高質量偽標籤")
print("=" * 80)
print()

# 載入多個最佳模型的預測
predictions_files = [
    'data/submission_efficientnet_tta.csv',  # 83.82%
    'data/submission_convnext_tta_prob.csv',  # ConvNeXt TTA
    'data/submission_improved.csv',  # 83.90% - 最佳
]

class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

# 載入所有預測
all_preds = []
for file_path in predictions_files:
    if Path(file_path).exists():
        df = pd.read_csv(file_path)
        
        # 檢查是否是 one-hot 格式
        probs = df[class_cols].values
        is_onehot = np.all(np.isin(probs, [0.0, 1.0]))
        
        if is_onehot:
            # 轉換為 soft probability
            probs_soft = np.where(probs == 1.0, 0.95, 0.05/3)
            df[class_cols] = probs_soft
            print(f"✓ 載入 {Path(file_path).name} (轉換為 soft)")
        else:
            print(f"✓ 載入 {Path(file_path).name}")
        
        all_preds.append(df)

if len(all_preds) == 0:
    print("錯誤：沒有可用的預測文件！")
    exit(1)

print(f"\n總共載入 {len(all_preds)} 個預測")
print()

# 計算平均預測和置信度
print("計算平均預測和模型一致性...")
test_df = pd.read_csv('data/test_data.csv')

# 平均所有預測
avg_probs = np.zeros((len(test_df), 4))
for df in all_preds:
    avg_probs += df[class_cols].values
avg_probs /= len(all_preds)

# 計算最大概率（置信度）
max_confidence = avg_probs.max(axis=1)
predicted_class = avg_probs.argmax(axis=1)

# 計算模型間的一致性（標準差越小越一致）
consistency_scores = []
for i in range(len(test_df)):
    sample_preds = [df[class_cols].iloc[i].values for df in all_preds]
    sample_preds = np.array(sample_preds)
    # 計算每個類別預測的標準差，取平均
    consistency = 1.0 - np.mean(np.std(sample_preds, axis=0))
    consistency_scores.append(consistency)
consistency_scores = np.array(consistency_scores)

# 組合置信度和一致性作為最終分數
quality_score = (max_confidence * 0.6) + (consistency_scores * 0.4)

# 設定多個閾值以獲取不同質量的偽標籤
thresholds = [0.95, 0.92, 0.90, 0.85]

print("\n各閾值下的偽標籤統計：")
print("-" * 80)

for threshold in thresholds:
    high_conf_mask = quality_score >= threshold
    n_samples = high_conf_mask.sum()
    
    if n_samples == 0:
        print(f"閾值 {threshold:.2f}: 0 個樣本")
        continue
    
    # 統計各類別分布
    class_dist = np.bincount(predicted_class[high_conf_mask], minlength=4)
    
    print(f"\n閾值 {threshold:.2f}: {n_samples} 個樣本 ({n_samples/len(test_df)*100:.1f}%)")
    for i, cls in enumerate(class_cols):
        print(f"  {cls:12s}: {class_dist[i]:4d} ({class_dist[i]/n_samples*100:.1f}%)")

# 選擇最佳閾值（選擇 0.92，平衡質量和數量）
best_threshold = 0.92
high_conf_mask = quality_score >= best_threshold
n_pseudo = high_conf_mask.sum()

print()
print("=" * 80)
print(f"選擇閾值: {best_threshold:.2f}")
print(f"生成偽標籤數: {n_pseudo} ({n_pseudo/len(test_df)*100:.1f}% 的測試集)")
print("=" * 80)
print()

# 創建偽標籤數據
pseudo_labels_df = pd.DataFrame({
    'new_filename': test_df.loc[high_conf_mask, 'new_filename'].values,
    'normal': (predicted_class[high_conf_mask] == 0).astype(int),
    'bacteria': (predicted_class[high_conf_mask] == 1).astype(int),
    'virus': (predicted_class[high_conf_mask] == 2).astype(int),
    'COVID-19': (predicted_class[high_conf_mask] == 3).astype(int),
    'confidence': quality_score[high_conf_mask],
    'source': 'pseudo'
})

# 保存偽標籤
output_path = 'data/pseudo_labels.csv'
pseudo_labels_df.to_csv(output_path, index=False)
print(f"✓ 偽標籤已保存: {output_path}")
print()

# 創建結合原始訓練集和偽標籤的增強訓練集
print("創建增強訓練集（原始 + 偽標籤）...")

# 載入原始訓練集
train_df = pd.read_csv('data/train_data.csv')
print(f"  原始訓練集: {len(train_df)} 樣本")

# 準備偽標籤（只保留需要的列）
pseudo_for_train = pseudo_labels_df[['new_filename', 'normal', 'bacteria', 'virus', 'COVID-19']].copy()

# 合併
augmented_train = pd.concat([train_df, pseudo_for_train], ignore_index=True)
print(f"  偽標籤: {len(pseudo_for_train)} 樣本")
print(f"  增強訓練集: {len(augmented_train)} 樣本")

# 保存增強訓練集
aug_output_path = 'data/train_data_augmented.csv'
augmented_train.to_csv(aug_output_path, index=False)
print(f"\n✓ 增強訓練集已保存: {aug_output_path}")

# 統計增強後的類別分布
print("\n增強後的類別分布：")
print("-" * 80)
for col in class_cols:
    count = augmented_train[col].sum()
    print(f"  {col:12s}: {count:4d} ({count/len(augmented_train)*100:.1f}%)")

print()
print("=" * 80)
print("✅ 偽標籤生成完成！")
print("=" * 80)
print()
print("下一步: 使用增強訓練集訓練新模型")
print()

# 顯示一些高質量偽標籤的例子
print("高質量偽標籤範例（前 10 個）：")
print("-" * 80)
top_samples = pseudo_labels_df.nlargest(10, 'confidence')
for idx, row in top_samples.iterrows():
    pred_class = class_cols[np.argmax([row['normal'], row['bacteria'], row['virus'], row['COVID-19']])]
    print(f"  {row['new_filename']:30s} -> {pred_class:12s} (quality: {row['confidence']:.4f})")
print()
