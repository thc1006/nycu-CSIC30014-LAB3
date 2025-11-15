#!/usr/bin/env python3
"""
投票集成：基於 0/1 預測的集成策略（無需概率）
"""

import pandas as pd
import numpy as np
import os

print("=" * 70)
print("🚀 投票集成優化器（0/1 預測）")
print("=" * 70)

# 頂級模型
top_models = [
    ('submission_hybrid_adaptive.csv', 87.574),  # 最佳
    ('submission_adaptive_confidence.csv', 86.683),
    ('submission_class_specific.csv', 86.638),
    ('submission_champion_arch_weighted.csv', 85.800),
    ('submission_champion_weighted_avg.csv', 85.780),
]

print("\n📊 加載模型預測...")

predictions = []
for filename, score in top_models:
    filepath = f'data/{filename}'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        predictions.append({
            'name': filename,
            'score': score,
            'df': df
        })
        print(f"  ✅ {filename}: {score}%")
    else:
        print(f"  ❌ {filename}: 不存在")

print(f"\n✅ 成功加載 {len(predictions)} 個模型")

# 提取預測
class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
all_preds = []

for pred in predictions:
    df = pred['df']
    # 每一行找到預測為 1 的類別
    preds_matrix = df[class_cols].values
    preds_classes = np.argmax(preds_matrix, axis=1)
    all_preds.append(preds_classes)

all_preds = np.array(all_preds)  # (num_models, num_samples)

# 策略 1: 多數表決
print("\n" + "=" * 70)
print("策略 1: 多數表決（Majority Voting）")
print("=" * 70)

from scipy import stats
majority_vote, _ = stats.mode(all_preds, axis=0, keepdims=False)

# 創建提交
submission_majority = predictions[0]['df'][['new_filename']].copy()
for i, cls in enumerate(class_cols):
    submission_majority[cls] = (majority_vote == i).astype(int)

submission_majority.to_csv('data/submission_ultra_majority_vote.csv', index=False)
print("\n✅ 已保存: data/submission_ultra_majority_vote.csv")

# 統計
print(f"\n預測分布:")
for i, cls in enumerate(class_cols):
    count = (majority_vote == i).sum()
    pct = count / len(majority_vote) * 100
    print(f"  {cls}: {count} ({pct:.1f}%)")

# 策略 2: 加權投票（基於分數）
print("\n" + "=" * 70)
print("策略 2: 加權投票（基於模型分數）")
print("=" * 70)

scores = np.array([p['score'] for p in predictions])
weights = scores / scores.sum()

print("\n權重分配:")
for pred, w in zip(predictions, weights):
    print(f"  {pred['name']}: {w:.4f}")

# 為每個類別累積投票
weighted_votes = np.zeros((len(all_preds[0]), 4))

for i, (preds, weight) in enumerate(zip(all_preds, weights)):
    for j, pred_class in enumerate(preds):
        weighted_votes[j, pred_class] += weight

# 選擇累積投票最高的類別
weighted_pred = np.argmax(weighted_votes, axis=1)

# 創建提交
submission_weighted = predictions[0]['df'][['new_filename']].copy()
for i, cls in enumerate(class_cols):
    submission_weighted[cls] = (weighted_pred == i).astype(int)

submission_weighted.to_csv('data/submission_ultra_weighted_vote.csv', index=False)
print("\n✅ 已保存: data/submission_ultra_weighted_vote.csv")

# 統計
print(f"\n預測分布:")
for i, cls in enumerate(class_cols):
    count = (weighted_pred == i).sum()
    pct = count / len(weighted_pred) * 100
    print(f"  {cls}: {count} ({pct:.1f}%)")

# 策略 3: Top-3 加權投票
print("\n" + "=" * 70)
print("策略 3: Top-3 模型加權投票")
print("=" * 70)

top3_preds = all_preds[:3]
top3_scores = scores[:3]
top3_weights = top3_scores / top3_scores.sum()

print("\nTop-3 權重:")
for pred, w in zip(predictions[:3], top3_weights):
    print(f"  {pred['name']}: {w:.4f}")

# 累積投票
top3_votes = np.zeros((len(top3_preds[0]), 4))

for preds, weight in zip(top3_preds, top3_weights):
    for j, pred_class in enumerate(preds):
        top3_votes[j, pred_class] += weight

top3_pred = np.argmax(top3_votes, axis=1)

# 創建提交
submission_top3 = predictions[0]['df'][['new_filename']].copy()
for i, cls in enumerate(class_cols):
    submission_top3[cls] = (top3_pred == i).astype(int)

submission_top3.to_csv('data/submission_ultra_top3_weighted.csv', index=False)
print("\n✅ 已保存: data/submission_ultra_top3_weighted.csv")

# 統計
print(f"\n預測分布:")
for i, cls in enumerate(class_cols):
    count = (top3_pred == i).sum()
    pct = count / len(top3_pred) * 100
    print(f"  {cls}: {count} ({pct:.1f}%)")

# 最終總結
print("\n" + "=" * 70)
print("🎉 投票集成完成！")
print("=" * 70)

print("\n生成的集成文件:")
print("  1. submission_ultra_majority_vote.csv - 多數表決")
print("  2. submission_ultra_weighted_vote.csv - 加權投票（所有模型）⭐")
print("  3. submission_ultra_top3_weighted.csv - Top-3 加權投票 ⭐⭐")

print("\n推薦提交: submission_ultra_top3_weighted.csv")
print("  理由: 專注於最佳模型，減少弱模型噪聲")

print("\n預期提升:")
print("  基於文獻：集成可提升 0.5-1.0%")
print("  預期分數：88.0-88.5%")

print("\n=" * 70)
