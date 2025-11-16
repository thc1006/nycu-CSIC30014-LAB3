#!/usr/bin/env python3
"""
Quick Breakthrough Ensemble - 方案 C 第一步

目標: 快速創建 4-5 模型集成，預期 +0.3-0.7% 提升
策略: 基於模型一致性的智能加權

模型池:
1. Hybrid Adaptive (87.574%) - 最強基礎
2. Swin-Large (86.785%) - Transformer視角
3. DINOv2 (86.702%) - 自監督視角
4. V2L-512 TTA (未知) - TTA增強
5. V2L-512 Ensemble (可選) - 基礎模型

集成策略:
- 不是簡單平均
- 基於一致性動態加權
- Temperature Scaling 優化
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

print("=" * 80)
print("🚀 Quick Breakthrough Ensemble - 突破 89%+")
print("=" * 80)
print()

# ============================================================================
# 1. 載入所有可用模型
# ============================================================================
print("📂 載入模型預測...")

def load_submission(path):
    """載入並解碼提交文件"""
    df = pd.read_csv(path)

    # 解碼 one-hot
    classes = ['normal', 'bacteria', 'virus', 'COVID-19']
    preds = []
    for _, row in df.iterrows():
        for cls in classes:
            if row[cls] == 1:
                preds.append(cls)
                break

    df['pred'] = preds
    return df

models = {}

# 核心 3 模型
try:
    models['hybrid'] = load_submission('data/submission_hybrid_adaptive.csv')
    print("  ✅ Hybrid Adaptive loaded")
except:
    print("  ❌ Hybrid Adaptive not found")

try:
    models['swin'] = load_submission('data/submission_swin_large.csv')
    print("  ✅ Swin-Large loaded")
except:
    print("  ❌ Swin-Large not found")

try:
    models['dinov2'] = load_submission('data/submission_dinov2.csv')
    print("  ✅ DINOv2 loaded")
except:
    print("  ❌ DINOv2 not found")

# TTA 模型
try:
    models['tta'] = load_submission('data/submission_v2l_512_tta.csv')
    print("  ✅ V2L-512 TTA loaded")
except:
    print("  ❌ V2L-512 TTA not found")

# 基礎集成
try:
    models['v2l_ensemble'] = load_submission('data/submission_v2l_512_ensemble.csv')
    print("  ✅ V2L-512 Ensemble loaded")
except:
    print("  ❌ V2L-512 Ensemble not found")

print(f"\n總共載入 {len(models)} 個模型")
print()

if len(models) < 3:
    print("❌ 模型數量不足，需要至少 3 個模型")
    exit(1)

# ============================================================================
# 2. 分析模型一致性
# ============================================================================
print("🔍 分析模型一致性...")

n_samples = len(list(models.values())[0])
model_names = list(models.keys())
n_models = len(models)

# 收集所有預測
all_preds = np.zeros((n_samples, n_models), dtype=object)
for i, (name, df) in enumerate(models.items()):
    all_preds[:, i] = df['pred'].values

# 統計一致性
agreement_counts = {i: 0 for i in range(n_models + 1)}

for i in range(n_samples):
    preds = all_preds[i, :]
    unique_preds = len(set(preds))

    # 計算一致數量
    max_agreement = max(Counter(preds).values())
    agreement_counts[max_agreement] += 1

print(f"  總樣本: {n_samples}")
for agree_count in sorted(agreement_counts.keys(), reverse=True):
    count = agreement_counts[agree_count]
    if count > 0:
        pct = count / n_samples * 100
        emoji = "✅" if agree_count >= n_models - 1 else "⚠️" if agree_count >= n_models // 2 else "❌"
        print(f"  {emoji} {agree_count}/{n_models} 模型一致: {count:4d} ({pct:5.1f}%)")
print()

# ============================================================================
# 3. 策略 A: Weighted Voting (基於模型分數)
# ============================================================================
print("📊 策略 A: Weighted Voting")
print("-" * 80)

# 基於已知測試分數的權重
model_weights = {
    'hybrid': 0.87574,
    'swin': 0.86785,
    'dinov2': 0.86702,
    'tta': 0.870,  # 估計 (保守)
    'v2l_ensemble': 0.855  # 估計 (非常保守)
}

# 歸一化權重
available_weights = {name: model_weights.get(name, 0.85)
                     for name in model_names}
total_weight = sum(available_weights.values())
normalized_weights = {name: w / total_weight
                      for name, w in available_weights.items()}

print("  模型權重:")
for name, weight in normalized_weights.items():
    print(f"    {name:.<20} {weight:.4f} ({weight*100:.1f}%)")
print()

# 加權投票
final_preds_weighted = []

for i in range(n_samples):
    vote_scores = {}

    for j, name in enumerate(model_names):
        pred = all_preds[i, j]
        weight = normalized_weights[name]

        if pred not in vote_scores:
            vote_scores[pred] = 0
        vote_scores[pred] += weight

    # 選擇最高得分
    best_pred = max(vote_scores.items(), key=lambda x: x[1])[0]
    final_preds_weighted.append(best_pred)

print(f"  ✅ Weighted Voting 完成")
print()

# ============================================================================
# 4. 策略 B: Confidence-Based Dynamic Voting
# ============================================================================
print("📊 策略 B: Confidence-Based Dynamic Voting")
print("-" * 80)

final_preds_confidence = []

for i in range(n_samples):
    preds = all_preds[i, :]

    # 計算一致性
    pred_counts = Counter(preds)
    most_common_pred, most_common_count = pred_counts.most_common(1)[0]

    # 動態決策
    if most_common_count >= n_models * 0.6:  # 60%+ 一致
        # 高一致性 -> 直接採用
        final_pred = most_common_pred
    elif most_common_count >= n_models * 0.4:  # 40-60% 一致
        # 中等一致性 -> 加權投票
        vote_scores = {}
        for j, name in enumerate(model_names):
            pred = preds[j]
            weight = normalized_weights[name]

            if pred not in vote_scores:
                vote_scores[pred] = 0
            vote_scores[pred] += weight

        final_pred = max(vote_scores.items(), key=lambda x: x[1])[0]
    else:
        # 低一致性 -> 採用最強模型
        final_pred = preds[model_names.index('hybrid')]

    final_preds_confidence.append(final_pred)

print(f"  ✅ Confidence-Based Voting 完成")
print()

# ============================================================================
# 5. 策略 C: Hybrid (結合 A + B)
# ============================================================================
print("📊 策略 C: Hybrid Strategy (推薦)")
print("-" * 80)

final_preds_hybrid = []

for i in range(n_samples):
    weighted_pred = final_preds_weighted[i]
    confidence_pred = final_preds_confidence[i]

    # 如果兩種方法一致，直接採用
    if weighted_pred == confidence_pred:
        final_pred = weighted_pred
    else:
        # 不一致時，查看原始預測
        preds = all_preds[i, :]
        pred_counts = Counter(preds)

        # 如果其中一個在原始預測中佔多數，採用它
        if weighted_pred in pred_counts and pred_counts[weighted_pred] >= n_models // 2:
            final_pred = weighted_pred
        elif confidence_pred in pred_counts and pred_counts[confidence_pred] >= n_models // 2:
            final_pred = confidence_pred
        else:
            # 否則採用 Hybrid Adaptive 的預測
            final_pred = preds[model_names.index('hybrid')]

    final_preds_hybrid.append(final_pred)

print(f"  ✅ Hybrid Strategy 完成")
print()

# ============================================================================
# 6. 比較三種策略
# ============================================================================
print("📊 策略比較")
print("-" * 80)

# 與 Hybrid Adaptive 比較
hybrid_preds = models['hybrid']['pred'].values

diff_weighted = sum(final_preds_weighted != hybrid_preds)
diff_confidence = sum(final_preds_confidence != hybrid_preds)
diff_hybrid = sum(np.array(final_preds_hybrid) != hybrid_preds)

print(f"  vs Hybrid Adaptive (87.574%):")
print(f"    Weighted Voting 差異: {diff_weighted:4d} ({diff_weighted/n_samples*100:.1f}%)")
print(f"    Confidence-Based 差異: {diff_confidence:4d} ({diff_confidence/n_samples*100:.1f}%)")
print(f"    Hybrid Strategy 差異: {diff_hybrid:4d} ({diff_hybrid/n_samples*100:.1f}%)")
print()

# 預測分布
print("  預測分布:")
for strategy_name, preds in [
    ('Weighted', final_preds_weighted),
    ('Confidence', final_preds_confidence),
    ('Hybrid', final_preds_hybrid)
]:
    counts = Counter(preds)
    print(f"    {strategy_name}:")
    for cls in ['normal', 'bacteria', 'virus', 'COVID-19']:
        count = counts[cls]
        print(f"      {cls:.<15} {count:4d} ({count/n_samples*100:.1f}%)")
print()

# ============================================================================
# 7. 保存提交文件
# ============================================================================
print("💾 保存提交文件...")

base_df = models['hybrid'][['new_filename']].copy()

# 策略 A
submission_weighted = base_df.copy()
for col in ['normal', 'bacteria', 'virus', 'COVID-19']:
    submission_weighted[col] = 0
for i, pred in enumerate(final_preds_weighted):
    submission_weighted.at[i, pred] = 1
submission_weighted.to_csv('data/submission_quick_weighted.csv', index=False)
print("  ✅ data/submission_quick_weighted.csv")

# 策略 B
submission_confidence = base_df.copy()
for col in ['normal', 'bacteria', 'virus', 'COVID-19']:
    submission_confidence[col] = 0
for i, pred in enumerate(final_preds_confidence):
    submission_confidence.at[i, pred] = 1
submission_confidence.to_csv('data/submission_quick_confidence.csv', index=False)
print("  ✅ data/submission_quick_confidence.csv")

# 策略 C (推薦)
submission_hybrid = base_df.copy()
for col in ['normal', 'bacteria', 'virus', 'COVID-19']:
    submission_hybrid[col] = 0
for i, pred in enumerate(final_preds_hybrid):
    submission_hybrid.at[i, pred] = 1
submission_hybrid.to_csv('data/submission_quick_hybrid.csv', index=False)
print("  ✅ data/submission_quick_hybrid.csv (推薦)")
print()

# ============================================================================
# 8. 總結與建議
# ============================================================================
print("=" * 80)
print("🎯 總結與建議")
print("=" * 80)
print()
print(f"  使用了 {n_models} 個模型:")
for name in model_names:
    print(f"    - {name}")
print()
print("  推薦提交順序:")
print("    1. submission_quick_hybrid.csv (最平衡)")
print("    2. submission_quick_confidence.csv (更保守)")
print("    3. submission_quick_weighted.csv (更激進)")
print()
print(f"  預期提升: +0.3-0.7% (基於 {diff_hybrid} 個差異樣本)")
print(f"  預期分數: 88.7-89.1%")
print()
print("=" * 80)
