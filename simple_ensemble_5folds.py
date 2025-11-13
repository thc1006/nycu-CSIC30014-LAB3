#!/usr/bin/env python3
"""
简单的 5-Fold 集成预测
平均所有 fold 的预测概率
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 输入目录
PRED_DIR = Path('outputs/breakthrough_20251113_004854/layer1_test_predictions')

# 输出文件
OUTPUT_FILE = Path('data/submission_efficientnet_v2_l_5fold.csv')

print("=" * 80)
print("🎯 5-Fold EfficientNet-V2-L 集成")
print("=" * 80)

# 收集所有 fold 的预测
all_preds = []

for fold in range(5):
    pred_file = PRED_DIR / f'efficientnet_v2_l_fold{fold}_test_pred.csv'

    if not pred_file.exists():
        print(f"❌ Fold {fold}: 文件不存在 - {pred_file}")
        continue

    df = pd.read_csv(pred_file)
    print(f"✅ Fold {fold}: 加载 {len(df)} 样本")

    # 提取概率列
    prob_cols = ['Normal', 'Bacteria', 'Virus', 'COVID-19']
    probs = df[prob_cols].values
    all_preds.append(probs)

# 平均所有预测
print(f"\n📊 总共 {len(all_preds)} 个模型")
avg_probs = np.mean(all_preds, axis=0)
print(f"📊 平均后形状: {avg_probs.shape}")

# 转换为 one-hot
pred_idx = avg_probs.argmax(axis=1)
one_hot = np.zeros_like(avg_probs)
one_hot[np.arange(one_hot.shape[0]), pred_idx] = 1

# 创建提交文件
result_df = pd.DataFrame(one_hot, columns=prob_cols)
result_df.insert(0, 'new_filename', df['new_filename'])

# 保存
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ 集成预测已保存至: {OUTPUT_FILE}")
print("=" * 80)

# 显示预测分布
pred_counts = pd.Series(pred_idx).value_counts()
class_names = ['Normal', 'Bacteria', 'Virus', 'COVID-19']
print("\n📊 预测分布:")
for class_idx, count in pred_counts.items():
    print(f"  {class_names[class_idx]}: {count} ({count/len(pred_idx)*100:.2f}%)")
