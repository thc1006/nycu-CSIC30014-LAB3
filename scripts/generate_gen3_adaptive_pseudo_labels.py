#!/usr/bin/env python3
"""
Gen3 自適應偽標籤生成器
策略：類別特定置信度閾值 + 難度自適應
目標：800-900 個高質量偽標籤
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("🎯 Gen3 自適應偽標籤生成器")
print("="*70)

# 載入 Gen2 的預測結果（5-Fold 平均）
print("\n載入 Gen2 預測...")
gen2_preds = []
for fold in range(5):
    pred_file = f'data/submission_v2l_512_gen2_fold{fold}.csv'
    if Path(pred_file).exists():
        df = pd.read_csv(pred_file)
        gen2_preds.append(df)
        print(f"  ✅ Fold {fold}: {len(df)} 樣本")
    else:
        print(f"  ⚠️  Fold {fold}: 文件不存在，將使用現有模型")
        # 如果 Gen2 還未完成，使用最佳現有模型
        pred_file = 'data/submission_hybrid_adaptive.csv'
        df = pd.read_csv(pred_file)
        gen2_preds.append(df)
        print(f"  📌 使用後備預測: {pred_file}")
        break

# 平均預測（如果有多個 fold）
if len(gen2_preds) > 1:
    print(f"\n平均 {len(gen2_preds)} 個模型的預測...")
    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

    # 確保所有預測順序一致
    base_filenames = gen2_preds[0]['new_filename'].values
    for i, pred_df in enumerate(gen2_preds[1:], 1):
        assert (pred_df['new_filename'].values == base_filenames).all(), \
            f"Fold {i} 文件名順序不一致"

    # 平均概率
    avg_probs = np.mean([df[class_cols].values for df in gen2_preds], axis=0)
    ensemble_df = pd.DataFrame(avg_probs, columns=class_cols)
    ensemble_df.insert(0, 'new_filename', base_filenames)
else:
    ensemble_df = gen2_preds[0].copy()
    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

print(f"✅ 集成預測完成: {len(ensemble_df)} 個測試樣本")

# 自適應閾值設定（根據類別難度）
adaptive_thresholds = {
    'normal': 0.92,      # 簡單類別：高閾值
    'bacteria': 0.90,    # 簡單類別：高閾值
    'virus': 0.85,       # 困難類別：中等閾值
    'COVID-19': 0.80     # 最困難：低閾值（確保獲得足夠樣本）
}

print("\n" + "="*70)
print("🔬 自適應閾值策略")
print("="*70)
for cls, thresh in adaptive_thresholds.items():
    print(f"  {cls:12s}: {thresh:.2f}")

# 計算每個樣本的預測類別和置信度
pred_classes = np.argmax(ensemble_df[class_cols].values, axis=1)
confidences = np.max(ensemble_df[class_cols].values, axis=1)

# 按類別分別篩選
high_conf_indices = []
high_conf_stats = {}

for cls_idx, cls_name in enumerate(class_cols):
    # 找到預測為此類別的樣本
    cls_mask = (pred_classes == cls_idx)
    cls_indices = np.where(cls_mask)[0]
    cls_confs = confidences[cls_mask]

    # 應用自適應閾值
    threshold = adaptive_thresholds[cls_name]
    high_conf_mask = cls_confs >= threshold
    selected_indices = cls_indices[high_conf_mask]

    high_conf_indices.extend(selected_indices.tolist())

    high_conf_stats[cls_name] = {
        'total': len(cls_indices),
        'selected': len(selected_indices),
        'avg_conf': cls_confs[high_conf_mask].mean() if high_conf_mask.any() else 0.0,
        'min_conf': cls_confs[high_conf_mask].min() if high_conf_mask.any() else 0.0,
        'max_conf': cls_confs[high_conf_mask].max() if high_conf_mask.any() else 0.0
    }

# 去重並排序
high_conf_indices = sorted(set(high_conf_indices))

print("\n" + "="*70)
print("📊 各類別偽標籤統計")
print("="*70)
for cls_name, stats in high_conf_stats.items():
    print(f"\n{cls_name}:")
    print(f"  預測總數: {stats['total']}")
    print(f"  高質量: {stats['selected']} ({stats['selected']/max(stats['total'],1)*100:.1f}%)")
    if stats['selected'] > 0:
        print(f"  置信度: {stats['min_conf']:.4f} - {stats['max_conf']:.4f} (平均: {stats['avg_conf']:.4f})")

# 創建偽標籤 DataFrame
pseudo_labels = []
for idx in high_conf_indices:
    row = ensemble_df.iloc[idx]
    filename = row['new_filename']
    cls_idx = pred_classes[idx]
    conf = confidences[idx]

    # 創建 one-hot 標籤
    label = [0, 0, 0, 0]
    label[cls_idx] = 1

    pseudo_labels.append({
        'new_filename': filename,
        'normal': label[0],
        'bacteria': label[1],
        'virus': label[2],
        'COVID-19': label[3],
        'source_dir': 'test_images',
        'class_label': class_cols[cls_idx],
        'confidence': conf,
        'source': 'pseudo_gen3'
    })

pseudo_df = pd.DataFrame(pseudo_labels)

# 保存偽標籤
output_path = 'data/pseudo_labels_gen3/adaptive.csv'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
pseudo_df.to_csv(output_path, index=False)

print("\n" + "="*70)
print("✅ Gen3 偽標籤生成完成")
print("="*70)
print(f"\n總數: {len(pseudo_df)} 個高質量偽標籤")
print(f"平均置信度: {pseudo_df['confidence'].mean():.4f}")
print(f"最低置信度: {pseudo_df['confidence'].min():.4f}")
print(f"保存位置: {output_path}")

# 預測分布
print("\n預測分布:")
for cls in class_cols:
    count = (pseudo_df['class_label'] == cls).sum()
    print(f"  {cls}: {count} ({count/len(pseudo_df)*100:.1f}%)")

# 與原始訓練數據合併並創建 5-Fold 訓練集
print("\n" + "="*70)
print("📦 創建 Gen3 訓練數據集 (5-Fold)")
print("="*70)

# 載入原始 K-Fold CSV
for fold in range(5):
    fold_train_csv = f'data/fold{fold}_train.csv'
    if not Path(fold_train_csv).exists():
        print(f"⚠️  {fold_train_csv} 不存在，跳過")
        continue

    # 載入原始訓練數據
    train_df = pd.read_csv(fold_train_csv)

    # 合併偽標籤
    gen3_train = pd.concat([train_df, pseudo_df], ignore_index=True)

    # 保存
    output_csv = f'data/fold{fold}_train_gen3.csv'
    gen3_train.to_csv(output_csv, index=False)

    print(f"Fold {fold}: {len(train_df)} 原始 + {len(pseudo_df)} 偽標籤 = {len(gen3_train)} 總計")

print("\n" + "="*70)
print("🎉 Gen3 數據準備完成！")
print("="*70)
print(f"\n下一步: bash START_GEN3_TRAINING.sh")
print(f"預期驗證 F1: 89.0-90.0%")
print(f"預期測試 F1: 89.5-91.0% 🎯")
