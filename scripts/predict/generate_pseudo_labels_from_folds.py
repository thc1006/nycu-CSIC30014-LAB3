#!/usr/bin/env python3
"""
從 V2-L 512 的 5 個 fold 預測生成偽標籤
策略：至少 3/5 fold 一致 + 平均置信度 ≥0.95
"""

import pandas as pd
import numpy as np
from scipy import stats

def main():
    print("="*70)
    print("🔍 從 5-Fold 預測生成偽標籤")
    print("="*70)

    # 5 個 fold 的預測文件
    fold_files = [f'data/submission_v2l_512_fold{i}.csv' for i in range(5)]
    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

    # 載入所有 fold 預測
    print("\n📂 載入 5-Fold 預測...")
    all_probs = []
    filenames = None

    for i, path in enumerate(fold_files):
        df = pd.read_csv(path)
        probs = df[class_cols].values
        all_probs.append(probs)
        if filenames is None:
            filenames = df['new_filename'].values
        print(f"  ✅ Fold {i}: {len(df)} 張，平均最大置信度 {probs.max(axis=1).mean():.4f}")

    all_probs = np.array(all_probs)  # (5, n_samples, 4)
    n_samples = all_probs.shape[1]

    print(f"\n📊 分析 5-Fold 一致性...")

    # 每個樣本的預測類別 (5個fold)
    pred_classes = np.argmax(all_probs, axis=2)  # (5, n_samples)

    # 計算每個樣本的眾數（最常預測的類別）
    mode_results = stats.mode(pred_classes, axis=0, keepdims=False)
    mode_class = mode_results.mode  # 眾數類別
    mode_count = mode_results.count  # 眾數出現次數

    # 計算每個樣本的平均置信度
    avg_probs = np.mean(all_probs, axis=0)  # (n_samples, 4)
    max_avg_conf = np.max(avg_probs, axis=1)  # 每個樣本的最大平均置信度

    # 統計一致性
    print(f"  5/5 一致: {(mode_count == 5).sum()} ({(mode_count == 5).sum()/n_samples*100:.1f}%)")
    print(f"  4/5 一致: {(mode_count == 4).sum()} ({(mode_count == 4).sum()/n_samples*100:.1f}%)")
    print(f"  3/5 一致: {(mode_count == 3).sum()} ({(mode_count == 3).sum()/n_samples*100:.1f}%)")

    # 篩選策略：
    # 1. Normal/Bacteria/Virus: 至少 4/5 一致 + 平均置信度 ≥0.95
    # 2. COVID-19: 至少 3/5 一致 + 平均置信度 ≥0.90 (放寬標準)

    mask_general = (mode_count >= 4) & (max_avg_conf >= 0.95) & (mode_class != 3)
    mask_covid = (mode_count >= 3) & (max_avg_conf >= 0.90) & (mode_class == 3)

    final_mask = mask_general | mask_covid

    print(f"\n✅ 篩選結果:")
    print(f"  一般類別 (4/5一致, conf≥0.95): {mask_general.sum()}")
    print(f"  COVID-19 (3/5一致, conf≥0.90): {mask_covid.sum()}")
    print(f"  總計: {final_mask.sum()} ({final_mask.sum()/n_samples*100:.1f}%)")

    # 創建偽標籤數據集
    pseudo_df = pd.DataFrame({
        'new_filename': filenames[final_mask],
        'label': mode_class[final_mask],
        'confidence': max_avg_conf[final_mask],
        'agreement': mode_count[final_mask]  # 記錄一致性程度
    })

    # 添加 one-hot 列
    for i, col in enumerate(class_cols):
        pseudo_df[col] = (pseudo_df['label'] == i).astype(int)

    # 按類別統計
    print(f"\n📈 偽標籤類別分布:")
    total = len(pseudo_df)
    for i, col in enumerate(class_cols):
        count = (pseudo_df['label'] == i).sum()
        avg_conf = pseudo_df[pseudo_df['label'] == i]['confidence'].mean() if count > 0 else 0
        avg_agree = pseudo_df[pseudo_df['label'] == i]['agreement'].mean() if count > 0 else 0
        print(f"  {col:12s}: {count:4d} ({count/total*100:5.1f}%) - 平均置信度: {avg_conf:.4f}, 平均一致性: {avg_agree:.1f}/5")

    # 保存
    output_path = 'data/test_pseudo_labels_5fold.csv'
    pseudo_df.to_csv(output_path, index=False)

    print(f"\n✅ 偽標籤已保存: {output_path}")
    print(f"   總樣本數: {len(pseudo_df)} (測試集的 {len(pseudo_df)/n_samples*100:.1f}%)")
    print(f"   平均置信度: {pseudo_df['confidence'].mean():.4f}")
    print(f"   平均一致性: {pseudo_df['agreement'].mean():.1f}/5")

    # 保存統計報告
    with open('data/pseudo_labels_5fold_stats.txt', 'w') as f:
        f.write(f"5-Fold 偽標籤統計報告\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"生成時間: 2025-11-15\n")
        f.write(f"來源: V2-L 512 @ 5-Fold CV\n")
        f.write(f"測試集總數: {n_samples}\n")
        f.write(f"高置信度樣本: {len(pseudo_df)} ({len(pseudo_df)/n_samples*100:.1f}%)\n\n")
        f.write(f"類別分布:\n")
        for i, col in enumerate(class_cols):
            count = (pseudo_df['label'] == i).sum()
            avg_conf = pseudo_df[pseudo_df['label'] == i]['confidence'].mean() if count > 0 else 0
            avg_agree = pseudo_df[pseudo_df['label'] == i]['agreement'].mean() if count > 0 else 0
            f.write(f"  {col}: {count} ({count/total*100:.1f}%) - conf: {avg_conf:.4f}, agree: {avg_agree:.1f}/5\n")
        f.write(f"\n整體統計:\n")
        f.write(f"  平均置信度: {pseudo_df['confidence'].mean():.4f}\n")
        f.write(f"  最低置信度: {pseudo_df['confidence'].min():.4f}\n")
        f.write(f"  平均一致性: {pseudo_df['agreement'].mean():.1f}/5\n")

    print(f"   統計報告: data/pseudo_labels_5fold_stats.txt")

    print(f"\n{'='*70}")
    print("🎉 Phase 1 完成！偽標籤品質更高且更多樣化")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
