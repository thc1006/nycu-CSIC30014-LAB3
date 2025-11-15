#!/usr/bin/env python3
"""
智能偽標籤集成 - 利用偽標籤信息優化集成策略
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_probs(path):
    df = pd.read_csv(path)
    filenames = df['new_filename'].values
    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
    values = df[class_cols].values

    if np.all((values == 0) | (values == 1)):
        probs = values.astype(float)
    else:
        probs = values

    return filenames, probs

def probs_to_onehot(probs):
    pred_classes = np.argmax(probs, axis=1)
    onehot = np.zeros((len(pred_classes), 4), dtype=int)
    onehot[np.arange(len(pred_classes)), pred_classes] = 1
    return onehot

def main():
    print("="*80)
    print("🧠 智能偽標籤集成")
    print("="*80)

    # 載入模型預測
    filenames, nih_probs = load_probs('data/submission_nih_stage4.csv')
    _, champion_probs = load_probs('data/FINAL_SUBMISSION_CORRECTED.csv')

    # 載入偽標籤信息
    pseudo_df = pd.read_csv('data/pseudo_labels_aggressive_0.80.csv')

    # 創建偽標籤字典（檔名 → 置信度）
    pseudo_conf_dict = {}
    for _, row in pseudo_df.iterrows():
        pseudo_conf_dict[row['new_filename']] = row['confidence']

    print(f"\n載入數據:")
    print(f"  測試樣本: {len(filenames)}")
    print(f"  偽標籤覆蓋: {len(pseudo_conf_dict)} ({len(pseudo_conf_dict)/len(filenames)*100:.1f}%)")

    # ════════════════════════════════════════════════════════════
    # 策略 1: 偽標籤置信度自適應加權
    # ════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("策略 1: 偽標籤置信度自適應加權")
    print("="*80)

    ensemble_adaptive = []

    for i, fname in enumerate(filenames):
        if fname in pseudo_conf_dict:
            conf = pseudo_conf_dict[fname]

            if conf >= 0.95:
                # 高置信度：使用 NIH (更激進)
                weight_nih = 0.70
            elif conf >= 0.90:
                weight_nih = 0.60
            elif conf >= 0.85:
                weight_nih = 0.50
            else:
                # 低置信度：使用保守集成
                weight_nih = 0.40
        else:
            # 沒有偽標籤：使用當前最佳
            weight_nih = 0.45

        prob = weight_nih * nih_probs[i] + (1 - weight_nih) * champion_probs[i]
        ensemble_adaptive.append(prob)

    ensemble_adaptive = np.array(ensemble_adaptive)

    # 保存
    onehot = probs_to_onehot(ensemble_adaptive)
    output_df = pd.DataFrame({'new_filename': filenames})
    for i, cn in enumerate(['normal', 'bacteria', 'virus', 'COVID-19']):
        output_df[cn] = onehot[:, i]

    output_df.to_csv('data/submission_adaptive_confidence.csv', index=False)

    stats = np.bincount(np.argmax(ensemble_adaptive, axis=1), minlength=4)
    conf = np.max(ensemble_adaptive, axis=1).mean()
    print(f"\n✅ submission_adaptive_confidence.csv")
    print(f"  置信度: {conf:.4f}")
    print(f"  分布: N={stats[0]} B={stats[1]} V={stats[2]} C={stats[3]}")

    # ════════════════════════════════════════════════════════════
    # 策略 2: 類別特定權重
    # ════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("策略 2: 類別特定權重")
    print("="*80)

    # 基於偽標籤分析，不同類別使用不同權重
    class_weights = {
        0: {'nih': 0.50, 'champion': 0.50},  # normal - 平衡
        1: {'nih': 0.60, 'champion': 0.40},  # bacteria - NIH 更強
        2: {'nih': 0.40, 'champion': 0.60},  # virus - Champion 更強
        3: {'nih': 0.70, 'champion': 0.30},  # COVID-19 - NIH 最強
    }

    # 對每個樣本，根據最可能的類別選擇權重
    ensemble_class_specific = []

    for i in range(len(filenames)):
        # 使用平均預測確定最可能的類別
        avg_pred = (nih_probs[i] + champion_probs[i]) / 2
        pred_class = np.argmax(avg_pred)

        # 使用該類別的特定權重
        weight_nih = class_weights[pred_class]['nih']
        prob = weight_nih * nih_probs[i] + (1 - weight_nih) * champion_probs[i]
        ensemble_class_specific.append(prob)

    ensemble_class_specific = np.array(ensemble_class_specific)

    # 保存
    onehot = probs_to_onehot(ensemble_class_specific)
    output_df = pd.DataFrame({'new_filename': filenames})
    for i, cn in enumerate(['normal', 'bacteria', 'virus', 'COVID-19']):
        output_df[cn] = onehot[:, i]

    output_df.to_csv('data/submission_class_specific.csv', index=False)

    stats = np.bincount(np.argmax(ensemble_class_specific, axis=1), minlength=4)
    conf = np.max(ensemble_class_specific, axis=1).mean()
    print(f"\n✅ submission_class_specific.csv")
    print(f"  置信度: {conf:.4f}")
    print(f"  分布: N={stats[0]} B={stats[1]} V={stats[2]} C={stats[3]}")

    # ════════════════════════════════════════════════════════════
    # 策略 3: 混合自適應
    # ════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("策略 3: 混合自適應（置信度 + 類別）")
    print("="*80)

    ensemble_hybrid = []

    for i, fname in enumerate(filenames):
        # 基礎權重（類別特定）
        avg_pred = (nih_probs[i] + champion_probs[i]) / 2
        pred_class = np.argmax(avg_pred)
        base_weight_nih = class_weights[pred_class]['nih']

        # 如果有偽標籤，根據置信度調整
        if fname in pseudo_conf_dict:
            conf = pseudo_conf_dict[fname]

            if conf >= 0.95:
                adjustment = +0.15  # 增加 NIH 權重
            elif conf >= 0.90:
                adjustment = +0.10
            elif conf >= 0.85:
                adjustment = +0.05
            else:
                adjustment = -0.05  # 降低 NIH 權重（不確定時保守）
        else:
            adjustment = 0

        weight_nih = np.clip(base_weight_nih + adjustment, 0.2, 0.8)
        prob = weight_nih * nih_probs[i] + (1 - weight_nih) * champion_probs[i]
        ensemble_hybrid.append(prob)

    ensemble_hybrid = np.array(ensemble_hybrid)

    # 保存
    onehot = probs_to_onehot(ensemble_hybrid)
    output_df = pd.DataFrame({'new_filename': filenames})
    for i, cn in enumerate(['normal', 'bacteria', 'virus', 'COVID-19']):
        output_df[cn] = onehot[:, i]

    output_df.to_csv('data/submission_hybrid_adaptive.csv', index=False)

    stats = np.bincount(np.argmax(ensemble_hybrid, axis=1), minlength=4)
    conf = np.max(ensemble_hybrid, axis=1).mean()
    print(f"\n✅ submission_hybrid_adaptive.csv")
    print(f"  置信度: {conf:.4f}")
    print(f"  分布: N={stats[0]} B={stats[1]} V={stats[2]} C={stats[3]}")

    # ════════════════════════════════════════════════════════════
    # 總結
    # ════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("✅ 智能集成完成！")
    print("="*80)

    print("\n生成的提交文件:")
    print("  1. submission_adaptive_confidence.csv - 基於偽標籤置信度")
    print("  2. submission_class_specific.csv - 基於類別特性")
    print("  3. submission_hybrid_adaptive.csv - 混合策略")

    print("\n建議提交順序:")
    print("  優先: submission_hybrid_adaptive.csv (最智能)")
    print("  備選: submission_adaptive_confidence.csv")
    print("  備選: submission_class_specific.csv")

    print("\n預期效果: 86.683% → 87-88% (+0.5-1.5%)")

if __name__ == '__main__':
    main()
