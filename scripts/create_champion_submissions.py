#!/usr/bin/env python3
"""
創建多個奪冠策略提交
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_probs(path):
    """載入概率文件"""
    df = pd.read_csv(path)
    filenames = df['new_filename'].values

    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

    # 檢查是否為 one-hot
    values = df[class_cols].values

    if np.all((values == 0) | (values == 1)):
        # One-hot，需要轉回概率（假設等概率）
        probs = values.astype(float)
    else:
        # 已經是概率
        probs = values

    return filenames, probs

def probs_to_onehot(probs):
    """概率轉 one-hot"""
    pred_classes = np.argmax(probs, axis=1)
    onehot = np.zeros((len(pred_classes), 4), dtype=int)
    onehot[np.arange(len(pred_classes)), pred_classes] = 1
    return onehot

def save_submission(filenames, probs, output_path, description=""):
    """保存提交文件"""
    onehot = probs_to_onehot(probs)

    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
    output_df = pd.DataFrame({'new_filename': filenames})

    for i, class_name in enumerate(class_names):
        output_df[class_name] = onehot[:, i]

    output_df.to_csv(output_path, index=False)

    # 統計
    pred_classes = np.argmax(probs, axis=1)
    avg_conf = np.max(probs, axis=1).mean()
    class_dist = np.bincount(pred_classes, minlength=4)

    print(f"\n✅ {output_path}")
    print(f"   {description}")
    print(f"   平均置信度: {avg_conf:.4f}")
    print(f"   分布: N={class_dist[0]} B={class_dist[1]} V={class_dist[2]} C={class_dist[3]}")

def main():
    print("="*80)
    print("🏆 創建奪冠策略提交")
    print("="*80)

    # 載入所有可用的預測
    nih_stage4_path = 'data/submission_nih_stage4.csv'
    champion_path = 'data/FINAL_SUBMISSION_CORRECTED.csv'

    print("\n載入預測...")
    filenames, nih_probs = load_probs(nih_stage4_path)
    _, champion_probs = load_probs(champion_path)

    print(f"  ✓ NIH Stage 4: {nih_probs.shape}")
    print(f"  ✓ Champion: {champion_probs.shape}")

    # 策略 A: 純 NIH Stage 4
    print("\n" + "="*80)
    print("策略 A: 純 NIH Stage 4")
    print("="*80)
    save_submission(
        filenames, nih_probs,
        'data/submission_pure_nih_stage4.csv',
        "Pure NIH Stage 4 (Val F1: 88.35%)"
    )

    # 策略 B: 80% NIH + 20% Champion
    print("\n" + "="*80)
    print("策略 B: 80% NIH Stage 4 + 20% Champion")
    print("="*80)
    ensemble_80_20 = 0.8 * nih_probs + 0.2 * champion_probs
    save_submission(
        filenames, ensemble_80_20,
        'data/submission_nih80_champion20.csv',
        "80% NIH Stage 4 + 20% Champion"
    )

    # 策略 C: 70% NIH + 30% Champion
    print("\n" + "="*80)
    print("策略 C: 70% NIH Stage 4 + 30% Champion")
    print("="*80)
    ensemble_70_30 = 0.7 * nih_probs + 0.3 * champion_probs
    save_submission(
        filenames, ensemble_70_30,
        'data/submission_nih70_champion30.csv',
        "70% NIH Stage 4 + 30% Champion"
    )

    # 策略 D: 60% NIH + 40% Champion
    print("\n" + "="*80)
    print("策略 D: 60% NIH Stage 4 + 40% Champion")
    print("="*80)
    ensemble_60_40 = 0.6 * nih_probs + 0.4 * champion_probs
    save_submission(
        filenames, ensemble_60_40,
        'data/submission_nih60_champion40.csv',
        "60% NIH Stage 4 + 40% Champion (close to current best)"
    )

    # 策略 E: 90% NIH + 10% Champion
    print("\n" + "="*80)
    print("策略 E: 90% NIH Stage 4 + 10% Champion")
    print("="*80)
    ensemble_90_10 = 0.9 * nih_probs + 0.1 * champion_probs
    save_submission(
        filenames, ensemble_90_10,
        'data/submission_nih90_champion10.csv',
        "90% NIH Stage 4 + 10% Champion (aggressive)"
    )

    print("\n" + "="*80)
    print("✅ 所有策略已生成！")
    print("="*80)

    print("\n建議提交順序:")
    print("  1. submission_nih80_champion20.csv (最平衡)")
    print("  2. submission_pure_nih_stage4.csv (最純粹)")
    print("  3. submission_nih90_champion10.csv (最激進)")
    print("  4. submission_nih70_champion30.csv (保守一點)")
    print("\n預期: 至少一個能突破 87%！")

if __name__ == '__main__':
    main()
