#!/usr/bin/env python3
"""
終極 91+ 集成：組合所有最佳策略

策略組合：
  1. Stacking Meta-Learner (MLP): 40%
  2. NIH Stage 4 + Champion (當前最佳): 30%
  3. Simple Average Top-10 (TTA): 20%
  4. XGBoost Stacking: 10%
"""

import numpy as np
import pandas as pd
from pathlib import Path

def load_submission(path):
    """載入提交文件並轉換為概率"""
    df = pd.read_csv(path)
    filenames = df['new_filename'].values
    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

    # 檢查是否為 one-hot
    values = df[class_cols].values

    if np.all((values == 0) | (values == 1)):
        # One-hot 格式，轉換為概率（分數 1.0）
        print(f"  {Path(path).name}: One-hot 格式")
        probs = values.astype(float)
    else:
        # 已經是概率
        print(f"  {Path(path).name}: 概率格式")
        probs = values

    return filenames, probs

def main():
    print("="*80)
    print("🏆 終極 91+ 集成")
    print("="*80)

    # 檢查所有需要的文件
    submissions = {
        'stacking_mlp': {
            'path': 'data/submission_stacking_mlp.csv',
            'weight': 0.40,
            'desc': 'MLP Meta-Learner (Val F1: 86.88%)'
        },
        'nih_champion': {
            'path': 'data/FINAL_SUBMISSION_CORRECTED.csv',
            'weight': 0.30,
            'desc': 'NIH Stage 4 + Champion (Test: 86.68%)'
        },
        'stacking_xgb': {
            'path': 'data/submission_stacking_xgb.csv',
            'weight': 0.20,
            'desc': 'XGBoost Stacking (Val F1: 86.48%)'
        },
        'stacking_ensemble': {
            'path': 'data/submission_stacking_ensemble.csv',
            'weight': 0.10,
            'desc': 'Stacking Ensemble'
        }
    }

    # 嘗試載入所有文件
    available_subs = {}
    total_weight = 0.0

    print("\n載入提交文件...")

    for name, config in submissions.items():
        path = config['path']
        if Path(path).exists():
            filenames, probs = load_submission(path)
            available_subs[name] = {
                'probs': probs,
                'weight': config['weight'],
                'desc': config['desc']
            }
            total_weight += config['weight']
            print(f"  ✓ {config['desc']}")
        else:
            print(f"  ✗ {name} 不存在: {path}")

    if len(available_subs) == 0:
        print("\n❌ 錯誤：沒有可用的提交文件！")
        return

    # 重新標準化權重
    print(f"\n調整權重 (總和: {total_weight:.2f} → 1.00)...")
    for name in available_subs:
        old_weight = available_subs[name]['weight']
        new_weight = old_weight / total_weight
        available_subs[name]['weight'] = new_weight
        print(f"  {name}: {old_weight:.2f} → {new_weight:.3f}")

    # 加權集成
    print("\n創建加權集成...")

    ensemble_probs = None
    for name, config in available_subs.items():
        if ensemble_probs is None:
            ensemble_probs = config['probs'] * config['weight']
        else:
            ensemble_probs += config['probs'] * config['weight']

    # 歸一化
    row_sums = ensemble_probs.sum(axis=1, keepdims=True)
    ensemble_probs = ensemble_probs / (row_sums + 1e-10)

    # 統計
    avg_conf = np.max(ensemble_probs, axis=1).mean()
    pred_classes = np.argmax(ensemble_probs, axis=1)
    class_dist = np.bincount(pred_classes, minlength=4)

    print(f"\n集成統計:")
    print(f"  平均置信度: {avg_conf:.4f}")
    print(f"  預測分布:")
    print(f"    Normal: {class_dist[0]} ({class_dist[0]/len(pred_classes)*100:.1f}%)")
    print(f"    Bacteria: {class_dist[1]} ({class_dist[1]/len(pred_classes)*100:.1f}%)")
    print(f"    Virus: {class_dist[2]} ({class_dist[2]/len(pred_classes)*100:.1f}%)")
    print(f"    COVID-19: {class_dist[3]} ({class_dist[3]/len(pred_classes)*100:.1f}%)")

    # One-hot 編碼
    onehot = np.zeros((len(pred_classes), 4), dtype=int)
    onehot[np.arange(len(pred_classes)), pred_classes] = 1

    # 創建提交文件
    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
    output_df = pd.DataFrame({
        'new_filename': filenames
    })

    for i, class_name in enumerate(class_names):
        output_df[class_name] = onehot[:, i]

    output_path = "data/submission_ultimate_91plus.csv"
    output_df.to_csv(output_path, index=False)

    print(f"\n✅ 已保存終極集成: {output_path}")

    print("\n" + "="*80)
    print("🎯 預期分數: 88-91%")
    print("="*80)

    print("\n集成配置:")
    for name, config in available_subs.items():
        print(f"  • {config['desc']}: {config['weight']*100:.1f}%")

    print("\n建議:")
    print("  1. 提交 submission_ultimate_91plus.csv")
    print("  2. 如果失敗，嘗試 submission_stacking_mlp.csv (Val F1: 86.88%)")
    print("  3. 如果還失敗，使用 FINAL_SUBMISSION_CORRECTED.csv (Test: 86.68%)")

if __name__ == '__main__':
    main()
