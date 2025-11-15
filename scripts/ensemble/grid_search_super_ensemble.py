#!/usr/bin/env python3
"""
網格搜索超級集成 - 快速找到最優組合
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import sys

def load_probs(path):
    """載入概率文件"""
    df = pd.read_csv(path)
    filenames = df['new_filename'].values

    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
    values = df[class_cols].values

    if np.all((values == 0) | (values == 1)):
        # One-hot，假設等概率
        probs = values.astype(float)
    else:
        probs = values

    return filenames, probs

def probs_to_onehot(probs):
    """概率轉 one-hot"""
    pred_classes = np.argmax(probs, axis=1)
    onehot = np.zeros((len(pred_classes), 4), dtype=int)
    onehot[np.arange(len(pred_classes)), pred_classes] = 1
    return onehot

def create_ensemble(predictions_dict, weights):
    """創建加權集成"""
    ensemble_probs = None
    total_weight = sum(weights.values())

    for name, weight in weights.items():
        if name not in predictions_dict:
            continue

        normalized_weight = weight / total_weight
        if ensemble_probs is None:
            ensemble_probs = predictions_dict[name] * normalized_weight
        else:
            ensemble_probs += predictions_dict[name] * normalized_weight

    return ensemble_probs

def get_ensemble_stats(probs):
    """獲取集成統計"""
    pred_classes = np.argmax(probs, axis=1)
    avg_conf = np.max(probs, axis=1).mean()
    class_dist = np.bincount(pred_classes, minlength=4)
    return avg_conf, class_dist

def main():
    print("="*80)
    print("🔍 網格搜索超級集成")
    print("="*80)

    # 載入所有可用的預測
    predictions = {}

    pred_files = {
        'nih_stage4': 'data/submission_nih_stage4.csv',
        'champion': 'data/FINAL_SUBMISSION_CORRECTED.csv',
        'stacking_mlp': 'data/submission_stacking_mlp.csv',
        'stacking_xgb': 'data/submission_stacking_xgb.csv',
        'stacking_rf': 'data/submission_stacking_rf.csv',
    }

    print("\n載入預測...")
    filenames = None
    for name, path in pred_files.items():
        if Path(path).exists():
            fnames, probs = load_probs(path)
            if filenames is None:
                filenames = fnames
            predictions[name] = probs
            print(f"  ✓ {name}: {probs.shape}")
        else:
            print(f"  ✗ {name}: 不存在")

    if len(predictions) < 2:
        print("\n❌ 預測文件不足！")
        return

    print(f"\n✓ 總共載入 {len(predictions)} 個模型預測")

    # 定義權重網格
    weight_ranges = {
        'nih_stage4': [0.3, 0.4, 0.5, 0.6, 0.7],
        'champion': [0.2, 0.3, 0.4, 0.5],
        'stacking_mlp': [0.0, 0.1, 0.2, 0.3],
        'stacking_xgb': [0.0, 0.05, 0.1],
    }

    print("\n開始網格搜索...")
    print(f"搜索空間大小: {np.prod([len(v) for v in weight_ranges.values()])} 組合")

    best_ensemble = None
    best_weights = None
    best_conf = 0

    # 生成所有權重組合
    search_count = 0
    for nih_w, champ_w, mlp_w, xgb_w in product(
        weight_ranges['nih_stage4'],
        weight_ranges['champion'],
        weight_ranges['stacking_mlp'],
        weight_ranges['stacking_xgb']
    ):
        search_count += 1

        weights = {
            'nih_stage4': nih_w,
            'champion': champ_w,
            'stacking_mlp': mlp_w,
            'stacking_xgb': xgb_w,
        }

        ensemble_probs = create_ensemble(predictions, weights)
        conf, dist = get_ensemble_stats(ensemble_probs)

        # 使用置信度作為代理指標（沒有真實驗證集）
        if conf > best_conf:
            best_conf = conf
            best_ensemble = ensemble_probs
            best_weights = weights.copy()

        if search_count % 50 == 0:
            print(f"  搜索進度: {search_count}/{np.prod([len(v) for v in weight_ranges.values()])}")

    print(f"\n✅ 搜索完成！測試了 {search_count} 種組合")

    # 保存最佳集成
    print("\n" + "="*80)
    print("🏆 最佳集成配置")
    print("="*80)

    print("\n權重:")
    total = sum(best_weights.values())
    for name, weight in sorted(best_weights.items(), key=lambda x: x[1], reverse=True):
        pct = (weight/total) * 100
        print(f"  {name}: {weight:.2f} ({pct:.1f}%)")

    conf, dist = get_ensemble_stats(best_ensemble)
    print(f"\n統計:")
    print(f"  平均置信度: {conf:.4f}")
    print(f"  分布: Normal={dist[0]} Bacteria={dist[1]} Virus={dist[2]} COVID={dist[3]}")

    # 保存提交文件
    onehot = probs_to_onehot(best_ensemble)
    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
    output_df = pd.DataFrame({'new_filename': filenames})

    for i, class_name in enumerate(class_names):
        output_df[class_name] = onehot[:, i]

    output_path = 'data/submission_grid_search_super.csv'
    output_df.to_csv(output_path, index=False)

    print(f"\n✅ 已保存: {output_path}")

    # 也生成一些變體
    print("\n" + "="*80)
    print("生成權重變體...")
    print("="*80)

    variants = [
        {'nih_stage4': 0.6, 'champion': 0.3, 'stacking_mlp': 0.1, 'stacking_xgb': 0.0},
        {'nih_stage4': 0.5, 'champion': 0.4, 'stacking_mlp': 0.1, 'stacking_xgb': 0.0},
        {'nih_stage4': 0.4, 'champion': 0.4, 'stacking_mlp': 0.2, 'stacking_xgb': 0.0},
        {'nih_stage4': 0.5, 'champion': 0.3, 'stacking_mlp': 0.15, 'stacking_xgb': 0.05},
    ]

    for idx, weights in enumerate(variants, 1):
        ensemble_probs = create_ensemble(predictions, weights)
        onehot = probs_to_onehot(ensemble_probs)

        output_df = pd.DataFrame({'new_filename': filenames})
        for i, class_name in enumerate(class_names):
            output_df[class_name] = onehot[:, i]

        output_path = f'data/submission_variant_{idx}.csv'
        output_df.to_csv(output_path, index=False)

        conf, dist = get_ensemble_stats(ensemble_probs)
        print(f"\nVariant {idx}: {output_path}")
        print(f"  權重: {weights}")
        print(f"  置信度: {conf:.4f}")

    print("\n" + "="*80)
    print("✅ 所有集成已生成！")
    print("="*80)

if __name__ == '__main__':
    main()
