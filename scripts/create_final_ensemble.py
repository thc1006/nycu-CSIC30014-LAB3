#!/usr/bin/env python3
"""
創建最終集成：NIH Stage 4 + Champion Balanced
"""

import pandas as pd
import numpy as np

def ensemble_predictions(submissions, weights, output_csv='data/submission_final_ensemble.csv'):
    """
    集成多個提交文件

    Args:
        submissions: dict of {name: csv_path}
        weights: dict of {name: weight} (should sum to 1.0)
    """

    print("="*80)
    print("最終集成")
    print("="*80)

    # Load all submissions
    dfs = {}
    for name, path in submissions.items():
        df = pd.read_csv(path)
        dfs[name] = df
        print(f"✓ {name}: {path}")
        print(f"  - 樣本數: {len(df)}")

    # Verify all have same filenames
    filenames = dfs[list(dfs.keys())[0]]['new_filename'].values
    for name, df in dfs.items():
        assert np.all(df['new_filename'].values == filenames), f"{name} has different filenames!"

    print(f"\n集成權重:")
    for name, weight in weights.items():
        print(f"  {name}: {weight*100:.1f}%")

    total_weight = sum(weights.values())
    print(f"總權重: {total_weight:.3f}")
    assert abs(total_weight - 1.0) < 0.001, f"權重總和必須為 1.0，當前為 {total_weight}"

    # Ensemble
    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

    # Start with zeros
    ensemble_probs = np.zeros((len(filenames), 4))

    for name, weight in weights.items():
        df = dfs[name]
        probs = df[class_names].values
        ensemble_probs += weight * probs

        print(f"\n{name} 預測分布:")
        pred_labels = np.argmax(probs, axis=1)
        for i, class_name in enumerate(class_names):
            count = np.sum(pred_labels == i)
            print(f"  {class_name}: {count} ({100*count/len(pred_labels):.1f}%)")

    # Create submission
    submission = pd.DataFrame({
        'new_filename': filenames
    })

    for i, class_name in enumerate(class_names):
        submission[class_name] = ensemble_probs[:, i]

    # Save
    submission.to_csv(output_csv, index=False)
    print(f"\n✅ 集成預測已保存: {output_csv}")

    # Final statistics
    pred_labels = np.argmax(ensemble_probs, axis=1)
    max_probs = np.max(ensemble_probs, axis=1)

    print(f"\n最終預測分布:")
    for i, class_name in enumerate(class_names):
        count = np.sum(pred_labels == i)
        print(f"  {class_name}: {count} ({100*count/len(pred_labels):.1f}%)")

    print(f"\n置信度統計:")
    print(f"  平均: {np.mean(max_probs):.4f}")
    print(f"  中位數: {np.median(max_probs):.4f}")
    print(f"  最小: {np.min(max_probs):.4f}")
    print(f"  最大: {np.max(max_probs):.4f}")

if __name__ == '__main__':
    import sys

    # Option 1: 兩路集成 (保守穩健)
    print("選項: 兩路集成 (NIH Stage 4 + Champion)")
    print("-" * 80)

    submissions = {
        'nih_stage4': 'data/submission_nih_stage4.csv',
        'champion': 'data/submission_ultimate_final.csv',  # Champion Balanced 的實際文件
    }

    weights = {
        'nih_stage4': 0.55,  # NIH Val F1 最高
        'champion': 0.45,    # Champion 已驗證高測試分數
    }

    ensemble_predictions(submissions, weights, 'data/submission_final_two_way.csv')

    print("\n" + "="*80)
    print("預期分數: 84.8-86.2%")
    print("="*80)
