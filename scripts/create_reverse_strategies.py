#!/usr/bin/env python3
"""
反向策略：更多 Champion，更少 NIH
"""

import pandas as pd
import numpy as np

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

def save_submission(filenames, probs, output_path, nih_pct, champ_pct):
    onehot = probs_to_onehot(probs)

    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
    output_df = pd.DataFrame({'new_filename': filenames})

    for i, class_name in enumerate(class_names):
        output_df[class_name] = onehot[:, i]

    output_df.to_csv(output_path, index=False)

    pred_classes = np.argmax(probs, axis=1)
    avg_conf = np.max(probs, axis=1).mean()
    class_dist = np.bincount(pred_classes, minlength=4)

    print(f"\n✅ {output_path}")
    print(f"   {nih_pct}% NIH + {champ_pct}% Champion")
    print(f"   置信度: {avg_conf:.4f}")
    print(f"   分布: N={class_dist[0]} B={class_dist[1]} V={class_dist[2]} C={class_dist[3]}")

def main():
    print("="*80)
    print("🔄 反向策略：Champion 為主")
    print("="*80)

    filenames, nih_probs = load_probs('data/submission_nih_stage4.csv')
    _, champion_probs = load_probs('data/FINAL_SUBMISSION_CORRECTED.csv')

    # 策略 1: 40% NIH + 60% Champion
    ensemble_40_60 = 0.4 * nih_probs + 0.6 * champion_probs
    save_submission(filenames, ensemble_40_60,
                   'data/submission_nih40_champion60.csv', 40, 60)

    # 策略 2: 30% NIH + 70% Champion
    ensemble_30_70 = 0.3 * nih_probs + 0.7 * champion_probs
    save_submission(filenames, ensemble_30_70,
                   'data/submission_nih30_champion70.csv', 30, 70)

    # 策略 3: 20% NIH + 80% Champion
    ensemble_20_80 = 0.2 * nih_probs + 0.8 * champion_probs
    save_submission(filenames, ensemble_20_80,
                   'data/submission_nih20_champion80.csv', 20, 80)

    # 策略 4: 45% NIH + 55% Champion (接近當前最佳的反向)
    ensemble_45_55 = 0.45 * nih_probs + 0.55 * champion_probs
    save_submission(filenames, ensemble_45_55,
                   'data/submission_nih45_champion55.csv', 45, 55)

    # 策略 5: 50% NIH + 50% Champion (完全平衡)
    ensemble_50_50 = 0.5 * nih_probs + 0.5 * champion_probs
    save_submission(filenames, ensemble_50_50,
                   'data/submission_nih50_champion50.csv', 50, 50)

    print("\n" + "="*80)
    print("✅ 反向策略已生成！")
    print("="*80)

    print("\n理論分析:")
    print("  當前最佳 (55% NIH): 86.683%")
    print("  趨勢: NIH 比重越高，分數越低")
    print("  \n  預測: Champion 比重增加可能帶來提升！")
    print("  \n  最有希望:")
    print("    • 45-55 (反向當前最佳)")
    print("    • 40-60 (更多 Champion)")
    print("    • 50-50 (完全平衡)")

if __name__ == '__main__':
    main()
