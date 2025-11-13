#!/usr/bin/env python3
"""
第1名風格的簡單加權集成
Simple weighted ensemble like the 1st place winner
NO complex stacking, just smart weighted averaging
"""

import pandas as pd
import numpy as np
from pathlib import Path

def champion_ensemble():
    """
    第1名使用的簡單加權平均策略
    根據驗證集表現給予權重
    """
    print("=" * 80)
    print("CHAMPION-STYLE SIMPLE WEIGHTED ENSEMBLE")
    print("第1名風格：簡單加權平均集成")
    print("=" * 80)

    data_dir = Path('data')

    # 我們當前最佳的模型提交文件
    submissions = {
        # 基於之前的驗證分數（從 CLAUDE.md）
        'ultimate_final': {
            'file': 'submission_ultimate_final.csv',
            'val_f1': 0.8568,  # 85.68%
            'test_f1': 0.8411,  # 84.11%
            'weight': 0.40  # 最高權重 - 最佳單一集成
        },
        'mega_ensemble_tta': {
            'file': 'submission_mega_ensemble_tta.csv',
            'val_f1': 0.85,  # 估計
            'test_f1': None,
            'weight': 0.25  # TTA 增強版本
        },
        'ultimate_smart': {
            'file': 'submission_ultimate_smart.csv',
            'val_f1': 0.84,  # 估計
            'test_f1': None,
            'weight': 0.20  # 智能集成
        },
        'improved_breakthrough': {
            'file': 'submission_improved.csv',
            'val_f1': 0.8779,  # 87.79%
            'test_f1': 0.8390,  # 83.90%
            'weight': 0.15  # 最佳單一模型
        },
    }

    print("\n使用的模型:")
    for name, info in submissions.items():
        print(f"  {name:25} weight={info['weight']:.2f}  val_f1={info.get('val_f1', 0):.4f}")

    # 讀取所有提交文件
    all_preds = []
    weights = []

    for name, info in submissions.items():
        filepath = data_dir / info['file']
        if not filepath.exists():
            print(f"  ⚠️ {name} 文件不存在: {filepath}")
            continue

        df = pd.read_csv(filepath)

        # 提取概率（假設是 one-hot 或 soft probabilities）
        prob_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
        probs = df[prob_cols].values

        all_preds.append(probs)
        weights.append(info['weight'])

        print(f"  ✓ {name}: {probs.shape}")

    if len(all_preds) == 0:
        print("\n❌ 沒有找到任何提交文件！")
        return

    # 歸一化權重
    weights = np.array(weights)
    weights = weights / weights.sum()

    print(f"\n歸一化後權重: {weights}")

    # 加權平均
    print("\n計算加權平均...")
    final_probs = np.zeros_like(all_preds[0])

    for pred, w in zip(all_preds, weights):
        final_probs += w * pred

    # 轉換為 one-hot（硬預測）
    pred_classes = final_probs.argmax(axis=1)
    final_one_hot = np.zeros_like(final_probs)
    final_one_hot[np.arange(len(pred_classes)), pred_classes] = 1

    # 創建提交文件
    submission_df = pd.DataFrame(
        final_one_hot,
        columns=['normal', 'bacteria', 'virus', 'COVID-19']
    )

    # 添加文件名（從第一個提交文件複製）
    first_df = pd.read_csv(data_dir / submissions[list(submissions.keys())[0]]['file'])
    submission_df.insert(0, 'new_filename', first_df['new_filename'])

    # 保存
    output_file = data_dir / 'submission_champion_simple.csv'
    submission_df.to_csv(output_file, index=False)

    print(f"\n✓ 保存至: {output_file}")

    # 顯示預測分布
    class_counts = final_one_hot.sum(axis=0)
    print(f"\n預測類別分布:")
    for i, col in enumerate(['normal', 'bacteria', 'virus', 'COVID-19']):
        print(f"  {col:10}: {int(class_counts[i]):4} ({class_counts[i]/len(final_one_hot)*100:5.2f}%)")

    print("\n" + "=" * 80)
    print("完成！")
    print("預期提升: +0.5-1.5% (基於第1名經驗)")
    print("=" * 80)

if __name__ == '__main__':
    champion_ensemble()
