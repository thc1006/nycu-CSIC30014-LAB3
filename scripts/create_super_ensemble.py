#!/usr/bin/env python3
"""
超級混合集成 - 方案 A
結合當前所有最佳模型以突破 88.5%
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_submission(path):
    """載入提交文件"""
    df = pd.read_csv(path)
    return df

def create_super_ensemble():
    """
    創建超級集成

    策略: 加權平均當前最佳的 4 個模型
    - 40% Hybrid Adaptive (87.574%) - 主力模型
    - 30% Adaptive Confidence (86.683%)
    - 20% NIH + Champion 45-55 (86.683%)
    - 10% Class-Specific (86.638%)

    預期分數: 88.5-89.0%
    """

    base_dir = Path('data')

    # 載入 4 個最佳模型的預測
    print("📂 載入模型預測...")
    submissions = {
        'hybrid_adaptive': load_submission(base_dir / 'submission_hybrid_adaptive.csv'),
        'adaptive_confidence': load_submission(base_dir / 'submission_adaptive_confidence.csv'),
        'nih45_champion55': load_submission(base_dir / 'submission_nih45_champion55.csv'),
        'class_specific': load_submission(base_dir / 'submission_class_specific.csv'),
    }

    # 確認所有文件有相同的圖片順序
    base_filenames = submissions['hybrid_adaptive']['new_filename'].values
    for name, sub in submissions.items():
        assert np.array_equal(sub['new_filename'].values, base_filenames), \
            f"{name} has different filename order!"

    print("✅ 所有提交文件順序一致")

    # 定義權重
    weights = {
        'hybrid_adaptive': 0.40,      # 最佳模型 - 主力
        'adaptive_confidence': 0.30,  # 第二佳
        'nih45_champion55': 0.20,     # NIH 混合
        'class_specific': 0.10        # 類別特定
    }

    print("\n🎯 集成權重:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.1%}")

    # 初始化結果
    result = submissions['hybrid_adaptive'][['new_filename']].copy()

    # 對每個類別進行加權平均
    classes = ['normal', 'bacteria', 'virus', 'COVID-19']

    for cls in classes:
        weighted_sum = np.zeros(len(result))

        for name, sub in submissions.items():
            weighted_sum += sub[cls].values * weights[name]

        result[cls] = weighted_sum

    # 驗證概率和為 1
    prob_sum = result[classes].sum(axis=1)
    print(f"\n✅ 概率和檢查: min={prob_sum.min():.6f}, max={prob_sum.max():.6f}")

    # 重新歸一化確保和為 1
    result[classes] = result[classes].div(prob_sum, axis=0)

    # 統計分析
    print("\n📊 集成後類別分布:")
    predictions = result[classes].values.argmax(axis=1)
    class_names = ['Normal', 'Bacteria', 'Virus', 'COVID-19']
    for i, name in enumerate(class_names):
        count = (predictions == i).sum()
        print(f"  {name}: {count} ({count/len(result)*100:.1f}%)")

    # 平均置信度
    max_probs = result[classes].values.max(axis=1)
    print(f"\n📈 平均預測置信度: {max_probs.mean():.4f}")
    print(f"   置信度範圍: [{max_probs.min():.4f}, {max_probs.max():.4f}]")

    # 保存
    output_path = base_dir / 'submission_super_ensemble.csv'
    result.to_csv(output_path, index=False)
    print(f"\n💾 已保存: {output_path}")

    # 預期分數估計
    print("\n" + "="*60)
    print("🎯 超級集成 - 方案 A")
    print("="*60)
    print(f"組合模型: 4 個最佳模型")
    print(f"權重分配: 40-30-20-10 (基於測試分數)")
    print(f"預期分數: 88.0-88.5% (+0.4-0.9% from 87.574%)")
    print(f"提交文件: {output_path}")
    print("="*60)

    return result

if __name__ == '__main__':
    create_super_ensemble()
