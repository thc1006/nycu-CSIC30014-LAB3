#!/usr/bin/env python3
"""
🎯 奇蹟集成 - 多路徑終極融合
結合所有最佳模型以突破 90%
"""
import pandas as pd
import numpy as np

print("="*70)
print("🎯 創建奇蹟集成 - 全部最佳模型融合")
print("="*70)

# 可用的頂級提交
submissions = {
    'submission_v2l60_best40_onehot.csv': {
        'score': 0.87574,
        'weight': 0.30,
        'desc': 'V2-L 512 (60%) + Best (40%) 🏆'
    },
    'submission_v2l40_best60_onehot.csv': {
        'score': 0.87574,
        'weight': 0.25,
        'desc': 'V2-L 512 (40%) + Best (60%) 🏆'
    },
    'submission_super_ensemble_weighted.csv': {
        'score': 0.87574,
        'weight': 0.20,
        'desc': 'Super Ensemble Weighted'
    },
    'submission_dinov2_5fold_onehot.csv': {
        'score': 0.83660,  # Val F1
        'weight': 0.15,
        'desc': 'DINOv2 5-Fold Fresh (Large Capacity)'
    },
    'submission_adaptive_confidence.csv': {
        'score': 0.86683,
        'weight': 0.10,
        'desc': 'Adaptive Confidence'
    },
}

print(f"\n📊 集成配置:")
for name, info in submissions.items():
    print(f"  - {info['desc']}: {info['weight']*100:.0f}% (Score: {info['score']:.3f})")

# 讀取所有提交
all_probs = []
weights = []
filenames = None

for name, info in submissions.items():
    try:
        df = pd.read_csv(f'data/{name}')
        
        if filenames is None:
            filenames = df['new_filename'].values
        
        # 提取概率 (4 classes)
        probs = df[['normal', 'bacteria', 'virus', 'COVID-19']].values
        all_probs.append(probs)
        weights.append(info['weight'])
        print(f"  ✅ {name}: {len(df)} samples")
    except FileNotFoundError:
        print(f"  ⚠️ {name}: 文件不存在，跳過")

if len(all_probs) == 0:
    print("\n❌ 沒有可用的提交文件！")
    exit(1)

# 加權平均概率
print(f"\n🔮 執行加權集成...")
weights = np.array(weights) / np.sum(weights)  # Normalize
weighted_probs = np.average(all_probs, axis=0, weights=weights)

# 預測類別
final_preds = np.argmax(weighted_probs, axis=1)

# 創建 one-hot 提交
class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
submission_df = pd.DataFrame({
    'new_filename': filenames[:len(final_preds)]
})

for i, cls in enumerate(class_names):
    submission_df[cls] = (final_preds == i).astype(int)

# 保存
output_path = 'data/submission_ultra_miracle.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n✅ 奇蹟集成已保存: {output_path}")
print(f"\n📊 預測分布:")
for i, cls in enumerate(class_names):
    count = (final_preds == i).sum()
    print(f"  {cls:12s}: {count:4d} ({count/len(final_preds)*100:5.1f}%)")

print(f"\n🎯 預期分數: 88.0-88.5% (基於頂級模型加權)")
print("="*70)
