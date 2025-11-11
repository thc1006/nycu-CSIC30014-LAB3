#!/usr/bin/env python3
"""
綜合醫學影像深度分析腳本
系統性分析所有類別的胸部X光影像，記錄臨床特徵
"""
import pandas as pd
import json
from pathlib import Path
import random

print("=" * 80)
print("綜合醫學影像深度分析")
print("=" * 80)

# 讀取 CSV
train_df = pd.read_csv('data/train_data.csv')
val_df = pd.read_csv('data/val_data.csv')

# 添加來源目錄
train_df['source'] = 'train_images'
val_df['source'] = 'val_images'

combined_df = pd.concat([train_df, val_df], ignore_index=True)

# 按類別分組
label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

categories = {}
for label in label_cols:
    mask = combined_df[label] == 1
    files = combined_df[mask][['new_filename', 'source']].values.tolist()
    categories[label] = files
    print(f"\n{label}: {len(files)} 張影像")

# 選擇分析樣本
analysis_samples = {
    'COVID-19': categories['COVID-19'],  # 全部
    'bacteria': random.sample(categories['bacteria'], min(15, len(categories['bacteria']))),
    'virus': random.sample(categories['virus'], min(15, len(categories['virus']))),
    'normal': random.sample(categories['normal'], min(10, len(categories['normal']))),
}

# 儲存分析計劃
analysis_plan = {
    'total_samples': sum(len(v) for v in analysis_samples.values()),
    'categories': {k: len(v) for k, v in analysis_samples.items()},
    'samples': {k: [(f[0], f[1]) for f in v] for k, v in analysis_samples.items()}
}

with open('data/comprehensive_analysis_plan.json', 'w') as f:
    json.dump(analysis_plan, f, indent=2)

print(f"\n{'='*80}")
print("分析計劃")
print(f"{'='*80}")
for label, samples in analysis_samples.items():
    print(f"{label:15s}: {len(samples):3d} 張")

print(f"\n總計: {analysis_plan['total_samples']} 張影像待分析")

# 生成分析檔案列表（用於後續視覺分析）
output_list = []
for label, samples in analysis_samples.items():
    for fname, source in samples:
        output_list.append({
            'label': label,
            'filename': fname,
            'path': f"{source}/{fname}"
        })

# 依類別分組保存
with open('data/images_for_visual_analysis.json', 'w') as f:
    json.dump({
        'by_category': analysis_samples,
        'flat_list': output_list
    }, f, indent=2, default=str)

print(f"\n✅ 分析計劃已保存至 data/comprehensive_analysis_plan.json")
print(f"✅ 影像列表已保存至 data/images_for_visual_analysis.json")
print(f"\n準備開始視覺分析...")
