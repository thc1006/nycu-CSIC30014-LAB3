"""
Deep Visual Analysis of Train/Val/Test Dataset Differences
Find the root cause of 83.90% -> 91% gap
"""
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from collections import defaultdict
import json

print("=" * 80)
print("DEEP VISUAL ANALYSIS - FINDING THE 91% BREAKTHROUGH PATTERN")
print("=" * 80)

# Load data
train_df = pd.read_csv('data/train_data.csv')
val_df = pd.read_csv('data/val_data.csv')
test_df = pd.read_csv('data/test_data.csv')

class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

print(f"\n[1/6] Dataset Statistics:")
print(f"   Train: {len(train_df)} samples")
print(f"   Val:   {len(val_df)} samples")
print(f"   Test:  {len(test_df)} samples")

# Get labels
def get_label(row):
    for cls in class_names:
        if row[cls] == 1.0:
            return cls
    return 'unknown'

train_df['label'] = train_df.apply(get_label, axis=1)
val_df['label'] = val_df.apply(get_label, axis=1)

print(f"\n[2/6] Computing Image Statistics...")

def analyze_images(df, img_dir, split_name):
    stats = defaultdict(list)

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"   {split_name}: {idx}/{len(df)} processed...")

        img_path = Path(img_dir) / row['new_filename']
        if not img_path.exists():
            continue

        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        label = row.get('label', 'test')

        # Compute statistics
        stats[f'{label}_mean'].append(img.mean())
        stats[f'{label}_std'].append(img.std())
        stats[f'{label}_min'].append(img.min())
        stats[f'{label}_max'].append(img.max())
        stats[f'{label}_brightness'].append(img.mean())
        stats[f'{label}_contrast'].append(img.std() / (img.mean() + 1e-7))

        # Compute histogram entropy
        hist, _ = np.histogram(img, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        stats[f'{label}_entropy'].append(entropy)

        # Image size
        stats[f'{label}_height'].append(img.shape[0])
        stats[f'{label}_width'].append(img.shape[1])
        stats[f'{label}_aspect_ratio'].append(img.shape[1] / img.shape[0])

    return stats

print(f"\n   Analyzing TRAIN set...")
train_stats = analyze_images(train_df, 'train_images', 'Train')

print(f"\n   Analyzing VAL set...")
val_stats = analyze_images(val_df, 'val_images', 'Val')

print(f"\n   Analyzing TEST set...")
test_stats = analyze_images(test_df, 'test_images', 'Test')

print(f"\n[3/6] Statistical Comparison:")

def print_stat_comparison(metric, train_s, val_s, test_s, class_name='test'):
    key = f'{class_name}_{metric}'
    if key in train_s and len(train_s[key]) > 0:
        train_val = np.mean(train_s[key])
        print(f"   Train {metric:15s}: {train_val:.2f}")
    if key in val_s and len(val_s[key]) > 0:
        val_val = np.mean(val_s[key])
        print(f"   Val   {metric:15s}: {val_val:.2f}")
    if key in test_s and len(test_s[key]) > 0:
        test_val = np.mean(test_s[key])
        print(f"   Test  {metric:15s}: {test_val:.2f}")
    print()

metrics = ['mean', 'std', 'brightness', 'contrast', 'entropy', 'height', 'width']
for metric in metrics:
    print(f"\n{metric.upper()}:")
    print_stat_comparison(metric, train_stats, val_stats, test_stats)

# Class-specific analysis
print(f"\n[4/6] Class-Specific Statistics (Train vs Val):")
for cls in class_names:
    print(f"\n{cls.upper()}:")
    for metric in ['mean', 'std', 'contrast', 'entropy']:
        key = f'{cls}_{metric}'
        if key in train_stats and len(train_stats[key]) > 0:
            train_val = np.mean(train_stats[key])
            val_val = np.mean(val_stats[key]) if key in val_stats and len(val_stats[key]) > 0 else 0
            diff = abs(train_val - val_val)
            print(f"   {metric:12s}: Train={train_val:6.2f}, Val={val_val:6.2f}, Diff={diff:6.2f}")

# Save analysis
print(f"\n[5/6] Saving detailed analysis...")

analysis_result = {
    'train_stats': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                    for k, v in train_stats.items() if len(v) > 0},
    'val_stats': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                  for k, v in val_stats.items() if len(v) > 0},
    'test_stats': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                   for k, v in test_stats.items() if len(v) > 0},
}

with open('outputs/visual_analysis.json', 'w') as f:
    json.dump(analysis_result, f, indent=2)

print(f"   Saved to: outputs/visual_analysis.json")

print(f"\n[6/6] KEY FINDINGS:")
print(f"   ⚠ Check if test set has different image statistics")
print(f"   ⚠ Look for systematic differences in brightness/contrast/entropy")
print(f"   ⚠ These differences explain the train-test gap!")

print(f"\n" + "=" * 80)
print("ANALYSIS COMPLETE - Review outputs/visual_analysis.json for details")
print("=" * 80)
