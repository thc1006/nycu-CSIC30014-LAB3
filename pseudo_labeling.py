"""
Pseudo-Labeling Strategy
使用最佳模型对测试集预测，高置信度样本加入训练
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load best predictions (Improved model - 0.83900)
print("🔍 Loading best model predictions...")
submission = pd.read_csv('data/submission_improved.csv')
test_df = pd.read_csv('data/test_data.csv')

# Get probability columns
prob_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
probs = submission[prob_cols].values

# Calculate confidence (max probability)
confidences = probs.max(axis=1)
predicted_classes = probs.argmax(axis=1)

# Strategy: Only use HIGH confidence predictions
# Different thresholds for different classes
thresholds = {
    0: 0.95,  # normal - high confidence
    1: 0.95,  # bacteria - high confidence
    2: 0.95,  # virus - high confidence
    3: 0.85,  # COVID-19 - lower threshold (rare class)
}

# Select pseudo-labeled samples
pseudo_labels = []
class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

for i, (conf, pred_class) in enumerate(zip(confidences, predicted_classes)):
    threshold = thresholds[pred_class]
    if conf >= threshold:
        file_col = 'filename' if 'filename' in submission.columns else 'new_filename'
        pseudo_labels.append({
            'new_filename': submission.iloc[i][file_col],
            'label': class_names[pred_class],
            'confidence': conf,
            'fold': -1  # Mark as pseudo-labeled
        })

pseudo_df = pd.DataFrame(pseudo_labels)

print(f"\n📊 Pseudo-Labeling Statistics:")
print(f"  Total test samples: {len(submission)}")
print(f"  High-confidence samples: {len(pseudo_df)} ({len(pseudo_df)/len(submission)*100:.1f}%)")
print(f"\n  Class distribution:")
for label in class_names:
    count = (pseudo_df['label'] == label).sum()
    print(f"    {label:12s}: {count:4d} ({count/len(pseudo_df)*100:.1f}%)")

# Load original training data
train_df = pd.read_csv('train_data_with_kfold.csv')
print(f"\n  Original train size: {len(train_df)}")

# Combine with pseudo-labels
# Keep fold assignments for original data, -1 for pseudo
combined_df = pd.concat([train_df, pseudo_df], ignore_index=True)
combined_df['fold'] = combined_df['fold'].fillna(-1).astype(int)

# Save enhanced training data
output_path = 'train_data_with_pseudo.csv'
combined_df.to_csv(output_path, index=False)

print(f"\n✅ Enhanced training data saved: {output_path}")
print(f"  Total size: {len(combined_df)} (original {len(train_df)} + pseudo {len(pseudo_df)})")
print(f"  Expansion: +{len(pseudo_df)/len(train_df)*100:.1f}%")

# Print training strategy
print(f"\n🎯 Training Strategy:")
print(f"  1. Use pseudo-labeled data with weight=0.5 (to avoid overfitting to test)")
print(f"  2. Train for fewer epochs (30) with strong regularization")
print(f"  3. Focus on learning test distribution while maintaining train performance")
