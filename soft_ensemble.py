"""
Soft Ensemble: Probability averaging for maximum performance
This is MUCH faster than pseudo-label retraining
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("SOFT ENSEMBLE - PROBABILITY AVERAGING FOR 91% BREAKTHROUGH")
print("=" * 70)

# Load probability predictions
eff_df = pd.read_csv('data/submission_efficientnet_tta.csv')
conv_df = pd.read_csv('data/submission_convnext_tta_prob.csv')

print(f"\n[1/5] Loaded predictions:")
print(f"   EfficientNet TTA: {len(eff_df)} samples")
print(f"   ConvNeXt TTA:     {len(conv_df)} samples")

# Verify same order
assert (eff_df['new_filename'] == conv_df['new_filename']).all(), "File order mismatch!"

# Class names
class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

# Get probabilities
eff_probs = eff_df[class_names].values
conv_probs = conv_df[class_names].values

print(f"\n[2/5] Model performance weights:")
# EfficientNet has higher val F1 (89.76% vs 88.91%), weight it more
eff_weight = 0.55  # Slightly favor better model
conv_weight = 0.45

print(f"   EfficientNet: {eff_weight} (Val F1 = 89.76%)")
print(f"   ConvNeXt:     {conv_weight} (Val F1 = 88.91%)")

# Weighted average
ensemble_probs = eff_weight * eff_probs + conv_weight * conv_probs

# Normalize to ensure sum = 1
ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

# Get predictions
predicted_idx = ensemble_probs.argmax(axis=1)
predicted_labels = [class_names[i] for i in predicted_idx]
max_confidence = ensemble_probs.max(axis=1)

print(f"\n[3/5] Ensemble statistics:")
print(f"   Mean confidence: {max_confidence.mean():.4f}")
print(f"   Median confidence: {np.median(max_confidence):.4f}")
print(f"   Samples > 0.95: {(max_confidence > 0.95).sum()} ({(max_confidence > 0.95).sum()/len(ensemble_probs)*100:.1f}%)")

# Class distribution
unique, counts = np.unique(predicted_labels, return_counts=True)
print(f"\n[4/5] Prediction distribution:")
for label, count in zip(unique, counts):
    print(f"   {label:12s}: {count:4d} ({count/len(predicted_labels)*100:.1f}%)")

# Create one-hot submission
onehot = np.zeros_like(ensemble_probs)
onehot[np.arange(len(ensemble_probs)), predicted_idx] = 1.0

submission = pd.DataFrame({
    'new_filename': eff_df['new_filename'],
    'normal': onehot[:, 0],
    'bacteria': onehot[:, 1],
    'virus': onehot[:, 2],
    'COVID-19': onehot[:, 3]
})

submission.to_csv('data/submission_soft_ensemble.csv', index=False)
print(f"\n[5/5] Saved soft ensemble submission:")
print(f"   File: data/submission_soft_ensemble.csv")

print(f"\n" + "=" * 70)
print("SOFT ENSEMBLE COMPLETE - READY TO SUBMIT!")
print("=" * 70)
print(f"\nExpected improvement:")
print(f"  • Single best model:  83.82% (EfficientNet TTA)")
print(f"  • Soft ensemble:      85-87%+ (probability averaging reduces variance)")
print(f"  • Closer to 91% goal!")
