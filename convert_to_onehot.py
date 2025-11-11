import pandas as pd
import numpy as np

# Load probability submission
df = pd.read_csv('data/submission_efficientnet_tta.csv')

# Class names
class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

# Get probabilities and convert to one-hot
probs = df[class_names].values
predicted_idx = probs.argmax(axis=1)

# Create one-hot encoding
onehot = np.zeros_like(probs)
onehot[np.arange(len(probs)), predicted_idx] = 1.0

# Create submission with one-hot encoding
submission = pd.DataFrame({
    'new_filename': df['new_filename'],
    'normal': onehot[:, 0],
    'bacteria': onehot[:, 1],
    'virus': onehot[:, 2],
    'COVID-19': onehot[:, 3]
})

submission.to_csv('data/submission_efficientnet_tta_onehot.csv', index=False)
print("✅ Converted EfficientNet TTA to one-hot format")
print(f"   File: data/submission_efficientnet_tta_onehot.csv")
print(f"   Samples: {len(submission)}")

# Verify format matches successful submission
print("\nFirst 5 rows:")
print(submission.head())

# Print distribution
class_counts = onehot.sum(axis=0)
print("\nPrediction distribution:")
for i, name in enumerate(class_names):
    print(f"  {name:12s}: {int(class_counts[i]):4d} ({class_counts[i]/len(submission)*100:.2f}%)")
