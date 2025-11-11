import pandas as pd
import numpy as np

# Load probability submission
df = pd.read_csv('data/submission_efficientnet_tta.csv')

# Class names
class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

# Get probabilities and convert to labels
probs = df[class_names].values
predicted_labels = [class_names[i] for i in probs.argmax(axis=1)]

# Create label submission
submission = pd.DataFrame({
    'new_filename': df['new_filename'],
    'label': predicted_labels
})

submission.to_csv('data/submission_efficientnet_tta_label.csv', index=False)
print("✅ Converted EfficientNet TTA to label format")
print(f"   File: data/submission_efficientnet_tta_label.csv")
print(f"   Samples: {len(submission)}")

# Print distribution
unique, counts = np.unique(predicted_labels, return_counts=True)
print("\nPrediction distribution:")
for label, count in zip(unique, counts):
    print(f"  {label:12s}: {count:4d} ({count/len(predicted_labels)*100:.2f}%)")
