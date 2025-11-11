import pandas as pd
import numpy as np

# Load submissions
convnext = pd.read_csv('data/submission_convnext_tta.csv')
efficientnet = pd.read_csv('data/submission_efficientnet_tta.csv')

print(f"ConvNeXt: {len(convnext)} samples")
print(f"EfficientNet: {len(efficientnet)} samples")

# Convert EfficientNet probs to labels
class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
prob_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

# Get predicted labels from probabilities
probs = efficientnet[prob_cols].values
efficientnet_labels = [class_names[i] for i in probs.argmax(axis=1)]

# Ensemble by voting
ensemble_labels = []
for i in range(len(convnext)):
    votes = [convnext.iloc[i]['label'], efficientnet_labels[i]]
    # Count votes
    from collections import Counter
    vote_counts = Counter(votes)
    majority = vote_counts.most_common(1)[0][0]
    ensemble_labels.append(majority)

# Create submission
submission = pd.DataFrame({
    'new_filename': convnext['new_filename'],
    'label': ensemble_labels
})

submission.to_csv('data/submission_2model_ensemble.csv', index=False)
print(f"\n✅ Saved ensemble: data/submission_2model_ensemble.csv")

# Print distribution
unique, counts = np.unique(ensemble_labels, return_counts=True)
print("\nEnsemble distribution:")
for label, count in zip(unique, counts):
    print(f"  {label:12s}: {count:4d} ({count/len(ensemble_labels)*100:.2f}%)")
