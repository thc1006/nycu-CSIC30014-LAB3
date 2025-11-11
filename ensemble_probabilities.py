"""
Ensemble probability-based predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def ensemble_prob_submissions(submission_files, output_path, method='average'):
    """
    Ensemble multiple probability-based submission files

    Args:
        submission_files: List of submission CSV paths
        output_path: Output path
        method: 'average' or 'geometric_mean'
    """
    print(f"🔥 Ensembling {len(submission_files)} probability-based submissions...")
    print(f"Method: {method}")
    print()

    # Load all submissions
    submissions = []
    for f in submission_files:
        df = pd.read_csv(f)
        print(f"  Loaded {Path(f).name}: {len(df)} samples")
        submissions.append(df)

    # Get file column and class columns
    file_col = 'filename' if 'filename' in submissions[0].columns else 'new_filename'
    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

    # Verify all have same files
    base_files = set(submissions[0][file_col])
    for i, sub in enumerate(submissions[1:], 1):
        if set(sub[file_col]) != base_files:
            print(f"⚠️ Warning: submission {i} has different files!")

    # Extract probability matrices
    prob_matrices = []
    for sub in submissions:
        probs = sub[class_cols].values
        prob_matrices.append(probs)

    # Ensemble probabilities
    if method == 'average':
        # Simple average
        ensemble_probs = np.mean(prob_matrices, axis=0)
    elif method == 'geometric_mean':
        # Geometric mean (better for probabilities)
        ensemble_probs = np.exp(np.mean(np.log(np.array(prob_matrices) + 1e-10), axis=0))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize to sum to 1
    ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

    # Create ensemble submission
    ensemble_df = pd.DataFrame(ensemble_probs, columns=class_cols)
    ensemble_df.insert(0, file_col, submissions[0][file_col].values)

    ensemble_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved ensemble to {output_path}")

    # Convert to class predictions for statistics
    predicted_classes = np.argmax(ensemble_probs, axis=1)
    predicted_labels = [class_cols[c] for c in predicted_classes]

    # Print statistics
    unique, counts = np.unique(predicted_labels, return_counts=True)
    print("\n📈 Ensemble prediction distribution:")
    for label, count in zip(unique, counts):
        pct = count / len(predicted_labels) * 100
        print(f"  {label:12s}: {count:4d} ({pct:5.2f}%)")

    # Print average probabilities
    print("\n📊 Average probabilities by class:")
    mean_probs = ensemble_probs.mean(axis=0)
    for i, cls in enumerate(class_cols):
        print(f"  {cls:12s}: {mean_probs[i]:.4f}")

    # Compare with individual submissions
    print("\n📊 Individual submission class distributions:")
    for idx, sub in enumerate(submissions):
        probs = sub[class_cols].values
        pred_classes = np.argmax(probs, axis=1)
        pred_labels = [class_cols[c] for c in pred_classes]
        unique, counts = np.unique(pred_labels, return_counts=True)
        print(f"\n  Submission {idx+1} ({Path(submission_files[idx]).name}):")
        for label, count in zip(unique, counts):
            pct = count / len(pred_labels) * 100
            print(f"    {label:12s}: {count:4d} ({pct:5.2f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submissions', nargs='+', required=True, help='List of submission files')
    parser.add_argument('--output', required=True, help='Output path')
    parser.add_argument('--method', default='average', choices=['average', 'geometric_mean'])

    args = parser.parse_args()

    ensemble_prob_submissions(args.submissions, args.output, args.method)
