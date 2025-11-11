"""
Simple but effective ensemble of multiple predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def ensemble_submissions(submission_files, output_path, method='voting'):
    """
    Ensemble multiple submission files

    Args:
        submission_files: List of submission CSV paths
        output_path: Output path
        method: 'voting' or 'average' (for prob-based)
    """
    print(f"🔥 Ensembling {len(submission_files)} submissions...")
    print(f"Method: {method}")

    # Load all submissions
    submissions = []
    for f in submission_files:
        df = pd.read_csv(f)
        print(f"  Loaded {Path(f).name}: {len(df)} samples")
        submissions.append(df)

    # Get file column name
    file_col = 'filename' if 'filename' in submissions[0].columns else 'new_filename'

    # Verify all have same files
    base_files = set(submissions[0][file_col])
    for i, sub in enumerate(submissions[1:], 1):
        if set(sub[file_col]) != base_files:
            print(f"⚠️ Warning: submission {i} has different files!")

    # Ensemble by majority voting
    ensemble_labels = []

    for idx in range(len(submissions[0])):
        # Get predictions from all models
        votes = [sub.iloc[idx]['label'] for sub in submissions]

        # Majority vote
        from collections import Counter
        vote_counts = Counter(votes)
        majority_label = vote_counts.most_common(1)[0][0]

        ensemble_labels.append(majority_label)

    # Create ensemble submission
    ensemble_df = pd.DataFrame({
        file_col: submissions[0][file_col],
        'label': ensemble_labels
    })

    ensemble_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved ensemble to {output_path}")

    # Print statistics
    unique, counts = np.unique(ensemble_labels, return_counts=True)
    print("\n📈 Ensemble prediction distribution:")
    for label, count in zip(unique, counts):
        pct = count / len(ensemble_labels) * 100
        print(f"  {label:12s}: {count:4d} ({pct:5.2f}%)")

    # Compare with individual submissions
    print("\n📊 Individual submission distributions:")
    for i, sub in enumerate(submissions):
        unique, counts = np.unique(sub['label'], return_counts=True)
        print(f"\n  Submission {i+1} ({Path(submission_files[i]).name}):")
        for label, count in zip(unique, counts):
            pct = count / len(sub) * 100
            print(f"    {label:12s}: {count:4d} ({pct:5.2f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submissions', nargs='+', required=True, help='List of submission files')
    parser.add_argument('--output', required=True, help='Output path')
    parser.add_argument('--method', default='voting', choices=['voting', 'average'])

    args = parser.parse_args()

    ensemble_submissions(args.submissions, args.output, args.method)
