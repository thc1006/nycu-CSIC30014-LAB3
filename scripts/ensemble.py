#!/usr/bin/env python
"""
Ensemble multiple submission files by weighted averaging probabilities
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def ensemble_submissions(submissions, weights, output_path):
    """
    Ensemble multiple submissions by weighted averaging

    Args:
        submissions: list of (csv_path, weight) tuples
        output_path: output CSV path
    """
    print(f"Ensemble {len(submissions)} submissions:")

    # Load all submissions
    dfs = []
    for csv_path, weight in submissions:
        df = pd.read_csv(csv_path)
        print(f"  - {csv_path} (weight={weight})")
        dfs.append((df, weight))

    # Get first filename column
    result = dfs[0][0].copy()
    filename_col = result.columns[0]  # Assuming first column is filename
    class_cols = result.columns[1:]

    # Weighted average
    print(f"\nAveraging {len(class_cols)} classes...")
    avg_probs = np.zeros_like(result[class_cols].values, dtype=np.float32)

    for df, weight in dfs:
        avg_probs += df[class_cols].values * weight

    # Normalize
    avg_probs = avg_probs / sum(w for _, w in submissions)

    # Create result
    result[class_cols] = avg_probs
    result.to_csv(output_path, index=False)

    print(f"\n✅ Ensembled submission saved: {output_path}")
    print(f"   Rows: {len(result)}")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble multiple submissions")
    parser.add_argument("submissions", nargs="+", help="submission.csv and weight pairs")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")

    args = parser.parse_args()

    # Parse submissions and weights
    if len(args.submissions) % 2 != 0:
        print("Error: Must provide pairs of (csv_path, weight)")
        exit(1)

    submissions = []
    for i in range(0, len(args.submissions), 2):
        csv_path = args.submissions[i]
        weight = float(args.submissions[i+1])
        submissions.append((csv_path, weight))

    # Verify weights sum to something reasonable
    total_weight = sum(w for _, w in submissions)
    print(f"Total weight: {total_weight}")

    ensemble_submissions(submissions, weights=None, output_path=args.output)
