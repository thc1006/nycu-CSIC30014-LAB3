"""
Create ensemble predictions from multiple submission files
"""
import pandas as pd
import numpy as np
import argparse

def soft_voting_ensemble(submissions, weights=None):
    """
    Ensemble using soft voting (averaging probabilities)

    Args:
        submissions: List of (name, DataFrame) tuples
        weights: List of weights for each submission (optional)
    """
    if weights is None:
        weights = [1.0] * len(submissions)

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    print(f"\nEnsemble Configuration:")
    for (name, _), w in zip(submissions, weights):
        print(f"  {name}: weight = {w:.3f}")

    # Get the first submission as template
    result = submissions[0][1].copy()

    # Average the probabilities
    prob_cols = ["normal", "bacteria", "virus", "COVID-19"]
    prob_matrix = np.zeros((len(result), 4))

    for (name, df), w in zip(submissions, weights):
        probs = df[prob_cols].values
        prob_matrix += w * probs

    # Convert to one-hot encoding (argmax)
    predictions = prob_matrix.argmax(axis=1)
    one_hot = np.zeros_like(prob_matrix)
    one_hot[np.arange(len(one_hot)), predictions] = 1.0

    # Update result with one-hot encoded predictions
    result[prob_cols] = one_hot

    return result

def main():
    parser = argparse.ArgumentParser(description="Create ensemble from multiple submissions")
    parser.add_argument("--original", type=str, default="data/submission.csv",
                       help="Original submission (80%%) - weight 1.0")
    parser.add_argument("--exp1", type=str, default="submission_exp1.csv",
                       help="Exp1 submission (76.15%%) - weight 0.8")
    parser.add_argument("--exp2", type=str, default="submission_exp2.csv",
                       help="Exp2 submission (71.95%%) - weight 0.6")
    parser.add_argument("--mode", type=str, default="2way", choices=["2way", "3way"],
                       help="Ensemble mode: 2way (original+exp1) or 3way (all)")
    parser.add_argument("--output", type=str, default="submission_ensemble.csv",
                       help="Output filename")

    args = parser.parse_args()

    print("="*80)
    print("ENSEMBLE CREATION")
    print("="*80)

    # Load submissions
    original = pd.read_csv(args.original)
    exp1 = pd.read_csv(args.exp1)
    exp2 = pd.read_csv(args.exp2)

    print(f"\nLoaded submissions:")
    print(f"  Original: {args.original} (80.00%)")
    print(f"  Exp1:     {args.exp1} (76.15%)")
    print(f"  Exp2:     {args.exp2} (71.95%)")

    if args.mode == "2way":
        # 2-way ensemble: Original + Exp1
        submissions = [
            ("Original (80%)", original),
            ("Exp1 (76.15%)", exp1)
        ]
        # Weight by performance
        weights = [1.0, 0.95]  # Original slightly higher weight

    else:  # 3way
        # 3-way ensemble: All three
        submissions = [
            ("Original (80%)", original),
            ("Exp1 (76.15%)", exp1),
            ("Exp2 (71.95%)", exp2)
        ]
        # Weight by performance
        weights = [1.0, 0.95, 0.90]  # Original highest, Exp2 lowest

    # Create ensemble
    result = soft_voting_ensemble(submissions, weights)

    # Save result
    result.to_csv(args.output, index=False)
    print(f"\n[OK] Ensemble saved to: {args.output}")

    # Show class distribution
    prob_cols = ["normal", "bacteria", "virus", "COVID-19"]
    predictions = result[prob_cols].values.argmax(axis=1)
    class_names = prob_cols

    print(f"\nEnsemble class distribution:")
    for i, name in enumerate(class_names):
        count = (predictions == i).sum()
        print(f"  {name:12s}: {count:4d} ({count/len(result)*100:.1f}%)")

    print("\n" + "="*80)
    print(f"RECOMMENDATION:")
    if args.mode == "2way":
        print("2-way ensemble (Original + Exp1) is safer.")
        print("Expected score: 80-81%")
    else:
        print("3-way ensemble includes Exp2 which performed poorly.")
        print("Expected score: 79-81% (uncertain)")
    print("="*80)

if __name__ == "__main__":
    main()
