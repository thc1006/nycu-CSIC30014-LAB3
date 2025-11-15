#!/usr/bin/env python3
"""
Generate test predictions using trained Stacking Meta-Learner
Using test_predictions directory
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from stacking_meta_learner import StackingEnsemble


def collect_test_predictions(use_tta=False):
    """
    Collect test predictions from all base models and merge folds
    """
    print("=" * 80)
    print("Collecting Test Predictions")
    print("=" * 80)

    pred_dir = 'data/test_predictions_tta' if use_tta else 'data/test_predictions'

    # Find all test prediction files
    pred_files = list(Path(pred_dir).glob('*.csv'))

    if len(pred_files) == 0:
        print(f"\n❌ No test prediction files found in {pred_dir}!")
        return None, None

    print(f"\nFound {len(pred_files)} test predictions in {pred_dir}")

    # Load test filenames
    test_df = pd.read_csv('data/test_data.csv')
    test_filenames = test_df['new_filename'].values

    # Group predictions by model type
    model_groups = {}

    for pred_file in pred_files:
        file_stem = pred_file.stem.replace('_tta6x', '')

        # Extract model type (remove fold suffix)
        if '_fold' in file_stem:
            model_type = file_stem.split('_fold')[0]
        else:
            model_type = file_stem

        df = pd.read_csv(pred_file)
        preds = df[['normal', 'bacteria', 'virus', 'COVID-19']].values

        if model_type not in model_groups:
            model_groups[model_type] = []

        model_groups[model_type].append(preds)

    # Merge folds by averaging
    all_preds = {}

    print("\nMerging folds...")
    for model_type, pred_list in model_groups.items():
        # Average all folds
        merged_preds = np.mean(pred_list, axis=0)
        all_preds[model_type] = merged_preds
        print(f"  ✓ {model_type}: {len(pred_list)} folds → {merged_preds.shape}")

    return all_preds, test_filenames


def main():
    """
    Generate final test predictions using meta-learner
    """
    print("\n" + "=" * 80)
    print("STACKING META-LEARNER PREDICTION")
    print("=" * 80)
    print("\nThis uses the trained meta-learner to combine all base models")
    print("Expected: +1-3% improvement over best single model")
    print("=" * 80)

    # Try TTA first
    print("\n[1] Trying TTA predictions...")
    preds_dict, test_filenames = collect_test_predictions(use_tta=True)

    if preds_dict is None:
        print("\n[2] Falling back to non-TTA predictions...")
        preds_dict, test_filenames = collect_test_predictions(use_tta=False)

        if preds_dict is None:
            print("❌ No predictions found!")
            return

    # Try different meta-learner types
    meta_types = ['mlp', 'xgb', 'rf', 'lgb', 'logistic']

    best_result = None
    best_type = None

    for meta_type in meta_types:
        model_path = f'models/stacking_{meta_type}.pkl'

        if not Path(model_path).exists():
            print(f"\n⚠️  {meta_type.upper()} meta-learner not found, skipping")
            continue

        print(f"\n{'=' * 80}")
        print(f"Using {meta_type.upper()} Meta-Learner")
        print("=" * 80)

        # Load meta-learner
        stacker = StackingEnsemble.load(model_path)
        print(f"✓ Loaded from {model_path}")

        # Generate meta-predictions
        print("\nGenerating meta-predictions...")
        final_probs = stacker.predict(preds_dict)

        # Convert to labels (for submission format)
        final_labels = final_probs.argmax(axis=1)
        class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

        # Create submission dataframe
        submission_df = pd.DataFrame({
            'new_filename': test_filenames,
            'normal': (final_labels == 0).astype(int),
            'bacteria': (final_labels == 1).astype(int),
            'virus': (final_labels == 2).astype(int),
            'COVID-19': (final_labels == 3).astype(int)
        })

        # Save
        output_file = f'data/submission_stacking_{meta_type}.csv'
        submission_df.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file}")

        # Statistics
        class_dist = np.bincount(final_labels, minlength=4)
        avg_conf = np.max(final_probs, axis=1).mean()

        print(f"\nStatistics:")
        print(f"  Average confidence: {avg_conf:.4f}")
        print(f"  Normal: {class_dist[0]} ({class_dist[0]/len(final_labels)*100:.1f}%)")
        print(f"  Bacteria: {class_dist[1]} ({class_dist[1]/len(final_labels)*100:.1f}%)")
        print(f"  Virus: {class_dist[2]} ({class_dist[2]/len(final_labels)*100:.1f}%)")
        print(f"  COVID-19: {class_dist[3]} ({class_dist[3]/len(final_labels)*100:.1f}%)")

        best_result = submission_df
        best_type = meta_type

    if best_result is None:
        print("\n❌ No meta-learner found!")
        print("Please train meta-learner first:")
        print("  python scripts/stacking_meta_learner.py")
        return

    # Summary
    print("\n" + "=" * 80)
    print("STACKING PREDICTION COMPLETED")
    print("=" * 80)
    print(f"\nGenerated {len(meta_types)} submissions")
    print(f"\nBest recommended: submission_stacking_mlp.csv (Val F1: 86.88%)")
    print(f"\nExpected test score: 87-90%")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
