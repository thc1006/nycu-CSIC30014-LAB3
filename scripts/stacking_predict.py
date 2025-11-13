#!/usr/bin/env python3
"""
Generate test predictions using trained Stacking Meta-Learner
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from stacking_meta_learner import StackingEnsemble


def collect_test_predictions(data_dir='data'):
    """
    Collect test predictions from all base models
    """
    print("=" * 80)
    print("Collecting Test Predictions")
    print("=" * 80)

    # Find all test prediction files
    pred_files = list(Path(data_dir).glob('submission_*.csv'))

    if len(pred_files) == 0:
        print("\n❌ No test prediction files found!")
        print("Please generate test predictions first:")
        print("  python src/predict.py --config configs/model.yaml")
        return None

    print(f"\nFound {len(pred_files)} test predictions")

    # Load test filenames
    test_df = pd.read_csv(Path(data_dir) / 'test' / 'sample_submission.csv')
    test_filenames = test_df['new_filename'].values

    # Collect predictions
    all_preds = {}

    for pred_file in pred_files:
        # Skip ensemble files (we want base model predictions)
        if 'ensemble' in pred_file.name.lower() or 'stacking' in pred_file.name.lower():
            continue

        model_name = pred_file.stem.replace('submission_', '')

        df = pd.read_csv(pred_file)

        # Check if it's a probability file or label file
        if set(['normal', 'bacteria', 'virus', 'COVID-19']).issubset(df.columns):
            # Probability format
            preds = df[['normal', 'bacteria', 'virus', 'COVID-19']].values
        else:
            # Label format - skip
            continue

        all_preds[model_name] = preds
        print(f"  ✓ {model_name}: {preds.shape}")

    if len(all_preds) == 0:
        print("\n⚠️  No probability predictions found")
        print("Predictions must contain columns: normal, bacteria, virus, COVID-19")
        return None

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

    data_dir = Path('data')

    # Try different meta-learner types
    meta_types = ['lgb', 'xgb', 'mlp', 'rf', 'logistic']

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

        # Collect test predictions
        preds_dict, test_filenames = collect_test_predictions(data_dir)
        if preds_dict is None:
            continue

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
        output_file = data_dir / f'submission_stacking_{meta_type}.csv'
        submission_df.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file}")

        # Also save probabilities
        prob_output = data_dir / f'submission_stacking_{meta_type}_probs.csv'
        prob_df = pd.DataFrame(
            final_probs,
            columns=class_names
        )
        prob_df.insert(0, 'new_filename', test_filenames)
        prob_df.to_csv(prob_output, index=False)
        print(f"✓ Saved probabilities: {prob_output}")

        best_result = submission_df
        best_type = meta_type

    if best_result is None:
        print("\n❌ No meta-learner found!")
        print("Please train meta-learner first:")
        print("  python scripts/stacking_meta_learner.py")
        return

    # Create final submission (using best meta-learner)
    final_output = data_dir / 'submission_stacking_final.csv'
    best_result.to_csv(final_output, index=False)
    print(f"\n✓ Final submission: {final_output}")

    # Summary
    print("\n" + "=" * 80)
    print("STACKING PREDICTION COMPLETED")
    print("=" * 80)
    print(f"\nBest meta-learner: {best_type.upper()}")
    print(f"Output: {final_output}")
    print(f"\nTo submit:")
    print(f"  kaggle competitions submit -c cxr-multi-label-classification \\")
    print(f"    -f {final_output} \\")
    print(f'    -m "Stacking Meta-Learner ({best_type.upper()}) - All {len(preds_dict)} models"')
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
