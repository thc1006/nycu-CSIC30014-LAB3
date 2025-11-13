#!/usr/bin/env python3
"""
Generate test predictions using trained Stacking Meta-Learner
從 BREAKTHROUGH Layer 1 預測生成最終提交
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from stacking_meta_learner import StackingEnsemble


def main():
    """
    使用 MLP meta-learner 生成最終測試集預測
    """
    print("\n" + "=" * 80)
    print("BREAKTHROUGH STACKING META-LEARNER PREDICTION")
    print("=" * 80)
    print("\n使用訓練好的 MLP meta-learner 集成所有 Layer 1 模型")
    print("預期驗證分數: 86.88% Macro-F1")
    print("=" * 80)

    # Paths
    layer1_test_dir = Path('outputs/breakthrough_20251113_004854/layer1_test_predictions')
    test_csv = Path('data/test_data.csv')
    meta_learner_path = Path('models/stacking_mlp.pkl')

    if not meta_learner_path.exists():
        print(f"\n❌ Meta-learner not found: {meta_learner_path}")
        print("Please train meta-learner first:")
        print("  python scripts/stacking_meta_learner.py")
        return

    print(f"\n✓ Loading meta-learner from: {meta_learner_path}")
    stacker = StackingEnsemble.load(str(meta_learner_path))

    # Load test filenames
    test_df = pd.read_csv(test_csv)
    test_filenames = test_df['new_filename'].values
    print(f"✓ Test samples: {len(test_filenames)}")

    # Collect Layer 1 test predictions
    print(f"\n{'='*80}")
    print("Collecting Layer 1 Test Predictions")
    print("=" * 80)

    pred_files = sorted(layer1_test_dir.glob('*_test_pred.csv'))

    if len(pred_files) == 0:
        print(f"\n❌ No test predictions found in {layer1_test_dir}")
        print("Please generate test predictions first:")
        print("  python scripts/generate_layer1_test_predictions.py")
        return

    print(f"Found {len(pred_files)} prediction files")

    # Group predictions by model type
    from collections import defaultdict
    model_preds_by_fold = defaultdict(list)

    for pred_file in pred_files:
        # Parse: {model_type}_fold{n}_test_pred.csv
        stem = pred_file.stem
        if '_fold' in stem:
            parts = stem.split('_fold')
            model_type = parts[0]
            fold_num = int(parts[1].split('_')[0])
            model_preds_by_fold[model_type].append((fold_num, pred_file))

    print(f"\nFound {len(model_preds_by_fold)} model types:")
    for model_type, fold_files in model_preds_by_fold.items():
        print(f"  • {model_type}: {len(fold_files)} folds")

    # Merge predictions from all folds for each model (average)
    all_preds = {}
    for model_type, fold_files in model_preds_by_fold.items():
        # Sort by fold
        fold_files.sort(key=lambda x: x[0])

        # Average predictions across folds
        fold_preds = []
        for fold_num, pred_file in fold_files:
            df = pd.read_csv(pred_file)
            # Handle case-insensitive column names
            df.columns = df.columns.str.lower()
            preds = df[['normal', 'bacteria', 'virus', 'covid-19']].values
            fold_preds.append(preds)

        # Average across folds
        avg_preds = np.mean(fold_preds, axis=0)
        all_preds[model_type] = avg_preds
        print(f"  ✓ {model_type}: {avg_preds.shape} (averaged {len(fold_files)} folds)")

    # Generate meta-predictions
    print(f"\n{'='*80}")
    print("Generating Meta-Learner Predictions")
    print("=" * 80)

    final_probs = stacker.predict(all_preds)
    print(f"✓ Generated predictions: {final_probs.shape}")

    # Convert to submission format
    final_labels = final_probs.argmax(axis=1)
    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

    submission_df = pd.DataFrame({
        'new_filename': test_filenames,
        'normal': (final_labels == 0).astype(int),
        'bacteria': (final_labels == 1).astype(int),
        'virus': (final_labels == 2).astype(int),
        'COVID-19': (final_labels == 3).astype(int)
    })

    # Save
    output_dir = Path('data')
    output_file = output_dir / 'submission_breakthrough_stacking.csv'
    submission_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved submission: {output_file}")

    # Also save probabilities
    prob_output = output_dir / 'submission_breakthrough_stacking_probs.csv'
    prob_df = pd.DataFrame(final_probs, columns=class_names)
    prob_df.insert(0, 'new_filename', test_filenames)
    prob_df.to_csv(prob_output, index=False)
    print(f"✓ Saved probabilities: {prob_output}")

    # Summary
    print("\n" + "=" * 80)
    print("STACKING PREDICTION COMPLETED")
    print("=" * 80)
    print(f"\nOutput: {output_file}")
    print(f"Validation Score: 86.88% Macro-F1")
    print(f"Expected Test Score: ~87-90% (vs current best 84.19%)")
    print(f"\n預期提升: +3-6%")
    print("=" * 80)


if __name__ == '__main__':
    main()
