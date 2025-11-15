#!/usr/bin/env python3
"""
Ensemble predictions from all 11 champion models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

def load_predictions(pred_dir='outputs/champion_predictions'):
    """Load all prediction files"""
    pred_files = sorted(glob.glob(f'{pred_dir}/*.csv'))

    print(f"Found {len(pred_files)} prediction files:")
    for f in pred_files:
        print(f"  - {Path(f).name}")

    predictions = []
    filenames = None

    for pred_file in pred_files:
        df = pd.read_csv(pred_file)

        if filenames is None:
            filenames = df['new_filename'].values
        else:
            assert np.all(filenames == df['new_filename'].values), "Filenames mismatch!"

        class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
        probs = df[class_cols].values
        predictions.append(probs)

    predictions = np.stack(predictions, axis=0)  # Shape: (n_models, n_samples, n_classes)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"  Models: {predictions.shape[0]}")
    print(f"  Samples: {predictions.shape[1]}")
    print(f"  Classes: {predictions.shape[2]}")

    return filenames, predictions

def probs_to_onehot(probs):
    """Convert probability predictions to one-hot labels"""
    pred_classes = np.argmax(probs, axis=1)
    onehot = np.zeros((len(pred_classes), 4), dtype=int)
    onehot[np.arange(len(pred_classes)), pred_classes] = 1
    return onehot

def save_submission(filenames, probs, output_path):
    """Save submission file"""
    onehot = probs_to_onehot(probs)

    output_df = pd.DataFrame({'new_filename': filenames})
    for i, cn in enumerate(['normal', 'bacteria', 'virus', 'COVID-19']):
        output_df[cn] = onehot[:, i]

    output_df.to_csv(output_path, index=False)

    stats = np.bincount(np.argmax(probs, axis=1), minlength=4)
    conf = np.max(probs, axis=1).mean()

    print(f"\nSaved: {output_path}")
    print(f"  Confidence: {conf:.4f}")
    print(f"  Distribution: N={stats[0]} B={stats[1]} V={stats[2]} C={stats[3]}")

    return output_path

def main():
    print("="*80)
    print("Champion Models Ensemble")
    print("="*80)

    # Load all predictions
    pred_dir = 'outputs/champion_predictions'
    pred_files = sorted(glob.glob(f'{pred_dir}/*.csv'))
    filenames, predictions = load_predictions(pred_dir)

    # ========================================================================
    # Strategy 1: Simple Average
    # ========================================================================
    print("\n" + "="*80)
    print("Strategy 1: Simple Average (Equal Weight)")
    print("="*80)

    avg_probs = predictions.mean(axis=0)
    save_submission(filenames, avg_probs, 'data/submission_champion_simple_avg.csv')

    # ========================================================================
    # Strategy 2: Weighted Average (by model capacity)
    # ========================================================================
    print("\n" + "="*80)
    print("Strategy 2: Weighted Average (by Model Capacity)")
    print("="*80)

    # Auto-detect models and assign weights based on naming
    n_models = predictions.shape[0]
    model_names = [Path(f).stem for f in pred_files]

    # Assign weights based on model type
    model_weights = []
    for name in model_names:
        if 'convnext' in name:
            model_weights.append(1.0)  # ConvNeXt-Large (200M)
        elif 'vit_large' in name:
            model_weights.append(1.3)  # ViT-Large (307M)
        elif 'beit_large' in name:
            model_weights.append(1.3)  # BEiT-Large (307M)
        elif 'maxvit' in name:
            model_weights.append(1.1)  # MaxViT-Large (212M)
        elif 'coatnet' in name:
            model_weights.append(0.9)  # CoAtNet-3 (168M)
        else:
            model_weights.append(1.0)  # Default

    model_weights = np.array(model_weights)
    model_weights = model_weights / model_weights.sum()  # Normalize

    print(f"Detected {n_models} models:")
    for i, (name, w) in enumerate(zip(model_names, model_weights)):
        print(f"  Model {i+1} ({name}): {w:.4f}")

    weighted_probs = np.average(predictions, axis=0, weights=model_weights)
    save_submission(filenames, weighted_probs, 'data/submission_champion_weighted_avg.csv')

    # ========================================================================
    # Strategy 3: Architecture-Based Weighting
    # ========================================================================
    print("\n" + "="*80)
    print("Strategy 3: Architecture-Based Weighting")
    print("="*80)

    # Higher weight for Transformer-based models (ViT, BEiT)
    # Lower weight for CNN-based (CoAtNet)
    arch_weights = []
    for name in model_names:
        if 'convnext' in name:
            arch_weights.append(1.0)  # ConvNeXt - Hybrid
        elif 'vit_large' in name:
            arch_weights.append(1.4)  # ViT-Large - Pure Transformer
        elif 'beit_large' in name:
            arch_weights.append(1.4)  # BEiT-Large - Transformer with better pretraining
        elif 'maxvit' in name:
            arch_weights.append(1.2)  # MaxViT - Hybrid (Transformer + Conv)
        elif 'coatnet' in name:
            arch_weights.append(0.8)  # CoAtNet - More CNN-like
        else:
            arch_weights.append(1.0)  # Default

    arch_weights = np.array(arch_weights)
    arch_weights = arch_weights / arch_weights.sum()

    print("Architecture weights:")
    for i, (name, w) in enumerate(zip(model_names, arch_weights)):
        print(f"  Model {i+1} ({name}): {w:.4f}")

    arch_probs = np.average(predictions, axis=0, weights=arch_weights)
    save_submission(filenames, arch_probs, 'data/submission_champion_arch_weighted.csv')

    # ========================================================================
    # Strategy 4: Majority Voting (Hard Voting)
    # ========================================================================
    print("\n" + "="*80)
    print("Strategy 4: Majority Voting")
    print("="*80)

    # Convert each model's prediction to hard labels
    hard_predictions = np.argmax(predictions, axis=2)  # Shape: (n_models, n_samples)

    # Count votes for each class
    voted_classes = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=4).argmax(),
        axis=0,
        arr=hard_predictions
    )

    # Convert to probabilities (all weight to voted class)
    voted_probs = np.zeros((len(voted_classes), 4))
    voted_probs[np.arange(len(voted_classes)), voted_classes] = 1.0

    save_submission(filenames, voted_probs, 'data/submission_champion_majority_vote.csv')

    # ========================================================================
    # Strategy 5: Median Ensemble
    # ========================================================================
    print("\n" + "="*80)
    print("Strategy 5: Median Ensemble (Robust to Outliers)")
    print("="*80)

    median_probs = np.median(predictions, axis=0)
    save_submission(filenames, median_probs, 'data/submission_champion_median.csv')

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("Ensemble Complete!")
    print("="*80)

    print("\nGenerated 5 ensemble submissions:")
    print("  1. submission_champion_simple_avg.csv - Equal weight")
    print("  2. submission_champion_weighted_avg.csv - Weighted by model size")
    print("  3. submission_champion_arch_weighted.csv - Weighted by architecture")
    print("  4. submission_champion_majority_vote.csv - Hard voting")
    print("  5. submission_champion_median.csv - Median ensemble")

    print("\nRecommended submission order:")
    print("  Priority 1: submission_champion_arch_weighted.csv (Transformer-focused)")
    print("  Priority 2: submission_champion_weighted_avg.csv (Capacity-based)")
    print("  Priority 3: submission_champion_simple_avg.csv (Baseline)")

    print("\nExpected performance: 88-90% (target: 91%)")
    print("="*80)

if __name__ == '__main__':
    main()
