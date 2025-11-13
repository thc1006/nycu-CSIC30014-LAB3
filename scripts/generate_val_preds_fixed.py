#!/usr/bin/env python3
"""
Generate validation predictions for all trained models
Fixed version that works with current codebase
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data import make_loader
from model import create_model
from sklearn.metrics import f1_score


def load_model_and_predict(checkpoint_path, project_root):
    """
    Load model and generate predictions on validation set
    """
    print(f"\nProcessing: {checkpoint_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    config = checkpoint.get('config', {})
    model_name = config.get('model', 'efficientnet_v2_s')
    img_size = config.get('img_size', 384)
    dropout = config.get('dropout', 0.25)

    print(f"  Model: {model_name}")
    print(f"  Image size: {img_size}")
    print(f"  Dropout: {dropout}")

    # Create validation loader
    data_dir = project_root / 'data'
    train_csv = data_dir / 'train.csv'
    train_dir = data_dir / 'train'

    # Read CSV and filter validation split
    df = pd.read_csv(train_csv)
    if 'split' not in df.columns:
        # If no split column, use last 20% as validation
        val_start = int(len(df) * 0.8)
        df['split'] = 'train'
        df.iloc[val_start:, df.columns.get_loc('split')] = 'val'

    val_df = df[df['split'] == 'val'].copy()

    # Save temp validation CSV
    temp_val_csv = data_dir / 'temp_val.csv'
    val_df.to_csv(temp_val_csv, index=False)

    print(f"  Validation samples: {len(val_df)}")

    # Create loader
    label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
    _, val_loader = make_loader(
        csv_path=str(temp_val_csv),
        images_dir=str(train_dir),
        file_col='new_filename',
        label_cols=label_cols,
        img_size=img_size,
        batch_size=32,
        num_workers=4,
        augment=False,  # No augmentation for validation
        shuffle=False,
        weighted=False
    )

    # Create model
    model = create_model(
        model_name=model_name,
        num_classes=4,
        pretrained=False,
        dropout=dropout
    )

    # Load weights
    state_dict = checkpoint['model_state_dict']

    # Handle torch.compile model (remove '_orig_mod.' prefix)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Generate predictions
    all_preds = []
    all_labels = []
    all_filenames = []

    print("  Generating predictions...")
    with torch.no_grad():
        for images, labels, filenames in tqdm(val_loader, desc='Validating'):
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_filenames.extend(filenames)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate F1 score
    pred_classes = all_preds.argmax(axis=1)

    # Filter out -1 labels (if any)
    valid_mask = all_labels != -1
    if valid_mask.sum() > 0:
        f1 = f1_score(all_labels[valid_mask], pred_classes[valid_mask], average='macro')
        print(f"  Validation F1: {f1:.4f}")
    else:
        f1 = 0.0
        print(f"  ⚠️ No valid labels found")

    # Clean up temp file
    temp_val_csv.unlink()

    return all_preds, all_labels, all_filenames, f1


def main():
    """
    Generate validation predictions for all models
    """
    print("=" * 80)
    print("GENERATE VALIDATION PREDICTIONS FOR STACKING")
    print("=" * 80)

    project_root = Path(__file__).parent.parent
    outputs_dir = project_root / 'outputs'
    data_dir = project_root / 'data'

    # Find all model checkpoints
    checkpoint_files = sorted(outputs_dir.glob('*/best.pt'))

    if len(checkpoint_files) == 0:
        print("\n❌ No model checkpoints found!")
        print("Please train models first.")
        return

    print(f"\nFound {len(checkpoint_files)} model checkpoints")
    print("\nModels:")
    for i, ckpt in enumerate(checkpoint_files, 1):
        print(f"  {i}. {ckpt.parent.name}")

    # Generate predictions for each model
    all_results = {}
    results_summary = []

    for ckpt_path in checkpoint_files:
        model_name = ckpt_path.parent.name

        try:
            preds, labels, filenames, f1 = load_model_and_predict(
                checkpoint_path=ckpt_path,
                project_root=project_root
            )

            # Save predictions
            output_file = data_dir / f'validation_predictions_{model_name}.csv'

            df = pd.DataFrame(
                preds,
                columns=['normal', 'bacteria', 'virus', 'COVID-19']
            )
            df['label'] = labels
            df['filename'] = filenames

            df.to_csv(output_file, index=False)
            print(f"  ✓ Saved: {output_file}")

            all_results[model_name] = preds
            results_summary.append({
                'model': model_name,
                'val_f1': f1,
                'num_samples': len(preds)
            })

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✓ Generated predictions for {len(all_results)} models\n")

    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df = summary_df.sort_values('val_f1', ascending=False)
        print(summary_df.to_string(index=False))
        print(f"\nBest model: {summary_df.iloc[0]['model']} (F1: {summary_df.iloc[0]['val_f1']:.4f})")

    print("\nFiles saved in: data/validation_predictions_*.csv")
    print("\nNext step:")
    print("  python3 scripts/stacking_meta_learner.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
