#!/usr/bin/env python3
"""
Generate validation predictions for all trained models
This is required for stacking/meta-learning
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset import ChestXRayDataset
from torch.utils.data import DataLoader


def load_model_and_predict(checkpoint_path, data_dir='data', batch_size=32):
    """
    Load model and generate predictions on validation set
    """
    print(f"\nProcessing: {checkpoint_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    config = checkpoint.get('config', {})
    img_size = config.get('img_size', 384)
    model_name = config.get('model', 'efficientnet_v2_s')

    print(f"  Model: {model_name}")
    print(f"  Image size: {img_size}")

    # Load validation data
    val_dataset = ChestXRayDataset(
        csv_path=Path(data_dir) / 'train.csv',
        img_dir=Path(data_dir) / 'train',
        split='val',
        img_size=img_size,
        augment=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    from model import create_model
    model = create_model(
        model_name=model_name,
        num_classes=4,
        pretrained=False,
        dropout=config.get('dropout', 0.25)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Generate predictions
    all_preds = []
    all_labels = []
    all_filenames = []

    print("  Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            images = batch['image'].to(device)
            labels = batch['label']

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

            if 'filename' in batch:
                all_filenames.extend(batch['filename'])

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate F1 score
    from sklearn.metrics import f1_score
    pred_classes = all_preds.argmax(axis=1)
    f1 = f1_score(all_labels, pred_classes, average='macro')

    print(f"  Validation F1: {f1:.4f}")

    return all_preds, all_labels, all_filenames


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
    checkpoint_files = list(outputs_dir.glob('*/best.pt'))

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

    for ckpt_path in checkpoint_files:
        model_name = ckpt_path.parent.name

        try:
            preds, labels, filenames = load_model_and_predict(
                checkpoint_path=ckpt_path,
                data_dir=data_dir
            )

            # Save predictions
            output_file = data_dir / f'validation_predictions_{model_name}.csv'

            df = pd.DataFrame(
                preds,
                columns=['normal', 'bacteria', 'virus', 'COVID-19']
            )
            df['label'] = labels

            if len(filenames) > 0:
                df['filename'] = filenames

            df.to_csv(output_file, index=False)
            print(f"  ✓ Saved: {output_file}")

            all_results[model_name] = preds

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✓ Generated predictions for {len(all_results)} models")
    print("\nFiles saved in: data/validation_predictions_*.csv")
    print("\nNext step:")
    print("  python scripts/stacking_meta_learner.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
