"""
Resplit train/val data using stratified split to ensure balanced classes.

This script:
1. Merges existing train_data.csv and val_data.csv
2. Uses stratified split to ensure each class is proportionally represented
3. Generates new train_data.csv and val_data.csv with better COVID-19 representation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    train_csv = data_dir / "train_data.csv"
    val_csv = data_dir / "val_data.csv"

    # Backup originals
    backup_dir = data_dir / "backup"
    backup_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("RESPLITTING DATA WITH STRATIFIED SPLIT")
    print("=" * 60)

    # Load existing data
    print("\n[1/6] Loading existing data...")
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    print(f"  Original train: {len(df_train)} samples")
    print(f"  Original val:   {len(df_val)} samples")

    # Merge all data
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    print(f"\n  Total samples: {len(df_all)}")

    # Create class labels for stratification
    # We'll use a single label that represents the one-hot encoding
    label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

    # Convert one-hot to single label
    df_all['class_label'] = df_all[label_cols].idxmax(axis=1)

    # Show class distribution
    print("\n[2/6] Current class distribution:")
    class_counts = df_all['class_label'].value_counts()
    for cls, count in class_counts.items():
        pct = 100 * count / len(df_all)
        print(f"  {cls:12s}: {count:4d} ({pct:.1f}%)")

    # Backup original files
    print("\n[3/6] Backing up original files...")
    import shutil
    shutil.copy(train_csv, backup_dir / "train_data.csv.backup")
    shutil.copy(val_csv, backup_dir / "val_data.csv.backup")
    print(f"  Backed up to: {backup_dir}/")

    # Stratified split (80-20)
    print("\n[4/6] Performing stratified split (80-20)...")

    # Use stratified split to ensure each class is proportionally represented
    train_idx, val_idx = train_test_split(
        np.arange(len(df_all)),
        test_size=0.2,
        random_state=42,
        stratify=df_all['class_label']
    )

    df_train_new = df_all.iloc[train_idx].copy()
    df_val_new = df_all.iloc[val_idx].copy()

    # Remove the temporary class_label column
    df_train_new = df_train_new.drop(columns=['class_label'])
    df_val_new = df_val_new.drop(columns=['class_label'])

    print(f"  New train: {len(df_train_new)} samples")
    print(f"  New val:   {len(df_val_new)} samples")

    # Show new distribution
    print("\n[5/6] New distribution:")
    print("\n  TRAINING SET:")
    train_counts = {
        'normal': df_train_new['normal'].sum(),
        'bacteria': df_train_new['bacteria'].sum(),
        'virus': df_train_new['virus'].sum(),
        'COVID-19': df_train_new['COVID-19'].sum()
    }
    for cls, count in train_counts.items():
        pct = 100 * count / len(df_train_new)
        print(f"    {cls:12s}: {int(count):4d} ({pct:.1f}%)")

    print("\n  VALIDATION SET:")
    val_counts = {
        'normal': df_val_new['normal'].sum(),
        'bacteria': df_val_new['bacteria'].sum(),
        'virus': df_val_new['virus'].sum(),
        'COVID-19': df_val_new['COVID-19'].sum()
    }
    for cls, count in val_counts.items():
        pct = 100 * count / len(df_val_new)
        print(f"    {cls:12s}: {int(count):4d} ({pct:.1f}%)")

    # Save new splits
    print("\n[6/6] Saving new splits...")
    df_train_new.to_csv(train_csv, index=False)
    df_val_new.to_csv(val_csv, index=False)

    print(f"  ✓ Saved: {train_csv}")
    print(f"  ✓ Saved: {val_csv}")

    # Summary
    print("\n" + "=" * 60)
    print("RESPLIT COMPLETE!")
    print("=" * 60)
    print(f"\nKey improvements:")
    print(f"  - COVID-19 in val: {int(val_counts['COVID-19'])} samples (was {7})")
    print(f"  - Stratified split ensures balanced classes")
    print(f"  - Original files backed up to: {backup_dir}/")
    print("\nYou can now retrain with better validation metrics!")
    print("=" * 60)

if __name__ == "__main__":
    main()
