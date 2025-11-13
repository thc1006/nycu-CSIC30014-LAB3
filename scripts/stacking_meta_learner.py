#!/usr/bin/env python3
"""
Stacking / Meta-Learning Ensemble
Trains a second-level model to learn optimal combinations of base models

This is the KEY technique to break through 91%!

Architecture:
    Level 0 (Base Models): 18+ models → predictions
    Level 1 (Meta-Learner): Learn to combine Level 0 predictions

Expected improvement: +1-3% over best single model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class StackingEnsemble:
    """
    Two-level stacking ensemble
    """

    def __init__(self, meta_model_type='lgb', use_features=True):
        """
        Args:
            meta_model_type: 'lgb', 'xgb', 'mlp', 'rf', 'logistic'
            use_features: Whether to use additional features (entropy, confidence, etc.)
        """
        self.meta_model_type = meta_model_type
        self.use_features = use_features
        self.meta_models = []  # One per class
        self.base_model_paths = []
        self.feature_names = []

    def collect_base_predictions(self, data_dir='data', val_preds_dir=None):
        """
        Collect predictions from all base models on validation set

        Args:
            data_dir: Directory containing train.csv (ground truth)
            val_preds_dir: Directory containing validation prediction files
        """
        print("=" * 80)
        print("Collecting Base Model Predictions")
        print("=" * 80)

        # Try multiple locations for validation predictions
        search_dirs = []
        if val_preds_dir:
            search_dirs.append(Path(val_preds_dir))

        # Use absolute paths to avoid CWD issues
        import os
        project_root = Path(os.getcwd())
        search_dirs.extend([
            project_root / 'outputs' / 'breakthrough_20251113_004854' / 'layer1_val_predictions',
            project_root / 'outputs' / 'layer1_val_predictions',
            Path(data_dir),
            Path('.'),
        ])

        pred_files = []
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            # Look for various prediction file patterns
            patterns = [
                'validation_predictions_*.csv',
                '*_val_pred.csv',
                '*_fold*_val*.csv',
            ]
            for pattern in patterns:
                pred_files.extend(list(search_dir.glob(pattern)))

        # Remove duplicates
        pred_files = list(set(pred_files))

        if len(pred_files) == 0:
            print("⚠️ No validation prediction files found!")
            print("Searched in:")
            for d in search_dirs:
                print(f"  - {d}")
            print("\nPlease generate validation predictions for all models first.")
            return None, None

        print(f"Found {len(pred_files)} prediction files from K-Fold CV")

        # Group predictions by model type (not fold)
        # e.g., efficientnet_v2_l_fold0_val_pred.csv -> efficientnet_v2_l
        from collections import defaultdict
        model_preds_by_fold = defaultdict(list)

        for pred_file in pred_files:
            # Parse filename: {model_type}_fold{n}_val_pred.csv
            stem = pred_file.stem
            if '_fold' in stem:
                parts = stem.split('_fold')
                model_type = parts[0]
                fold_num = int(parts[1].split('_')[0])
                model_preds_by_fold[model_type].append((fold_num, pred_file))
            else:
                print(f"  ⚠️ Skipping {pred_file.name} (unexpected format)")

        if not model_preds_by_fold:
            print("❌ No valid K-Fold prediction files found!")
            return None, None

        print(f"\nFound {len(model_preds_by_fold)} model types:")
        for model_type, fold_files in model_preds_by_fold.items():
            print(f"  • {model_type}: {len(fold_files)} folds")

        # Merge all folds to create complete validation predictions
        # Load ground truth from all K-fold validation splits
        kfold_dir = Path('data/kfold_splits')
        all_val_data = []

        for fold in range(5):
            fold_csv = kfold_dir / f'fold{fold}_val.csv'
            if fold_csv.exists():
                fold_df = pd.read_csv(fold_csv)
                all_val_data.append(fold_df)

        if not all_val_data:
            print("❌ No K-fold validation CSVs found in data/kfold_splits/")
            return None, None

        val_df = pd.concat(all_val_data, ignore_index=True)
        print(f"\n✓ Loaded ground truth: {len(val_df)} samples from {len(all_val_data)} folds")

        # Merge predictions from all folds for each model
        all_preds = {}
        for model_type, fold_files in model_preds_by_fold.items():
            # Sort by fold number
            fold_files.sort(key=lambda x: x[0])

            # Concatenate predictions from all folds
            fold_preds = []
            for fold_num, pred_file in fold_files:
                df = pd.read_csv(pred_file)
                fold_preds.append(df[['normal', 'bacteria', 'virus', 'COVID-19']].values)

            # Stack vertically
            merged_preds = np.concatenate(fold_preds, axis=0)
            all_preds[model_type] = merged_preds
            print(f"  ✓ {model_type}: {merged_preds.shape} (merged from {len(fold_files)} folds)")

        self.base_model_names = list(all_preds.keys())
        print(f"\n✓ Total base models: {len(self.base_model_names)}")
        return all_preds, val_df

    def create_meta_features(self, predictions_dict):
        """
        Create meta-features from base model predictions

        Features:
        1. Raw probabilities from each model
        2. Entropy (uncertainty)
        3. Max probability (confidence)
        4. Probability std across models
        5. Pairwise disagreements
        """
        n_samples = len(next(iter(predictions_dict.values())))
        n_models = len(predictions_dict)
        n_classes = 4

        # Stack all predictions: (n_models, n_samples, n_classes)
        pred_stack = np.stack(list(predictions_dict.values()), axis=0)

        features = []
        feature_names = []

        # 1. Raw probabilities
        for i, model_name in enumerate(predictions_dict.keys()):
            features.append(pred_stack[i])  # (n_samples, n_classes)
            for c in range(n_classes):
                feature_names.append(f'{model_name}_class{c}')

        features = np.concatenate(features, axis=1)  # (n_samples, n_models*n_classes)

        if self.use_features:
            # 2. Entropy (per sample)
            entropy = -np.sum(pred_stack * np.log(pred_stack + 1e-10), axis=2)  # (n_models, n_samples)
            entropy_mean = entropy.mean(axis=0, keepdims=True).T  # (n_samples, 1)
            entropy_std = entropy.std(axis=0, keepdims=True).T
            features = np.concatenate([features, entropy_mean, entropy_std], axis=1)
            feature_names.extend(['entropy_mean', 'entropy_std'])

            # 3. Confidence (max probability)
            max_prob = pred_stack.max(axis=2)  # (n_models, n_samples)
            confidence_mean = max_prob.mean(axis=0, keepdims=True).T
            confidence_std = max_prob.std(axis=0, keepdims=True).T
            features = np.concatenate([features, confidence_mean, confidence_std], axis=1)
            feature_names.extend(['confidence_mean', 'confidence_std'])

            # 4. Agreement (std of predictions across models for each class)
            pred_std = pred_stack.std(axis=0)  # (n_samples, n_classes)
            features = np.concatenate([features, pred_std], axis=1)
            for c in range(n_classes):
                feature_names.append(f'disagreement_class{c}')

        self.feature_names = feature_names
        print(f"\n✓ Created {features.shape[1]} meta-features")

        return features

    def train_meta_model(self, X_meta, y_true, cv_folds=5):
        """
        Train meta-learner using cross-validation on base predictions
        """
        print("\n" + "=" * 80)
        print(f"Training Meta-Learner ({self.meta_model_type.upper()})")
        print("=" * 80)

        n_classes = 4
        class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Train one model per class (One-vs-Rest)
        self.meta_models = []
        cv_scores = []

        for class_idx, class_name in enumerate(class_names):
            print(f"\n[{class_idx+1}/{n_classes}] Training for {class_name}...")

            y_binary = (y_true == class_idx).astype(int)

            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y_binary)):
                X_train, X_val = X_meta[train_idx], X_meta[val_idx]
                y_train, y_val = y_binary[train_idx], y_binary[val_idx]

                # Create meta-model
                if self.meta_model_type == 'lgb':
                    model = lgb.LGBMClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=5,
                        num_leaves=31,
                        min_child_samples=20,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42 + fold,
                        verbose=-1
                    )
                elif self.meta_model_type == 'xgb':
                    model = xgb.XGBClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42 + fold,
                        eval_metric='logloss'
                    )
                elif self.meta_model_type == 'mlp':
                    model = MLPClassifier(
                        hidden_layer_sizes=(128, 64, 32),
                        activation='relu',
                        alpha=0.001,
                        learning_rate_init=0.001,
                        max_iter=500,
                        random_state=42 + fold
                    )
                elif self.meta_model_type == 'rf':
                    model = RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_split=10,
                        random_state=42 + fold
                    )
                else:  # logistic
                    model = LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        random_state=42 + fold
                    )

                # Train
                model.fit(X_train, y_train)

                # Validate
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
                f1 = f1_score(y_val, y_pred)
                fold_scores.append(f1)

                if fold == 0:  # Use first fold model
                    self.meta_models.append(model)

            avg_score = np.mean(fold_scores)
            cv_scores.append(avg_score)
            print(f"  Cross-val F1: {avg_score:.4f} (±{np.std(fold_scores):.4f})")

        macro_f1 = np.mean(cv_scores)
        print(f"\n✓ Meta-learner Macro-F1: {macro_f1:.4f}")
        print(f"  Per-class: {', '.join([f'{s:.4f}' for s in cv_scores])}")

        return macro_f1

    def predict(self, predictions_dict):
        """
        Use trained meta-learner to make final predictions
        """
        X_meta = self.create_meta_features(predictions_dict)

        n_samples = X_meta.shape[0]
        n_classes = 4

        final_probs = np.zeros((n_samples, n_classes))

        for class_idx, model in enumerate(self.meta_models):
            final_probs[:, class_idx] = model.predict_proba(X_meta)[:, 1]

        # Normalize
        final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)

        return final_probs

    def save(self, path='models/stacking_meta_learner.pkl'):
        """Save trained meta-learner"""
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"\n✓ Saved meta-learner to {path}")

    @staticmethod
    def load(path='models/stacking_meta_learner.pkl'):
        """Load trained meta-learner"""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


def main():
    """
    Main training pipeline
    """
    print("\n" + "=" * 80)
    print("STACKING META-LEARNER TRAINING")
    print("=" * 80)
    print("\nThis will train a second-level model to optimally combine")
    print("all base models' predictions.")
    print("\nExpected improvement: +1-3% over best single model")
    print("=" * 80)

    # Try different meta-learners
    meta_types = ['lgb', 'xgb', 'mlp', 'rf', 'logistic']
    results = {}

    for meta_type in meta_types:
        print(f"\n\n{'=' * 80}")
        print(f"Testing {meta_type.upper()} as meta-learner")
        print("=" * 80)

        stacker = StackingEnsemble(meta_model_type=meta_type, use_features=True)

        # Collect base predictions
        result = stacker.collect_base_predictions()
        if result is None or result[0] is None:
            print(f"❌ Failed to collect predictions for {meta_type}")
            continue
        preds_dict, val_df = result

        # Create meta-features
        X_meta = stacker.create_meta_features(preds_dict)

        # Get ground truth labels (map class names to indices)
        class_to_idx = {'normal': 0, 'bacteria': 1, 'virus': 2, 'COVID-19': 3}
        y_true = val_df['class_label'].map(class_to_idx).values

        # Train meta-learner
        score = stacker.train_meta_model(X_meta, y_true, cv_folds=5)
        results[meta_type] = score

        # Save
        stacker.save(f'models/stacking_{meta_type}.pkl')

    # Summary
    print("\n\n" + "=" * 80)
    print("META-LEARNER COMPARISON")
    print("=" * 80)
    for meta_type, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{meta_type:>10s}: {score:.4f}")

    best_type = max(results, key=results.get)
    print(f"\n✓ Best meta-learner: {best_type.upper()} ({results[best_type]:.4f})")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Generate test predictions using the meta-learner:")
    print("   python scripts/stacking_predict.py")
    print("\n2. Expected test score: ~87-90% (vs current 84.19%)")
    print("\n3. This should get you much closer to 91%!")
    print("=" * 80)


if __name__ == '__main__':
    main()
