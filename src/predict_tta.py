"""
Test Time Augmentation (TTA) Prediction
增強測試時預測精度
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models import get_model
import pandas as pd
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import os


class TTAPredictor:
    """Test Time Augmentation Predictor"""

    def __init__(self, model, device, tta_transforms=5):
        self.model = model
        self.device = device
        self.tta_transforms = tta_transforms

    def get_tta_transforms(self, base_size=384):
        """Generate TTA transformations"""
        transforms = []

        # 1. Original
        transforms.append(T.Compose([
            T.Resize((base_size, base_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))

        # 2. Horizontal flip
        transforms.append(T.Compose([
            T.Resize((base_size, base_size)),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))

        # 3. Slight rotation (+5 degrees)
        transforms.append(T.Compose([
            T.Resize((base_size, base_size)),
            T.RandomRotation(degrees=(5, 5)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))

        # 4. Slight rotation (-5 degrees)
        transforms.append(T.Compose([
            T.Resize((base_size, base_size)),
            T.RandomRotation(degrees=(-5, -5)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))

        # 5. Scale up slightly
        transforms.append(T.Compose([
            T.Resize((int(base_size * 1.05), int(base_size * 1.05))),
            T.CenterCrop(base_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))

        return transforms[:self.tta_transforms]

    def predict_with_tta(self, image, img_size=384):
        """Predict with TTA"""
        transforms = self.get_tta_transforms(img_size)
        predictions = []

        self.model.eval()
        with torch.no_grad():
            for transform in transforms:
                # Apply transform
                img_tensor = transform(image).unsqueeze(0).to(self.device)

                # Predict
                logits = self.model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())

        # Average predictions
        avg_pred = np.mean(predictions, axis=0)
        return avg_pred


def predict_with_tta_ensemble(
    checkpoints,
    test_csv,
    test_images_dir,
    output_path,
    img_size=384,
    tta_transforms=5,
    batch_size=16,
    num_workers=8,
    device='cuda'
):
    """
    Predict using multiple checkpoints + TTA

    Args:
        checkpoints: List of checkpoint paths
        test_csv: Test CSV path
        test_images_dir: Test images directory
        output_path: Output submission path
        img_size: Image size
        tta_transforms: Number of TTA transforms
        batch_size: Batch size
        num_workers: Number of workers
        device: Device
    """

    print(f"🔥 TTA Ensemble Prediction")
    print(f"  • Checkpoints: {len(checkpoints)}")
    print(f"  • TTA transforms: {tta_transforms}")
    print(f"  • Image size: {img_size}")
    print(f"  • Device: {device}")
    print()

    # Load test data
    test_df = pd.read_csv(test_csv)
    file_col = 'filename' if 'filename' in test_df.columns else 'new_filename'

    # Prepare predictions storage
    all_predictions = []

    # For each checkpoint
    for ckpt_idx, ckpt_path in enumerate(checkpoints):
        print(f"[{ckpt_idx+1}/{len(checkpoints)}] Loading {Path(ckpt_path).parent.name}/{Path(ckpt_path).name}...")

        # Load model
        state = torch.load(ckpt_path, map_location=device)
        # Try to get model_name from different locations
        if 'model_name' in state:
            model_name = state['model_name']
            num_classes = state.get('num_classes', 4)
        elif 'cfg' in state:
            model_name = state['cfg']['model']['name']
            num_classes = state['cfg']['data']['num_classes']
        else:
            model_name = 'efficientnet_v2_s'
            num_classes = 4

        model = get_model(model_name, weights='DEFAULT')

        # Modify classifier
        if hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'head'):
            in_features = model.head.in_features if hasattr(model.head, 'in_features') else model.head.fc.in_features
            model.head = nn.Linear(in_features, num_classes)

        model.load_state_dict(state['model'])
        model = model.to(device)
        model.eval()

        # Create TTA predictor
        tta_predictor = TTAPredictor(model, device, tta_transforms)

        # Predict
        fold_predictions = []

        with torch.no_grad():
            for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Fold {ckpt_idx}"):
                img_path = os.path.join(test_images_dir, row[file_col])
                image = Image.open(img_path).convert('RGB')

                # Predict with TTA
                pred = tta_predictor.predict_with_tta(image, img_size)
                fold_predictions.append(pred[0])

        all_predictions.append(np.array(fold_predictions))
        print(f"  ✅ Fold {ckpt_idx} complete. Shape: {np.array(fold_predictions).shape}")

    # Average predictions across all folds
    print()
    print("📊 Ensembling predictions...")
    final_predictions = np.mean(all_predictions, axis=0)
    print(f"  Final shape: {final_predictions.shape}")

    # Create submission with probabilities
    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

    # Create DataFrame with probabilities
    submission = pd.DataFrame({
        file_col: test_df[file_col],
    })

    # Add probability columns
    for i, class_name in enumerate(class_names):
        submission[class_name] = final_predictions[:, i]

    submission.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}")

    # Print statistics
    predicted_classes = np.argmax(final_predictions, axis=1)
    predicted_labels = [class_names[c] for c in predicted_classes]
    unique, counts = np.unique(predicted_labels, return_counts=True)
    print()
    print("📈 Prediction distribution:")
    for label, count in zip(unique, counts):
        pct = count / len(predicted_labels) * 100
        print(f"  {label:12s}: {count:4d} ({pct:5.2f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True, help='List of checkpoint paths')
    parser.add_argument('--test-csv', default='data/test_data.csv')
    parser.add_argument('--test-images', default='test_images')
    parser.add_argument('--output', default='data/submission_tta_ensemble.csv')
    parser.add_argument('--img-size', type=int, default=384)
    parser.add_argument('--tta-transforms', type=int, default=5)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    predict_with_tta_ensemble(
        checkpoints=args.checkpoints,
        test_csv=args.test_csv,
        test_images_dir=args.test_images,
        output_path=args.output,
        img_size=args.img_size,
        tta_transforms=args.tta_transforms,
        device=args.device
    )
