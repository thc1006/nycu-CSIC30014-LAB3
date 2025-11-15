#!/usr/bin/env python3
"""
生成 Layer 1 所有模型的驗證集預測
用於訓練 Layer 2 meta-learners
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class SimpleDataset(Dataset):
    """簡單的數據集類"""
    def __init__(self, csv_path, images_dir, img_size=384):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.img_size = img_size

        self.transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Support K-Fold split with source_dir column
        if 'source_dir' in row.index and pd.notna(row['source_dir']):
            img_path = os.path.join(row['source_dir'], row['new_filename'])
        else:
            img_path = os.path.join(self.images_dir, row['new_filename'])

        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        return image, row['new_filename']

def load_model(checkpoint_path, model_type, img_size, num_classes=4):
    """加載模型"""
    print(f"Loading model from {checkpoint_path}")

    # Build model based on type
    if model_type == 'efficientnet_v2_l':
        from torchvision import models
        model = models.efficientnet_v2_l(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    elif model_type.startswith('swin_'):
        import timm
        model = timm.create_model(
            'swin_large_patch4_window12_384',
            pretrained=False,
            num_classes=num_classes,
            img_size=img_size
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    elif 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)

    return model

@torch.no_grad()
def predict(model, dataloader, device):
    """生成預測"""
    model.eval()
    all_probs = []
    all_filenames = []

    for images, filenames in tqdm(dataloader, desc="Predicting"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_filenames.extend(filenames)

    all_probs = np.concatenate(all_probs, axis=0)
    return all_probs, all_filenames

def generate_predictions_for_model(checkpoint_path, val_csv, images_dir, output_path,
                                   model_type, img_size=384, batch_size=32):
    """為單個模型生成驗證集預測"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(checkpoint_path, model_type, img_size)
    model = model.to(device)

    # Create dataset and dataloader
    dataset = SimpleDataset(val_csv, images_dir, img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Validation samples: {len(dataset)}")

    # Generate predictions
    probs, filenames = predict(model, dataloader, device)

    # Create output DataFrame
    df = pd.DataFrame(probs, columns=['normal', 'bacteria', 'virus', 'COVID-19'])
    df.insert(0, 'new_filename', filenames)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved predictions to {output_path}")

    return df

def main():
    """主函數：為所有 Layer 1 模型生成驗證集預測"""

    # Configuration
    breakthrough_dir = Path('outputs/breakthrough_20251113_004854')
    layer1_dir = breakthrough_dir / 'layer1'
    output_dir = breakthrough_dir / 'layer1_val_predictions'
    output_dir.mkdir(parents=True, exist_ok=True)

    kfold_dir = Path('data/kfold_splits')
    images_dir = 'data/train'

    print("=" * 80)
    print("生成 Layer 1 驗證集預測")
    print("=" * 80)

    # Model configurations
    models_config = {
        'efficientnet_v2_l': {
            'img_size': 480,
            'batch_size': 8,
        },
        'swin_large': {
            'img_size': 384,
            'batch_size': 12,
        }
    }

    total_models = 0
    completed_models = 0

    # Process each model type
    for model_type, config in models_config.items():
        model_dir = layer1_dir / model_type

        if not model_dir.exists():
            print(f"⚠️ Model directory not found: {model_dir}")
            continue

        # Process each fold
        for fold in range(5):
            fold_dir = model_dir / f'fold{fold}'
            checkpoint = fold_dir / 'best.pt'

            if not checkpoint.exists():
                print(f"⚠️ Checkpoint not found: {checkpoint}")
                continue

            total_models += 1

            # Output file
            output_file = output_dir / f'{model_type}_fold{fold}_val_pred.csv'

            # Skip if already exists
            if output_file.exists():
                print(f"✅ Already exists: {output_file.name}")
                completed_models += 1
                continue

            # Validation CSV for this fold
            val_csv = kfold_dir / f'fold{fold}_val.csv'

            if not val_csv.exists():
                print(f"⚠️ Validation CSV not found: {val_csv}")
                continue

            print(f"\n{'='*80}")
            print(f"Processing: {model_type} Fold {fold}")
            print(f"{'='*80}")
            print(f"Checkpoint: {checkpoint}")
            print(f"Val CSV: {val_csv}")
            print(f"Output: {output_file}")

            try:
                # Generate predictions
                generate_predictions_for_model(
                    checkpoint_path=str(checkpoint),
                    val_csv=str(val_csv),
                    images_dir=images_dir,
                    output_path=str(output_file),
                    model_type=model_type,
                    img_size=config['img_size'],
                    batch_size=config['batch_size']
                )
                completed_models += 1

            except Exception as e:
                print(f"❌ Error processing {model_type} fold {fold}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Summary
    print(f"\n{'='*80}")
    print("總結")
    print(f"{'='*80}")
    print(f"完成: {completed_models} / {total_models} 模型")

    # List generated files
    pred_files = sorted(output_dir.glob('*_val_pred.csv'))
    print(f"\n生成的預測文件 ({len(pred_files)}):")
    for f in pred_files:
        size = f.stat().st_size / 1024  # KB
        print(f"  ✓ {f.name} ({size:.1f} KB)")

    if completed_models == total_models and total_models > 0:
        print(f"\n✅ 所有驗證集預測已生成！")
        print(f"\n下一步:")
        print(f"  python3 scripts/stacking_meta_learner.py")
    else:
        print(f"\n⚠️ 部分模型預測失敗，請檢查錯誤")

    print("=" * 80)

if __name__ == '__main__':
    main()
