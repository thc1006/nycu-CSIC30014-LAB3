#!/usr/bin/env python3
"""
🚀 MEGA ENSEMBLE with TTA - Target: 91%+
Combines ALL available models with Test-Time Augmentation
"""
import os, torch, pandas as pd, numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
from PIL import Image
from tqdm import tqdm
import timm

class TTADataset(Dataset):
    """Dataset with Test-Time Augmentation"""
    def __init__(self, image_dir, img_size, tta_transforms=5):
        self.files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpeg', '.jpg', '.png'))])
        self.image_dir = image_dir
        self.img_size = img_size
        self.tta_transforms = tta_transforms

        # Base transforms
        self.normalize = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')

        # Generate TTA variants
        tta_imgs = []

        # 1. Original (center crop)
        tta_imgs.append(self._transform(img, flip=False, rotate=0))

        if self.tta_transforms >= 2:
            # 2. Horizontal flip
            tta_imgs.append(self._transform(img, flip=True, rotate=0))

        if self.tta_transforms >= 4:
            # 3-4. Small rotations
            tta_imgs.append(self._transform(img, flip=False, rotate=5))
            tta_imgs.append(self._transform(img, flip=False, rotate=-5))

        if self.tta_transforms >= 6:
            # 5-6. Flip + rotations
            tta_imgs.append(self._transform(img, flip=True, rotate=5))
            tta_imgs.append(self._transform(img, flip=True, rotate=-5))

        return torch.stack(tta_imgs), fname

    def _transform(self, img, flip=False, rotate=0):
        """Apply single transform variant"""
        # Resize
        img = T.Resize((self.img_size, self.img_size))(img)

        # Rotate
        if rotate != 0:
            img = T.functional.rotate(img, rotate)

        # Flip
        if flip:
            img = T.functional.hflip(img)

        # To tensor and normalize
        img = T.ToTensor()(img)
        img = self.normalize(img)

        return img

def load_checkpoint(model, ckpt_path, device):
    """Load model checkpoint safely"""
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'model' in state:
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
        model.eval()
        return True
    except Exception as e:
        print(f"  ❌ Failed to load {ckpt_path}: {e}")
        return False

@torch.no_grad()
def predict_tta(model, loader, device):
    """Generate predictions with TTA"""
    all_probs = []
    for tta_imgs, _ in tqdm(loader, desc="  Predicting", leave=False):
        # tta_imgs shape: (batch, tta_count, C, H, W)
        batch_size, tta_count, C, H, W = tta_imgs.shape

        # Flatten TTA dimension
        tta_imgs = tta_imgs.view(batch_size * tta_count, C, H, W).to(device)

        # Predict
        logits = model(tta_imgs)
        probs = torch.softmax(logits, dim=1)

        # Reshape and average TTA
        probs = probs.view(batch_size, tta_count, -1).mean(dim=1)

        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs)

def build_model(model_name, img_size, num_classes=4):
    """Build model architecture"""
    if model_name == 'densenet121':
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'vit_base_patch16_384':
        model = timm.create_model('vit_base_patch16_384', pretrained=False, num_classes=num_classes)
    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == 'regnet_y_3_2gf':
        model = models.regnet_y_3_2gf(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'convnext_base':
        model = models.convnext_base(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 MEGA ENSEMBLE with TTA")
    print(f"Device: {device}\n")

    # Model configurations with estimated weights based on expected performance
    models_config = [
        {
            'name': 'Medical DenseNet121',
            'arch': 'densenet121',
            'ckpt': 'outputs/medical_pretrained/best.pt',
            'img_size': 384,
            'weight': 1.5,  # Medical pretrained gets higher weight
            'tta': 4
        },
        {
            'name': 'Vision Transformer',
            'arch': 'vit_base_patch16_384',
            'ckpt': 'outputs/vit_ultimate/best.pt',
            'img_size': 384,
            'weight': 1.3,  # ViT is powerful
            'tta': 2  # ViT is slow, use less TTA
        },
        {
            'name': 'EfficientNet-V2-S (Improved)',
            'arch': 'efficientnet_v2_s',
            'ckpt': 'outputs/improved_breakthrough/best.pt',
            'img_size': 384,
            'weight': 1.2,
            'tta': 4
        },
        {
            'name': 'EfficientNet-V2-S (CLAHE)',
            'arch': 'efficientnet_v2_s',
            'ckpt': 'outputs/breakthrough_clahe/best.pt',
            'img_size': 384,
            'weight': 1.1,
            'tta': 4
        },
        {
            'name': 'RegNet-Y-3.2GF',
            'arch': 'regnet_y_3_2gf',
            'ckpt': 'outputs/diverse_model2/best.pt',
            'img_size': 384,
            'weight': 1.0,
            'tta': 4
        },
        {
            'name': 'EfficientNet-V2-S @ 320px',
            'arch': 'efficientnet_v2_s',
            'ckpt': 'outputs/diverse_model3/best.pt',
            'img_size': 320,
            'weight': 1.0,
            'tta': 4
        },
        {
            'name': 'ConvNeXt-Base @ 448px',
            'arch': 'convnext_base',
            'ckpt': 'outputs/run1/best.pt',
            'img_size': 448,
            'weight': 1.1,  # High resolution gets bonus
            'tta': 2  # Large model, less TTA
        },
    ]

    # Add 5-fold models if available
    for fold in range(5):
        fold_path = f'outputs/final_optimized/fold{fold}/best.pt'
        if os.path.exists(fold_path):
            models_config.append({
                'name': f'EfficientNet-V2-S Fold-{fold}',
                'arch': 'efficientnet_v2_s',
                'ckpt': fold_path,
                'img_size': 384,
                'weight': 0.8,  # Fold models get lower weight
                'tta': 2
            })

    test_dir = 'test_images'

    # Collect predictions
    all_predictions = []
    all_weights = []
    filenames = None

    print(f"📊 Loading and predicting with {len(models_config)} models:\n")

    for i, config in enumerate(models_config, 1):
        print(f"[{i}/{len(models_config)}] {config['name']}")
        print(f"  Architecture: {config['arch']} @ {config['img_size']}px")
        print(f"  TTA transforms: {config['tta']}")
        print(f"  Weight: {config['weight']}")

        # Check if checkpoint exists
        if not os.path.exists(config['ckpt']):
            print(f"  ⚠️  Checkpoint not found, skipping...")
            continue

        try:
            # Build model
            model = build_model(config['arch'], config['img_size'])
            model = model.to(device)

            # Load checkpoint
            if not load_checkpoint(model, config['ckpt'], device):
                continue

            # Create dataset and loader
            dataset = TTADataset(test_dir, config['img_size'], tta_transforms=config['tta'])
            if filenames is None:
                filenames = dataset.files

            loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

            # Predict with TTA
            probs = predict_tta(model, loader, device)
            all_predictions.append(probs)
            all_weights.append(config['weight'])

            print(f"  ✅ Predictions shape: {probs.shape}")
            print(f"  Mean confidence: {probs.max(axis=1).mean():.4f}\n")

            # Clean up
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ Error processing model: {e}\n")
            continue

    if len(all_predictions) == 0:
        print("❌ No models were successfully loaded!")
        return

    # Weighted ensemble
    print(f"\n{'='*60}")
    print(f"🎯 Creating weighted ensemble from {len(all_predictions)} models...")
    print(f"{'='*60}")

    # Normalize weights
    all_weights = np.array(all_weights)
    all_weights = all_weights / all_weights.sum()

    # Weighted average
    ensemble_probs = np.average(all_predictions, axis=0, weights=all_weights)

    # Temperature scaling (calibration)
    temperature = 1.1  # Slightly sharpen predictions
    ensemble_probs = np.power(ensemble_probs, 1/temperature)
    ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

    pred_labels = ensemble_probs.argmax(axis=1)

    # Convert to one-hot
    one_hot = np.zeros_like(ensemble_probs)
    one_hot[np.arange(len(one_hot)), pred_labels] = 1

    # Create submission
    label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
    df = pd.DataFrame(one_hot, columns=label_cols)
    df.insert(0, 'new_filename', filenames)

    # Save
    output_path = 'data/submission_mega_ensemble_tta.csv'
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"✅ MEGA ENSEMBLE COMPLETE!")
    print(f"{'='*60}")
    print(f"  Output: {output_path}")
    print(f"  Models used: {len(all_predictions)}")
    print(f"  Total samples: {len(df)}")
    print(f"\n  Label distribution:")
    for i, col in enumerate(label_cols):
        count = (pred_labels == i).sum()
        print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\n  Mean confidence: {ensemble_probs.max(axis=1).mean():.4f}")
    print(f"  Min confidence: {ensemble_probs.max(axis=1).min():.4f}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
