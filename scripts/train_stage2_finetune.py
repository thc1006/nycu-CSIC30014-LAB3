#!/usr/bin/env python3
"""
終極突破訓練腳本 - 完全獨立，無相對導入依賴
支持 DINOv2-Large, EfficientNet-V2-L, Swin-Large
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import math
import time

# ============================================================================
# Dataset
# ============================================================================

class CXRDataset(Dataset):
    """胸部 X 光數據集"""
    def __init__(self, csv_path, images_dir, img_size=224, augment=False):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.img_size = img_size
        self.label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

        # Transforms
        if augment:
            self.transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1),
                T.ToTensor(),
                T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 支持 K-Fold split (source_dir 欄位)
        if 'source_dir' in row.index and pd.notna(row['source_dir']):
            img_path = os.path.join(row['source_dir'], row['new_filename'])
        else:
            img_path = os.path.join(self.images_dir, row['new_filename'])

        # Load image
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        # Label (one-hot -> class index)
        if all(col in row.index for col in self.label_cols):
            label_vec = row[self.label_cols].values.astype(float)
            if not np.any(np.isnan(label_vec)):
                label = int(np.argmax(label_vec))
            else:
                label = -1  # Test data
        else:
            label = -1

        return image, label, row['new_filename']

# ============================================================================
# Loss Functions
# ============================================================================

class ImprovedFocalLoss(nn.Module):
    """Improved Focal Loss with label smoothing"""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, target):
        num_classes = logits.size(-1)
        device = logits.device

        # Label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_target = torch.zeros_like(logits)
                smooth_target.fill_(self.label_smoothing / (num_classes - 1))
                smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)

            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -(smooth_target * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(logits, target, reduction='none')

        # Focal term
        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma

        # Class weights
        if self.alpha is not None:
            if self.alpha.device != device:
                self.alpha = self.alpha.to(device)
            alpha_t = self.alpha[target]
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss

        return loss.mean()

# ============================================================================
# Model Builder
# ============================================================================

def build_model(model_name, num_classes=4, dropout=0.3, img_size=384):
    """構建模型"""
    print(f"[Model] Building {model_name} (img_size={img_size})...")

    if model_name == 'efficientnet_v2_l':
        from torchvision import models
        model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)

        # 修改 classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        print(f"[Model] EfficientNet-V2-L loaded (params: ~120M)")

    elif model_name.startswith('dinov2_'):
        from transformers import Dinov2ForImageClassification, Dinov2Config

        dinov2_models = {
            'dinov2_small': 'facebook/dinov2-small',
            'dinov2_base': 'facebook/dinov2-base',
            'dinov2_large': 'facebook/dinov2-large',
        }

        if model_name not in dinov2_models:
            raise ValueError(f"Unknown DINOv2: {model_name}")

        pretrained_name = dinov2_models[model_name]

        # Load config and modify
        config = Dinov2Config.from_pretrained(pretrained_name)
        config.num_labels = num_classes

        # Load model
        model = Dinov2ForImageClassification.from_pretrained(
            pretrained_name,
            config=config,
            ignore_mismatched_sizes=True
        )

        # Add dropout
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )

        print(f"[Model] DINOv2-{model_name.split('_')[1]} loaded (params: ~300M for large)")

    elif model_name.startswith('swin_'):
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=dropout,
            img_size=img_size
        )
        print(f"[Model] Swin Transformer ({model_name}) loaded (params: ~200M for large)")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler=None, mixup_prob=0.0, mixup_alpha=1.0):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets, _) in enumerate(loader):
        if targets[0] == -1:  # Skip test data
            continue

        images = images.to(device)
        targets = targets.to(device)

        # Mixup augmentation
        if mixup_prob > 0 and np.random.rand() < mixup_prob:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            batch_size = images.size(0)
            index = torch.randperm(batch_size).to(device)

            mixed_images = lam * images + (1 - lam) * images[index]
            targets_a, targets_b = targets, targets[index]

            # Forward
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(mixed_images)
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(mixed_images)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                loss.backward()
                optimizer.step()
        else:
            # Normal training
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    """評估模型"""
    model.eval()
    all_preds = []
    all_targets = []

    for images, targets, _ in loader:
        if targets[0] == -1:  # Skip test data
            continue

        images = images.to(device)

        outputs = model(images)
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits

        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())

    if len(all_preds) == 0:
        return 0.0, None

    # Macro F1
    f1 = f1_score(all_targets, all_preds, average='macro')

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    return f1, cm

# ============================================================================
# Main Training
# ============================================================================

def train_model(config_path, fold=None):
    """訓練模型"""

    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override fold if specified
    if fold is not None:
        cfg['fold'] = fold

    print("=" * 80)
    print(f"🚀 開始訓練: {cfg['model']}")
    if 'fold' in cfg:
        print(f"📁 Fold: {cfg['fold']}")
    print("=" * 80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Seed
    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build model
    model = build_model(
        cfg['model'],
        num_classes=cfg.get('num_classes', 4),
        dropout=cfg.get('dropout', 0.3),
        img_size=cfg.get('img_size', 384)
    )
    model = model.to(device)

    # Data
    if 'fold' in cfg and 'kfold_csv_dir' in cfg:
        # K-Fold data
        train_csv = f"{cfg['kfold_csv_dir']}/fold{cfg['fold']}_train.csv"
        val_csv = f"{cfg['kfold_csv_dir']}/fold{cfg['fold']}_val.csv"
        images_dir = cfg.get('data_dir', 'data/train')
    else:
        # Standard train/val split
        train_csv = cfg.get('train_csv', 'data/train.csv')
        val_csv = cfg.get('val_csv', 'data/val.csv')
        images_dir = cfg.get('data_dir', 'data/train')

    print(f"Train CSV: {train_csv}")
    print(f"Val CSV: {val_csv}")

    train_dataset = CXRDataset(train_csv, images_dir, cfg['img_size'], augment=True)
    val_dataset = CXRDataset(val_csv, images_dir, cfg['img_size'], augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'] * 2,
        shuffle=False,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Loss
    if cfg.get('loss', 'crossentropy') == 'improved_focal':
        criterion = ImprovedFocalLoss(
            alpha=cfg.get('focal_alpha', [1.0, 1.5, 2.0, 12.0]),
            gamma=cfg.get('focal_gamma', 3.5),
            label_smoothing=cfg.get('label_smoothing', 0.12)
        )
        print(f"Loss: Improved Focal (α={cfg.get('focal_alpha')}, γ={cfg.get('focal_gamma')})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Loss: CrossEntropy")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg.get('weight_decay', 0.0001)
    )
    print(f"Optimizer: AdamW (lr={cfg['lr']}, wd={cfg.get('weight_decay')})")

    # Scheduler (Cosine with warmup)
    total_steps = len(train_loader) * cfg['epochs']
    warmup_steps = len(train_loader) * cfg.get('warmup_epochs', 3)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Scheduler: Cosine with {cfg.get('warmup_epochs', 3)} epochs warmup")

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if cfg.get('mixed_precision', True) else None
    if scaler:
        print("Mixed precision: Enabled (FP16)")

    # Output directory
    output_dir = Path(cfg.get('output_dir', 'outputs/model'))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    patience = cfg.get('patience', 15)

    print("\n" + "=" * 80)
    print("開始訓練")
    print("=" * 80 + "\n")

    for epoch in range(cfg['epochs']):
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler,
            mixup_prob=cfg.get('mixup_prob', 0.0),
            mixup_alpha=cfg.get('mixup_alpha', 1.0)
        )

        # Validate
        val_f1, val_cm = evaluate(model, val_loader, device)

        elapsed = time.time() - start_time

        # Log
        print(f"Epoch {epoch+1}/{cfg['epochs']} ({elapsed:.1f}s) | "
              f"Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_cm is not None and epoch % 5 == 0:
            print(f"  Confusion Matrix:\n{val_cm}")

        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'config': cfg
            }
            torch.save(checkpoint, output_dir / 'best.pt')
            print(f"  ✅ Best model saved (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping (patience={patience})")
                break

    print("\n" + "=" * 80)
    print(f"✅ 訓練完成！Best Val F1: {best_f1:.4f}")
    print("=" * 80)

    return best_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config YAML file')
    parser.add_argument('--fold', type=int, default=None, help='Override fold number')
    args = parser.parse_args()

    train_model(args.config, fold=args.fold)

if __name__ == '__main__':
    main()
