#!/usr/bin/env python3
"""
🏆 冠軍模型訓練 - 榨乾 GPU
使用 1065 個偽標籤 + 原始數據訓練多個大型模型

策略: 訓練 5 個不同架構的大型模型，每個使用最大 batch size
- ConvNeXt-Large (200M params)
- ViT-Large (307M params)
- MaxViT-Large (212M params)
- BEiT-Large (307M params)
- CoAtNet-3 (168M params)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import timm
import math
import argparse

# ============================================================================
# Dataset with Pseudo-labels Support
# ============================================================================

class CXRDatasetWithPseudo(Dataset):
    """胸部 X 光數據集 + 偽標籤支持"""
    def __init__(self, train_csv, pseudo_csv=None, images_dir='data/images',
                 img_size=384, augment=True, pseudo_weight=0.5):
        # Load training data
        train_df = pd.read_csv(train_csv)

        # Add pseudo-labels if provided
        if pseudo_csv and os.path.exists(pseudo_csv):
            pseudo_df = pd.read_csv(pseudo_csv)
            print(f"[OK] Loading {len(pseudo_df)} pseudo-labels from {pseudo_csv}")

            # Add sample_weight column
            train_df['sample_weight'] = 1.0
            pseudo_df['sample_weight'] = pseudo_weight

            # Combine
            self.df = pd.concat([train_df, pseudo_df], ignore_index=True)
            print(f"[OK] Total samples: {len(self.df)} ({len(train_df)} real + {len(pseudo_df)} pseudo)")
        else:
            train_df['sample_weight'] = 1.0
            self.df = train_df

        self.images_dir = images_dir
        self.img_size = img_size
        self.label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

        # Augmentation
        if augment:
            self.transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=20),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                T.ToTensor(),
                T.RandomErasing(p=0.4, scale=(0.02, 0.2)),
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

        # Image path
        if 'source_dir' in row.index and pd.notna(row['source_dir']):
            img_path = os.path.join(row['source_dir'], row['new_filename'])
        else:
            img_path = os.path.join(self.images_dir, row['new_filename'])

        # Load image
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        # Label
        label_vec = row[self.label_cols].values.astype(float)
        label = int(np.argmax(label_vec))

        # Sample weight
        weight = row['sample_weight']

        return image, label, weight

# ============================================================================
# Weighted Focal Loss
# ============================================================================

class WeightedFocalLoss(nn.Module):
    """Focal Loss with per-sample weighting for pseudo-labels"""
    def __init__(self, alpha=[1.0, 1.5, 2.0, 15.0], gamma=3.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target, sample_weights=None):
        num_classes = logits.size(-1)

        # Label smoothing
        target_one_hot = torch.zeros_like(logits)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        target_one_hot = target_one_hot * (1 - self.label_smoothing) + \
                         self.label_smoothing / num_classes

        # Focal loss
        probs = torch.softmax(logits, dim=1)
        ce_loss = -target_one_hot * torch.log(probs + 1e-10)

        # Modulation
        pt = (target_one_hot * probs).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        # Class weights (move alpha to same device as target)
        alpha_t = self.alpha.to(target.device)[target]

        loss = alpha_t * focal_weight * ce_loss.sum(dim=1)

        # Apply sample weights (for pseudo-labels)
        if sample_weights is not None:
            loss = loss * sample_weights

        return loss.mean()

# ============================================================================
# Model Builder
# ============================================================================

def build_model(model_name, num_classes=4, pretrained=True):
    """Build large models with timm"""

    print(f"\n[Model] Building {model_name}...")

    if model_name == 'convnext_large':
        model = timm.create_model('convnext_large', pretrained=pretrained, num_classes=num_classes)
        print(f"[Model] ConvNeXt-Large loaded (~200M params)")

    elif model_name == 'vit_large':
        model = timm.create_model('vit_large_patch16_384', pretrained=pretrained,
                                   num_classes=num_classes, img_size=384)
        print(f"[Model] ViT-Large loaded (~307M params)")

    elif model_name == 'maxvit_large':
        model = timm.create_model('maxvit_large_tf_384', pretrained=pretrained, num_classes=num_classes)
        print(f"[Model] MaxViT-Large loaded (~212M params)")

    elif model_name == 'beit_large':
        model = timm.create_model('beit_large_patch16_384', pretrained=pretrained, num_classes=num_classes)
        print(f"[Model] BEiT-Large loaded (~307M params)")

    elif model_name == 'coatnet':
        model = timm.create_model('coatnet_3_rw_224', pretrained=pretrained, num_classes=num_classes)
        print(f"[Model] CoAtNet-3 loaded (~168M params)")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

# ============================================================================
# Training Function
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, scheduler=None, scaler=None, use_channels_last=True):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for images, labels, weights in loader:
        # Move to GPU with channels_last format (if enabled)
        if use_channels_last:
            images = images.to(device, memory_format=torch.channels_last)
        else:
            images = images.to(device)
        labels = labels.to(device)
        weights = weights.to(device)

        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels, weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels, weights)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(loader), f1

def validate(model, loader, criterion, device, use_channels_last=True):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels, weights in loader:
            # Move to GPU with channels_last format (if enabled)
            if use_channels_last:
                images = images.to(device, memory_format=torch.channels_last)
            else:
                images = images.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            # Use AMP for validation too
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels, weights)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(loader), f1

# ============================================================================
# Main Training
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['convnext_large', 'vit_large', 'maxvit_large',
                                'beit_large', 'coatnet'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--pseudo_weight', type=float, default=0.5)
    args = parser.parse_args()

    print("="*80)
    print(f"Training {args.model.upper()} - Fold {args.fold}")
    print("="*80)
    print(f"  Image size: {args.img_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Pseudo-label weight: {args.pseudo_weight}")
    print("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Datasets
    train_csv = f'data/kfold_splits/fold{args.fold}_train.csv'
    val_csv = f'data/kfold_splits/fold{args.fold}_val.csv'
    pseudo_csv = 'data/pseudo_labels_for_training_0.80.csv'

    print(f"\nLoading data:")
    print(f"  Train: {train_csv}")
    print(f"  Val: {val_csv}")
    print(f"  Pseudo: {pseudo_csv}")

    train_dataset = CXRDatasetWithPseudo(
        train_csv, pseudo_csv=pseudo_csv,
        img_size=args.img_size, augment=True,
        pseudo_weight=args.pseudo_weight
    )
    val_dataset = CXRDatasetWithPseudo(
        val_csv, pseudo_csv=None,
        img_size=args.img_size, augment=False
    )

    # Optimize DataLoader for GPU utilization
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=8, pin_memory=True,
                             persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2,
                           shuffle=False, num_workers=8, pin_memory=True,
                           persistent_workers=True, prefetch_factor=4)

    # Model with GPU optimizations
    model = build_model(args.model, num_classes=4, pretrained=True)
    model = model.to(device)

    # Enable mixed precision training
    torch.backends.cudnn.benchmark = True

    # Use channels_last for better GPU utilization
    # Skip for MaxViT and CoAtNet which have compatibility issues
    use_channels_last = args.model not in ['maxvit_large', 'coatnet']
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("Channels last: ENABLED")
    else:
        print("Channels last: DISABLED (model incompatible)")

    # Compile model for faster execution (PyTorch 2.0+)
    # Skip compilation for MaxViT and CoAtNet due to compatibility issues
    if hasattr(torch, 'compile') and args.model not in ['maxvit_large', 'coatnet']:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("[OK] Model compiled with torch.compile() (reduce-overhead mode)")
        except:
            print("[WARNING] torch.compile() not available, skipping")
    else:
        if args.model in ['maxvit_large', 'coatnet']:
            print("[INFO] Skipping torch.compile() for model compatibility")

    # Enable TF32 for faster training on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Loss & Optimizer
    criterion = WeightedFocalLoss(
        alpha=[1.0, 1.5, 2.0, 15.0],
        gamma=3.0,
        label_smoothing=0.1
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0002)

    # Cosine scheduler with warmup
    warmup_epochs = 3
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_f1 = 0
    output_dir = Path(f'outputs/champion_{args.model}/fold{args.fold}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training...")
    print(f"   Mixed precision: ENABLED")
    print(f"   Channels last: {'ENABLED' if use_channels_last else 'DISABLED'}")
    print(f"   TF32: ENABLED")
    print(f"   Workers: 8 with prefetch")
    print(f"Output: {output_dir}\n")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_f1 = train_epoch(model, train_loader, criterion,
                                           optimizer, device, scheduler, scaler, use_channels_last)
        val_loss, val_f1 = validate(model, val_loader, criterion, device, use_channels_last)

        print(f"  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
            }, output_dir / 'best.pt')
            print(f"  [BEST] New best F1: {best_f1:.4f}")

        print()

    print("="*80)
    print(f"[DONE] Training complete!")
    print(f"   Best Val F1: {best_f1:.4f}")
    print(f"   Model saved: {output_dir / 'best.pt'}")
    print("="*80)

if __name__ == '__main__':
    main()
