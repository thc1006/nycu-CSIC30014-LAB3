#!/usr/bin/env python3
"""
🚀 DINOv2 突破訓練 - 方案 A
基於自監督預訓練的 Vision Transformer 實現 90% 目標

DINOv2 優勢:
- 142M 圖片自監督預訓練 (vs ImageNet 1.2M)
- 強大的 few-shot 學習能力
- 文獻支持：醫學影像 +2-4%
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
from sklearn.metrics import f1_score, classification_report
import timm
import math
import argparse
from tqdm import tqdm

# ============================================================================
# Dataset
# ============================================================================

class CXRDataset(Dataset):
    """胸部 X 光數據集"""
    def __init__(self, csv_path, images_dir='data/images', img_size=518, augment=True):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.img_size = img_size
        self.label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

        # DINOv2 推薦的輸入解析度是 518x518
        if augment:
            self.transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.3),  # 醫學影像: 減少水平翻轉
                T.RandomRotation(degrees=15),   # 溫和旋轉
                T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.88, 1.12)),
                T.ColorJitter(brightness=0.25, contrast=0.25),
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

        # Handle source_dir column (train_images/ or data/images/)
        if 'source_dir' in row.index and pd.notna(row['source_dir']):
            img_path = os.path.join(row['source_dir'], row['new_filename'])
        else:
            img_path = os.path.join(self.images_dir, row['new_filename'])

        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        label_vec = row[self.label_cols].values.astype(float)
        label = int(np.argmax(label_vec))

        return image, label

# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss with class weights"""
    def __init__(self, alpha=[1.0, 1.5, 2.0, 15.0], gamma=3.0, label_smoothing=0.08):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        num_classes = logits.size(-1)

        # Label smoothing
        target_one_hot = torch.zeros_like(logits)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        target_one_hot = target_one_hot * (1 - self.label_smoothing) + \
                         self.label_smoothing / num_classes

        # Focal loss
        probs = torch.softmax(logits, dim=1)
        ce_loss = -target_one_hot * torch.log(probs + 1e-10)

        pt = (target_one_hot * probs).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        alpha_t = self.alpha.to(target.device)[target]
        loss = alpha_t * focal_weight * ce_loss.sum(dim=1)

        return loss.mean()

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating', leave=False):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    return f1_macro * 100, f1_per_class * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=16)  # DINOv2 needs smaller batch
    parser.add_argument('--img_size', type=int, default=518)   # DINOv2 optimal size
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--output_dir', type=str, default='outputs/dinov2_breakthrough')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"🚀 DINOv2 突破訓練 - Fold {args.fold}")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.img_size}")
    print(f"Learning Rate: {args.lr}")

    # Output directory
    fold_dir = Path(args.output_dir) / f'fold{args.fold}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_csv = f'data/fold{args.fold}_train.csv'
    val_csv = f'data/fold{args.fold}_val.csv'

    print(f"\nLoading data...")
    train_dataset = CXRDataset(train_csv, img_size=args.img_size, augment=True)
    val_dataset = CXRDataset(val_csv, img_size=args.img_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2,
                           shuffle=False, num_workers=4, pin_memory=True)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Model - DINOv2 Base
    print(f"\nCreating DINOv2 model...")
    model = timm.create_model('vit_base_patch14_dinov2', pretrained=True, num_classes=4)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params/1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params/1e6:.1f}M")

    # Loss & Optimizer
    criterion = FocalLoss(alpha=[1.0, 1.5, 2.0, 15.0], gamma=3.0, label_smoothing=0.08)

    # AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    # Cosine schedule with warmup
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * 3  # 3 epochs warmup

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print(f"\n{'='*70}")
    print(f"Training started...")
    print(f"{'='*70}\n")

    best_f1 = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)

        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device, scheduler)
        val_f1, val_f1_per_class = validate(model, val_loader, device)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val F1 (macro): {val_f1:.2f}%")
        print(f"  Val F1 per class: Normal={val_f1_per_class[0]:.1f}% "
              f"Bacteria={val_f1_per_class[1]:.1f}% "
              f"Virus={val_f1_per_class[2]:.1f}% "
              f"COVID-19={val_f1_per_class[3]:.1f}%")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'f1_per_class': val_f1_per_class,
            }, fold_dir / 'best.pt')
            print(f"  ✅ New best F1: {best_f1:.2f}% (saved)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping triggered!")
                break

    print(f"\n{'='*70}")
    print(f"✅ Fold {args.fold} training complete!")
    print(f"   Best Val F1: {best_f1:.2f}%")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
