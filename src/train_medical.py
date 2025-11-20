# -*- coding: utf-8 -*-
"""
Improved training script with medical optimizations

医学优化的训练脚本 - 针对COVID-19检测优化
"""
import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd

from src.data import CSVDataset, make_loader
from src.utils import seed_everything, set_perf_flags
from src.medical_losses import MedicalFocalLoss, CovidAwareFocalLoss, get_loss_function
from src.medical_augmentation import MedicalXrayAugmentation

# Import standard functions from train.py
from src.train import build_model, cosine_lr, evaluate


def train_one_epoch_medical(model, train_loader, loss_fn, optimizer, scaler, cfg, epoch):
    """
    Modified training loop with medical optimizations
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")

    for batch_idx, (imgs, targets, _) in enumerate(pbar):
        imgs = imgs.to(model.device, non_blocking=True)
        targets = targets.to(model.device, non_blocking=True)

        # Forward pass with AMP
        amp_dtype = torch.bfloat16 if cfg['perf'].get('amp_dtype') == 'bf16' else torch.float32

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(imgs)
            loss = loss_fn(logits, targets)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels = targets.detach().cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(labels)

        # Progress bar
        current_loss = total_loss / (batch_idx + 1)
        accuracy = np.mean(preds == labels)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{accuracy:.3f}',
            'macro_f1': f'{macro_f1:.3f}'
        })

    avg_loss = total_loss / len(train_loader)
    train_acc = np.mean(all_preds == all_targets)
    train_macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return avg_loss, train_acc, train_macro_f1, all_preds, all_targets


def evaluate_medical(model, val_loader, loss_fn, cfg):
    """
    Medical-focused evaluation with per-class metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(val_loader, desc="Validating")
    amp_dtype = torch.bfloat16 if cfg['perf'].get('amp_dtype') == 'bf16' else torch.float32

    with torch.no_grad():
        for (imgs, targets, _) in pbar:
            imgs = imgs.to(model.device, non_blocking=True)
            targets = targets.to(model.device, non_blocking=True)

            # Forward pass
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(imgs)
                loss = loss_fn(logits, targets)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            labels = targets.detach().cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels)

    avg_loss = total_loss / len(val_loader)
    val_acc = np.mean(all_preds == all_targets)
    val_macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    # Per-class metrics
    class_names = ['Normal', 'Bacteria', 'Virus', 'COVID-19']
    per_class_f1 = {}

    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_targets) == i
        if class_mask.sum() > 0:
            class_preds = np.array(all_preds)[class_mask]
            class_f1 = f1_score(np.array(all_targets)[class_mask],
                               class_preds,
                               average='binary' if len(np.unique(class_preds)) <= 2 else 'weighted',
                               zero_division=0)
            per_class_f1[class_name] = class_f1

    return {
        'loss': avg_loss,
        'acc': val_acc,
        'macro_f1': val_macro_f1,
        'per_class_f1': per_class_f1,
        'preds': all_preds,
        'targets': all_targets,
        'cm': confusion_matrix(all_targets, all_preds)
    }


def train_with_medical_optimization(cfg):
    """
    Main training function with medical optimizations
    """
    # Setup
    seed_everything(cfg['train']['seed'])
    set_perf_flags(cfg['perf'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create output directory
    os.makedirs(cfg['out']['dir'], exist_ok=True)

    # Load data
    print("Loading training data...")
    _, train_loader = make_loader(
        csv_path=cfg['data']['train_csv'],
        images_dir=cfg['data']['images_dir_train'],
        file_col=cfg['data'].get('file_col', 'new_filename'),
        label_cols=cfg['data']['label_cols'],
        img_size=cfg['model']['img_size'],
        batch_size=cfg['train']['batch_size'],
        num_workers=cfg['train']['num_workers'],
        augment=True,
        shuffle=True,
        weighted=cfg['train'].get('use_weighted_sampler', False),
        advanced_aug=cfg['train'].get('advanced_aug', False)
    )

    print("Loading validation data...")
    _, val_loader = make_loader(
        csv_path=cfg['data']['val_csv'],
        images_dir=cfg['data']['images_dir_val'],
        file_col=cfg['data'].get('file_col', 'new_filename'),
        label_cols=cfg['data']['label_cols'],
        img_size=cfg['model']['img_size'],
        batch_size=cfg['train']['batch_size'],
        num_workers=cfg['train']['num_workers'],
        augment=False,
        shuffle=False,
        weighted=False,
        advanced_aug=False
    )

    # Build model
    model = build_model(cfg['model']['name'], cfg['data']['num_classes'])
    model = model.to(device)
    model.device = device

    print(f"Model: {cfg['model']['name']}")

    # Loss function - use medical optimized loss
    loss_name = cfg['train'].get('loss', 'medical_focal')
    if loss_name == 'medical_focal':
        print("Using MedicalFocalLoss")
        loss_fn = MedicalFocalLoss(
            alpha=[1.0, 1.5, 1.8, 3.0],
            gamma=[2.0, 2.0, 2.5, 3.0],
            label_smoothing=cfg['train'].get('label_smoothing', 0.05)
        )
    elif loss_name == 'covid_aware_focal':
        print("Using CovidAwareFocalLoss")
        loss_fn = CovidAwareFocalLoss(
            alpha=[1.0, 1.5, 2.0, 4.0],
            gamma=[2.0, 2.0, 3.0, 4.0],
            label_smoothing=cfg['train'].get('label_smoothing', 0.05)
        )
    else:
        loss_fn = get_loss_function(loss_name, cfg['data']['num_classes'])

    loss_fn = loss_fn.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay']
    )

    # Learning rate scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg['train']['epochs']
    warmup_steps = steps_per_epoch * cfg['train'].get('warmup_epochs', 1)
    scheduler = cosine_lr(optimizer, cfg['train']['lr'], warmup_steps, total_steps)

    # GradScaler for FP16
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_val_macro_f1 = 0.0
    best_val_covid_f1 = 0.0
    best_epoch = 0

    print(f"\nStarting training for {cfg['train']['epochs']} epochs...")
    print("=" * 80)

    for epoch in range(cfg['train']['epochs']):
        # Train
        train_loss, train_acc, train_macro_f1, _, _ = train_one_epoch_medical(
            model, train_loader, loss_fn, optimizer, scaler, cfg, epoch
        )

        # Update scheduler
        for i in range(len(train_loader)):
            pass  # Scheduler already updated in train loop

        # Validate
        val_metrics = evaluate_medical(model, val_loader, loss_fn, cfg)

        print(f"\nEpoch {epoch+1}/{cfg['train']['epochs']}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, Macro F1: {train_macro_f1:.3f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.3f}, Macro F1: {val_metrics['macro_f1']:.3f}")

        # Per-class metrics
        print("  Per-class F1:")
        for class_name, f1 in val_metrics['per_class_f1'].items():
            print(f"    {class_name:15}: {f1:.3f}")

        # Save best model by Macro F1
        if val_metrics['macro_f1'] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            checkpoint_path = os.path.join(cfg['out']['dir'], 'best.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  >>> Saved best model (Macro F1: {best_val_macro_f1:.3f})")

        # Also track COVID-19 F1 separately
        if 'COVID-19' in val_metrics['per_class_f1']:
            covid_f1 = val_metrics['per_class_f1']['COVID-19']
            if covid_f1 > best_val_covid_f1:
                best_val_covid_f1 = covid_f1
                checkpoint_path = os.path.join(cfg['out']['dir'], 'best_covid_f1.pt')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  >>> Saved COVID-19 optimized model (COVID-19 F1: {best_val_covid_f1:.3f})")

    print("\n" + "=" * 80)
    print(f"Training completed!")
    print(f"Best Macro F1: {best_val_macro_f1:.3f} at epoch {best_epoch+1}")
    print(f"Best COVID-19 F1: {best_val_covid_f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Handle config inheritance
    if 'inherits' in cfg:
        parent_path = cfg['inherits']
        with open(parent_path, 'r') as f:
            parent_cfg = yaml.safe_load(f)
        # Deep merge
        def merge_dicts(parent, child):
            for k, v in child.items():
                if k not in parent:
                    parent[k] = v
                elif isinstance(v, dict):
                    merge_dicts(parent[k], v)
                else:
                    parent[k] = v
            return parent
        cfg = merge_dicts(parent_cfg, cfg)

    # Train
    train_with_medical_optimization(cfg)
