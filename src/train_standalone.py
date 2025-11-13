#!/usr/bin/env python3
"""
Standalone training script (no relative imports)
"""
import os, sys, math, argparse, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import f1_score
from torchvision import models

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
import data
import losses
import aug
import utils

def build_model(name: str, num_classes: int):
    """Build model"""
    if name == "efficientnet_v2_l":
        m = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name.startswith('dinov2_'):
        try:
            from transformers import Dinov2ForImageClassification
        except ImportError:
            raise ImportError("transformers required for DINOv2")
        dinov2_models = {
            'dinov2_small': 'facebook/dinov2-small',
            'dinov2_base': 'facebook/dinov2-base',
            'dinov2_large': 'facebook/dinov2-large',
        }
        if name not in dinov2_models:
            raise ValueError(f"Unknown DINOv2: {name}")
        model_name = dinov2_models[name]
        m = Dinov2ForImageClassification.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif name.startswith('swin_'):
        try:
            import timm
        except ImportError:
            raise ImportError("timm required for Swin")
        m = timm.create_model(name, pretrained=True, num_classes=num_classes)
    elif name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m

def train_model(config_path):
    """Train model with given config"""
    cfg = utils.load_config(config_path)
    utils.seed_everything(cfg.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model
    model = build_model(cfg['model'], cfg.get('num_classes', 4))
    model = model.to(device)

    # Data loaders
    train_loader = data.make_loader(
        csv_path=cfg.get('data_csv', 'data/train.csv'),
        img_dir=cfg.get('data_dir', 'data/train'),
        split='train',
        batch_size=cfg['batch_size'],
        img_size=cfg['img_size'],
        augment=True,
        num_workers=cfg.get('num_workers', 4)
    )

    val_loader = data.make_loader(
        csv_path=cfg.get('data_csv', 'data/train.csv'),
        img_dir=cfg.get('data_dir', 'data/train'),
        split='val',
        batch_size=cfg['batch_size'],
        img_size=cfg['img_size'],
        augment=False,
        num_workers=cfg.get('num_workers', 4)
    )

    # Loss function
    loss_type = cfg.get('loss', 'crossentropy')
    if loss_type == 'improved_focal':
        criterion = losses.ImprovedFocalLoss(
            alpha=cfg.get('focal_alpha', [1.0, 1.5, 2.0, 12.0]),
            gamma=cfg.get('focal_gamma', 3.5),
            label_smoothing=cfg.get('label_smoothing', 0.12)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg.get('weight_decay', 0.0001)
    )

    # Scheduler
    total_steps = len(train_loader) * cfg['epochs']
    warmup_steps = len(train_loader) * cfg.get('warmup_epochs', 3)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_f1 = 0.0
    output_dir = Path(cfg.get('output_dir', 'outputs/model'))
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler() if cfg.get('mixed_precision', True) else None

    for epoch in range(cfg['epochs']):
        # Train
        model.train()
        train_loss = 0.0

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    if hasattr(outputs, 'logits'):  # For transformers
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
            train_loss += loss.item()

        # Validate
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.numpy())

        val_f1 = f1_score(val_targets, val_preds, average='macro')
        avg_train_loss = train_loss / len(train_loader)

        print(f"[Epoch {epoch+1}/{cfg['epochs']}] "
              f"Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")

        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'config': cfg
            }, output_dir / 'best.pt')
            print(f"  → Saved best model (F1: {val_f1:.4f})")

    print(f"\nTraining completed! Best Val F1: {best_f1:.4f}")
    return best_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config YAML path')
    args = parser.parse_args()

    train_model(args.config)
