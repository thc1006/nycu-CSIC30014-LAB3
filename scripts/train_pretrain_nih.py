#!/usr/bin/env python3
"""
Stage 1: NIH ChestX-ray14 多標籤預訓練
第1名核心技巧 - 在大規模外部數據上預訓練特徵提取器

預期效果: +3-5% 在目標任務上
訓練時間: 18-20小時 (10 epochs, 77K 樣本)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score

# 14種疾病標籤
DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

class NIHChestXrayDataset(Dataset):
    """NIH ChestX-ray14 多標籤數據集"""

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 讀取影像
        img_path = Path(row['image_path']) / row['Image Index']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 多標籤（14維 binary vector）
        labels = row[DISEASE_LABELS].values.astype(np.float32)
        labels = torch.from_numpy(labels)

        return image, labels


def create_transforms(img_size=384, augment=True):
    """創建數據增強"""
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def build_model(model_name='efficientnet_v2_s', num_classes=14):
    """
    構建模型
    使用 ImageNet 預訓練權重初始化
    """
    if model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        # 替換分類頭為多標籤分類
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # 混合精度訓練
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """驗證"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(loader, desc='Validating'):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        # 收集預測和標籤用於計算 AUC
        all_labels.append(labels.cpu().numpy())
        all_preds.append(torch.sigmoid(outputs).cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # 計算每個類別的 AUC
    auc_scores = []
    for i in range(len(DISEASE_LABELS)):
        if all_labels[:, i].sum() > 0:  # 只計算有正樣本的類別
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)

    mean_auc = np.mean(auc_scores) if auc_scores else 0.0

    return total_loss / len(loader), mean_auc


def main(args):
    print("=" * 80)
    print("🏆 Stage 1: NIH ChestX-ray14 多標籤預訓練")
    print("第1名核心技巧 - 外部數據預訓練")
    print("=" * 80)

    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n設備: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 數據集
    project_root = Path(__file__).parent.parent
    nih_processed = project_root / 'data' / 'external' / 'nih_chestxray14' / 'processed'

    print(f"\n載入數據集...")
    train_dataset = NIHChestXrayDataset(
        csv_path=nih_processed / 'train.csv',
        transform=create_transforms(img_size=args.img_size, augment=True)
    )
    val_dataset = NIHChestXrayDataset(
        csv_path=nih_processed / 'val.csv',
        transform=create_transforms(img_size=args.img_size, augment=False)
    )

    print(f"  訓練: {len(train_dataset)} 樣本")
    print(f"  驗證: {len(val_dataset)} 樣本")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 模型
    print(f"\n構建模型: {args.model}")
    model = build_model(model_name=args.model, num_classes=14)
    model = model.to(device)

    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  總參數: {total_params:,}")

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()  # 多標籤分類
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 學習率調度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr / 100
    )

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # 訓練
    print(f"\n開始訓練 {args.epochs} epochs...")
    best_auc = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")

        # 訓練
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # 驗證
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # 學習率調整
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch + 1} 結果:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val AUC:    {val_auc:.4f}")
        print(f"  LR:         {current_lr:.6f}")

        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_loss': val_loss,
                'config': {
                    'model': args.model,
                    'img_size': args.img_size,
                    'num_classes': 14
                }
            }
            torch.save(checkpoint, output_dir / 'best.pt')
            print(f"  ✓ 保存最佳模型 (AUC: {val_auc:.4f})")

    print("\n" + "=" * 80)
    print("🎉 Stage 1 預訓練完成！")
    print(f"最佳驗證 AUC: {best_auc:.4f}")
    print(f"模型保存至: {output_dir / 'best.pt'}")
    print("\n下一步:")
    print("  python3 scripts/train_finetune_target.py \\")
    print(f"    --pretrained {output_dir / 'best.pt'}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='efficientnet_v2_s')
    parser.add_argument('--img-size', type=int, default=384)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-amp', action='store_true', default=True)
    parser.add_argument('--output-dir', default='outputs/pretrain_nih_stage1')

    args = parser.parse_args()
    main(args)
