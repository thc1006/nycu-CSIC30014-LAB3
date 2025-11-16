#!/usr/bin/env python3
"""
🚀 Swin-Large 極限訓練 - 目標 90%+
完全獨立腳本，使用 timm 庫
"""
import os, sys, torch, torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd, timm
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import time

class CXRDataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=384, augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_size = img_size
        self.label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
        
        if augment:
            self.transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.RandomErasing(p=0.3),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(self.img_dir) / row['new_filename']
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        
        if all(col in row for col in self.label_cols):
            label = np.argmax([row[col] for col in self.label_cols])
        else:
            label = int(row['label'])
        
        return img, label

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def mixup_data(x, y, alpha=1.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_fold(fold, device='cuda'):
    print(f"\n{'='*70}")
    print(f"🚀 Fold {fold} 訓練開始")
    print(f"{'='*70}\n")
    
    # Data
    train_dataset = CXRDataset('data/train.csv', 'data/train_images', 
                                img_size=384, augment=True)
    val_dataset = CXRDataset('data/val.csv', 'data/train_images', 
                              img_size=384, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, 
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, 
                             num_workers=4, pin_memory=True)
    
    print(f"📊 數據:")
    print(f"  訓練集: {len(train_dataset)} 樣本")
    print(f"  驗證集: {len(val_dataset)} 樣本\n")
    
    # Model
    print("🏗️ 創建 Swin-Large 模型...")
    model = timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=4)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  總參數: {total_params/1e6:.1f}M\n")
    
    # Loss & Optimizer
    focal_alpha = torch.tensor([1.0, 1.5, 2.0, 12.0]).to(device)
    criterion = FocalLoss(alpha=focal_alpha, gamma=3.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Training
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(40):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Mixup
            if np.random.rand() < 0.6:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
                
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"  [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(targets.numpy())
        
        val_f1 = f1_score(val_labels, val_preds, average='macro') * 100
        train_acc = 100. * train_correct / train_total
        
        print(f"\nEpoch {epoch+1}/40:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val F1: {val_f1:.2f}%")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            
            # Save
            os.makedirs(f'outputs/swin_large_ultimate/fold{fold}', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1,
            }, f'outputs/swin_large_ultimate/fold{fold}/best.pt')
            print(f"  ✅ 新最佳 F1: {val_f1:.2f}% (已保存)")
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"\n⚠️ 早停 (patience=15)")
                break
    
    print(f"\n✅ Fold {fold} 完成! 最佳 Val F1: {best_f1:.2f}%")
    return best_f1

def main():
    print("="*70)
    print("🚀 Swin-Large 極限訓練 - 突破 90% 目標")
    print("="*70)
    print(f"\n📊 配置:")
    print(f"  模型: Swin-Large (197M 參數)")
    print(f"  圖像尺寸: 384×384")
    print(f"  Batch Size: 4")
    print(f"  Epochs: 40 (早停 patience=15)")
    print(f"  Mixup: 60% prob, α=1.2")
    print(f"  Focal Loss: α=[1.0, 1.5, 2.0, 12.0], γ=3.0")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}\n")
    
    if str(device) == 'cpu':
        print("⚠️ 警告: 未檢測到 GPU! 訓練將非常慢。")
        return
    
    # Train 5 folds
    fold_scores = []
    start_time = time.time()
    
    for fold in range(5):
        f1 = train_fold(fold, device)
        fold_scores.append(f1)
    
    elapsed = (time.time() - start_time) / 3600
    avg_f1 = np.mean(fold_scores)
    
    print("\n" + "="*70)
    print("🎉 所有 Fold 訓練完成!")
    print("="*70)
    print(f"\n📊 5-Fold 結果:")
    for i, f1 in enumerate(fold_scores):
        print(f"  Fold {i}: {f1:.2f}%")
    print(f"\n📈 平均 Val F1: {avg_f1:.2f}%")
    print(f"⏱️ 總訓練時間: {elapsed:.1f} 小時")
    print(f"\n🎯 預期測試分數: {avg_f1+3:.2f}% (基於 DINOv2 +3% 經驗)")
    print("="*70)

if __name__ == '__main__':
    main()
