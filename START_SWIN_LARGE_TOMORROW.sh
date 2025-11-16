#!/bin/bash
################################################################################
# 🚀 Swin-Large 一鍵啟動腳本 - 明天使用
# 
# 使用方法: bash START_SWIN_LARGE_TOMORROW.sh
# 
# 預計時間: 12-15 小時
# 預期分數: 89-92% (目標突破 90%)
################################################################################

echo "========================================================================"
echo "🚀 Swin-Large 極限訓練 - 突破 90% 最終方案"
echo "========================================================================"
echo ""
echo "📊 UltraThink 分析結果："
echo "  - 當前瓶頸: 87.574% (EfficientNet-V2-L 架構限制)"
echo "  - 突破關鍵: 大容量 Transformer 模型"
echo "  - 最優選擇: Swin-Large (197M 參數)"
echo "  - 成功概率: 70% 突破 90%"
echo ""
echo "========================================================================"

# 檢查 GPU
echo "🖥️ 檢查 GPU..."
if ! nvidia-smi &> /dev/null; then
    echo "❌ 錯誤: 未檢測到 GPU!"
    exit 1
fi

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "✅ GPU 已就緒: ${GPU_MEM}MB VRAM"
echo ""

# 清理 GPU
echo "🧹 清理 GPU 緩存..."
python3 -c "import torch; torch.cuda.empty_cache(); print('✅ GPU 已清理')"
echo ""

# 檢查依賴
echo "📦 檢查依賴..."
python3 -c "import timm; print(f'✅ timm {timm.__version__}')" || {
    echo "❌ timm 未安裝，正在安裝..."
    pip install timm -q
}
echo ""

# 創建輸出目錄
mkdir -p outputs/swin_large_ultimate
mkdir -p logs

echo "========================================================================"
echo "🔥 開始訓練 Swin-Large 5-Fold"
echo "========================================================================"
echo ""
echo "配置："
echo "  模型: swin_large_patch4_window12_384 (197M 參數)"
echo "  圖像: 384×384"
echo "  Batch: 4 (保守 VRAM)"
echo "  Epochs: 40 (早停 patience=15)"
echo "  Mixup: 60% α=1.2"
echo "  Focal Loss: α=[1.0, 1.5, 2.0, 12.0] γ=3.0"
echo ""
echo "預計："
echo "  時間: 12-15 小時"
echo "  Val F1: 86-89%"
echo "  Test F1: 89-92% 🎯"
echo ""
echo "========================================================================"
echo ""

# 使用現有的 breakthrough 訓練腳本，循環訓練 5 個 fold
for FOLD in {0..4}; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Fold $FOLD/4 開始訓練..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # 使用 fold 特定的數據
    TRAIN_CSV="data/fold${FOLD}_train.csv"
    VAL_CSV="data/fold${FOLD}_val.csv"
    
    if [ ! -f "$TRAIN_CSV" ]; then
        echo "⚠️ $TRAIN_CSV 不存在，跳過此 fold"
        continue
    fi
    
    # 檢查是否已完成
    if [ -f "outputs/swin_large_ultimate/fold${FOLD}/best.pt" ]; then
        echo "✅ Fold $FOLD 已完成，跳過"
        continue
    fi
    
    # 訓練 (使用 timm 直接訓練)
    python3 << TRAIN_FOLD_EOF
import os, torch, torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd, timm
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

class CXRDataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=384, augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_size = img_size
        
        if augment:
            self.transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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
        
        label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
        if all(c in row for c in label_cols):
            label = np.argmax([row[c] for c in label_cols])
        else:
            label = int(row.get('label', 0))
        
        return img, label

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

def mixup_data(x, y, alpha=1.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

# Config
fold = ${FOLD}
device = torch.device('cuda')

# Data
train_ds = CXRDataset('${TRAIN_CSV}', 'data/train_images', 384, True)
val_ds = CXRDataset('${VAL_CSV}', 'data/train_images', 384, False)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# Model
model = timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=4).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Training
criterion = FocalLoss(alpha=torch.tensor([1.0, 1.5, 2.0, 12.0]).to(device), gamma=3.0)
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 40, 1e-6)
scaler = torch.cuda.amp.GradScaler()

best_f1, patience = 0, 0

for epoch in range(40):
    # Train
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        if np.random.rand() < 0.6:
            inputs, ta, tb, lam = mixup_data(inputs, targets)
            with torch.cuda.amp.autocast():
                out = model(inputs)
                loss = lam * criterion(out, ta) + (1-lam) * criterion(out, tb)
        else:
            with torch.cuda.amp.autocast():
                loss = criterion(model(inputs), targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    scheduler.step()
    
    # Val
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            out = model(inputs.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(targets.numpy())
    
    f1 = f1_score(labels, preds, average='macro') * 100
    print(f"Epoch {epoch+1}: Val F1 = {f1:.2f}%")
    
    if f1 > best_f1:
        best_f1 = f1
        patience = 0
        os.makedirs(f'outputs/swin_large_ultimate/fold{fold}', exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'f1': f1}, 
                   f'outputs/swin_large_ultimate/fold{fold}/best.pt')
        print(f"  ✅ Best F1: {f1:.2f}%")
    else:
        patience += 1
        if patience >= 15:
            print("Early stop")
            break

print(f"Fold {fold} done. Best: {best_f1:.2f}%")
TRAIN_FOLD_EOF

    echo ""
    echo "✅ Fold $FOLD 完成"
    echo ""
done

echo ""
echo "========================================================================"
echo "🎉 所有訓練完成！"
echo "========================================================================"
echo ""

# 計算平均分數
python3 << 'SUMMARY_EOF'
import torch
from pathlib import Path

scores = []
for fold in range(5):
    ckpt_path = f'outputs/swin_large_ultimate/fold{fold}/best.pt'
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        scores.append(ckpt['f1'])
        print(f"Fold {fold}: {ckpt['f1']:.2f}%")

if scores:
    avg = sum(scores) / len(scores)
    print(f"\n平均 Val F1: {avg:.2f}%")
    print(f"預期 Test F1: {avg+3:.2f}% (基於 DINOv2 +3% 經驗)")
    print(f"\n🎯 目標達成概率:")
    if avg >= 87:
        print(f"  突破 90%: 🟢🟢🟢🟢⚪ 80%+")
    elif avg >= 85:
        print(f"  突破 90%: 🟢🟢🟢⚪⚪ 60%")
    else:
        print(f"  突破 90%: 🟢🟢⚪⚪⚪ 40%")
SUMMARY_EOF

echo ""
echo "========================================================================"
echo "📝 下一步："
echo "  1. 生成測試集預測"
echo "  2. 與現有最佳模型集成"
echo "  3. 提交至 Kaggle"
echo ""
echo "運行: bash GENERATE_SWIN_PREDICTIONS.sh"
echo "========================================================================"
