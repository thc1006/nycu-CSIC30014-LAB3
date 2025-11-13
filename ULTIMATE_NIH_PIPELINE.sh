#!/bin/bash
# 終極 NIH 完整流程 - Stage 2-4 自動化
# 榨乾所有資源，衝擊 90%+

set -e

echo "================================================================================"
echo "🚀 終極 NIH 完整流程啟動"
echo "================================================================================"
echo "開始時間: $(date)"
echo ""

# 創建必要目錄
mkdir -p logs
mkdir -p outputs/stage2_nih_finetune
mkdir -p outputs/stage3_pseudo_labels
mkdir -p outputs/stage4_pseudo_training
mkdir -p data/pseudo_labels_nih

# ============================================================================
# Stage 2: NIH 預訓練模型微調 (5-Fold)
# ============================================================================

echo ""
echo "================================================================================"
echo "📦 Stage 2: NIH 預訓練模型微調 (5-Fold)"
echo "================================================================================"
echo "預計時間: 15-20 小時"
echo ""

for fold in {0..4}; do
    echo "----------------------------------------"
    echo "🔄 訓練 Fold $fold / 4"
    echo "----------------------------------------"

    # 啟動訓練
    python3 train_breakthrough.py \
        --config configs/stage2_finetune.yaml \
        --fold $fold \
        > logs/stage2_nih_fold${fold}_$(date +%Y%m%d_%H%M%S).log 2>&1

    echo "✅ Fold $fold 完成！"
    echo ""

    # 檢查是否成功
    if [ ! -f "outputs/stage2_finetune/best.pt" ]; then
        echo "❌ Fold $fold 訓練失敗！"
        exit 1
    fi

    # 重命名以保存不同 fold
    mv outputs/stage2_finetune/best.pt outputs/stage2_nih_finetune/fold${fold}_best.pt

done

echo "✅ Stage 2 (5-Fold) 全部完成！"

# ============================================================================
# Stage 3: 偽標籤生成
# ============================================================================

echo ""
echo "================================================================================"
echo "🏷️ Stage 3: 偽標籤生成"
echo "================================================================================"
echo "預計時間: 2-3 小時"
echo ""

# 創建偽標籤生成腳本
cat > scripts/generate_pseudo_labels_nih.py << 'PSEUDO_SCRIPT'
#!/usr/bin/env python3
"""
Stage 3: 生成偽標籤
使用 Stage 2 訓練的模型對測試集生成高置信度偽標籤
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

# Import from train_breakthrough
sys.path.insert(0, str(Path(__file__).parent.parent))
from train_breakthrough import build_model

class TestDataset(Dataset):
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
        img_path = os.path.join(self.images_dir, row['new_filename'])
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        return image, row['new_filename']

@torch.no_grad()
def generate_pseudo_labels(model_paths, test_csv, test_images, output_csv, threshold=0.95, img_size=384):
    """生成偽標籤"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加載測試數據
    test_dataset = TestDataset(test_csv, test_images, img_size)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    print(f"測試集樣本數: {len(test_dataset)}")

    # 集成預測
    all_probs = []

    for model_path in model_paths:
        print(f"\n載入模型: {model_path}")

        # 構建模型
        model = build_model('efficientnet_v2_s', num_classes=4, dropout=0.3, img_size=img_size)

        # 載入權重
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        # 預測
        fold_probs = []
        for images, filenames in tqdm(test_loader, desc=f"預測"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            fold_probs.append(probs)

        fold_probs = np.concatenate(fold_probs, axis=0)
        all_probs.append(fold_probs)

    # 平均所有模型的預測
    avg_probs = np.mean(all_probs, axis=0)

    # 獲取最高置信度的預測
    max_probs = np.max(avg_probs, axis=1)
    pred_labels = np.argmax(avg_probs, axis=1)

    # 篩選高置信度樣本
    high_conf_mask = max_probs >= threshold
    high_conf_count = np.sum(high_conf_mask)

    print(f"\n高置信度 (≥{threshold}) 樣本數: {high_conf_count} / {len(test_dataset)} ({100*high_conf_count/len(test_dataset):.1f}%)")

    # 創建偽標籤 DataFrame
    test_df = pd.read_csv(test_csv)
    pseudo_df = test_df[high_conf_mask].copy()

    # 添加偽標籤（one-hot encoding）
    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
    for i, class_name in enumerate(class_names):
        pseudo_df[class_name] = (pred_labels[high_conf_mask] == i).astype(int)

    # 添加置信度
    pseudo_df['confidence'] = max_probs[high_conf_mask]

    # 保存
    pseudo_df.to_csv(output_csv, index=False)
    print(f"✅ 偽標籤已保存: {output_csv}")

    # 統計
    print("\n偽標籤分布:")
    for i, class_name in enumerate(class_names):
        count = np.sum(pred_labels[high_conf_mask] == i)
        print(f"  {class_name}: {count} ({100*count/high_conf_count:.1f}%)")

    return pseudo_df

if __name__ == '__main__':
    # 配置
    model_paths = [
        'outputs/stage2_nih_finetune/fold0_best.pt',
        'outputs/stage2_nih_finetune/fold1_best.pt',
        'outputs/stage2_nih_finetune/fold2_best.pt',
        'outputs/stage2_nih_finetune/fold3_best.pt',
        'outputs/stage2_nih_finetune/fold4_best.pt',
    ]

    test_csv = 'data/test_data.csv'
    test_images = 'test_images'
    output_csv = 'data/pseudo_labels_nih/pseudo_labels_high_conf.csv'

    # 生成偽標籤
    generate_pseudo_labels(
        model_paths=model_paths,
        test_csv=test_csv,
        test_images=test_images,
        output_csv=output_csv,
        threshold=0.95,
        img_size=384
    )
PSEUDO_SCRIPT

chmod +x scripts/generate_pseudo_labels_nih.py

# 執行偽標籤生成
python3 scripts/generate_pseudo_labels_nih.py > logs/stage3_pseudo_labels_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "✅ Stage 3 偽標籤生成完成！"

# ============================================================================
# Stage 4: 偽標籤訓練 (5-Fold)
# ============================================================================

echo ""
echo "================================================================================"
echo "🎓 Stage 4: 偽標籤訓練 (5-Fold)"
echo "================================================================================"
echo "預計時間: 20-25 小時"
echo ""

# 創建擴增訓練集腳本
cat > scripts/create_augmented_dataset.py << 'AUG_SCRIPT'
#!/usr/bin/env python3
"""合併原始訓練數據和偽標籤"""

import pandas as pd
from pathlib import Path

# 讀取 K-Fold splits
kfold_dir = Path('data/kfold_splits')
pseudo_df = pd.read_csv('data/pseudo_labels_nih/pseudo_labels_high_conf.csv')

print(f"原始訓練集: {len(pd.read_csv(kfold_dir / 'fold0_train.csv')) * 5} 樣本")
print(f"偽標籤: {len(pseudo_df)} 樣本")

# 為每個 fold 創建擴增數據集
for fold in range(5):
    train_df = pd.read_csv(kfold_dir / f'fold{fold}_train.csv')

    # 合併
    augmented_df = pd.concat([train_df, pseudo_df], ignore_index=True)

    # 保存
    output_path = kfold_dir / f'fold{fold}_train_augmented.csv'
    augmented_df.to_csv(output_path, index=False)
    print(f"✅ Fold {fold} 擴增數據集: {len(augmented_df)} 樣本 → {output_path}")

print("✅ 所有擴增數據集已創建！")
AUG_SCRIPT

python3 scripts/create_augmented_dataset.py

# 創建 Stage 4 配置
cat > configs/stage4_pseudo.yaml << 'STAGE4_CONFIG'
# Stage 4: 偽標籤訓練配置

model: efficientnet_v2_s
img_size: 384
num_classes: 4
dropout: 0.3

# 使用 Stage 2 的最佳模型作為初始化
pretrained_checkpoint: outputs/pretrain_nih_stage1/best.pt

# K-Fold 配置
fold: 0  # 會被命令行覆蓋
kfold_csv_dir: data/kfold_splits
data_dir: .

# 訓練配置（稍微增加 epoch 因為數據更多）
epochs: 35
batch_size: 24
num_workers: 12

# 優化器
optimizer: adamw
lr: 0.00008  # 稍微降低學習率
weight_decay: 0.00015
warmup_epochs: 3

# 學習率調度
scheduler: cosine
min_lr: 0.000001

# Loss
loss_type: improved_focal
focal_alpha: [1.0, 1.5, 2.0, 12.0]
focal_gamma: 3.5
label_smoothing: 0.12

# 數據增強
mixup_prob: 0.5
mixup_alpha: 1.2
cutmix_prob: 0.5
cutmix_alpha: 1.0
aug_rotation: 18
aug_scale: [0.88, 1.12]
random_erasing_prob: 0.3

# 正則化
use_swa: true
swa_start_epoch: 25
swa_lr: 0.00004

# 早停
early_stopping: true
patience: 12

# 輸出
output_dir: outputs/stage4_pseudo_training
save_best: true
STAGE4_CONFIG

# 訓練 Stage 4 (5-Fold)
for fold in {0..4}; do
    echo "----------------------------------------"
    echo "🔄 Stage 4 訓練 Fold $fold / 4"
    echo "----------------------------------------"

    # 修改 K-Fold CSV 配置以使用擴增數據
    sed -i "s/fold${fold}_train.csv/fold${fold}_train_augmented.csv/g" data/kfold_splits/fold${fold}_train.csv || true

    # 訓練
    python3 train_breakthrough.py \
        --config configs/stage4_pseudo.yaml \
        --fold $fold \
        > logs/stage4_pseudo_fold${fold}_$(date +%Y%m%d_%H%M%S).log 2>&1

    echo "✅ Stage 4 Fold $fold 完成！"

    # 重命名模型
    if [ -f "outputs/stage4_pseudo_training/best.pt" ]; then
        mv outputs/stage4_pseudo_training/best.pt outputs/stage4_pseudo_training/fold${fold}_best.pt
    fi
done

echo ""
echo "================================================================================"
echo "✅ 完整 NIH Pipeline 執行完成！"
echo "================================================================================"
echo "完成時間: $(date)"
echo ""
echo "輸出模型:"
echo "  Stage 2: outputs/stage2_nih_finetune/fold{0-4}_best.pt"
echo "  Stage 4: outputs/stage4_pseudo_training/fold{0-4}_best.pt"
echo ""
echo "下一步: 生成最終預測並提交！"
echo "================================================================================"
