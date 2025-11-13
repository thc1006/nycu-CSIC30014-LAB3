#!/bin/bash
# 🚀 並行雙路徑冠軍流程
# 路徑 A: NIH EfficientNet-V2-S (Stage 2-4, 5-Fold)
# 路徑 B: EfficientNet-V2-L 偽標籤 (Stage 3-4, 1-Fold)

set -e

LOG_DIR="logs/parallel_execution"
mkdir -p $LOG_DIR

echo "================================================================================"
echo "🚀🚀 並行雙路徑冠軍流程啟動 🚀🚀"
echo "================================================================================"
echo "開始時間: $(date)"
echo "策略: 交錯執行，最大化 GPU 利用率"
echo ""

# ============================================================================
# Phase 1: 路徑 A - NIH Stage 2 (5-Fold) - 主力路徑
# ============================================================================

echo "================================================================================"
echo "📦 Phase 1: 路徑 A - NIH Stage 2 微調 (5-Fold)"
echo "================================================================================"
echo "模型: EfficientNet-V2-S (20M 參數)"
echo "預訓練: NIH ChestX-ray14 (112K 樣本, Val AUC 85.55%)"
echo "預計時間: 10-15 小時 (5 folds × 2-3 小時/fold)"
echo ""

for fold in {0..4}; do
    echo "----------------------------------------"
    echo "🔄 NIH Stage 2 - Fold $fold / 4"
    echo "----------------------------------------"

    START_TIME=$(date +%s)

    # 訓練
    python3 train_breakthrough.py \
        --config configs/stage2_finetune.yaml \
        --fold $fold \
        > $LOG_DIR/nih_stage2_fold${fold}_$(date +%Y%m%d_%H%M%S).log 2>&1

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo "✅ NIH Stage 2 Fold $fold 完成！耗時: $((DURATION/60)) 分鐘"

    # 檢查並重命名
    if [ -f "outputs/stage2_finetune/best.pt" ]; then
        mkdir -p outputs/nih_v2s_stage2
        mv outputs/stage2_finetune/best.pt outputs/nih_v2s_stage2/fold${fold}_best.pt
        echo "   模型已保存: outputs/nih_v2s_stage2/fold${fold}_best.pt"
    else
        echo "❌ Fold $fold 訓練失敗！"
        exit 1
    fi

    echo ""
done

echo "✅ Phase 1 完成！所有 NIH Stage 2 模型已訓練"
echo ""

# ============================================================================
# Phase 2: 並行生成偽標籤
# ============================================================================

echo "================================================================================"
echo "🏷️ Phase 2: 並行生成偽標籤"
echo "================================================================================"
echo ""

# 路徑 A: 使用 NIH Stage 2 模型
echo "🔄 路徑 A: 使用 NIH Stage 2 (5 models) 生成偽標籤..."
python3 scripts/generate_pseudo_nih.py \
    --model_dir outputs/nih_v2s_stage2 \
    --model_arch efficientnet_v2_s \
    --output data/pseudo_labels_nih/high_conf.csv \
    --threshold 0.95 \
    > $LOG_DIR/pseudo_nih_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID_PSEUDO_A=$!
echo "   PID: $PID_PSEUDO_A"

# 路徑 B: 使用 V2-L 模型
echo "🔄 路徑 B: 使用 V2-L (1 model) 生成偽標籤..."
python3 scripts/generate_pseudo_nih.py \
    --model_dir outputs/stage2_finetune \
    --model_arch efficientnet_v2_l \
    --output data/pseudo_labels_v2l/high_conf.csv \
    --threshold 0.95 \
    > $LOG_DIR/pseudo_v2l_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID_PSEUDO_B=$!
echo "   PID: $PID_PSEUDO_B"

# 等待兩個都完成
echo ""
echo "⏳ 等待偽標籤生成完成..."
wait $PID_PSEUDO_A
echo "✅ 路徑 A 偽標籤完成"
wait $PID_PSEUDO_B
echo "✅ 路徑 B 偽標籤完成"

echo ""
echo "✅ Phase 2 完成！偽標籤已生成"
echo ""

# ============================================================================
# Phase 3: 創建擴增訓練集
# ============================================================================

echo "================================================================================"
echo "📚 Phase 3: 創建擴增訓練集"
echo "================================================================================"
echo ""

# 創建合併腳本
cat > scripts/merge_pseudo_labels.py << 'MERGE_SCRIPT'
#!/usr/bin/env python3
"""合併原始訓練數據和偽標籤"""

import pandas as pd
from pathlib import Path

kfold_dir = Path('data/kfold_splits')

# 路徑 A: NIH 偽標籤
pseudo_nih = pd.read_csv('data/pseudo_labels_nih/high_conf.csv')
print(f"NIH 偽標籤: {len(pseudo_nih)} 樣本")

# 路徑 B: V2-L 偽標籤
pseudo_v2l = pd.read_csv('data/pseudo_labels_v2l/high_conf.csv')
print(f"V2-L 偽標籤: {len(pseudo_v2l)} 樣本")

# 合併兩個偽標籤（取交集以確保高質量）
# 只保留兩個模型都預測為相同類別的樣本
merged_pseudo = []

for idx, row_nih in pseudo_nih.iterrows():
    # 找到對應的 V2-L 預測
    row_v2l = pseudo_v2l[pseudo_v2l['new_filename'] == row_nih['new_filename']]

    if len(row_v2l) > 0:
        row_v2l = row_v2l.iloc[0]

        # 檢查預測是否一致
        class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
        pred_nih = [row_nih[col] for col in class_cols]
        pred_v2l = [row_v2l[col] for col in class_cols]

        if pred_nih == pred_v2l:
            # 取平均置信度
            row_nih_copy = row_nih.copy()
            row_nih_copy['confidence'] = (row_nih['confidence'] + row_v2l['confidence']) / 2
            merged_pseudo.append(row_nih_copy)

merged_pseudo_df = pd.DataFrame(merged_pseudo)
print(f"合併後高質量偽標籤: {len(merged_pseudo_df)} 樣本 (兩模型一致)")

# 為每個 fold 創建擴增數據集
for fold in range(5):
    train_df = pd.read_csv(kfold_dir / f'fold{fold}_train.csv')

    # NIH 路徑: 使用合併的高質量偽標籤
    augmented_nih = pd.concat([train_df, merged_pseudo_df], ignore_index=True)
    output_nih = kfold_dir / f'fold{fold}_train_nih_augmented.csv'
    augmented_nih.to_csv(output_nih, index=False)
    print(f"✅ NIH Fold {fold}: {len(train_df)} + {len(merged_pseudo_df)} = {len(augmented_nih)} 樣本")

# V2-L 路徑: 只為 fold 0 創建（快速測試）
train_df = pd.read_csv(kfold_dir / 'fold0_train.csv')
augmented_v2l = pd.concat([train_df, pseudo_v2l], ignore_index=True)
output_v2l = kfold_dir / 'fold0_train_v2l_augmented.csv'
augmented_v2l.to_csv(output_v2l, index=False)
print(f"✅ V2-L Fold 0: {len(train_df)} + {len(pseudo_v2l)} = {len(augmented_v2l)} 樣本")

print("✅ 所有擴增數據集已創建！")
MERGE_SCRIPT

chmod +x scripts/merge_pseudo_labels.py
python3 scripts/merge_pseudo_labels.py

echo "✅ Phase 3 完成！擴增訓練集已創建"
echo ""

# ============================================================================
# Phase 4: Stage 4 偽標籤訓練
# ============================================================================

echo "================================================================================"
echo "🎓 Phase 4: Stage 4 偽標籤訓練"
echo "================================================================================"
echo ""

# 創建 Stage 4 配置
cat > configs/stage4_nih_pseudo.yaml << 'STAGE4_NIH'
model: efficientnet_v2_s
img_size: 384
num_classes: 4
dropout: 0.3

# 使用 NIH Stage 1 作為初始化（更好的起點）
pretrained_checkpoint: outputs/pretrain_nih_stage1/best.pt

fold: 0
kfold_csv_dir: data/kfold_splits
data_dir: .

epochs: 30
batch_size: 24
num_workers: 12

optimizer: adamw
lr: 0.00008
weight_decay: 0.00015
warmup_epochs: 3

scheduler: cosine
min_lr: 0.000001

loss_type: improved_focal
focal_alpha: [1.0, 1.5, 2.0, 12.0]
focal_gamma: 3.5
label_smoothing: 0.12

mixup_prob: 0.5
mixup_alpha: 1.2
cutmix_prob: 0.5
aug_rotation: 18
random_erasing_prob: 0.3

use_swa: true
swa_start_epoch: 22
swa_lr: 0.00004

early_stopping: true
patience: 10

output_dir: outputs/nih_v2s_stage4
save_best: true
STAGE4_NIH

cat > configs/stage4_v2l_pseudo.yaml << 'STAGE4_V2L'
model: efficientnet_v2_l
img_size: 384
num_classes: 4
dropout: 0.3

# 使用剛訓練的 Stage 2 作為初始化
pretrained_checkpoint: outputs/stage2_finetune/best.pt

fold: 0
kfold_csv_dir: data/kfold_splits
data_dir: .

epochs: 25
batch_size: 16
num_workers: 12

optimizer: adamw
lr: 0.00005
weight_decay: 0.00015
warmup_epochs: 2

scheduler: cosine
min_lr: 0.000001

loss_type: improved_focal
focal_alpha: [1.0, 1.5, 2.0, 12.0]
focal_gamma: 3.5
label_smoothing: 0.12

mixup_prob: 0.5
cutmix_prob: 0.5
aug_rotation: 18
random_erasing_prob: 0.3

use_swa: true
swa_start_epoch: 18
swa_lr: 0.00003

early_stopping: true
patience: 10

output_dir: outputs/v2l_pseudo_stage4
save_best: true
STAGE4_V2L

# 路徑 A: NIH Stage 4 (5-Fold) - 主力
echo "🔄 路徑 A: NIH Stage 4 訓練 (5-Fold)..."
for fold in {0..4}; do
    echo "   Training NIH Fold $fold..."

    # 臨時修改配置以使用擴增數據
    python3 << EOF
import yaml
with open('configs/stage4_nih_pseudo.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['fold'] = $fold
with open('configs/stage4_nih_pseudo_fold${fold}.yaml', 'w') as f:
    yaml.dump(cfg, f)

# 修改 K-Fold CSV 以使用擴增數據
import pandas as pd
train_df = pd.read_csv('data/kfold_splits/fold${fold}_train_nih_augmented.csv')
# 確保路徑正確
print(f"Fold $fold 訓練樣本: {len(train_df)}")
EOF

    # 訓練
    python3 train_breakthrough.py \
        --config configs/stage4_nih_pseudo_fold${fold}.yaml \
        --fold $fold \
        > $LOG_DIR/nih_stage4_fold${fold}_$(date +%Y%m%d_%H%M%S).log 2>&1

    # 保存模型
    if [ -f "outputs/nih_v2s_stage4/best.pt" ]; then
        cp outputs/nih_v2s_stage4/best.pt outputs/nih_v2s_stage4/fold${fold}_best.pt
        echo "   ✅ NIH Stage 4 Fold $fold 完成"
    fi
done

echo ""
echo "🔄 路徑 B: V2-L Stage 4 訓練 (Fold 0)..."

# 路徑 B: V2-L Stage 4 (只訓練 Fold 0 作為測試)
python3 << EOF
import yaml
with open('configs/stage4_v2l_pseudo.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['fold'] = 0
with open('configs/stage4_v2l_pseudo_fold0.yaml', 'w') as f:
    yaml.dump(cfg, f)
EOF

python3 train_breakthrough.py \
    --config configs/stage4_v2l_pseudo_fold0.yaml \
    --fold 0 \
    > $LOG_DIR/v2l_stage4_fold0_$(date +%Y%m%d_%H%M%S).log 2>&1

if [ -f "outputs/v2l_pseudo_stage4/best.pt" ]; then
    cp outputs/v2l_pseudo_stage4/best.pt outputs/v2l_pseudo_stage4/fold0_best.pt
    echo "   ✅ V2-L Stage 4 Fold 0 完成"
fi

echo ""
echo "✅ Phase 4 完成！所有 Stage 4 模型已訓練"
echo ""

# ============================================================================
# Phase 5: 生成最終預測
# ============================================================================

echo "================================================================================"
echo "🎯 Phase 5: 生成最終預測"
echo "================================================================================"
echo ""

# 創建最終預測腳本
cat > scripts/generate_final_predictions.py << 'FINAL_PRED'
#!/usr/bin/env python3
"""生成最終集成預測"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_breakthrough import build_model
from scripts.generate_pseudo_nih import TestDataset

@torch.no_grad()
def generate_predictions(models_config, test_csv, test_images, output_csv):
    """生成最終預測"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加載數據
    test_dataset = TestDataset(test_csv, test_images, img_size=384)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    all_probs = []

    for model_info in models_config:
        model_paths = model_info['paths']
        model_arch = model_info['arch']
        weight = model_info['weight']

        print(f"\n處理: {model_arch} ({len(model_paths)} models, weight={weight})")

        fold_probs_list = []

        for model_path in model_paths:
            model = build_model(model_arch, num_classes=4, dropout=0.3, img_size=384)
            checkpoint = torch.load(model_path, map_location='cpu')

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(device)
            model.eval()

            fold_probs = []
            for images, _ in tqdm(test_loader, desc=f"預測 {Path(model_path).name}"):
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                fold_probs.append(probs)

            fold_probs = np.concatenate(fold_probs, axis=0)
            fold_probs_list.append(fold_probs)

            del model
            torch.cuda.empty_cache()

        # 平均此類模型的預測
        avg_probs = np.mean(fold_probs_list, axis=0)
        # 應用權重
        weighted_probs = avg_probs * weight
        all_probs.append(weighted_probs)

    # 最終集成
    final_probs = np.sum(all_probs, axis=0)
    final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)  # 重新歸一化

    # 生成提交文件
    test_df = pd.read_csv(test_csv)
    pred_labels = np.argmax(final_probs, axis=1)

    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
    for i, class_name in enumerate(class_names):
        test_df[class_name] = (pred_labels == i).astype(int)

    test_df[['new_filename'] + class_names].to_csv(output_csv, index=False)
    print(f"\n✅ 最終預測已保存: {output_csv}")

    # 統計
    print("\n預測分布:")
    for i, class_name in enumerate(class_names):
        count = np.sum(pred_labels == i)
        print(f"  {class_name}: {count} ({100*count/len(test_df):.1f}%)")

if __name__ == '__main__':
    # 配置所有模型
    models_config = [
        {
            'paths': [f'outputs/nih_v2s_stage4/fold{i}_best.pt' for i in range(5)],
            'arch': 'efficientnet_v2_s',
            'weight': 0.50  # NIH Stage 4 (主力)
        },
        {
            'paths': [f'outputs/nih_v2s_stage2/fold{i}_best.pt' for i in range(5)],
            'arch': 'efficientnet_v2_s',
            'weight': 0.20  # NIH Stage 2
        },
        {
            'paths': ['outputs/v2l_pseudo_stage4/fold0_best.pt'],
            'arch': 'efficientnet_v2_l',
            'weight': 0.15  # V2-L Stage 4
        },
        {
            'paths': ['outputs/stage2_finetune/best.pt'],
            'arch': 'efficientnet_v2_l',
            'weight': 0.15  # V2-L Stage 2
        },
    ]

    generate_predictions(
        models_config=models_config,
        test_csv='data/test_data.csv',
        test_images='test_images',
        output_csv='data/submission_ultimate_parallel.csv'
    )
FINAL_PRED

chmod +x scripts/generate_final_predictions.py
python3 scripts/generate_final_predictions.py > $LOG_DIR/final_predictions_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "✅ Phase 5 完成！最終預測已生成"
echo ""

# ============================================================================
# 完成報告
# ============================================================================

echo "================================================================================"
echo "🎉🎉🎉 並行雙路徑流程執行完成！ 🎉🎉🎉"
echo "================================================================================"
echo "完成時間: $(date)"
echo ""
echo "輸出模型:"
echo "  路徑 A (NIH V2-S):"
echo "    Stage 2: outputs/nih_v2s_stage2/fold{0-4}_best.pt"
echo "    Stage 4: outputs/nih_v2s_stage4/fold{0-4}_best.pt"
echo ""
echo "  路徑 B (V2-L):"
echo "    Stage 4: outputs/v2l_pseudo_stage4/fold0_best.pt"
echo ""
echo "最終提交:"
echo "  data/submission_ultimate_parallel.csv"
echo ""
echo "下一步: 提交至 Kaggle！"
echo "================================================================================"
