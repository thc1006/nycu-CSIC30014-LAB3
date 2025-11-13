#!/bin/bash
# NIH Stage 3-4 訓練: 使用 Stage 2 模型 + 偽標籤進一步微調
# 預計時間: 15 epochs × 5 folds = ~50-60 分鐘

echo "================================================================================"
echo "🚀 NIH Stage 3-4 Training - Pseudo-Label Fine-tuning"
echo "================================================================================"
echo "開始時間: $(date)"
echo ""
echo "配置:"
echo "  - 基礎模型: NIH Stage 2 (已訓練 30 epochs)"
echo "  - 偽標籤: 562 個高置信度樣本 (≥ 0.95)"
echo "  - 訓練策略: 15 epochs 微調"
echo "  - 學習率: 0.00005 (較低，微調)"
echo ""

# 創建輸出目錄
mkdir -p outputs/nih_v2s_stage3_4
mkdir -p logs/stage3_4

# 訓練 5 個 fold
for fold in {0..4}; do
    echo "================================================================================"
    echo "🔄 Training Fold $fold / 4"
    echo "================================================================================"

    start_time=$(date +%s)

    # 從 Stage 2 檢查點開始
    stage2_checkpoint="outputs/nih_v2s_stage2/fold${fold}_best.pt"

    if [ ! -f "$stage2_checkpoint" ]; then
        echo "❌ Stage 2 checkpoint not found: $stage2_checkpoint"
        exit 1
    fi

    echo "📦 使用 Stage 2 檢查點: $stage2_checkpoint"
    echo "📦 訓練數據: data/fold${fold}_train_with_pseudo.csv (3279-3280 樣本)"
    echo ""

    # 創建臨時配置文件 (指定 pretrained_checkpoint)
    temp_config="configs/stage3_4_pseudo_fold${fold}.yaml"

    cat > "$temp_config" <<EOF
# NIH Stage 3-4: Fold $fold
model: efficientnet_v2_s
img_size: 384
num_classes: 4
dropout: 0.3

# 從 Stage 2 繼續訓練
pretrained_checkpoint: $stage2_checkpoint

# K-Fold 設置
fold: $fold
kfold_csv_dir: data/kfold_splits

# 偽標籤設置
use_pseudo_labels: true

# 訓練參數
epochs: 15
batch_size: 16
lr: 0.00005
weight_decay: 0.0002

# 優化器
optimizer: adamw
scheduler: cosine
warmup_epochs: 2

# Loss 配置
loss_type: improved_focal
focal_alpha: [1.0, 1.5, 2.0, 12.0]
focal_gamma: 3.5
label_smoothing: 0.10

# 數據增強 (較保守)
mixup_prob: 0.3
mixup_alpha: 0.8
cutmix_prob: 0.3
aug_rotation: 12
aug_scale: [0.90, 1.10]
random_erasing_prob: 0.20

# SWA
use_swa: true
swa_start_epoch: 10
swa_lr: 0.00002

# 早停
patience: 8
min_delta: 0.001

# 輸出
output_dir: outputs/nih_v2s_stage3_4
save_best_only: true
EOF

    # 訓練
    python3 train_breakthrough.py --config "$temp_config" --fold $fold \
        > logs/stage3_4/fold${fold}_$(date +%Y%m%d_%H%M%S).log 2>&1

    # 檢查是否成功
    if [ $? -eq 0 ]; then
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        minutes=$((elapsed / 60))

        echo "✅ Fold $fold 完成！耗時: ${minutes} 分鐘"
        echo "   模型已保存: outputs/nih_v2s_stage3_4/fold${fold}_best.pt"
        echo ""
    else
        echo "❌ Fold $fold 訓練失敗！"
        exit 1
    fi

    # 刪除臨時配置文件
    rm -f "$temp_config"
done

echo "================================================================================"
echo "✅ Stage 3-4 Training Complete!"
echo "================================================================================"
echo "結束時間: $(date)"
echo ""
echo "所有模型已保存:"
for fold in {0..4}; do
    if [ -f "outputs/nih_v2s_stage3_4/fold${fold}_best.pt" ]; then
        size=$(du -h "outputs/nih_v2s_stage3_4/fold${fold}_best.pt" | cut -f1)
        echo "  - Fold $fold: $size"
    fi
done
echo ""
echo "下一步: 使用這 5 個模型生成測試集預測"
echo "  python3 scripts/generate_pseudo_nih.py --model_dir outputs/nih_v2s_stage3_4 --model_arch efficientnet_v2_s --output data/submission_nih_stage4.csv"
