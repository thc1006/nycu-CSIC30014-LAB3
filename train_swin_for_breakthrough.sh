#!/bin/bash
# 訓練 Swin-Large 5-Fold Models for BREAKTHROUGH_STACKING Pipeline
# 使用現有的 timestamp 目錄以保持一致性

set -e

# 使用現有的 breakthrough 目錄
BREAKTHROUGH_DIR="outputs/breakthrough_20251113_004854"
OUTPUT_BASE="${BREAKTHROUGH_DIR}/layer1/swin_large"
LOG_DIR="outputs/swin_breakthrough_logs_$(date +%Y%m%d_%H%M%S)"
TEMP_CONFIGS_DIR="temp_swin_configs_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"
mkdir -p "$TEMP_CONFIGS_DIR"

echo "================================================================================"
echo "🚀 開始訓練 Swin-Large 5-Fold Models (BREAKTHROUGH Pipeline)"
echo "================================================================================"
echo "開始時間: $(date)"
echo "日誌目錄: $LOG_DIR"
echo "輸出目錄: $OUTPUT_BASE"
echo "Pipeline 目錄: $BREAKTHROUGH_DIR"
echo "================================================================================"

# 記錄 GPU 資訊
nvidia-smi > "$LOG_DIR/gpu_info.txt"
echo "GPU 資訊已保存"

# 訓練每個 fold
for FOLD in {0..4}; do
    echo ""
    echo "================================================================================"
    echo "📊 訓練 Fold $FOLD / 4"
    echo "================================================================================"

    FOLD_OUTPUT_DIR="${OUTPUT_BASE}/fold${FOLD}"
    FOLD_LOG="${LOG_DIR}/fold${FOLD}.log"
    FOLD_CONFIG="${TEMP_CONFIGS_DIR}/swin_fold${FOLD}.yaml"

    # 檢查是否已完成
    if [ -f "${FOLD_OUTPUT_DIR}/best.pt" ]; then
        echo "✅ Fold $FOLD 已完成，跳過"
        SIZE=$(du -h "${FOLD_OUTPUT_DIR}/best.pt" | cut -f1)
        echo "   模型大小: $SIZE"
        continue
    fi

    # 創建輸出目錄
    mkdir -p "$FOLD_OUTPUT_DIR"

    # 創建 fold 特定的配置文件
    cat > "$FOLD_CONFIG" << EOF
# Swin Transformer Large Configuration - Fold ${FOLD}
# For BREAKTHROUGH_STACKING Pipeline

# K-Fold Training
fold: ${FOLD}
kfold_csv_dir: data/kfold_splits

# Model
model: swin_large_patch4_window12_384
img_size: 384
num_classes: 4

# Training
epochs: 40
batch_size: 6
lr: 0.00005
optimizer: adamw
weight_decay: 0.00025

# Loss
loss: improved_focal
focal_alpha: [1.0, 1.5, 2.0, 12.0]
focal_gamma: 3.5
label_smoothing: 0.12

# Scheduler
scheduler: cosine
warmup_epochs: 5
min_lr: 0.0000005

# Regularization
dropout: 0.3
drop_path_rate: 0.2

# Data Augmentation
mixup_prob: 0.6
mixup_alpha: 1.2
cutmix_prob: 0.5
cutmix_alpha: 1.0

# Geometric augmentation
aug_rotation: 15
aug_translate: 0.1
aug_scale: [0.9, 1.1]
aug_hflip: true
aug_vflip: false

# Advanced augmentation
random_erasing_prob: 0.3
color_jitter: 0.2
gaussian_blur_prob: 0.1

# SWA
use_swa: true
swa_start_epoch: 32
swa_lr: 0.00002

# Early stopping
patience: 12
min_delta: 0.0001

# Output
output_dir: ${FOLD_OUTPUT_DIR}
save_best_only: true

# GPU Optimization
mixed_precision: true
cudnn_benchmark: true
channels_last: true
gradient_accumulation_steps: 2
gradient_checkpointing: true
EOF

    echo "輸出目錄: $FOLD_OUTPUT_DIR"
    echo "日誌文件: $FOLD_LOG"
    echo "配置文件: $FOLD_CONFIG"
    echo "開始時間: $(date)"

    # 運行訓練
    python3 train_breakthrough.py --config "$FOLD_CONFIG" --fold $FOLD > "$FOLD_LOG" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Fold $FOLD 訓練成功"
        echo "完成時間: $(date)"

        # 檢查最佳模型
        if [ -f "${FOLD_OUTPUT_DIR}/best.pt" ]; then
            SIZE=$(du -h "${FOLD_OUTPUT_DIR}/best.pt" | cut -f1)
            echo "   最佳模型大小: $SIZE"

            # 提取驗證 F1 分數
            BEST_F1=$(grep -E "Val.*F1|Best.*F1" "$FOLD_LOG" | tail -1 || echo "未找到")
            echo "   $BEST_F1"
        else
            echo "   ⚠️ 未找到 best.pt"
        fi
    else
        echo "❌ Fold $FOLD 訓練失敗 (退出碼: $EXIT_CODE)"
        echo "查看日誌: $FOLD_LOG"
        echo ""
        echo "最後 20 行錯誤:"
        tail -20 "$FOLD_LOG"

        # 繼續下一個 fold 而不是退出
        echo "⚠️ 繼續訓練下一個 fold..."
    fi

    echo "================================================================================"
done

echo ""
echo "================================================================================"
echo "🎉 所有 Fold 訓練完成"
echo "================================================================================"
echo "結束時間: $(date)"
echo ""
echo "📊 訓練總結:"
echo "---"

# 統計成功的模型
SUCCESS_COUNT=0
for FOLD in {0..4}; do
    FOLD_OUTPUT_DIR="${OUTPUT_BASE}/fold${FOLD}"
    if [ -f "${FOLD_OUTPUT_DIR}/best.pt" ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✅ Fold $FOLD: 成功"

        # 顯示驗證 F1
        FOLD_LOG="${LOG_DIR}/fold${FOLD}.log"
        if [ -f "$FOLD_LOG" ]; then
            BEST_F1=$(grep -E "Val.*F1|Best.*F1" "$FOLD_LOG" | tail -1 || echo "")
            if [ -n "$BEST_F1" ]; then
                echo "   $BEST_F1"
            fi
        fi
    else
        echo "❌ Fold $FOLD: 失敗或未完成"
    fi
done

echo "---"
echo "成功: $SUCCESS_COUNT / 5 models"
echo ""
echo "📁 文件位置:"
echo "   模型: ${OUTPUT_BASE}/fold*/best.pt"
echo "   日誌: ${LOG_DIR}/"
echo "   臨時配置: ${TEMP_CONFIGS_DIR}/ (可刪除)"
echo ""
echo "✅ Pipeline 狀態:"
EFF_COUNT=$(find ${BREAKTHROUGH_DIR}/layer1/efficientnet_v2_l -name "best.pt" 2>/dev/null | wc -l)
SWIN_COUNT=$(find ${OUTPUT_BASE} -name "best.pt" 2>/dev/null | wc -l)
TOTAL=$((EFF_COUNT + SWIN_COUNT))
echo "   EfficientNet-V2-L: $EFF_COUNT / 5"
echo "   Swin-Large: $SWIN_COUNT / 5"
echo "   總計: $TOTAL / 10 Layer 1 模型"
echo ""
echo "🔮 下一步:"
if [ $TOTAL -eq 10 ]; then
    echo "   ✅ Layer 1 完成！可以開始 Layer 2 (meta-learners)"
    echo "   執行: python3 scripts/stacking_meta_learner.py"
else
    echo "   ⏳ 等待 Layer 1 完成 ($TOTAL / 10)"
fi
echo "================================================================================"
