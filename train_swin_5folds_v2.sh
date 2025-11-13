#!/bin/bash
# 训练 Swin-Large 5-Fold Models (v2 - 使用临时配置文件)
# 预计总时间: ~20 小时 (每个 fold 约 4 小时)

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/swin_training_logs_${TIMESTAMP}"
OUTPUT_BASE="outputs/swin_breakthrough_${TIMESTAMP}"
TEMP_CONFIGS_DIR="temp_configs_${TIMESTAMP}"

mkdir -p "$LOG_DIR"
mkdir -p "$TEMP_CONFIGS_DIR"

echo "================================================================================"
echo "🚀 开始训练 Swin-Large 5-Fold Models"
echo "================================================================================"
echo "开始时间: $(date)"
echo "日志目录: $LOG_DIR"
echo "输出目录: $OUTPUT_BASE"
echo "临时配置: $TEMP_CONFIGS_DIR"
echo "================================================================================"

# 记录系统信息
nvidia-smi > "$LOG_DIR/gpu_info.txt"
echo "GPU 信息已保存"

# 训练每个 fold
for FOLD in {0..4}; do
    echo ""
    echo "================================================================================"
    echo "📊 训练 Fold $FOLD / 4"
    echo "================================================================================"

    FOLD_OUTPUT_DIR="${OUTPUT_BASE}/fold${FOLD}"
    FOLD_LOG="${LOG_DIR}/fold${FOLD}.log"
    FOLD_CONFIG="${TEMP_CONFIGS_DIR}/swin_fold${FOLD}.yaml"

    # 创建 fold 特定的配置文件
    cat > "$FOLD_CONFIG" << EOF
# Swin Transformer Large Configuration - Fold ${FOLD}
# Auto-generated config

# K-Fold Training
fold: ${FOLD}
kfold_csv_dir: data/kfold_splits

# Model
model: swin_large_patch4_window7_224
img_size: 384
num_classes: 4

# Training
epochs: 45
batch_size: 10
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
swa_start_epoch: 35
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
EOF

    echo "输出目录: $FOLD_OUTPUT_DIR"
    echo "日志文件: $FOLD_LOG"
    echo "配置文件: $FOLD_CONFIG"
    echo "开始时间: $(date)"

    # 运行训练
    python3 -m src.train_v2 --config "$FOLD_CONFIG" > "$FOLD_LOG" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Fold $FOLD 训练成功"
        echo "完成时间: $(date)"

        # 检查最佳模型
        if [ -f "${FOLD_OUTPUT_DIR}/best.pt" ]; then
            SIZE=$(du -h "${FOLD_OUTPUT_DIR}/best.pt" | cut -f1)
            echo "   最佳模型大小: $SIZE"

            # 提取验证 F1 分数
            BEST_F1=$(grep -E "Val.*F1|Best.*F1" "$FOLD_LOG" | tail -1 || echo "未找到")
            echo "   $BEST_F1"
        else
            echo "   ⚠️ 未找到 best.pt"
        fi
    else
        echo "❌ Fold $FOLD 训练失败 (退出码: $EXIT_CODE)"
        echo "查看日志: $FOLD_LOG"
        echo ""
        echo "最后 20 行错误:"
        tail -20 "$FOLD_LOG"
    fi

    echo "================================================================================"
done

echo ""
echo "================================================================================"
echo "🎉 所有 Fold 训练完成"
echo "================================================================================"
echo "结束时间: $(date)"
echo ""
echo "📊 训练总结:"
echo "---"

# 统计成功的模型
SUCCESS_COUNT=0
for FOLD in {0..4}; do
    FOLD_OUTPUT_DIR="${OUTPUT_BASE}/fold${FOLD}"
    if [ -f "${FOLD_OUTPUT_DIR}/best.pt" ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✅ Fold $FOLD: 成功"

        # 显示验证 F1
        FOLD_LOG="${LOG_DIR}/fold${FOLD}.log"
        if [ -f "$FOLD_LOG" ]; then
            BEST_F1=$(grep -E "Val.*F1|Best.*F1" "$FOLD_LOG" | tail -1 || echo "")
            if [ -n "$BEST_F1" ]; then
                echo "   $BEST_F1"
            fi
        fi
    else
        echo "❌ Fold $FOLD: 失败或未完成"
    fi
done

echo "---"
echo "成功: $SUCCESS_COUNT / 5 models"
echo ""
echo "📁 文件位置:"
echo "   模型: ${OUTPUT_BASE}/fold*/best.pt"
echo "   日志: ${LOG_DIR}/"
echo "   配置: ${TEMP_CONFIGS_DIR}/ (可删除)"
echo ""
echo "🧹 清理临时配置:"
echo "   rm -rf ${TEMP_CONFIGS_DIR}"
echo ""
echo "🔮 下一步:"
echo "   1. 生成测试预测: python3 generate_swin_predictions.py"
echo "   2. 集成所有模型: python3 ensemble_all_models.py"
echo "================================================================================"
