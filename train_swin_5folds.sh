#!/bin/bash
#  训练 Swin-Large 5-Fold Models
# 预计总时间: ~20 小时 (每个 fold 约 4 小时)

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/swin_training_logs_${TIMESTAMP}"
OUTPUT_BASE="outputs/swin_breakthrough_${TIMESTAMP}"

mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "🚀 开始训练 Swin-Large 5-Fold Models"
echo "================================================================================"
echo "开始时间: $(date)"
echo "日志目录: $LOG_DIR"
echo "输出目录: $OUTPUT_BASE"
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

    echo "输出目录: $FOLD_OUTPUT_DIR"
    echo "日志文件: $FOLD_LOG"
    echo "开始时间: $(date)"

    # 运行训练
    python3 -m src.train_v2 \
        --config configs/swin_large.yaml \
        --fold $FOLD \
        --output_dir "$FOLD_OUTPUT_DIR" \
        > "$FOLD_LOG" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Fold $FOLD 训练成功"
        echo "完成时间: $(date)"

        # 检查最佳模型
        if [ -f "${FOLD_OUTPUT_DIR}/best.pt" ]; then
            SIZE=$(du -h "${FOLD_OUTPUT_DIR}/best.pt" | cut -f1)
            echo "   最佳模型大小: $SIZE"

            # 提取验证 F1 分数
            BEST_F1=$(grep "Best Val F1" "$FOLD_LOG" | tail -1 || echo "未找到")
            echo "   $BEST_F1"
        else
            echo "   ⚠️ 未找到 best.pt"
        fi
    else
        echo "❌ Fold $FOLD 训练失败 (退出码: $EXIT_CODE)"
        echo "查看日志: $FOLD_LOG"
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
            BEST_F1=$(grep "Best Val F1" "$FOLD_LOG" | tail -1 || echo "")
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
echo ""
echo "🔮 下一步:"
echo "   1. 生成测试预测: python3 generate_swin_predictions.py"
echo "   2. 集成所有模型: python3 ensemble_all_models.py"
echo "================================================================================"
