#!/bin/bash
# 自動執行 Layer 2 Pipeline
# 1. 等待 Swin Fold 4 訓練完成
# 2. 生成所有 Layer 1 驗證集預測
# 3. 訓練 Layer 2 meta-learners
# 4. 生成最終集成提交

set -e

echo "================================================================================"
echo "🚀 Layer 2 Pipeline 自動執行"
echo "================================================================================"
echo "開始時間: $(date)"
echo

# Step 1: 等待 Fold 4 完成
echo "1️⃣ 檢查 Swin Fold 4 訓練狀態..."
FOLD4_MODEL="outputs/breakthrough_20251113_004854/layer1/swin_large/fold4/best.pt"

if [ ! -f "$FOLD4_MODEL" ]; then
    echo "⏳ Fold 4 尚未完成，等待訓練..."

    # 等待訓練完成（最多等待2小時）
    TIMEOUT=$((2 * 60 * 60))  # 2 hours in seconds
    ELAPSED=0
    WAIT_INTERVAL=30

    while [ ! -f "$FOLD4_MODEL" ] && [ $ELAPSED -lt $TIMEOUT ]; do
        sleep $WAIT_INTERVAL
        ELAPSED=$((ELAPSED + WAIT_INTERVAL))

        # 每5分鐘打印一次狀態
        if [ $((ELAPSED % 300)) -eq 0 ]; then
            MINUTES=$((ELAPSED / 60))
            echo "   已等待 $MINUTES 分鐘..."

            # 檢查訓練進程是否還在運行
            if ! ps aux | grep "train_breakthrough.py.*fold4" | grep -v grep > /dev/null; then
                echo "   ⚠️ 訓練進程似乎已停止"
                break
            fi
        fi
    done

    if [ ! -f "$FOLD4_MODEL" ]; then
        echo "❌ Fold 4 訓練未完成或超時"
        echo "請手動檢查訓練狀態: ./MONITOR_SWIN_TRAINING.sh"
        exit 1
    fi
fi

echo "✅ Fold 4 已完成！"
SIZE=$(du -h "$FOLD4_MODEL" | cut -f1)
echo "   模型大小: $SIZE"
echo

# 等待 GPU 釋放
echo "⏳ 等待 GPU 釋放..."
while true; do
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if [ "$GPU_UTIL" -lt 20 ]; then
        echo "✅ GPU 已釋放 (使用率: ${GPU_UTIL}%)"
        break
    fi
    echo "   GPU 使用率: ${GPU_UTIL}%，等待中..."
    sleep 10
done
echo

# Step 2: 生成 Layer 1 驗證集預測
echo "================================================================================"
echo "2️⃣ 生成 Layer 1 驗證集預測"
echo "================================================================================"
echo "開始時間: $(date)"
echo

python3 scripts/generate_layer1_val_predictions.py

if [ $? -ne 0 ]; then
    echo "❌ 生成驗證集預測失敗"
    exit 1
fi

echo
echo "✅ 驗證集預測生成完成"
echo "完成時間: $(date)"
echo

# 檢查生成的文件
VAL_PRED_DIR="outputs/breakthrough_20251113_004854/layer1_val_predictions"
VAL_PRED_COUNT=$(ls -1 "$VAL_PRED_DIR"/*.csv 2>/dev/null | wc -l)
echo "生成的預測文件數: $VAL_PRED_COUNT"

if [ $VAL_PRED_COUNT -lt 10 ]; then
    echo "⚠️ 預期 10 個預測文件，但只找到 $VAL_PRED_COUNT 個"
    echo "繼續執行，但可能影響 meta-learner 性能"
fi
echo

# Step 3: 訓練 Layer 2 Meta-Learners
echo "================================================================================"
echo "3️⃣ 訓練 Layer 2 Meta-Learners"
echo "================================================================================"
echo "開始時間: $(date)"
echo

python3 scripts/stacking_meta_learner.py

if [ $? -ne 0 ]; then
    echo "❌ Meta-learner 訓練失敗"
    exit 1
fi

echo
echo "✅ Meta-learner 訓練完成"
echo "完成時間: $(date)"
echo

# Step 4: 生成最終提交
echo "================================================================================"
echo "4️⃣ 生成最終提交"
echo "================================================================================"
echo "開始時間: $(date)"
echo

# TODO: 根據實際的最終集成腳本調整
if [ -f "scripts/stacking_predict.py" ]; then
    python3 scripts/stacking_predict.py
    echo "✅ 最終提交已生成"
elif [ -f "ULTIMATE_ENSEMBLE_NOW.py" ]; then
    python3 ULTIMATE_ENSEMBLE_NOW.py
    echo "✅ 最終提交已生成"
else
    echo "⚠️ 未找到提交生成腳本"
    echo "請手動生成最終提交"
fi

echo
echo "================================================================================"
echo "🎉 Layer 2 Pipeline 完成！"
echo "================================================================================"
echo "完成時間: $(date)"
echo
echo "下一步："
echo "  1. 檢查生成的提交文件"
echo "  2. 驗證提交格式"
echo "  3. 上傳到 Kaggle"
echo "================================================================================"
