#!/bin/bash
# Gen2 訓練監控腳本

echo "════════════════════════════════════════════════════════════"
echo "🔍 Gen2 訓練監控"
echo "════════════════════════════════════════════════════════════"
echo ""

# 檢查訓練進程
TRAIN_PIDS=$(pgrep -f "train_v2.py.*gen2" | wc -l)
if [ "$TRAIN_PIDS" -gt 0 ]; then
    echo "✅ 訓練進程運行中 ($TRAIN_PIDS 個進程)"
else
    echo "⚠️  沒有訓練進程運行"
fi

# GPU 狀態
echo ""
echo "📊 GPU 狀態:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  使用率: %s%% | 記憶體: %s/%s MB | 溫度: %s°C\n", $1, $2, $3, $4}'

# 訓練進度
echo ""
echo "📈 各 Fold 訓練進度:"
for i in 0 1 2 3 4; do
    LOG="outputs/v2l_512_gen2/logs/fold${i}.log"
    if [ -f "$LOG" ]; then
        LAST_EPOCH=$(grep -oP '\[epoch \K[0-9]+' "$LOG" | tail -1)
        BEST_F1=$(grep -oP 'val macro-F1=\K[0-9.]+' "$LOG" | tail -1)
        if [ -n "$LAST_EPOCH" ]; then
            printf "  Fold %d: Epoch %s/50 | 最佳 Val F1: %s%%\n" "$i" "$LAST_EPOCH" "$(echo "$BEST_F1 * 100" | bc -l | xargs printf "%.2f")"
        else
            echo "  Fold $i: 初始化中..."
        fi
    else
        echo "  Fold $i: 未開始"
    fi
done

# 訓練時間
echo ""
echo "⏱️  訓練時間:"
if [ -f "logs/gen2_training_fixed.log" ]; then
    START_TIME=$(stat -c %Y logs/gen2_training_fixed.log)
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    HOURS=$((MINUTES / 60))
    MINS=$((MINUTES % 60))
    printf "  已運行: %d 小時 %d 分鐘\n" "$HOURS" "$MINS"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
