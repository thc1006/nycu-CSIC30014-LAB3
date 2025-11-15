#!/bin/bash
# 🚀 Gen3 訓練 - 自適應偽標籤最終衝刺

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_BASE="configs/efficientnet_v2l_512_gen3.yaml"
OUTPUT_BASE="outputs/v2l_512_gen3"
LOG_DIR="$OUTPUT_BASE/logs"

mkdir -p "$LOG_DIR"

echo "🚀 Gen3 訓練開始 - 最終衝刺 90%+"
echo "數據: 原始 + Gen3 自適應偽標籤 (800-900 樣本)"
echo "策略: 類別特定閾值 + 對比學習"
echo "預計: 7-8 小時"
echo ""

START_TIME=$(date +%s)

for FOLD in 0 1 2 3 4; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 訓練 Fold $FOLD / 4"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    FOLD_START=$(date +%s)
    FOLD_CONFIG="configs/gen3_fold${FOLD}.yaml"

    # 創建 fold 專用配置
    sed "s/{fold}/$FOLD/g" "$CONFIG_BASE" > "$FOLD_CONFIG"

    # 訓練
    python3 src/train_v2.py \
        --config "$FOLD_CONFIG" \
        2>&1 | tee "$LOG_DIR/fold${FOLD}.log"

    FOLD_END=$(date +%s)
    FOLD_MIN=$(( ($FOLD_END - $FOLD_START) / 60 ))

    echo "✅ Fold $FOLD 完成！用時: ${FOLD_MIN} 分鐘"
    echo ""

    rm -f "$FOLD_CONFIG"
done

END_TIME=$(date +%s)
TOTAL_MIN=$(( ($END_TIME - $START_TIME) / 60 ))
TOTAL_HR=$(( $TOTAL_MIN / 60 ))
TOTAL_MIN_REMAIN=$(( $TOTAL_MIN % 60 ))

echo "🎉 Gen3 訓練全部完成！"
echo "總用時: ${TOTAL_HR} 小時 ${TOTAL_MIN_REMAIN} 分鐘"
echo ""
echo "下一步: python3 scripts/generate_gen3_predictions.py"
echo "預期測試 F1: 89.5-91.0% 🎯"
