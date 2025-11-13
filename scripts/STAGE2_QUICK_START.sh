#!/bin/bash
# Stage 2: NIH 預訓練模型微調 - 快速啟動腳本

set -e

echo "=========================================="
echo "🚀 Stage 2: NIH 微調訓練"
echo "=========================================="
echo "開始時間: $(date)"
echo

# 創建日誌目錄
mkdir -p logs

# 使用 train_breakthrough.py (NIH 預訓練暫時跳過，直接用 ImageNet 預訓練)
nohup python3 train_breakthrough.py \
  --config configs/stage2_finetune.yaml \
  --fold 0 \
  > logs/stage2_finetune_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo $PID > logs/stage2.pid

echo "✅ Stage 2 訓練已啟動！"
echo "PID: $PID"
echo "日誌: logs/stage2_finetune_*.log"
echo
echo "監控命令:"
echo "  tail -f logs/stage2_finetune_*.log"
echo "  nvidia-smi"
echo "  ps aux | grep $PID"
echo
echo "預計完成: 6-8 小時後"
echo "=========================================="
