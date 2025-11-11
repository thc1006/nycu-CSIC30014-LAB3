#!/bin/bash
# Real-time training monitor

echo "================================================================================"
echo "📊 Training Monitor"
echo "================================================================================"
echo ""

# Check GPU status
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s\n  Memory: %s / %s MB (%.1f%%)\n  Utilization: %s%%\n  Power: %.0fW\n\n",
                 $1, $2, $3, $4, ($3/$4*100), $5, $6}'

# Check ConvNeXt training
echo "---"
if ps aux | grep -q "[p]ython3 -m src.train_v2 --config configs/ultra_optimized.yaml"; then
    echo "✅ ConvNeXt-Base: TRAINING"
    echo ""
    echo "Latest epochs:"
    tail -100 outputs/convnext_ultra_train.log | grep -E "epoch [0-9]+" | tail -5
    echo ""
    echo "Best model so far:"
    tail -100 outputs/convnext_ultra_train.log | grep "saved new best" | tail -1
else
    echo "⏸️  ConvNeXt-Base: NOT RUNNING"
fi

echo ""
echo "---"

# Check EfficientNet training
if ps aux | grep -q "[p]ython3 -m src.train_v2 --config configs/efficientnet_v2_l.yaml"; then
    echo "✅ EfficientNet-V2-L: TRAINING"
    echo ""
    echo "Latest epochs:"
    tail -100 outputs/effnet_v2_l_train.log 2>/dev/null | grep -E "epoch [0-9]+" | tail -5 || echo "  (log not yet available)"
else
    echo "⏸️  EfficientNet-V2-L: NOT RUNNING"
fi

echo ""
echo "================================================================================"
echo "監控指令:"
echo "  watch -n 5 ./monitor_training.sh     # 每5秒自動刷新"
echo "  tail -f outputs/convnext_ultra_train.log"
echo "================================================================================"
