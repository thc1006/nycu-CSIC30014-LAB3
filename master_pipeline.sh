#!/bin/bash
# Master Pipeline to Reach 91%+ Accuracy
# Based on ultra-deep data analysis findings

set -e  # Exit on error

echo "================================================================================"
echo "🎯 Master Pipeline: 目標 91%+ Accuracy"
echo "================================================================================"
echo ""

# ============================================================================
# Phase 1: Train Ultra-Optimized Models
# ============================================================================
echo "📊 Phase 1: Training Ultra-Optimized Models"
echo "-------------------------------------------"
echo ""

# Check if ConvNeXt is still training
if ps aux | grep -q "[p]ython3 -m src.train_v2 --config configs/ultra_optimized.yaml"; then
    echo "⏳ ConvNeXt-Base is currently training..."
    echo "   Waiting for completion before proceeding..."

    # Wait for ConvNeXt training to complete
    while ps aux | grep -q "[p]ython3 -m src.train_v2 --config configs/ultra_optimized.yaml"; do
        sleep 30
        tail -1 outputs/convnext_ultra_train.log | grep -E "epoch [0-9]+" || true
    done

    echo "✅ ConvNeXt-Base training completed!"
else
    echo "⚠️  ConvNeXt-Base not running. Please start training first."
    exit 1
fi

echo ""
echo "📈 ConvNeXt-Base training summary:"
tail -20 outputs/convnext_ultra_train.log | grep -E "best|val"
echo ""

# Train EfficientNet-V2-L (if not already trained)
if [ ! -f "outputs/efficientnet_v2_l/best.pt" ]; then
    echo "🔥 Starting EfficientNet-V2-L training..."
    nohup python3 -m src.train_v2 --config configs/efficientnet_v2_l.yaml > outputs/effnet_v2_l_train.log 2>&1 &
    EFFNET_PID=$!
    echo "   PID: $EFFNET_PID"
    echo "   Log: tail -f outputs/effnet_v2_l_train.log"
    echo ""

    # Wait for EfficientNet training
    echo "⏳ Waiting for EfficientNet-V2-L training..."
    wait $EFFNET_PID
    echo "✅ EfficientNet-V2-L training completed!"
else
    echo "✅ EfficientNet-V2-L already trained (best.pt exists)"
fi

echo ""
echo "✅ Phase 1 Complete: Both models trained"
echo ""

# ============================================================================
# Phase 2: Test Time Augmentation (TTA)
# ============================================================================
echo "📊 Phase 2: Applying Test Time Augmentation"
echo "-------------------------------------------"
echo ""

# TTA on ConvNeXt-Base
echo "🔄 Applying TTA to ConvNeXt-Base..."
python3 -m src.predict_tta \
    --checkpoints outputs/ultra_optimized/best.pt \
    --test-csv data/test_data.csv \
    --test-images test_images \
    --output data/submission_convnext_tta.csv \
    --img-size 448 \
    --tta-transforms 5 \
    --device cuda

echo "✅ ConvNeXt TTA complete: data/submission_convnext_tta.csv"
echo ""

# TTA on EfficientNet-V2-L
echo "🔄 Applying TTA to EfficientNet-V2-L..."
python3 -m src.predict_tta \
    --checkpoints outputs/efficientnet_v2_l/best.pt \
    --test-csv data/test_data.csv \
    --test-images test_images \
    --output data/submission_effnetv2l_tta.csv \
    --img-size 480 \
    --tta-transforms 5 \
    --device cuda

echo "✅ EfficientNet-V2-L TTA complete: data/submission_effnetv2l_tta.csv"
echo ""

# TTA on existing best model (submission_improved.csv was from a model)
if [ -f "outputs/improved_breakthrough/best.pt" ]; then
    echo "🔄 Applying TTA to previous best model..."
    python3 -m src.predict_tta \
        --checkpoints outputs/improved_breakthrough/best.pt \
        --test-csv data/test_data.csv \
        --test-images test_images \
        --output data/submission_improved_tta.csv \
        --img-size 384 \
        --tta-transforms 5 \
        --device cuda

    echo "✅ Previous best TTA complete: data/submission_improved_tta.csv"
else
    echo "⚠️  Previous best checkpoint not found, skipping TTA"
    # Use existing submission_improved.csv instead
    echo "   Using existing submission_improved.csv"
fi

echo ""
echo "✅ Phase 2 Complete: TTA applied to all models"
echo ""

# ============================================================================
# Phase 3: Advanced Ensemble
# ============================================================================
echo "📊 Phase 3: Creating Advanced Ensemble"
echo "---------------------------------------"
echo ""

# Collect all available TTA submissions
SUBMISSIONS=""
[ -f "data/submission_convnext_tta.csv" ] && SUBMISSIONS="$SUBMISSIONS data/submission_convnext_tta.csv"
[ -f "data/submission_effnetv2l_tta.csv" ] && SUBMISSIONS="$SUBMISSIONS data/submission_effnetv2l_tta.csv"
[ -f "data/submission_improved_tta.csv" ] && SUBMISSIONS="$SUBMISSIONS data/submission_improved_tta.csv"
[ -f "data/submission_improved.csv" ] && [ ! -f "data/submission_improved_tta.csv" ] && SUBMISSIONS="$SUBMISSIONS data/submission_improved.csv"

echo "📁 Submissions to ensemble:"
for sub in $SUBMISSIONS; do
    echo "   - $sub"
done
echo ""

# Create ensemble using geometric mean (better for probabilities)
echo "🔀 Creating ensemble with geometric mean..."
python3 ensemble_probabilities.py \
    --submissions $SUBMISSIONS \
    --output data/submission_ultra_ensemble.csv \
    --method geometric_mean

echo ""
echo "✅ Phase 3 Complete: Ensemble created"
echo ""

# ============================================================================
# Phase 4: Submission Summary
# ============================================================================
echo "📊 Phase 4: Submission Summary"
echo "------------------------------"
echo ""

echo "📈 Final ensemble distribution:"
python3 -c "
import pandas as pd
df = pd.read_csv('data/submission_ultra_ensemble.csv')
class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
pred_classes = df[class_cols].values.argmax(axis=1)
pred_labels = [class_cols[c] for c in pred_classes]
from collections import Counter
counts = Counter(pred_labels)
total = len(pred_labels)
for label, count in sorted(counts.items()):
    pct = count / total * 100
    print(f'  {label:12s}: {count:4d} ({pct:5.2f}%)')
print()
print(f'  Total samples: {total}')
avg_conf = df[class_cols].max(axis=1).mean()
print(f'  Avg confidence: {avg_conf:.4f}')
"

echo ""
echo "================================================================================"
echo "✅ Pipeline Complete!"
echo "================================================================================"
echo ""
echo "📦 Generated submissions:"
echo "  • data/submission_convnext_tta.csv       (ConvNeXt-Base + TTA)"
echo "  • data/submission_effnetv2l_tta.csv      (EfficientNet-V2-L + TTA)"
echo "  • data/submission_ultra_ensemble.csv     (Final Ensemble) ⭐"
echo ""
echo "🎯 Next Steps:"
echo "  1. Submit data/submission_ultra_ensemble.csv to Kaggle"
echo "  2. If score < 91%, consider pseudo-labeling (Phase 5)"
echo "  3. Monitor leaderboard and iterate"
echo ""
echo "監控當前訓練進度:"
echo "  tail -f outputs/convnext_ultra_train.log"
echo ""
