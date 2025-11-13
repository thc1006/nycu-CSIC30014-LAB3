#!/bin/bash
# Complete Breakthrough Pipeline to Achieve 91%+ Macro-F1
# Automated execution of all breakthrough strategies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0;33m'

# Log file
LOG_FILE="outputs/breakthrough_91plus_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs

echo "=========================================="
echo "BREAKTHROUGH 91%+ PIPELINE"
echo "=========================================="
echo "Current best: 84.19%"
echo "Target: 91.085%"
echo "Gap: 6.895%"
echo ""
echo "Strategy:"
echo "  1. Train large models (DINOv2, EfficientNet-V2-L, Swin-Large)"
echo "  2. Download external data (background)"
echo "  3. MedSAM ROI extraction"
echo "  4. Stacking/Meta-learning"
echo "  5. Semi-supervised learning"
echo ""
echo "Estimated time: 12-24 hours"
echo "Log: $LOG_FILE"
echo "=========================================="
echo ""

# Function to run command with logging
run_step() {
    local step_name=$1
    local step_num=$2
    local total_steps=$3
    shift 3
    local cmd="$@"

    echo ""
    echo "=========================================="
    echo "[$step_num/$total_steps] $step_name"
    echo "=========================================="
    echo "Command: $cmd"
    echo "Started: $(date)"
    echo ""

    if $cmd 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✓ Completed: $step_name${NC}"
    else
        echo -e "${RED}✗ Failed: $step_name${NC}"
        exit 1
    fi

    echo "Finished: $(date)"
}

TOTAL_STEPS=8

# ==========================================
# PHASE 1: Train New Large Models (6-8 hours)
# ==========================================
echo ""
echo "=========================================="
echo "PHASE 1: Training Large Models"
echo "=========================================="
echo ""

# Check GPU
nvidia-smi || { echo "GPU not available!"; exit 1; }

# Step 1: DINOv2-Large
run_step "Train DINOv2-Large" 1 $TOTAL_STEPS \
    python src/train_v2.py --config configs/dinov2_large.yaml

# Step 2: EfficientNet-V2-L
run_step "Train EfficientNet-V2-L" 2 $TOTAL_STEPS \
    python src/train_v2.py --config configs/efficientnetv2_l.yaml

# Step 3: Swin-Large
run_step "Train Swin-Large" 3 $TOTAL_STEPS \
    python src/train_v2.py --config configs/swin_large.yaml

# ==========================================
# PHASE 2: Generate Predictions for Stacking
# ==========================================
echo ""
echo "=========================================="
echo "PHASE 2: Generate Validation Predictions"
echo "=========================================="
echo ""

# Generate predictions from all models for meta-learning
run_step "Generate validation predictions for stacking" 4 $TOTAL_STEPS \
    python scripts/generate_validation_predictions.py

# ==========================================
# PHASE 3: Stacking/Meta-Learning (CRITICAL!)
# ==========================================
echo ""
echo "=========================================="
echo "PHASE 3: Stacking Meta-Learner"
echo "=========================================="
echo ""
echo "This is the KEY breakthrough technique!"
echo "Expected: +1-3% improvement"
echo ""

run_step "Train stacking meta-learner" 5 $TOTAL_STEPS \
    python scripts/stacking_meta_learner.py

# Generate test predictions with stacking
run_step "Generate stacking ensemble predictions" 6 $TOTAL_STEPS \
    python scripts/stacking_predict.py

# ==========================================
# PHASE 4: MedSAM ROI Extraction (if available)
# ==========================================
echo ""
echo "=========================================="
echo "PHASE 4: MedSAM ROI Extraction (Optional)"
echo "=========================================="
echo ""

if [ -f "external_data/medsam_vit_b.pth" ]; then
    echo "✓ MedSAM checkpoint found"
    run_step "Extract lung ROIs with MedSAM" 7 $TOTAL_STEPS \
        python scripts/medsam_roi_extraction.py

    echo "Training on ROI-focused data..."
    run_step "Train model on MedSAM ROIs" 8 $TOTAL_STEPS \
        python src/train_v2.py --config configs/dinov2_large_medsam_roi.yaml
else
    echo "⚠️  MedSAM not available, skipping"
    echo "To download: bash scripts/download_external_data.sh"
fi

# ==========================================
# PHASE 5: Final Ensemble
# ==========================================
echo ""
echo "=========================================="
echo "PHASE 5: Final Super-Ensemble"
echo "=========================================="
echo ""
echo "Combining:"
echo "  - 18 original models"
echo "  - 3 new large models (DINOv2, EfficientNet-V2-L, Swin-Large)"
echo "  - 1 MedSAM ROI model (if available)"
echo "  - Stacking meta-learner"
echo ""

python scripts/final_super_ensemble.py

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "BREAKTHROUGH PIPELINE COMPLETED"
echo "=========================================="
echo ""

# Show latest submission
if [ -f "data/submission_stacking_final.csv" ]; then
    echo -e "${GREEN}✓ Final submission ready:${NC}"
    echo "  data/submission_stacking_final.csv"
    echo ""
    echo "To submit:"
    echo "  kaggle competitions submit -c cxr-multi-label-classification \\"
    echo "    -f data/submission_stacking_final.csv \\"
    echo "    -m 'Stacking Meta-Learner + Large Models (DINOv2, EffNet-V2-L, Swin-L)'"
    echo ""
fi

# Show expected improvement
echo ""
echo "Expected Performance:"
echo "  Current best: 84.19%"
echo "  After stacking: ~87-90%"
echo "  After external data: ~90-93%"
echo ""
echo "If score < 91%, next steps:"
echo "  1. Download external data: bash scripts/download_external_data.sh"
echo "  2. Train with external pretraining"
echo "  3. Semi-supervised learning with pseudo-labels"
echo ""

echo "Pipeline completed: $(date)"
echo "Full log: $LOG_FILE"
echo "=========================================="
