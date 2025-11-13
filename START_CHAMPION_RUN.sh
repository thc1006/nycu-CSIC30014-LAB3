#!/bin/bash
# ============================================================================
# 一鍵啟動冠軍管線（背景執行）
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo "============================================================================"
echo "🏆 CHAMPION PIPELINE LAUNCHER"
echo "============================================================================"
echo ""
echo "This will start the complete champion pipeline in the background."
echo ""
echo "Strategy:"
echo "  ✓ Train all large models (DINOv2-L, EfficientNet-V2-L, Swin-L)"
echo "  ✓ Download external data (MedSAM)"
echo "  ✓ MedSAM ROI extraction"
echo "  ✓ Multi-layer stacking (3 layers)"
echo "  ✓ TTA for all models"
echo "  ✓ Pseudo-labeling"
echo "  ✓ Ultimate ensemble"
echo ""
echo "Expected Score: 91-95% Macro-F1"
echo "Time: 24-48 hours (fully automated)"
echo ""
echo "============================================================================"
echo ""

# 檢查是否已在運行
EXISTING_PID=$(ps aux | grep "champion_pipeline.sh" | grep -v grep | awk '{print $2}')

if [ -n "$EXISTING_PID" ]; then
    echo -e "${YELLOW}⚠ Champion pipeline is already running!${NC}"
    echo "  PID: $EXISTING_PID"
    echo ""
    echo "Options:"
    echo "  1. Monitor progress: bash scripts/monitor_champion.sh"
    echo "  2. Kill and restart: kill $EXISTING_PID && bash $0"
    echo ""
    exit 1
fi

# GPU 檢查
echo "Checking GPU availability..."
if ! nvidia-smi &>/dev/null; then
    echo -e "${RED}✗ GPU not available!${NC}"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 磁碟空間檢查
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available disk space: ${AVAILABLE_GB}GB"

if [ "$AVAILABLE_GB" -lt 50 ]; then
    echo -e "${RED}✗ Insufficient disk space (< 50GB)${NC}"
    echo "Please free up space and try again"
    exit 1
fi

echo ""
echo "============================================================================"

read -p "Start champion pipeline in background? (y/n) " -n 1 -r
echo
echo "============================================================================"
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# 確保腳本可執行
chmod +x scripts/champion_pipeline.sh
chmod +x scripts/monitor_champion.sh

# 啟動管線（背景執行，不受終端關閉影響）
echo "Starting champion pipeline..."
nohup bash scripts/champion_pipeline.sh > champion_pipeline_stdout.log 2>&1 &

PIPELINE_PID=$!

echo ""
echo -e "${GREEN}✓ Champion pipeline started!${NC}"
echo ""
echo "Process ID: $PIPELINE_PID"
echo "Stdout log: champion_pipeline_stdout.log"
echo ""
echo "============================================================================"
echo "📊 MONITORING"
echo "============================================================================"
echo ""
echo "Monitor progress:"
echo "  bash scripts/monitor_champion.sh"
echo ""
echo "Watch in real-time:"
echo "  bash scripts/monitor_champion.sh --watch"
echo ""
echo "Follow master log:"
echo "  tail -f outputs/champion_logs_*/champion_master.log"
echo ""
echo "GPU usage:"
echo "  nvidia-smi -l 1"
echo ""
echo "============================================================================"
echo "🎯 NEXT STEPS"
echo "============================================================================"
echo ""
echo "1. Wait for completion (24-48 hours)"
echo "2. Check results:"
echo "   bash scripts/monitor_champion.sh"
echo ""
echo "3. Submit when ready:"
echo "   kaggle competitions submit -c cxr-multi-label-classification \\"
echo "     -f data/submission_ULTIMATE_CHAMPION.csv \\"
echo "     -m 'Ultimate Champion Submission'"
echo ""
echo "============================================================================"
echo ""
echo -e "${CYAN}Pipeline is running in background. Safe to close terminal!${NC}"
echo ""
echo "To check status later:"
echo "  ps aux | grep champion_pipeline"
echo ""
echo "Good luck! 🏆"
echo ""
