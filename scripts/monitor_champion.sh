#!/bin/bash
# ============================================================================
# Champion Pipeline Monitor
# 實時監控訓練進度，自動恢復
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 找到最新的 log 目錄
LATEST_LOG=$(ls -td outputs/champion_logs_* 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No champion pipeline logs found"
    echo "Start the pipeline with: nohup bash scripts/champion_pipeline.sh &"
    exit 1
fi

MASTER_LOG="$LATEST_LOG/champion_master.log"
PROGRESS_FILE="$LATEST_LOG/progress.txt"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

clear

echo "============================================================================"
echo "🏆 CHAMPION PIPELINE MONITOR"
echo "============================================================================"
echo ""
echo "Log Directory: $LATEST_LOG"
echo "Started: $(ls -l "$LATEST_LOG" | head -2 | tail -1 | awk '{print $6, $7, $8}')"
echo ""

# 檢查進程
PIPELINE_PID=$(ps aux | grep "champion_pipeline.sh" | grep -v grep | grep -v monitor | awk '{print $2}')

if [ -n "$PIPELINE_PID" ]; then
    echo -e "${GREEN}✓ Pipeline is running${NC} (PID: $PIPELINE_PID)"
else
    echo -e "${YELLOW}⚠ Pipeline not running${NC}"
fi

echo ""
echo "============================================================================"
echo "📊 PROGRESS"
echo "============================================================================"
echo ""

if [ -f "$PROGRESS_FILE" ]; then
    while IFS= read -r line; do
        echo -e "${GREEN}✓${NC} $line"
    done < "$PROGRESS_FILE"
else
    echo "No progress recorded yet"
fi

echo ""
echo "============================================================================"
echo "🎯 CURRENT STATUS"
echo "============================================================================"
echo ""

# GPU 狀態
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s%% utilized, %sMB / %sMB\n", $1, $3, $4, $5}'

echo ""

# 訓練中的模型
TRAINING_MODELS=$(find "$LATEST_LOG" -name "*_train.log" -type f -mmin -30 | wc -l)
if [ "$TRAINING_MODELS" -gt 0 ]; then
    echo -e "${CYAN}Currently Training:${NC}"
    find "$LATEST_LOG" -name "*_train.log" -type f -mmin -30 | while read log; do
        model=$(basename "$log" | sed 's/_train.log//')
        last_line=$(tail -1 "$log" 2>/dev/null)
        echo "  - $model: $last_line"
    done
else
    echo "No active training (last 30 min)"
fi

echo ""

# 完成的模型
COMPLETED_MODELS=$(find "$LATEST_LOG" -name "*_done" -type f | wc -l)
echo "Completed Models: $COMPLETED_MODELS"

# 生成的提交文件
SUBMISSIONS=$(ls data/submission_*.csv 2>/dev/null | wc -l)
echo "Generated Submissions: $SUBMISSIONS"

echo ""
echo "============================================================================"
echo "📝 RECENT LOG (last 20 lines)"
echo "============================================================================"
echo ""

if [ -f "$MASTER_LOG" ]; then
    tail -20 "$MASTER_LOG"
else
    echo "Master log not found"
fi

echo ""
echo "============================================================================"
echo "⌨️  COMMANDS"
echo "============================================================================"
echo ""
echo "  tail -f $MASTER_LOG"
echo "    → Follow master log in real-time"
echo ""
echo "  nvidia-smi -l 1"
echo "    → Monitor GPU usage"
echo ""
echo "  watch -n 5 'bash scripts/monitor_champion.sh'"
echo "    → Auto-refresh this monitor every 5 seconds"
echo ""
echo "  bash scripts/monitor_champion.sh --watch"
echo "    → Auto-refresh mode"
echo ""

# Auto-refresh mode
if [ "$1" == "--watch" ]; then
    echo "Auto-refresh enabled (Ctrl+C to exit)"
    while true; do
        sleep 5
        bash "$0"
    done
fi
