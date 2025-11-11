#!/bin/bash
# 🔥 終極榨乾 GPU - GPU Caching + 大 Batch

set -e

echo "=========================================="
echo "🔥🔥🔥 終極榨乾 GPU 模式 🔥🔥🔥"
echo "=========================================="
echo "GPU Caching: ✅ (消除 I/O)"
echo "Batch Size: 64"
echo "模型: EfficientNet-B0 (FP16)"
echo "影像: 320px"
echo "開始: $(date)"
echo ""

mkdir -p outputs/ultimate_gpu outputs/ultimate_logs

for fold in 0 1 2 3 4; do
    echo ""
    echo "=========================================="
    echo "🔥 Fold $fold/5 - 終極模式"
    echo "=========================================="

    # 生成配置
    python3 - <<GENCONFIG
fold = $fold
with open('configs/ultimate_gpu.yaml', 'r') as f:
    config = f.read()

config = config.replace('train_csv: data/fold0_train.csv', f'train_csv: data/fold{fold}_train.csv')
config = config.replace('val_csv: data/fold0_val.csv', f'val_csv: data/fold{fold}_val.csv')
config = config.replace('dir: outputs/ultimate_gpu/fold{fold_id}', f'dir: outputs/ultimate_gpu/fold{fold}')
config = config.replace('checkpoint_path: outputs/ultimate_gpu/fold{fold_id}/best.pt', f'checkpoint_path: outputs/ultimate_gpu/fold{fold}/best.pt')
config = config.replace('submission_path: data/submission_ultimate_fold{fold_id}.csv', f'submission_path: data/submission_ultimate_fold{fold}.csv')

with open(f'configs/ultimate_gpu_fold{fold}.yaml', 'w') as f:
    f.write(config)
print(f"✅ Fold {fold} 終極配置 (GPU Cache + Batch 64)")
GENCONFIG

    echo "🚀 載入影像至 GPU VRAM..."
    python3 -m src.train_v2 --config configs/ultimate_gpu_fold$fold.yaml 2>&1 | tee outputs/ultimate_logs/fold${fold}_train.log

    echo "✅ Fold $fold 完成: $(date)"
    nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
done

echo ""
echo "=========================================="
echo "🔮 生成預測"
echo "=========================================="

for fold in 0 1 2 3 4; do
    python3 -m src.predict \
        --config configs/ultimate_gpu_fold$fold.yaml \
        --ckpt outputs/ultimate_gpu/fold$fold/best.pt \
        2>&1 | tee outputs/ultimate_logs/fold${fold}_predict.log
done

echo ""
echo "=========================================="
echo "🎯 集成"
echo "=========================================="

python3 - <<ENSEMBLE
import pandas as pd
import numpy as np
from pathlib import Path

predictions = []
for fold in range(5):
    f = f'data/submission_ultimate_fold{fold}.csv'
    if Path(f).exists():
        predictions.append(pd.read_csv(f))
        print(f"  ✅ Fold {fold}")

ensemble_df = predictions[0].copy()
label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

for col in label_cols:
    arr = np.array([p[col].values for p in predictions])
    ensemble_df[col] = arr.mean(axis=0)

ensemble_df[label_cols] = ensemble_df[label_cols].div(ensemble_df[label_cols].sum(axis=1), axis=0)

for idx, row in ensemble_df.iterrows():
    max_col = row[label_cols].idxmax()
    for col in label_cols:
        ensemble_df.at[idx, col] = 1.0 if col == max_col else 0.0

ensemble_df.to_csv('data/submission_ultimate_ensemble.csv', index=False)
print(f"\n🔥 終極預測: data/submission_ultimate_ensemble.csv")
ENSEMBLE

echo ""
echo "=========================================="
echo "🔥 GPU 已完全榨乾！"
echo "=========================================="
echo "完成: $(date)"
echo "提交: data/submission_ultimate_ensemble.csv"
