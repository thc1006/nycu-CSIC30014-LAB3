#!/bin/bash
# 🔥 最終優化訓練 - 榨乾 GPU

set -e

echo "=========================================="
echo "🔥 最終優化訓練 - 榨乾 GPU"
echo "=========================================="
echo "Batch 56 + EfficientNet-V2-S @ 352px"
echo "醫學預處理: CLAHE + Unsharp Masking"
echo "開始: $(date)"
echo ""

mkdir -p outputs/final_optimized outputs/final_logs

for fold in 0 1 2 3 4; do
    echo ""
    echo "=========================================="
    echo "🔥 Fold $fold/5"
    echo "=========================================="

    python3 - <<GENCONFIG
fold = $fold
with open('configs/final_optimized.yaml', 'r') as f:
    config = f.read()

config = config.replace('train_csv: data/fold0_train.csv', f'train_csv: data/fold{fold}_train.csv')
config = config.replace('val_csv: data/fold0_val.csv', f'val_csv: data/fold{fold}_val.csv')
config = config.replace('dir: outputs/final_optimized/fold{fold_id}', f'dir: outputs/final_optimized/fold{fold}')
config = config.replace('checkpoint_path: outputs/final_optimized/fold{fold_id}/best.pt', f'checkpoint_path: outputs/final_optimized/fold{fold}/best.pt')
config = config.replace('submission_path: data/submission_final_fold{fold_id}.csv', f'submission_path: data/submission_final_fold{fold}.csv')

with open(f'configs/final_optimized_fold{fold}.yaml', 'w') as f:
    f.write(config)
print(f"✅ Fold {fold}")
GENCONFIG

    python3 -m src.train_v2 --config configs/final_optimized_fold$fold.yaml 2>&1 | tee outputs/final_logs/fold${fold}_train.log

    echo "✅ Fold $fold: $(date)"
    nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
done

echo ""
echo "=========================================="
echo "🔮 生成預測"
echo "=========================================="

for fold in 0 1 2 3 4; do
    python3 -m src.predict \
        --config configs/final_optimized_fold$fold.yaml \
        --ckpt outputs/final_optimized/fold$fold/best.pt \
        2>&1 | tee outputs/final_logs/fold${fold}_predict.log
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
    f = f'data/submission_final_fold{fold}.csv'
    if Path(f).exists():
        predictions.append(pd.read_csv(f))

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

ensemble_df.to_csv('data/submission_final.csv', index=False)
print(f"\n🚀 最終提交: data/submission_final.csv")
print(f"   樣本: {len(ensemble_df)}")
ENSEMBLE

echo ""
echo "🎯 完成: data/submission_final.csv"
echo "完成時間: $(date)"
