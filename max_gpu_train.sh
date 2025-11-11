#!/bin/bash
# 榨乾 GPU 的自動化訓練流程
# 使用更大的 batch size 和模型

set -e

echo "=========================================="
echo "🚀 榨乾 GPU - 5-Fold CV 訓練"
echo "=========================================="
echo "配置: EfficientNet-V2-S, 384px, Batch 48"
echo "開始時間: $(date)"
echo ""

mkdir -p outputs/kfold_max_gpu outputs/max_gpu_logs

for fold in 0 1 2 3 4; do
    echo ""
    echo "=========================================="
    echo "🔥 Fold $fold/5 - 榨乾 GPU"
    echo "=========================================="

    # 生成配置
    python3 - <<GENCONFIG
fold = $fold
with open('configs/kfold_max_gpu.yaml', 'r') as f:
    config = f.read()

config = config.replace('train_csv: data/fold0_train.csv', f'train_csv: data/fold{fold}_train.csv')
config = config.replace('val_csv: data/fold0_val.csv', f'val_csv: data/fold{fold}_val.csv')
config = config.replace('dir: outputs/kfold_max_gpu/fold{fold_id}', f'dir: outputs/kfold_max_gpu/fold{fold}')
config = config.replace('checkpoint_path: outputs/kfold_max_gpu/fold{fold_id}/best.pt', f'checkpoint_path: outputs/kfold_max_gpu/fold{fold}/best.pt')
config = config.replace('submission_path: data/submission_max_gpu_fold{fold_id}.csv', f'submission_path: data/submission_max_gpu_fold{fold}.csv')

with open(f'configs/kfold_max_gpu_fold{fold}.yaml', 'w') as f:
    f.write(config)
print(f"✅ Fold {fold} 配置已生成 (Batch 48, EfficientNet-V2-S, 384px)")
GENCONFIG

    # 訓練
    echo "🚀 開始訓練 Fold $fold (預計 25-30 分鐘)..."
    python3 -m src.train_v2 --config configs/kfold_max_gpu_fold$fold.yaml 2>&1 | tee outputs/max_gpu_logs/fold${fold}_train.log

    echo "✅ Fold $fold 完成: $(date)"

    # 顯示 GPU 峰值使用
    echo "📊 GPU 峰值統計:"
    nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
done

echo ""
echo "=========================================="
echo "🔮 生成測試集預測"
echo "=========================================="

for fold in 0 1 2 3 4; do
    echo "Fold $fold 預測..."
    python3 -m src.predict \
        --config configs/kfold_max_gpu_fold$fold.yaml \
        --ckpt outputs/kfold_max_gpu/fold$fold/best.pt \
        2>&1 | tee outputs/max_gpu_logs/fold${fold}_predict.log
done

echo ""
echo "=========================================="
echo "🎯 集成預測"
echo "=========================================="

python3 - <<ENSEMBLE
import pandas as pd
import numpy as np
from pathlib import Path

predictions = []
for fold in range(5):
    pred_file = f'data/submission_max_gpu_fold{fold}.csv'
    if Path(pred_file).exists():
        df = pd.read_csv(pred_file)
        predictions.append(df)
        print(f"  ✅ Fold {fold}: {len(df)} 預測")

ensemble_df = predictions[0].copy()
label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

for col in label_cols:
    predictions_array = np.array([pred[col].values for pred in predictions])
    ensemble_df[col] = predictions_array.mean(axis=0)

ensemble_df[label_cols] = ensemble_df[label_cols].div(ensemble_df[label_cols].sum(axis=1), axis=0)

for idx, row in ensemble_df.iterrows():
    max_col = row[label_cols].idxmax()
    for col in label_cols:
        ensemble_df.at[idx, col] = 1.0 if col == max_col else 0.0

ensemble_df.to_csv('data/submission_max_gpu_ensemble.csv', index=False)
print(f"\n🚀 最終預測: data/submission_max_gpu_ensemble.csv")
print(f"   樣本數: {len(ensemble_df)}")

for col in label_cols:
    count = ensemble_df[col].sum()
    print(f"  {col:15s}: {int(count):4d} ({count/len(ensemble_df)*100:5.2f}%)")
ENSEMBLE

echo ""
echo "=========================================="
echo "✅ 完成！GPU 已榨乾！"
echo "=========================================="
echo "結束時間: $(date)"
echo ""
echo "🎯 最終提交: data/submission_max_gpu_ensemble.csv"
echo ""
