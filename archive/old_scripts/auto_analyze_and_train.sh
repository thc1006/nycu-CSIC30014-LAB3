#!/bin/bash
# 完全自動化的分析和訓練流程
# 這個腳本會在你睡覺時自動運行完整個流程

set -e  # 遇到錯誤立即停止

echo "=========================================="
echo "全自動化 K-Fold CV 訓練流程"
echo "=========================================="
echo "開始時間: $(date)"
echo ""

# 創建輸出目錄
mkdir -p outputs/kfold_run outputs/auto_analysis_logs

# Step 1: 訓練所有 5 個 folds
echo "=========================================="
echo "Step 1/3: 訓練 5-Fold Cross Validation"
echo "=========================================="

for fold in 0 1 2 3 4; do
    echo ""
    echo "=========================================="
    echo "訓練 Fold $fold/5"
    echo "=========================================="
    echo "開始時間: $(date)"

    # 生成配置
    python3 - <<GENCONFIG
import sys
fold = $fold
with open('configs/kfold_medical_enhanced.yaml', 'r') as f:
    config = f.read()

config = config.replace('train_csv: data/fold0_train.csv', f'train_csv: data/fold{fold}_train.csv')
config = config.replace('val_csv: data/fold0_val.csv', f'val_csv: data/fold{fold}_val.csv')
config = config.replace('dir: outputs/kfold_run/fold{fold_id}', f'dir: outputs/kfold_run/fold{fold}')
config = config.replace('checkpoint_path: outputs/kfold_run/fold{fold_id}/best.pt', f'checkpoint_path: outputs/kfold_run/fold{fold}/best.pt')
config = config.replace('submission_path: data/submission_kfold_fold{fold_id}.csv', f'submission_path: data/submission_kfold_fold{fold}.csv')

with open(f'configs/kfold_fold{fold}.yaml', 'w') as f:
    f.write(config)
print(f"✅ Fold {fold} 配置已生成")
GENCONFIG

    # 訓練
    python3 -m src.train_v2 --config configs/kfold_fold$fold.yaml 2>&1 | tee outputs/auto_analysis_logs/fold${fold}_train.log

    echo "✅ Fold $fold 訓練完成: $(date)"
    echo ""
done

echo ""
echo "=========================================="
echo "Step 2/3: 生成每個 Fold 的測試集預測"
echo "=========================================="

for fold in 0 1 2 3 4; do
    echo "生成 Fold $fold 的預測..."
    python3 -m src.predict \
        --config configs/kfold_fold$fold.yaml \
        --ckpt outputs/kfold_run/fold$fold/best.pt \
        2>&1 | tee outputs/auto_analysis_logs/fold${fold}_predict.log
    echo "✅ Fold $fold 預測完成"
done

echo ""
echo "=========================================="
echo "Step 3/3: 集成 5 個模型的預測"
echo "=========================================="

python3 - <<ENSEMBLE
import pandas as pd
import numpy as np
from pathlib import Path

print("讀取所有 fold 的預測...")

# 讀取所有預測
predictions = []
for fold in range(5):
    pred_file = f'data/submission_kfold_fold{fold}.csv'
    if Path(pred_file).exists():
        df = pd.read_csv(pred_file)
        predictions.append(df)
        print(f"  ✅ Fold {fold}: {len(df)} 個預測")
    else:
        print(f"  ⚠️  Fold {fold} 預測檔案不存在，跳過")

if len(predictions) == 0:
    print("❌ 沒有找到任何預測檔案！")
    exit(1)

# 集成預測 (平均)
print(f"\n集成 {len(predictions)} 個模型的預測...")

ensemble_df = predictions[0].copy()
label_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

for col in label_cols:
    # 對每個類別，計算所有模型的平均預測
    predictions_array = np.array([pred[col].values for pred in predictions])
    ensemble_df[col] = predictions_array.mean(axis=0)

# 重新歸一化（確保每行和為1）
ensemble_df[label_cols] = ensemble_df[label_cols].div(ensemble_df[label_cols].sum(axis=1), axis=0)

# One-hot 編碼
for idx, row in ensemble_df.iterrows():
    max_col = row[label_cols].idxmax()
    for col in label_cols:
        ensemble_df.at[idx, col] = 1.0 if col == max_col else 0.0

# 儲存
ensemble_df.to_csv('data/submission_kfold_ensemble.csv', index=False)
print(f"\n✅ 集成預測已保存至: data/submission_kfold_ensemble.csv")
print(f"   總計 {len(ensemble_df)} 個測試樣本")

# 統計
print(f"\n預測統計:")
for col in label_cols:
    count = ensemble_df[col].sum()
    print(f"  {col:15s}: {int(count):4d} ({count/len(ensemble_df)*100:5.2f}%)")

ENSEMBLE

echo ""
echo "=========================================="
echo "訓練完成！"
echo "=========================================="
echo "結束時間: $(date)"
echo ""
echo "輸出檔案:"
echo "  - 最終提交檔案: data/submission_kfold_ensemble.csv"
echo "  - 模型檢查點: outputs/kfold_run/fold*/best.pt"
echo "  - 訓練日誌: outputs/auto_analysis_logs/"
echo ""
echo "下一步: 將 data/submission_kfold_ensemble.csv 上傳至 Kaggle"
echo ""
