#!/usr/bin/env python3
"""
K-Fold Cross Validation 訓練腳本
自動訓練所有 5 個 fold 並集成預測
"""
import subprocess
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

print("=" * 80)
print("K-Fold Cross Validation 訓練")
print("=" * 80)

# 配置
N_FOLDS = 5
BASE_CONFIG = "configs/kfold_medical_enhanced.yaml"
OUTPUT_BASE = Path("outputs/kfold_run")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# 讀取 fold 資訊
with open('data/kfold_info.json', 'r') as f:
    fold_info = json.load(f)

print(f"\n總共 {N_FOLDS} 個 folds")
print(f"配置檔案: {BASE_CONFIG}")
print(f"輸出目錄: {OUTPUT_BASE}")

# 訓練每個 fold
fold_results = []

for fold_idx in range(N_FOLDS):
    print(f"\n{'='*80}")
    print(f"訓練 Fold {fold_idx + 1}/{N_FOLDS}")
    print(f"{'='*80}")

    # 創建 fold 特定的配置
    fold_config_path = f"configs/kfold_fold{fold_idx}.yaml"

    # 讀取基礎配置
    with open(BASE_CONFIG, 'r') as f:
        config_content = f.read()

    # 替換 fold 相關的路徑
    config_content = config_content.replace(
        'train_csv: data/fold0_train.csv',
        f'train_csv: data/fold{fold_idx}_train.csv'
    )
    config_content = config_content.replace(
        'val_csv: data/fold0_val.csv',
        f'val_csv: data/fold{fold_idx}_val.csv'
    )
    config_content = config_content.replace(
        'dir: outputs/kfold_run/fold{fold_id}',
        f'dir: outputs/kfold_run/fold{fold_idx}'
    )
    config_content = config_content.replace(
        'checkpoint_path: outputs/kfold_run/fold{fold_id}/best.pt',
        f'checkpoint_path: outputs/kfold_run/fold{fold_idx}/best.pt'
    )
    config_content = config_content.replace(
        'submission_path: data/submission_kfold_fold{fold_id}.csv',
        f'submission_path: data/submission_kfold_fold{fold_idx}.csv'
    )

    # 儲存 fold 配置
    with open(fold_config_path, 'w') as f:
        f.write(config_content)

    print(f"✅ 配置檔案已生成: {fold_config_path}")

    # 訓練此 fold
    cmd = [
        sys.executable, '-m', 'src.train_v2',
        '--config', fold_config_path
    ]

    print(f"🚀 開始訓練...")
    print(f"指令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ Fold {fold_idx} 訓練完成！")

        # 記錄結果
        fold_results.append({
            'fold': fold_idx,
            'status': 'success',
            'checkpoint': f'outputs/kfold_run/fold{fold_idx}/best.pt'
        })

    except subprocess.CalledProcessError as e:
        print(f"❌ Fold {fold_idx} 訓練失敗: {e}")
        fold_results.append({
            'fold': fold_idx,
            'status': 'failed',
            'error': str(e)
        })
        continue

# 儲存訓練結果
with open('data/kfold_training_results.json', 'w') as f:
    json.dump(fold_results, f, indent=2)

print(f"\n{'='*80}")
print("訓練結果摘要")
print(f"{'='*80}")

success_count = sum(1 for r in fold_results if r['status'] == 'success')
print(f"成功: {success_count}/{N_FOLDS}")
print(f"失敗: {N_FOLDS - success_count}/{N_FOLDS}")

if success_count == 0:
    print("\n❌ 所有 fold 訓練都失敗了！")
    sys.exit(1)

print(f"\n✅ K-Fold 訓練完成！")
print(f"下一步: 運行集成預測腳本")
