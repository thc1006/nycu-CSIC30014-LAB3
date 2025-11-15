#!/usr/bin/env python3
"""
終極 Stacking 預測：結合原始 + TTA 預測
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def load_test_predictions(use_tta=False):
    """載入所有測試集預測"""
    pred_dir = 'data/test_predictions_tta' if use_tta else 'data/test_predictions'

    if not os.path.exists(pred_dir):
        print(f"⚠️ 預測目錄不存在: {pred_dir}")
        return None

    pred_files = sorted(Path(pred_dir).glob('*.csv'))

    if len(pred_files) == 0:
        print(f"⚠️ 沒有找到預測文件: {pred_dir}")
        return None

    print(f"\n從 {pred_dir} 載入預測...")
    print(f"找到 {len(pred_files)} 個預測文件")

    # 讀取所有預測
    all_preds = {}
    filenames = None

    for pred_file in pred_files:
        df = pd.read_csv(pred_file)

        # 保存文件名（第一次）
        if filenames is None:
            filenames = df['new_filename'].values

        # 提取預測概率
        class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
        probs = df[class_cols].values

        # 使用文件名作為 key
        model_name = pred_file.stem.replace('_tta6x', '')
        all_preds[model_name] = probs

    print(f"✓ 載入 {len(all_preds)} 個模型的預測")
    print(f"  樣本數: {len(filenames)}")

    return all_preds, filenames

def create_meta_features(all_preds):
    """創建元特徵"""
    n_samples = len(next(iter(all_preds.values())))
    n_classes = 4

    # 收集所有模型的預測
    model_names = sorted(all_preds.keys())
    n_models = len(model_names)

    # 基礎特徵：每個模型的預測概率
    base_features = np.zeros((n_samples, n_models * n_classes))
    for i, model_name in enumerate(model_names):
        base_features[:, i*n_classes:(i+1)*n_classes] = all_preds[model_name]

    # 統計特徵
    all_probs = np.stack([all_preds[name] for name in model_names], axis=0)  # (n_models, n_samples, n_classes)

    # 每個類別的平均預測
    mean_probs = np.mean(all_probs, axis=0)  # (n_samples, n_classes)

    # 每個類別的標準差
    std_probs = np.std(all_probs, axis=0)  # (n_samples, n_classes)

    # 每個類別的最大預測
    max_probs = np.max(all_probs, axis=0)  # (n_samples, n_classes)

    # 每個類別的最小預測
    min_probs = np.min(all_probs, axis=0)  # (n_samples, n_classes)

    # 組合所有特徵
    meta_features = np.concatenate([
        base_features,      # n_models * n_classes
        mean_probs,         # n_classes
        std_probs,          # n_classes
        max_probs,          # n_classes
        min_probs           # n_classes
    ], axis=1)

    print(f"\n✓ 創建元特徵: {meta_features.shape}")
    print(f"  基礎特徵: {base_features.shape[1]}")
    print(f"  統計特徵: {meta_features.shape[1] - base_features.shape[1]}")

    return meta_features

def predict_with_stacking(meta_learner_path, meta_features):
    """使用 Stacking 模型預測"""
    print(f"\n載入 Meta-Learner: {meta_learner_path}")

    with open(meta_learner_path, 'rb') as f:
        meta_models = pickle.load(f)

    # 每個類別獨立預測
    n_samples = meta_features.shape[0]
    predictions = np.zeros((n_samples, 4))

    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

    for i, class_name in enumerate(class_names):
        model = meta_models[class_name]

        # 預測概率
        if hasattr(model, 'predict_proba'):
            # 分類器
            probs = model.predict_proba(meta_features)
            if probs.shape[1] == 2:
                predictions[:, i] = probs[:, 1]  # 正類概率
            else:
                predictions[:, i] = probs[:, 0]
        else:
            # 回歸器
            predictions[:, i] = model.predict(meta_features)

    # 歸一化確保總和為 1
    predictions = np.clip(predictions, 0, 1)
    row_sums = predictions.sum(axis=1, keepdims=True)
    predictions = predictions / (row_sums + 1e-10)

    return predictions

def main():
    print("="*80)
    print("終極 Stacking 預測")
    print("="*80)

    # 嘗試載入 TTA 預測
    print("\n[1] 嘗試載入 TTA 預測...")
    tta_preds, filenames = load_test_predictions(use_tta=True)

    if tta_preds is None:
        print("\n⚠️ TTA 預測不可用，使用原始預測")
        tta_preds, filenames = load_test_predictions(use_tta=False)
        if tta_preds is None:
            print("❌ 錯誤：沒有可用的預測！")
            return

    # 創建元特徵
    print("\n[2] 創建元特徵...")
    meta_features = create_meta_features(tta_preds)

    # 嘗試所有可用的 Meta-Learner
    meta_learner_paths = {
        'MLP': 'models/stacking_mlp.pkl',
        'XGB': 'models/stacking_xgb.pkl',
        'RF': 'models/stacking_rf.pkl',
        'LGB': 'models/stacking_lgb.pkl',
    }

    all_stacking_preds = {}

    for name, path in meta_learner_paths.items():
        if not os.path.exists(path):
            print(f"⚠️ {name} 不存在: {path}")
            continue

        print(f"\n[3] 使用 {name} Meta-Learner 預測...")
        preds = predict_with_stacking(path, meta_features)
        all_stacking_preds[name] = preds

        # 統計
        avg_conf = np.max(preds, axis=1).mean()
        pred_classes = np.argmax(preds, axis=1)
        class_dist = np.bincount(pred_classes, minlength=4)

        print(f"✓ {name} 預測完成")
        print(f"  平均置信度: {avg_conf:.4f}")
        print(f"  預測分布: Normal={class_dist[0]}, Bacteria={class_dist[1]}, "
              f"Virus={class_dist[2]}, COVID-19={class_dist[3]}")

    # 保存所有 Stacking 預測
    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

    for name, preds in all_stacking_preds.items():
        # One-hot 編碼
        pred_classes = np.argmax(preds, axis=1)
        onehot = np.zeros((len(pred_classes), 4), dtype=int)
        onehot[np.arange(len(pred_classes)), pred_classes] = 1

        # 創建提交文件
        output_df = pd.DataFrame({
            'new_filename': filenames
        })

        for i, class_name in enumerate(class_names):
            output_df[class_name] = onehot[:, i]

        output_path = f"data/submission_stacking_{name.lower()}.csv"
        output_df.to_csv(output_path, index=False)

        print(f"\n✅ 已保存 {name} 提交: {output_path}")

    # 集成所有 Stacking 預測
    if len(all_stacking_preds) > 1:
        print("\n[4] 集成所有 Stacking 預測...")

        ensemble_preds = np.mean(list(all_stacking_preds.values()), axis=0)

        # One-hot 編碼
        pred_classes = np.argmax(ensemble_preds, axis=1)
        onehot = np.zeros((len(pred_classes), 4), dtype=int)
        onehot[np.arange(len(pred_classes)), pred_classes] = 1

        output_df = pd.DataFrame({
            'new_filename': filenames
        })

        for i, class_name in enumerate(class_names):
            output_df[class_name] = onehot[:, i]

        output_path = "data/submission_stacking_ensemble.csv"
        output_df.to_csv(output_path, index=False)

        avg_conf = np.max(ensemble_preds, axis=1).mean()
        class_dist = np.bincount(pred_classes, minlength=4)

        print(f"\n✅ 已保存集成提交: {output_path}")
        print(f"  平均置信度: {avg_conf:.4f}")
        print(f"  預測分布: Normal={class_dist[0]}, Bacteria={class_dist[1]}, "
              f"Virus={class_dist[2]}, COVID-19={class_dist[3]}")

    print("\n" + "="*80)
    print("✅ Stacking 預測完成！")
    print("="*80)
    print("\n建議提交:")
    print("  • 最佳單一: submission_stacking_mlp.csv (Val F1: 86.88%)")
    print("  • 集成所有: submission_stacking_ensemble.csv")
    print("\n預期測試分數: 87-90%")

if __name__ == '__main__':
    main()
