#!/usr/bin/env python3
"""
深度分析胸部X光資料集
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_csv_data():
    """分析 CSV 檔案的統計資訊"""
    data_dir = Path("data")

    results = {
        "csv_analysis": {},
        "class_distribution": {},
        "file_info": {}
    }

    # 分析訓練集
    print("=" * 80)
    print("分析訓練集 (train_data.csv)")
    print("=" * 80)
    train_df = pd.read_csv(data_dir / "train_data.csv")
    print(f"\n訓練集大小: {len(train_df)} 筆")
    print(f"欄位: {list(train_df.columns)}")

    # 類別分布
    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
    train_class_dist = {}
    for col in class_cols:
        count = train_df[col].sum()
        percentage = (count / len(train_df)) * 100
        train_class_dist[col] = {
            "count": int(count),
            "percentage": round(percentage, 2)
        }
        print(f"  {col}: {count} 筆 ({percentage:.2f}%)")

    results["csv_analysis"]["train"] = {
        "total": len(train_df),
        "class_distribution": train_class_dist
    }

    # 分析驗證集
    print("\n" + "=" * 80)
    print("分析驗證集 (val_data.csv)")
    print("=" * 80)
    val_df = pd.read_csv(data_dir / "val_data.csv")
    print(f"\n驗證集大小: {len(val_df)} 筆")

    val_class_dist = {}
    for col in class_cols:
        count = val_df[col].sum()
        percentage = (count / len(val_df)) * 100
        val_class_dist[col] = {
            "count": int(count),
            "percentage": round(percentage, 2)
        }
        print(f"  {col}: {count} 筆 ({percentage:.2f}%)")

    results["csv_analysis"]["val"] = {
        "total": len(val_df),
        "class_distribution": val_class_dist
    }

    # 分析測試集
    print("\n" + "=" * 80)
    print("分析測試集 (test_data.csv)")
    print("=" * 80)
    test_df = pd.read_csv(data_dir / "test_data.csv")
    print(f"\n測試集大小: {len(test_df)} 筆")
    print(f"欄位: {list(test_df.columns)}")

    results["csv_analysis"]["test"] = {
        "total": len(test_df),
        "columns": list(test_df.columns)
    }

    # 檢查檔案名稱格式
    print("\n" + "=" * 80)
    print("檔案名稱格式分析")
    print("=" * 80)

    all_files = pd.concat([
        train_df['new_filename'],
        val_df['new_filename'],
        test_df['new_filename']
    ])

    # 檔案副檔名統計
    extensions = all_files.str.split('.').str[-1].value_counts()
    print(f"\n檔案副檔名分布:")
    for ext, count in extensions.items():
        print(f"  .{ext}: {count} 個檔案")

    # 檔案名稱是否為數字
    numeric_names = all_files.str.replace('.jpeg', '').str.replace('.jpg', '').str.isnumeric().sum()
    print(f"\n數字命名的檔案: {numeric_names}/{len(all_files)}")

    results["file_info"] = {
        "total_files": len(all_files),
        "extensions": {k: int(v) for k, v in extensions.items()},
        "numeric_names": int(numeric_names)
    }

    # 類別不平衡分析
    print("\n" + "=" * 80)
    print("類別不平衡分析")
    print("=" * 80)

    total_samples = len(train_df)
    imbalance_ratios = {}

    for col in class_cols:
        count = train_df[col].sum()
        ratio = count / total_samples
        imbalance = total_samples / count if count > 0 else float('inf')
        imbalance_ratios[col] = {
            "samples": int(count),
            "ratio": round(ratio, 4),
            "imbalance_factor": round(imbalance, 2)
        }
        print(f"\n{col}:")
        print(f"  樣本數: {count}")
        print(f"  比例: {ratio:.4f}")
        print(f"  不平衡因子: {imbalance:.2f}x")

    results["class_distribution"]["imbalance"] = imbalance_ratios

    # 建議的類別權重 (用於 Focal Loss)
    print("\n" + "=" * 80)
    print("建議的損失函數類別權重")
    print("=" * 80)

    # 方法1: 逆頻率
    inverse_freq_weights = {}
    max_count = max([train_df[col].sum() for col in class_cols])

    print("\n方法1: 逆頻率權重 (Inverse Frequency)")
    for col in class_cols:
        count = train_df[col].sum()
        weight = max_count / count if count > 0 else 1.0
        inverse_freq_weights[col] = round(weight, 2)
        print(f"  {col}: {weight:.2f}")

    # 方法2: 有效樣本數 (Effective Number of Samples)
    beta = 0.9999
    effective_weights = {}

    print(f"\n方法2: 有效樣本數權重 (β={beta})")
    for col in class_cols:
        count = train_df[col].sum()
        if count > 0:
            weight = (1 - beta) / (1 - beta ** count)
            weight = weight / max([1 for _ in class_cols])  # 正規化
        else:
            weight = 1.0
        effective_weights[col] = round(weight, 4)
        print(f"  {col}: {weight:.4f}")

    results["suggested_weights"] = {
        "inverse_frequency": inverse_freq_weights,
        "effective_samples": effective_weights
    }

    # 檢查重複檔案
    print("\n" + "=" * 80)
    print("資料品質檢查")
    print("=" * 80)

    duplicates_train = train_df['new_filename'].duplicated().sum()
    duplicates_val = val_df['new_filename'].duplicated().sum()
    duplicates_test = test_df['new_filename'].duplicated().sum()

    print(f"\n重複檔案名稱:")
    print(f"  訓練集: {duplicates_train}")
    print(f"  驗證集: {duplicates_val}")
    print(f"  測試集: {duplicates_test}")

    # 檢查訓練集和驗證集是否有重疊
    train_files = set(train_df['new_filename'])
    val_files = set(val_df['new_filename'])
    test_files = set(test_df['new_filename'])

    train_val_overlap = train_files & val_files
    train_test_overlap = train_files & test_files
    val_test_overlap = val_files & test_files

    print(f"\n資料洩漏檢查:")
    print(f"  訓練集 ∩ 驗證集: {len(train_val_overlap)} 個檔案")
    print(f"  訓練集 ∩ 測試集: {len(train_test_overlap)} 個檔案")
    print(f"  驗證集 ∩ 測試集: {len(val_test_overlap)} 個檔案")

    if len(train_val_overlap) > 0:
        print(f"  ⚠️  警告: 訓練集和驗證集有重疊!")

    results["data_quality"] = {
        "duplicates": {
            "train": int(duplicates_train),
            "val": int(duplicates_val),
            "test": int(duplicates_test)
        },
        "data_leakage": {
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "val_test_overlap": len(val_test_overlap)
        }
    }

    # 儲存結果
    with open("data/csv_analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("分析完成！結果已儲存至 data/csv_analysis_report.json")
    print("=" * 80)

    return results

if __name__ == "__main__":
    analyze_csv_data()
