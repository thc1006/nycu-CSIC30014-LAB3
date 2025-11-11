#!/usr/bin/env python3
"""
深度分析胸部X光資料集 (使用標準函式庫)
"""
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

def read_csv_file(filepath):
    """讀取 CSV 檔案"""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data

def analyze_csv_data():
    """分析 CSV 檔案的統計資訊"""
    data_dir = Path("data")

    results = {
        "csv_analysis": {},
        "class_distribution": {},
        "file_info": {},
        "suggested_weights": {},
        "data_quality": {}
    }

    # 分析訓練集
    print("=" * 80)
    print("分析訓練集 (train_data.csv)")
    print("=" * 80)

    train_data = read_csv_file(data_dir / "train_data.csv")
    print(f"\n訓練集大小: {len(train_data)} 筆")

    if train_data:
        print(f"欄位: {list(train_data[0].keys())}")

    # 類別分布
    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
    train_class_dist = {}

    for col in class_cols:
        count = sum(1 for row in train_data if row.get(col) == '1')
        percentage = (count / len(train_data)) * 100 if train_data else 0
        train_class_dist[col] = {
            "count": count,
            "percentage": round(percentage, 2)
        }
        print(f"  {col}: {count} 筆 ({percentage:.2f}%)")

    results["csv_analysis"]["train"] = {
        "total": len(train_data),
        "class_distribution": train_class_dist
    }

    # 分析驗證集
    print("\n" + "=" * 80)
    print("分析驗證集 (val_data.csv)")
    print("=" * 80)

    val_data = read_csv_file(data_dir / "val_data.csv")
    print(f"\n驗證集大小: {len(val_data)} 筆")

    val_class_dist = {}
    for col in class_cols:
        count = sum(1 for row in val_data if row.get(col) == '1')
        percentage = (count / len(val_data)) * 100 if val_data else 0
        val_class_dist[col] = {
            "count": count,
            "percentage": round(percentage, 2)
        }
        print(f"  {col}: {count} 筆 ({percentage:.2f}%)")

    results["csv_analysis"]["val"] = {
        "total": len(val_data),
        "class_distribution": val_class_dist
    }

    # 分析測試集
    print("\n" + "=" * 80)
    print("分析測試集 (test_data.csv)")
    print("=" * 80)

    test_data = read_csv_file(data_dir / "test_data.csv")
    print(f"\n測試集大小: {len(test_data)} 筆")

    if test_data:
        print(f"欄位: {list(test_data[0].keys())}")

    results["csv_analysis"]["test"] = {
        "total": len(test_data),
        "columns": list(test_data[0].keys()) if test_data else []
    }

    # 檔案名稱格式分析
    print("\n" + "=" * 80)
    print("檔案名稱格式分析")
    print("=" * 80)

    all_files = []
    all_files.extend([row['new_filename'] for row in train_data])
    all_files.extend([row['new_filename'] for row in val_data])
    all_files.extend([row['new_filename'] for row in test_data])

    # 檔案副檔名統計
    extensions = Counter()
    for filename in all_files:
        ext = filename.split('.')[-1] if '.' in filename else 'unknown'
        extensions[ext] += 1

    print(f"\n檔案副檔名分布:")
    for ext, count in extensions.most_common():
        print(f"  .{ext}: {count} 個檔案")

    # 檔案名稱是否為數字
    numeric_names = sum(1 for f in all_files if f.replace('.jpeg', '').replace('.jpg', '').replace('.png', '').isdigit())
    print(f"\n數字命名的檔案: {numeric_names}/{len(all_files)}")

    results["file_info"] = {
        "total_files": len(all_files),
        "extensions": dict(extensions),
        "numeric_names": numeric_names
    }

    # 類別不平衡分析
    print("\n" + "=" * 80)
    print("類別不平衡分析 (基於訓練集)")
    print("=" * 80)

    total_samples = len(train_data)
    imbalance_ratios = {}

    class_counts = {}
    for col in class_cols:
        count = sum(1 for row in train_data if row.get(col) == '1')
        class_counts[col] = count

    for col in class_cols:
        count = class_counts[col]
        ratio = count / total_samples if total_samples > 0 else 0
        imbalance = total_samples / count if count > 0 else float('inf')
        imbalance_ratios[col] = {
            "samples": count,
            "ratio": round(ratio, 4),
            "imbalance_factor": round(imbalance, 2) if imbalance != float('inf') else "inf"
        }
        print(f"\n{col}:")
        print(f"  樣本數: {count}")
        print(f"  比例: {ratio:.4f}")
        print(f"  不平衡因子: {imbalance:.2f}x" if imbalance != float('inf') else "  不平衡因子: inf")

    results["class_distribution"]["imbalance"] = imbalance_ratios

    # 建議的類別權重
    print("\n" + "=" * 80)
    print("建議的損失函數類別權重")
    print("=" * 80)

    # 方法1: 逆頻率
    inverse_freq_weights = {}
    max_count = max(class_counts.values()) if class_counts else 1

    print("\n方法1: 逆頻率權重 (Inverse Frequency)")
    print("  公式: weight[i] = max_count / count[i]")
    for col in class_cols:
        count = class_counts[col]
        weight = max_count / count if count > 0 else 1.0
        inverse_freq_weights[col] = round(weight, 2)
        print(f"  {col}: {weight:.2f}")

    # 方法2: 平方根逆頻率 (較溫和)
    sqrt_inv_weights = {}
    print(f"\n方法2: 平方根逆頻率權重 (較溫和)")
    print("  公式: weight[i] = sqrt(max_count / count[i])")
    for col in class_cols:
        count = class_counts[col]
        weight = (max_count / count) ** 0.5 if count > 0 else 1.0
        sqrt_inv_weights[col] = round(weight, 2)
        print(f"  {col}: {weight:.2f}")

    # 方法3: 有效樣本數
    beta = 0.9999
    effective_weights = {}
    print(f"\n方法3: 有效樣本數權重 (Effective Number, β={beta})")
    print("  公式: weight[i] = (1 - β) / (1 - β^count[i])")

    eff_nums = {}
    for col in class_cols:
        count = class_counts[col]
        if count > 0:
            eff_num = (1 - beta) / (1 - beta ** count)
            eff_nums[col] = eff_num
        else:
            eff_nums[col] = 1.0

    # 正規化
    max_eff = max(eff_nums.values()) if eff_nums else 1.0
    for col in class_cols:
        weight = max_eff / eff_nums[col]
        effective_weights[col] = round(weight, 4)
        print(f"  {col}: {weight:.4f}")

    results["suggested_weights"] = {
        "inverse_frequency": inverse_freq_weights,
        "sqrt_inverse_frequency": sqrt_inv_weights,
        "effective_samples": effective_weights
    }

    # 資料品質檢查
    print("\n" + "=" * 80)
    print("資料品質檢查")
    print("=" * 80)

    # 檢查重複
    train_files = [row['new_filename'] for row in train_data]
    val_files = [row['new_filename'] for row in val_data]
    test_files = [row['new_filename'] for row in test_data]

    duplicates_train = len(train_files) - len(set(train_files))
    duplicates_val = len(val_files) - len(set(val_files))
    duplicates_test = len(test_files) - len(set(test_files))

    print(f"\n重複檔案名稱:")
    print(f"  訓練集: {duplicates_train}")
    print(f"  驗證集: {duplicates_val}")
    print(f"  測試集: {duplicates_test}")

    # 檢查資料洩漏
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    print(f"\n資料洩漏檢查:")
    print(f"  訓練集 ∩ 驗證集: {len(train_val_overlap)} 個檔案")
    print(f"  訓練集 ∩ 測試集: {len(train_test_overlap)} 個檔案")
    print(f"  驗證集 ∩ 測試集: {len(val_test_overlap)} 個檔案")

    if len(train_val_overlap) > 0:
        print(f"  ⚠️  警告: 訓練集和驗證集有重疊!")
        print(f"  重疊的檔案 (前5個): {list(train_val_overlap)[:5]}")

    results["data_quality"] = {
        "duplicates": {
            "train": duplicates_train,
            "val": duplicates_val,
            "test": duplicates_test
        },
        "data_leakage": {
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "val_test_overlap": len(val_test_overlap)
        }
    }

    # 額外統計
    print("\n" + "=" * 80)
    print("資料集統計摘要")
    print("=" * 80)

    total_train_val = len(train_data) + len(val_data)
    total_all = total_train_val + len(test_data)

    print(f"\n總檔案數: {total_all}")
    print(f"  訓練集: {len(train_data)} ({len(train_data)/total_all*100:.1f}%)")
    print(f"  驗證集: {len(val_data)} ({len(val_data)/total_all*100:.1f}%)")
    print(f"  測試集: {len(test_data)} ({len(test_data)/total_all*100:.1f}%)")

    print(f"\n訓練/驗證分割比例: {len(train_data)/(total_train_val)*100:.1f}% / {len(val_data)/(total_train_val)*100:.1f}%")

    # 儲存結果
    output_path = data_dir / "csv_analysis_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"分析完成！結果已儲存至 {output_path}")
    print("=" * 80)

    return results

if __name__ == "__main__":
    analyze_csv_data()
