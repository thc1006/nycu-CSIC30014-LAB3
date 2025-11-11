#!/usr/bin/env python3
"""
深度分析胸部X光影像的視覺特性
此腳本需要在有影像檔案的環境中執行（Windows 機器或 Colab）
"""
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter
import sys

# 嘗試導入影像處理庫
try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  警告: PIL/Pillow 未安裝。部分分析功能將無法使用。")
    print("   安裝指令: pip install Pillow numpy")

def analyze_single_image(image_path):
    """分析單張影像的詳細資訊"""
    if not PIL_AVAILABLE:
        return None

    try:
        img = Image.open(image_path)
        img_array = np.array(img)

        info = {
            "path": str(image_path),
            "format": img.format,
            "mode": img.mode,
            "size": img.size,  # (width, height)
            "width": img.size[0],
            "height": img.size[1],
            "aspect_ratio": round(img.size[0] / img.size[1], 3) if img.size[1] > 0 else 0,
            "file_size_kb": round(image_path.stat().st_size / 1024, 2),
        }

        # 色彩空間分析
        if img.mode == 'RGB':
            info["channels"] = 3
            info["is_grayscale"] = False
            # 檢查是否實際上是灰階（RGB 三通道相同）
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            is_actually_gray = np.allclose(r, g) and np.allclose(g, b)
            info["rgb_but_grayscale"] = is_actually_gray
        elif img.mode == 'L':
            info["channels"] = 1
            info["is_grayscale"] = True
        else:
            info["channels"] = len(img.getbands())
            info["is_grayscale"] = img.mode in ['L', '1']

        # 像素值統計
        if len(img_array.shape) == 3:
            # 彩色影像，取平均
            pixel_values = img_array.mean(axis=2)
        else:
            pixel_values = img_array

        info["pixel_stats"] = {
            "min": int(pixel_values.min()),
            "max": int(pixel_values.max()),
            "mean": round(float(pixel_values.mean()), 2),
            "std": round(float(pixel_values.std()), 2),
            "median": int(np.median(pixel_values))
        }

        # 對比度分析
        info["contrast_ratio"] = round(info["pixel_stats"]["std"] / info["pixel_stats"]["mean"], 3) if info["pixel_stats"]["mean"] > 0 else 0

        # 亮度分析
        if info["pixel_stats"]["mean"] < 85:
            brightness = "暗"
        elif info["pixel_stats"]["mean"] > 170:
            brightness = "亮"
        else:
            brightness = "中等"
        info["brightness_level"] = brightness

        return info

    except Exception as e:
        print(f"處理影像時出錯 {image_path}: {e}")
        return None

def analyze_image_dataset(base_path, csv_files):
    """分析整個影像資料集"""

    if not PIL_AVAILABLE:
        print("\n" + "=" * 80)
        print("錯誤: 需要 PIL/Pillow 來分析影像")
        print("=" * 80)
        print("\n請在有影像的環境中執行此腳本（Windows 機器或 Colab）")
        print("安裝指令: pip install Pillow numpy\n")
        return None

    base_path = Path(base_path)
    results = {
        "summary": {},
        "by_split": {},
        "by_class": {},
        "format_analysis": {},
        "size_distribution": {},
        "quality_issues": []
    }

    print("=" * 80)
    print("開始深度影像分析")
    print("=" * 80)

    for split_name, csv_file in csv_files.items():
        print(f"\n處理 {split_name} 集...")

        # 讀取 CSV
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # 確定影像目錄
        if split_name == 'train':
            img_dir = base_path / "train_images"
        elif split_name == 'val':
            img_dir = base_path / "val_images"
        elif split_name == 'test':
            img_dir = base_path / "test_images"
        else:
            continue

        if not img_dir.exists():
            print(f"  ⚠️  警告: 影像目錄不存在: {img_dir}")
            continue

        split_stats = {
            "total_images": 0,
            "processed": 0,
            "failed": 0,
            "formats": Counter(),
            "modes": Counter(),
            "sizes": [],
            "widths": [],
            "heights": [],
            "aspect_ratios": [],
            "file_sizes": [],
            "pixel_stats": defaultdict(list),
            "by_class": defaultdict(lambda: {
                "count": 0,
                "sizes": [],
                "pixel_means": [],
                "pixel_stds": []
            })
        }

        # 分析每張影像
        for i, row in enumerate(rows):
            filename = row['new_filename']
            img_path = img_dir / filename

            if not img_path.exists():
                split_stats["failed"] += 1
                results["quality_issues"].append({
                    "type": "missing_file",
                    "file": str(img_path),
                    "split": split_name
                })
                continue

            info = analyze_single_image(img_path)
            if info is None:
                split_stats["failed"] += 1
                continue

            split_stats["processed"] += 1
            split_stats["formats"][info["format"]] += 1
            split_stats["modes"][info["mode"]] += 1
            split_stats["sizes"].append(f"{info['width']}x{info['height']}")
            split_stats["widths"].append(info["width"])
            split_stats["heights"].append(info["height"])
            split_stats["aspect_ratios"].append(info["aspect_ratio"])
            split_stats["file_sizes"].append(info["file_size_kb"])

            for key, value in info["pixel_stats"].items():
                split_stats["pixel_stats"][key].append(value)

            # 按類別統計
            if split_name in ['train', 'val']:
                for class_name in ['normal', 'bacteria', 'virus', 'COVID-19']:
                    if row.get(class_name) == '1':
                        class_stats = split_stats["by_class"][class_name]
                        class_stats["count"] += 1
                        class_stats["sizes"].append(f"{info['width']}x{info['height']}")
                        class_stats["pixel_means"].append(info["pixel_stats"]["mean"])
                        class_stats["pixel_stds"].append(info["pixel_stats"]["std"])

            # 檢查異常
            if info["width"] < 100 or info["height"] < 100:
                results["quality_issues"].append({
                    "type": "too_small",
                    "file": filename,
                    "size": f"{info['width']}x{info['height']}",
                    "split": split_name
                })

            if info["file_size_kb"] > 5000:  # > 5MB
                results["quality_issues"].append({
                    "type": "too_large",
                    "file": filename,
                    "size_kb": info["file_size_kb"],
                    "split": split_name
                })

            # 進度顯示
            if (i + 1) % 100 == 0:
                print(f"  已處理 {i + 1}/{len(rows)} 張影像...")

        split_stats["total_images"] = len(rows)

        # 計算統計摘要
        if split_stats["processed"] > 0:
            split_summary = {
                "total": split_stats["total_images"],
                "processed": split_stats["processed"],
                "failed": split_stats["failed"],
                "formats": dict(split_stats["formats"]),
                "modes": dict(split_stats["modes"]),
                "size_stats": {
                    "unique_sizes": len(set(split_stats["sizes"])),
                    "most_common_size": Counter(split_stats["sizes"]).most_common(1)[0] if split_stats["sizes"] else None,
                    "width_range": [min(split_stats["widths"]), max(split_stats["widths"])],
                    "height_range": [min(split_stats["heights"]), max(split_stats["heights"])],
                    "width_mean": round(np.mean(split_stats["widths"]), 1),
                    "height_mean": round(np.mean(split_stats["heights"]), 1),
                    "aspect_ratio_mean": round(np.mean(split_stats["aspect_ratios"]), 3),
                },
                "file_size_stats": {
                    "min_kb": round(min(split_stats["file_sizes"]), 2),
                    "max_kb": round(max(split_stats["file_sizes"]), 2),
                    "mean_kb": round(np.mean(split_stats["file_sizes"]), 2),
                    "total_mb": round(sum(split_stats["file_sizes"]) / 1024, 2)
                },
                "pixel_intensity": {
                    "mean": round(np.mean(split_stats["pixel_stats"]["mean"]), 2),
                    "std": round(np.mean(split_stats["pixel_stats"]["std"]), 2),
                    "min_observed": min(split_stats["pixel_stats"]["min"]),
                    "max_observed": max(split_stats["pixel_stats"]["max"])
                }
            }

            # 按類別統計
            if split_name in ['train', 'val']:
                class_analysis = {}
                for class_name, class_data in split_stats["by_class"].items():
                    if class_data["count"] > 0:
                        class_analysis[class_name] = {
                            "count": class_data["count"],
                            "unique_sizes": len(set(class_data["sizes"])),
                            "pixel_mean_avg": round(np.mean(class_data["pixel_means"]), 2),
                            "pixel_std_avg": round(np.mean(class_data["pixel_stds"]), 2)
                        }
                split_summary["by_class"] = class_analysis

            results["by_split"][split_name] = split_summary

            print(f"  ✓ {split_name} 集分析完成")
            print(f"    處理: {split_stats['processed']}/{split_stats['total_images']}")
            print(f"    格式: {dict(split_stats['formats'])}")
            print(f"    尺寸範圍: {split_summary['size_stats']['width_range']}x{split_summary['size_stats']['height_range']}")

    # 整體統計
    total_processed = sum(s["processed"] for s in results["by_split"].values())
    total_images = sum(s["total"] for s in results["by_split"].values())

    results["summary"] = {
        "total_images": total_images,
        "total_processed": total_processed,
        "total_failed": total_images - total_processed,
        "quality_issues_count": len(results["quality_issues"])
    }

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"總影像數: {total_images}")
    print(f"成功處理: {total_processed}")
    print(f"品質問題: {len(results['quality_issues'])}")

    # 儲存結果
    output_path = Path("data") / "image_analysis_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n詳細報告已儲存至: {output_path}")

    return results

def generate_markdown_report(results):
    """生成 Markdown 格式的分析報告"""

    if results is None:
        return "# 影像分析報告\n\n無法生成報告：需要在有影像的環境中執行分析腳本。\n"

    md = "# 胸部X光影像深度分析報告\n\n"
    md += f"生成時間: {import datetime; datetime.datetime.now()}\n\n"
    md += "---\n\n"

    # 總體統計
    md += "## 總體統計\n\n"
    summary = results["summary"]
    md += f"- **總影像數**: {summary['total_images']:,}\n"
    md += f"- **成功處理**: {summary['total_processed']:,}\n"
    md += f"- **處理失敗**: {summary['total_failed']}\n"
    md += f"- **品質問題**: {summary['quality_issues_count']}\n\n"

    # 各資料集分析
    md += "## 各資料集詳細分析\n\n"

    for split_name, split_data in results["by_split"].items():
        md += f"### {split_name.upper()} 集\n\n"

        md += f"#### 基本資訊\n\n"
        md += f"- 總數: {split_data['total']}\n"
        md += f"- 處理成功: {split_data['processed']}\n"
        md += f"- 處理失敗: {split_data['failed']}\n\n"

        md += f"#### 影像格式\n\n"
        md += "| 格式 | 數量 |\n"
        md += "|------|------|\n"
        for fmt, count in split_data['formats'].items():
            md += f"| {fmt} | {count} |\n"
        md += "\n"

        md += f"#### 色彩模式\n\n"
        md += "| 模式 | 數量 | 說明 |\n"
        md += "|------|------|------|\n"
        for mode, count in split_data['modes'].items():
            desc = {
                'RGB': '彩色（3通道）',
                'L': '灰階',
                'RGBA': '彩色+透明度',
                'P': '調色板模式'
            }.get(mode, mode)
            md += f"| {mode} | {count} | {desc} |\n"
        md += "\n"

        md += f"#### 影像尺寸統計\n\n"
        size_stats = split_data['size_stats']
        md += f"- **唯一尺寸數**: {size_stats['unique_sizes']}\n"
        if size_stats['most_common_size']:
            md += f"- **最常見尺寸**: {size_stats['most_common_size'][0]} ({size_stats['most_common_size'][1]} 張)\n"
        md += f"- **寬度範圍**: {size_stats['width_range'][0]} - {size_stats['width_range'][1]} px\n"
        md += f"- **高度範圍**: {size_stats['height_range'][0]} - {size_stats['height_range'][1]} px\n"
        md += f"- **平均寬度**: {size_stats['width_mean']} px\n"
        md += f"- **平均高度**: {size_stats['height_mean']} px\n"
        md += f"- **平均長寬比**: {size_stats['aspect_ratio_mean']}\n\n"

        md += f"#### 檔案大小\n\n"
        file_stats = split_data['file_size_stats']
        md += f"- **最小**: {file_stats['min_kb']} KB\n"
        md += f"- **最大**: {file_stats['max_kb']} KB\n"
        md += f"- **平均**: {file_stats['mean_kb']} KB\n"
        md += f"- **總大小**: {file_stats['total_mb']} MB\n\n"

        md += f"#### 像素強度分析\n\n"
        pixel_stats = split_data['pixel_intensity']
        md += f"- **平均強度**: {pixel_stats['mean']} (0-255)\n"
        md += f"- **平均標準差**: {pixel_stats['std']}\n"
        md += f"- **最小觀測值**: {pixel_stats['min_observed']}\n"
        md += f"- **最大觀測值**: {pixel_stats['max_observed']}\n\n"

        # 按類別分析
        if 'by_class' in split_data:
            md += f"#### 按類別分析\n\n"
            md += "| 類別 | 數量 | 唯一尺寸 | 平均像素強度 | 平均標準差 |\n"
            md += "|------|------|----------|--------------|------------|\n"
            for class_name, class_data in split_data['by_class'].items():
                md += f"| {class_name} | {class_data['count']} | "
                md += f"{class_data['unique_sizes']} | "
                md += f"{class_data['pixel_mean_avg']} | "
                md += f"{class_data['pixel_std_avg']} |\n"
            md += "\n"

    # 品質問題
    if results["quality_issues"]:
        md += "## 品質問題\n\n"
        md += f"發現 {len(results['quality_issues'])} 個潛在品質問題：\n\n"

        # 按類型分組
        issues_by_type = defaultdict(list)
        for issue in results["quality_issues"]:
            issues_by_type[issue["type"]].append(issue)

        for issue_type, issues in issues_by_type.items():
            md += f"### {issue_type} ({len(issues)} 個)\n\n"
            for issue in issues[:10]:  # 只顯示前10個
                md += f"- {issue}\n"
            if len(issues) > 10:
                md += f"- ... 還有 {len(issues) - 10} 個\n"
            md += "\n"

    return md

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="深度分析胸部X光影像")
    parser.add_argument("--base-path", type=str,
                        default="C:/Users/thc1006/Desktop/114-1/nycu-CSIC30014-LAB3",
                        help="影像檔案的基礎路徑")
    parser.add_argument("--output", type=str, default="影像分析詳細報告.md",
                        help="Markdown 報告輸出路徑")

    args = parser.parse_args()

    # CSV 檔案
    csv_files = {
        "train": "data/train_data.csv",
        "val": "data/val_data.csv",
        "test": "data/test_data.csv"
    }

    # 執行分析
    results = analyze_image_dataset(args.base_path, csv_files)

    if results:
        # 生成 Markdown 報告
        md_report = generate_markdown_report(results)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(md_report)
        print(f"\nMarkdown 報告已儲存至: {args.output}")
