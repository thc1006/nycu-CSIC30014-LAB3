#!/usr/bin/env python3
"""
統計所有影像的尺寸、格式、色彩分佈
"""
from PIL import Image
import numpy as np
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

def get_image_stats(image_path):
    """獲取單張影像的統計資訊"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)

        stats = {
            'width': img.size[0],
            'height': img.size[1],
            'format': img.format,
            'mode': img.mode,
            'size_kb': image_path.stat().st_size / 1024,
        }

        # 像素統計
        if len(img_array.shape) == 3:
            pixel_data = img_array.mean(axis=2)
        else:
            pixel_data = img_array

        stats['pixel_mean'] = float(pixel_data.mean())
        stats['pixel_std'] = float(pixel_data.std())
        stats['pixel_min'] = int(pixel_data.min())
        stats['pixel_max'] = int(pixel_data.max())

        return stats
    except Exception as e:
        print(f"錯誤: {image_path}: {e}")
        return None

def main():
    base_path = Path('.')

    results = {
        'train': {'images': [], 'by_class': defaultdict(list)},
        'val': {'images': [], 'by_class': defaultdict(list)},
        'test': {'images': []}
    }

    # 處理訓練集和驗證集
    for split in ['train', 'val']:
        print(f"\n處理 {split} 集...")
        csv_file = f'data/{split}_data.csv'
        img_dir = Path(f'{split}_images')

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for i, row in enumerate(rows):
            filename = row['new_filename']
            img_path = img_dir / filename

            if not img_path.exists():
                continue

            stats = get_image_stats(img_path)
            if stats:
                stats['filename'] = filename
                results[split]['images'].append(stats)

                # 按類別分組
                for class_name in ['normal', 'bacteria', 'virus', 'COVID-19']:
                    if row.get(class_name) == '1':
                        results[split]['by_class'][class_name].append(stats)

            if (i + 1) % 500 == 0:
                print(f"  已處理 {i + 1}/{len(rows)}")

    # 處理測試集
    print(f"\n處理 test 集...")
    img_dir = Path('test_images')
    test_files = list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.jpg'))

    for i, img_path in enumerate(test_files):
        stats = get_image_stats(img_path)
        if stats:
            stats['filename'] = img_path.name
            results['test']['images'].append(stats)

        if (i + 1) % 500 == 0:
            print(f"  已處理 {i + 1}/{len(test_files)}")

    # 統計摘要
    print("\n" + "=" * 80)
    print("統計摘要")
    print("=" * 80)

    summary = {}

    for split in ['train', 'val', 'test']:
        images = results[split]['images']

        if not images:
            continue

        widths = [img['width'] for img in images]
        heights = [img['height'] for img in images]
        sizes_kb = [img['size_kb'] for img in images]
        formats = [img['format'] for img in images]
        modes = [img['mode'] for img in images]
        pixel_means = [img['pixel_mean'] for img in images]
        pixel_stds = [img['pixel_std'] for img in images]

        summary[split] = {
            'count': len(images),
            'width': {
                'min': min(widths),
                'max': max(widths),
                'mean': round(np.mean(widths), 1),
                'std': round(np.std(widths), 1)
            },
            'height': {
                'min': min(heights),
                'max': max(heights),
                'mean': round(np.mean(heights), 1),
                'std': round(np.std(heights), 1)
            },
            'size_kb': {
                'min': round(min(sizes_kb), 2),
                'max': round(max(sizes_kb), 2),
                'mean': round(np.mean(sizes_kb), 2),
                'total_mb': round(sum(sizes_kb) / 1024, 2)
            },
            'formats': dict(Counter(formats)),
            'modes': dict(Counter(modes)),
            'pixel_intensity': {
                'mean': round(np.mean(pixel_means), 2),
                'std': round(np.mean(pixel_stds), 2),
                'min_observed': min([img['pixel_min'] for img in images]),
                'max_observed': max([img['pixel_max'] for img in images])
            },
            'aspect_ratios': {
                'square': sum(1 for w, h in zip(widths, heights) if abs(w - h) < 10),
                'landscape': sum(1 for w, h in zip(widths, heights) if w > h + 10),
                'portrait': sum(1 for w, h in zip(widths, heights) if h > w + 10)
            }
        }

        print(f"\n{split.upper()} 集:")
        print(f"  影像數: {summary[split]['count']}")
        print(f"  尺寸範圍: {summary[split]['width']['min']}x{summary[split]['height']['min']} ~ {summary[split]['width']['max']}x{summary[split]['height']['max']}")
        print(f"  平均尺寸: {summary[split]['width']['mean']:.0f}x{summary[split]['height']['mean']:.0f}")
        print(f"  格式: {summary[split]['formats']}")
        print(f"  色彩模式: {summary[split]['modes']}")
        print(f"  平均檔案大小: {summary[split]['size_kb']['mean']:.2f} KB")
        print(f"  總大小: {summary[split]['size_kb']['total_mb']:.2f} MB")
        print(f"  像素強度均值: {summary[split]['pixel_intensity']['mean']:.2f}")

    # 按類別統計 (訓練集)
    if results['train']['by_class']:
        print(f"\n訓練集按類別統計:")
        print("-" * 80)

        for class_name in ['normal', 'bacteria', 'virus', 'COVID-19']:
            class_images = results['train']['by_class'][class_name]
            if not class_images:
                continue

            pixel_means = [img['pixel_mean'] for img in class_images]
            pixel_stds = [img['pixel_std'] for img in class_images]
            widths = [img['width'] for img in class_images]
            heights = [img['height'] for img in class_images]

            print(f"\n{class_name}:")
            print(f"  數量: {len(class_images)}")
            print(f"  平均尺寸: {np.mean(widths):.0f}x{np.mean(heights):.0f}")
            print(f"  像素強度均值: {np.mean(pixel_means):.2f} ± {np.std(pixel_means):.2f}")
            print(f"  像素標準差均值: {np.mean(pixel_stds):.2f}")

    # 儲存結果
    output = {
        'summary': summary,
        'detailed': results
    }

    with open('data/image_statistics_report.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("統計完成！結果已儲存至 data/image_statistics_report.json")
    print("=" * 80)

if __name__ == '__main__':
    main()
