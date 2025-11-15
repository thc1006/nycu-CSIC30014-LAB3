#!/usr/bin/env python3
"""
簡化版 CAPR 偽標籤生成器 - 直接使用現有項目結構
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import timm
from PIL import Image
from torchvision import transforms

class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

# 類別性能（基於已知的驗證結果）
class_performance = {
    'normal': 0.95,
    'bacteria': 0.92,
    'virus': 0.85,
    'COVID-19': 0.70
}

def compute_adaptive_threshold(class_name, base_threshold=0.90):
    """計算自適應閾值"""
    performance = class_performance[class_name]

    if performance >= 0.90:
        threshold = base_threshold + 0.05
    elif performance >= 0.80:
        threshold = base_threshold
    else:
        threshold = base_threshold - 0.10 * (1 - performance)

    return np.clip(threshold, 0.70, 0.98)

def load_model(model_path, device='cuda'):
    """加載模型"""
    checkpoint = torch.load(model_path, map_location=device)

    # 創建模型
    model = timm.create_model('efficientnet_v2_l', pretrained=False, num_classes=4)

    # 加載權重
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model

def predict_ensemble(models, test_df, img_dir, device='cuda'):
    """集成預測"""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_preds = []

    print(f"[集成預測] 使用 {len(models)} 個模型...")

    with torch.no_grad():
        for img_name in tqdm(test_df['new_filename'].values):
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            # 每個模型的預測
            model_preds = []
            for model in models:
                output = model(img_tensor)
                prob = F.softmax(output, dim=1).cpu().numpy()[0]
                model_preds.append(prob)

            # 平均預測
            mean_pred = np.mean(model_preds, axis=0)

            # 不確定性（標準差）
            std_pred = np.std(model_preds, axis=0)
            uncertainty = std_pred.max()

            all_preds.append({
                'image_name': img_name,
                'probs': mean_pred,
                'uncertainty': uncertainty
            })

    return all_preds

def quality_score(confidence, uncertainty, class_idx):
    """計算質量評分"""
    base_quality = confidence * (1 - uncertainty)

    # 困難類別放寬標準
    class_name = class_names[class_idx]
    performance = class_performance[class_name]

    if performance < 0.80:
        adjustment = 1.05
    elif performance < 0.90:
        adjustment = 1.02
    else:
        adjustment = 1.0

    return np.clip(base_quality * adjustment, 0, 1)

def generate_capr_pseudo_labels(
    predictions,
    max_per_class=[150, 200, 150, 50],
    min_quality_score=0.85
):
    """生成 CAPR 偽標籤"""
    pseudo_labels = []

    print("\n[CAPR] 生成類別自適應偽標籤...")

    for cls_idx, cls_name in enumerate(class_names):
        threshold = compute_adaptive_threshold(cls_name)

        # 該類別的候選
        candidates = []
        for pred in predictions:
            probs = pred['probs']
            pred_class = np.argmax(probs)

            if pred_class == cls_idx:
                conf = probs[cls_idx]
                unc = pred['uncertainty']
                qual = quality_score(conf, unc, cls_idx)

                if conf >= threshold and qual >= min_quality_score:
                    candidates.append({
                        'image_name': pred['image_name'],
                        'confidence': conf,
                        'quality_score': qual
                    })

        # 按質量排序，取 top-K
        candidates.sort(key=lambda x: x['quality_score'], reverse=True)
        selected = candidates[:max_per_class[cls_idx]]

        # 添加到偽標籤
        for s in selected:
            pseudo_labels.append({
                'image_name': s['image_name'],
                'class': cls_name,
                'class_idx': cls_idx,
                'confidence': s['confidence'],
                'quality_score': s['quality_score'],
                'threshold_used': threshold
            })

        print(f"  {cls_name}: 閾值={threshold:.3f}, 候選={len(candidates)}, 選擇={len(selected)}")

    return pd.DataFrame(pseudo_labels)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-paths', nargs='+', required=True)
    parser.add_argument('--test-csv', default='data/test_data.csv')
    parser.add_argument('--img-dir', default='data/test_images')
    parser.add_argument('--output', default='data/pseudo_labels_capr.csv')
    parser.add_argument('--max-per-class', type=int, nargs=4, default=[150, 200, 150, 50])
    parser.add_argument('--min-quality', type=float, default=0.85)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    print("=" * 60)
    print("CAPR 類別自適應偽標籤生成器")
    print("=" * 60)

    # 加載模型
    print(f"\n[加載] {len(args.model_paths)} 個模型...")
    models = []
    for path in args.model_paths:
        model = load_model(path, args.device)
        models.append(model)
        print(f"  ✅ {path}")

    # 加載測試集
    test_df = pd.read_csv(args.test_csv)
    print(f"\n[加載] 測試集: {len(test_df)} 張影像")

    # 集成預測
    predictions = predict_ensemble(models, test_df, args.img_dir, args.device)

    # 生成 CAPR 偽標籤
    pseudo_df = generate_capr_pseudo_labels(
        predictions,
        max_per_class=args.max_per_class,
        min_quality_score=args.min_quality
    )

    # 保存
    pseudo_df.to_csv(args.output, index=False)
    print(f"\n[保存] {len(pseudo_df)} 個偽標籤已保存至: {args.output}")

    # 統計
    print("\n" + "=" * 60)
    print("偽標籤統計:")
    print("=" * 60)
    for cls in class_names:
        cls_df = pseudo_df[pseudo_df['class'] == cls]
        if len(cls_df) > 0:
            print(f"\n{cls.upper()}:")
            print(f"  數量: {len(cls_df)}")
            print(f"  置信度: {cls_df['confidence'].mean():.4f} ± {cls_df['confidence'].std():.4f}")
            print(f"  質量評分: {cls_df['quality_score'].mean():.4f} ± {cls_df['quality_score'].std():.4f}")

    print("\n✅ CAPR 偽標籤生成完成！")

if __name__ == '__main__':
    main()
