#!/usr/bin/env python3
"""
CAPR (Category-Adaptive Pseudo-label Regulation) Generator

基於 2024 文獻的類別自適應偽標籤生成器
- 動態調整每個類別的置信度閾值
- 基於實時學習進度自適應調整
- 質量控制與噪聲預防
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import yaml

# 添加項目路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_model
from src.dataset import ChestXRayDataset
from torch.utils.data import DataLoader


class CategoryAdaptivePseudoLabeler:
    """類別自適應偽標籤生成器"""

    def __init__(
        self,
        model_paths,
        config_path,
        device='cuda',
        base_threshold=0.90,
        enable_quality_control=True
    ):
        self.device = device
        self.base_threshold = base_threshold
        self.enable_quality_control = enable_quality_control

        # 加載配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 類別名稱
        self.class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
        self.num_classes = len(self.class_names)

        # 加載模型集成
        print(f"[CAPR] 加載 {len(model_paths)} 個模型...")
        self.models = []
        for model_path in model_paths:
            model = self._load_model(model_path)
            self.models.append(model)

        # 類別性能統計（基於驗證集）
        self.class_performance = {
            'normal': 0.95,      # 簡單類別
            'bacteria': 0.92,
            'virus': 0.85,       # 中等困難
            'COVID-19': 0.70     # 最困難
        }

        print(f"[CAPR] 初始化完成")
        print(f"[CAPR] 基礎閾值: {base_threshold}")
        print(f"[CAPR] 質量控制: {'啟用' if enable_quality_control else '禁用'}")

    def _load_model(self, model_path):
        """加載單個模型"""
        model = create_model(
            model_name=self.config['model']['name'],
            num_classes=self.num_classes,
            pretrained=False,
            dropout=self.config['model'].get('dropout', 0.3)
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        return model

    def compute_adaptive_threshold(self, class_name, stage='initial'):
        """
        計算類別自適應閾值

        基於文獻: "動態調整閾值基於每個類別的實時學習進度"

        Args:
            class_name: 類別名稱
            stage: 訓練階段 ('initial', 'progressive', 'final')

        Returns:
            adaptive_threshold: 自適應閾值
        """
        # 基礎性能
        performance = self.class_performance[class_name]

        # 階段因子
        stage_factors = {
            'initial': 1.05,      # 初期更保守
            'progressive': 1.00,  # 中期平衡
            'final': 0.95        # 後期略放寬
        }
        stage_factor = stage_factors.get(stage, 1.0)

        # 動態閾值公式（基於 CAPR 論文）
        if performance >= 0.90:
            # 學習良好的類別：高閾值
            threshold = self.base_threshold + 0.05
        elif performance >= 0.80:
            # 中等類別：基礎閾值
            threshold = self.base_threshold
        else:
            # 困難類別：降低閾值（確保足夠樣本）
            # 性能越低，閾值越低
            threshold = self.base_threshold - 0.10 * (1 - performance)

        # 應用階段因子
        threshold = threshold * stage_factor

        # 限制範圍 [0.70, 0.98]
        threshold = np.clip(threshold, 0.70, 0.98)

        return threshold

    def predict_ensemble(self, dataloader):
        """
        集成預測

        Returns:
            predictions: (N, num_classes) 預測概率
            uncertainties: (N,) 預測不確定性
        """
        all_predictions = []

        print(f"[CAPR] 使用 {len(self.models)} 個模型進行集成預測...")

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="集成預測"):
                images = images.to(self.device)

                # 每個模型的預測
                model_preds = []
                for model in self.models:
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    model_preds.append(probs.cpu().numpy())

                # 堆疊: (num_models, batch_size, num_classes)
                model_preds = np.stack(model_preds, axis=0)

                # 平均預測
                mean_preds = model_preds.mean(axis=0)
                all_predictions.append(mean_preds)

        # 合併所有批次
        predictions = np.vstack(all_predictions)

        # 計算不確定性（標準差）
        all_model_preds = []
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="計算不確定性"):
                images = images.to(self.device)
                batch_preds = []
                for model in self.models:
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    batch_preds.append(probs.cpu().numpy())
                all_model_preds.append(np.stack(batch_preds, axis=0))

        all_model_preds = np.concatenate(all_model_preds, axis=1)  # (num_models, N, num_classes)
        uncertainties = all_model_preds.std(axis=0).max(axis=1)  # (N,) 每個樣本的最大不確定性

        return predictions, uncertainties

    def quality_score(self, confidence, uncertainty, class_idx):
        """
        計算偽標籤質量評分

        基於文獻: "質量評分 = 置信度 × (1 - 不確定性)"

        Args:
            confidence: 預測置信度
            uncertainty: 預測不確定性
            class_idx: 類別索引

        Returns:
            quality_score: 質量評分 [0, 1]
        """
        # 基礎質量
        base_quality = confidence * (1 - uncertainty)

        # 類別調整因子（困難類別放寬標準）
        class_name = self.class_names[class_idx]
        performance = self.class_performance[class_name]

        # 困難類別的質量評分略微提升（避免過度嚴格）
        if performance < 0.80:
            adjustment = 1.05
        elif performance < 0.90:
            adjustment = 1.02
        else:
            adjustment = 1.0

        quality = base_quality * adjustment

        return np.clip(quality, 0, 1)

    def generate_pseudo_labels(
        self,
        test_dataset,
        stage='initial',
        max_per_class=None,
        min_quality_score=0.85
    ):
        """
        生成類別自適應偽標籤

        Args:
            test_dataset: 測試數據集
            stage: 訓練階段
            max_per_class: 每個類別最大樣本數（防止頭部類別主導）
            min_quality_score: 最小質量評分

        Returns:
            pseudo_labels: DataFrame with columns ['image_name', 'class', 'confidence', 'quality_score', 'threshold_used']
        """
        # 創建 DataLoader
        dataloader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 集成預測
        predictions, uncertainties = self.predict_ensemble(dataloader)

        # 獲取預測類別和置信度
        pred_classes = predictions.argmax(axis=1)
        confidences = predictions.max(axis=1)

        # 按類別生成偽標籤
        pseudo_labels_list = []
        class_counts = {cls: 0 for cls in self.class_names}

        print(f"\n[CAPR] 使用自適應閾值生成偽標籤 (階段: {stage})...")

        for cls_idx, cls_name in enumerate(self.class_names):
            # 計算自適應閾值
            threshold = self.compute_adaptive_threshold(cls_name, stage)

            # 該類別的樣本
            cls_mask = (pred_classes == cls_idx)
            cls_indices = np.where(cls_mask)[0]

            if len(cls_indices) == 0:
                print(f"  {cls_name}: 閾值={threshold:.3f}, 候選=0, 選擇=0")
                continue

            # 候選樣本
            cls_confs = confidences[cls_indices]
            cls_uncs = uncertainties[cls_indices]

            # 應用閾值
            high_conf_mask = cls_confs >= threshold

            # 質量評分
            if self.enable_quality_control:
                quality_scores = np.array([
                    self.quality_score(conf, unc, cls_idx)
                    for conf, unc in zip(cls_confs, cls_uncs)
                ])

                # 質量過濾
                quality_mask = quality_scores >= min_quality_score
                final_mask = high_conf_mask & quality_mask
            else:
                quality_scores = cls_confs  # 簡單使用置信度
                final_mask = high_conf_mask

            selected_indices = cls_indices[final_mask]

            # 限制每類數量（防止頭部類別主導）
            if max_per_class is not None and len(selected_indices) > max_per_class[cls_idx]:
                # 按質量評分排序，取 top-K
                selected_qualities = quality_scores[final_mask]
                top_k_indices = np.argsort(selected_qualities)[-max_per_class[cls_idx]:]
                selected_indices = selected_indices[top_k_indices]

            # 添加到偽標籤列表
            for idx in selected_indices:
                img_name = test_dataset.image_names[idx]
                conf = confidences[idx]
                unc = uncertainties[idx]
                qual = self.quality_score(conf, unc, cls_idx)

                pseudo_labels_list.append({
                    'image_name': img_name,
                    'class': cls_name,
                    'class_idx': cls_idx,
                    'confidence': conf,
                    'uncertainty': unc,
                    'quality_score': qual,
                    'threshold_used': threshold
                })

                class_counts[cls_name] += 1

            print(f"  {cls_name}: 閾值={threshold:.3f}, 候選={len(cls_indices)}, "
                  f"高置信={high_conf_mask.sum()}, 高質量={final_mask.sum()}, "
                  f"選擇={len(selected_indices)}")

        # 轉換為 DataFrame
        pseudo_labels_df = pd.DataFrame(pseudo_labels_list)

        # 統計
        total = len(pseudo_labels_df)
        avg_conf = pseudo_labels_df['confidence'].mean() if total > 0 else 0
        avg_qual = pseudo_labels_df['quality_score'].mean() if total > 0 else 0

        print(f"\n[CAPR] 偽標籤生成完成:")
        print(f"  總數: {total}")
        print(f"  平均置信度: {avg_conf:.4f}")
        print(f"  平均質量評分: {avg_qual:.4f}")
        print(f"  類別分布: {dict(class_counts)}")

        return pseudo_labels_df


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description='CAPR 偽標籤生成器')
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--model-paths', type=str, nargs='+', required=True, help='模型檢查點路徑')
    parser.add_argument('--test-csv', type=str, default='data/test.csv', help='測試集 CSV')
    parser.add_argument('--output', type=str, default='data/pseudo_labels_capr.csv', help='輸出文件')
    parser.add_argument('--stage', type=str, default='initial', choices=['initial', 'progressive', 'final'])
    parser.add_argument('--max-per-class', type=int, nargs=4, default=[150, 200, 150, 50],
                        help='每類最大樣本數 (normal bacteria virus COVID-19)')
    parser.add_argument('--min-quality', type=float, default=0.85, help='最小質量評分')
    parser.add_argument('--device', type=str, default='cuda', help='設備')

    args = parser.parse_args()

    print("=" * 60)
    print("CAPR 類別自適應偽標籤生成器")
    print("=" * 60)
    print(f"配置: {args.config}")
    print(f"模型數量: {len(args.model_paths)}")
    print(f"階段: {args.stage}")
    print(f"每類最大: {args.max_per_class}")
    print(f"最小質量: {args.min_quality}")
    print("=" * 60)

    # 加載配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 創建測試數據集
    test_df = pd.read_csv(args.test_csv)
    test_dataset = ChestXRayDataset(
        df=test_df,
        img_dir='data/test_images',
        transform=None,  # 不使用增強
        is_training=False
    )

    print(f"\n[加載] 測試集: {len(test_dataset)} 張影像")

    # 創建 CAPR 生成器
    capr = CategoryAdaptivePseudoLabeler(
        model_paths=args.model_paths,
        config_path=args.config,
        device=args.device,
        base_threshold=0.90,
        enable_quality_control=True
    )

    # 生成偽標籤
    pseudo_labels_df = capr.generate_pseudo_labels(
        test_dataset=test_dataset,
        stage=args.stage,
        max_per_class=args.max_per_class,
        min_quality_score=args.min_quality
    )

    # 保存
    pseudo_labels_df.to_csv(args.output, index=False)
    print(f"\n[保存] 偽標籤已保存至: {args.output}")

    # 詳細統計
    print("\n" + "=" * 60)
    print("偽標籤詳細統計:")
    print("=" * 60)

    for cls_name in ['normal', 'bacteria', 'virus', 'COVID-19']:
        cls_df = pseudo_labels_df[pseudo_labels_df['class'] == cls_name]
        if len(cls_df) > 0:
            print(f"\n{cls_name.upper()}:")
            print(f"  數量: {len(cls_df)}")
            print(f"  置信度: {cls_df['confidence'].mean():.4f} ± {cls_df['confidence'].std():.4f}")
            print(f"  質量評分: {cls_df['quality_score'].mean():.4f} ± {cls_df['quality_score'].std():.4f}")
            print(f"  不確定性: {cls_df['uncertainty'].mean():.4f} ± {cls_df['uncertainty'].std():.4f}")
            print(f"  閾值: {cls_df['threshold_used'].iloc[0]:.4f}")

    print("\n✅ CAPR 偽標籤生成完成！")


if __name__ == '__main__':
    main()
