#!/usr/bin/env python3
"""
Ultimate Ensemble - 融合所有最佳模型達到 91%+
自動載入所有模型，生成 TTA 預測，加權融合
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, 'src')
from models import build_model

print("=" * 80)
print("ULTIMATE ENSEMBLE - 自動融合所有模型")
print("=" * 80)
print()

# ============================================================================
# 配置
# ============================================================================

# 所有可用的模型及其配置
MODEL_CONFIGS = [
    # 現有的最佳模型
    {
        'name': 'improved_breakthrough',
        'checkpoint': 'outputs/improved_breakthrough_run/best.pt',
        'config': 'configs/improved_breakthrough.yaml',
        'weight': 0.20,  # 當前最佳單模型
        'img_size': 384
    },
    {
        'name': 'breakthrough_clahe',
        'checkpoint': 'outputs/breakthrough_clahe/best.pt',
        'config': 'configs/breakthrough_clahe.yaml',
        'weight': 0.12,
        'img_size': 384
    },
    {
        'name': 'run1_convnext',
        'checkpoint': 'outputs/run1/best.pt',
        'config': 'configs/ultra_optimized.yaml',
        'weight': 0.15,
        'img_size': 448
    },
    {
        'name': 'fast_efficientnet',
        'checkpoint': 'outputs/fast_efficientnet_run/best.pt',
        'config': 'configs/fast_efficientnet.yaml',
        'weight': 0.10,
        'img_size': 320
    },
    # 新訓練的醫學預訓練模型
    {
        'name': 'medical_pretrained',
        'checkpoint': 'outputs/medical_pretrained/best.pt',
        'config': 'configs/medical_pretrained.yaml',
        'weight': 0.18,  # 醫學預訓練權重較高
        'img_size': 384
    },
    # 新訓練的 Vision Transformer
    {
        'name': 'vit_ultimate',
        'checkpoint': 'outputs/vit_ultimate/best.pt',
        'config': 'configs/vit_ultimate.yaml',
        'weight': 0.25,  # ViT 架構最高權重
        'img_size': 384
    }
]

# 測試數據配置
TEST_CSV = 'data/test_data.csv'
TEST_IMAGES_DIR = 'test_images'
OUTPUT_PATH = 'data/submission_ultimate_ensemble.csv'

# TTA 配置
TTA_TRANSFORMS = 5  # 每個模型使用 5 次增強

# ============================================================================
# 載入模型函數
# ============================================================================

def load_model_with_config(checkpoint_path, config_path, device='cuda'):
    """載入模型和配置"""
    # 讀取配置
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # 構建模型
    model = build_model(
        model_name=cfg['model']['name'],
        num_classes=cfg['data']['num_classes'],
        pretrained=False,
        img_size=cfg['model'].get('img_size', 384),
        dropout=cfg['model'].get('dropout', 0.3)
    )

    # 載入權重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, cfg

# ============================================================================
# TTA 預測函數
# ============================================================================

def get_tta_transforms(img_size):
    """獲取 TTA 變換"""
    base_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # TTA 變換：原圖 + 水平翻轉 + 輕微旋轉 + 亮度調整
    transforms = [
        base_transform,  # 1. 原圖
        T.Compose([T.RandomHorizontalFlip(p=1.0), base_transform]),  # 2. 水平翻轉
        T.Compose([T.RandomRotation(5), base_transform]),  # 3. 輕微旋轉
        T.Compose([T.ColorJitter(brightness=0.1), base_transform]),  # 4. 亮度
        T.Compose([T.RandomAffine(degrees=0, translate=(0.05, 0.05)), base_transform]),  # 5. 平移
    ]

    return transforms[:TTA_TRANSFORMS]

def predict_with_tta(model, image_path, img_size, device='cuda'):
    """使用 TTA 預測單張圖片"""
    img = Image.open(image_path).convert('RGB')

    transforms = get_tta_transforms(img_size)
    predictions = []

    with torch.no_grad():
        for transform in transforms:
            img_tensor = transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            predictions.append(probs)

    # 平均所有 TTA 預測
    return np.mean(predictions, axis=0)

# ============================================================================
# 主程序
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    print()

    # 載入測試數據
    test_df = pd.read_csv(TEST_CSV)
    print(f"測試數據: {len(test_df)} 張影像")
    print()

    # 檢查並載入所有可用模型
    available_models = []
    total_weight = 0.0

    print("檢查可用模型...")
    for model_cfg in MODEL_CONFIGS:
        ckpt_path = Path(model_cfg['checkpoint'])
        cfg_path = Path(model_cfg['config'])

        if ckpt_path.exists() and cfg_path.exists():
            print(f"  ✓ {model_cfg['name']} (權重: {model_cfg['weight']:.2f})")
            available_models.append(model_cfg)
            total_weight += model_cfg['weight']
        else:
            print(f"  ✗ {model_cfg['name']} (缺少文件，跳過)")

    print()
    print(f"總共找到 {len(available_models)} 個可用模型")
    print(f"總權重: {total_weight:.2f}")
    print()

    # 標準化權重
    for model_cfg in available_models:
        model_cfg['weight'] /= total_weight

    # 為每個模型生成預測
    all_predictions = []

    for i, model_cfg in enumerate(available_models):
        print(f"[{i+1}/{len(available_models)}] 處理 {model_cfg['name']}...")
        print(f"  載入模型: {model_cfg['checkpoint']}")

        # 載入模型
        model, cfg = load_model_with_config(
            model_cfg['checkpoint'],
            model_cfg['config'],
            device
        )

        print(f"  生成 TTA 預測 (每張 {TTA_TRANSFORMS} 次增強)...")

        # 對每張測試圖片生成預測
        predictions = []
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  {model_cfg['name']}"):
            img_path = Path(TEST_IMAGES_DIR) / row['new_filename']
            pred = predict_with_tta(model, img_path, model_cfg['img_size'], device)
            predictions.append(pred)

        predictions = np.array(predictions)  # (n_samples, 4)
        all_predictions.append({
            'name': model_cfg['name'],
            'predictions': predictions,
            'weight': model_cfg['weight']
        })

        print(f"  ✓ 完成 (權重: {model_cfg['weight']:.3f})")
        print()

        # 釋放顯存
        del model
        torch.cuda.empty_cache()

    # ========================================================================
    # 加權融合
    # ========================================================================
    print("=" * 80)
    print("加權融合所有模型...")
    print("=" * 80)
    print()

    ensemble_probs = np.zeros((len(test_df), 4))

    for pred_dict in all_predictions:
        print(f"  {pred_dict['name']:25s} 權重: {pred_dict['weight']:.3f}")
        ensemble_probs += pred_dict['weight'] * pred_dict['predictions']

    # 確保概率總和為 1
    ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

    # ========================================================================
    # 創建提交文件
    # ========================================================================
    print()
    print("創建最終提交文件...")

    class_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

    # 創建 soft probability 提交（保留概率分布）
    submission_soft = pd.DataFrame({
        'new_filename': test_df['new_filename'],
        'normal': ensemble_probs[:, 0],
        'bacteria': ensemble_probs[:, 1],
        'virus': ensemble_probs[:, 2],
        'COVID-19': ensemble_probs[:, 3]
    })

    # 轉換為 one-hot（競賽要求）
    predicted_idx = ensemble_probs.argmax(axis=1)
    onehot = np.zeros_like(ensemble_probs)
    onehot[np.arange(len(ensemble_probs)), predicted_idx] = 1.0

    submission_final = pd.DataFrame({
        'new_filename': test_df['new_filename'],
        'normal': onehot[:, 0],
        'bacteria': onehot[:, 1],
        'virus': onehot[:, 2],
        'COVID-19': onehot[:, 3]
    })

    # 保存
    submission_final.to_csv(OUTPUT_PATH, index=False)
    submission_soft.to_csv(OUTPUT_PATH.replace('.csv', '_soft.csv'), index=False)

    print(f"  ✓ One-hot 提交: {OUTPUT_PATH}")
    print(f"  ✓ Soft 提交:    {OUTPUT_PATH.replace('.csv', '_soft.csv')}")
    print()

    # ========================================================================
    # 預測統計
    # ========================================================================
    print("=" * 80)
    print("預測分布:")
    print("=" * 80)
    for i, cls in enumerate(class_cols):
        count = (predicted_idx == i).sum()
        print(f"  {cls:12s}: {count:4d} ({count/len(predicted_idx)*100:.1f}%)")

    print()
    print("平均預測置信度:")
    confidence = ensemble_probs.max(axis=1).mean()
    print(f"  {confidence:.4f}")
    print()

    print("=" * 80)
    print("✅ ULTIMATE ENSEMBLE 完成！")
    print("=" * 80)
    print()
    print(f"融合模型數: {len(available_models)}")
    print(f"TTA 倍數: {TTA_TRANSFORMS}x")
    print(f"總預測次數: {len(available_models)} × {len(test_df)} × {TTA_TRANSFORMS} = {len(available_models) * len(test_df) * TTA_TRANSFORMS:,}")
    print()
    print(f"提交文件已準備: {OUTPUT_PATH}")
    print()

if __name__ == '__main__':
    main()
