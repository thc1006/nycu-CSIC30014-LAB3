#!/usr/bin/env python3
"""
生成 EfficientNet-V2-L @ 512 的 5-Fold 預測並集成
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import from src
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from predict import predict_logits, TestSet
from utils import load_config

def generate_fold_predictions(fold_id, config_path, checkpoint_path, device):
    """生成單個 fold 的測試集預測 (返回概率而非 one-hot)"""
    print(f"\n{'='*70}")
    print(f"🔮 生成 Fold {fold_id} 預測")
    print(f"{'='*70}")

    # Load config
    config = load_config(config_path)

    # Create test dataset - use existing test_data.csv
    test_csv = 'data/test_data.csv'
    test_dir = 'test_images'  # test images directory
    img_size = config['model']['img_size']
    batch_size = config['train']['batch_size']

    test_dataset = TestSet(test_csv, test_dir, 'new_filename', img_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    from torch import nn
    from torchvision import models

    model = models.efficientnet_v2_l(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        best_f1 = checkpoint.get('best_f1', 'N/A')
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        best_f1 = 'N/A'
    else:
        model.load_state_dict(checkpoint)
        best_f1 = 'N/A'

    model = model.to(device)

    print(f"✅ 模型已加載: {checkpoint_path}")
    if best_f1 != 'N/A':
        print(f"   最佳驗證分數: {best_f1:.4f}")

    # Generate predictions (logits)
    logits, filenames = predict_logits(model, test_loader, device)

    # Convert to probabilities
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    print(f"✅ Fold {fold_id} 預測完成: {len(filenames)} 張影像")
    print(f"   預測形狀: {probs.shape}")

    return filenames, probs

def create_submission(filenames, predictions, output_path, class_names):
    """創建提交檔案 (概率分布)"""
    df = pd.DataFrame(predictions, columns=class_names)
    df.insert(0, 'new_filename', filenames)
    df.to_csv(output_path, index=False)
    print(f"✅ 提交檔案已保存: {output_path}")

    # Print statistics
    pred_classes = np.argmax(predictions, axis=1)
    class_counts = np.bincount(pred_classes, minlength=len(class_names))
    print(f"\n預測分布:")
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        print(f"  {name}: {count} ({count/len(predictions)*100:.1f}%)")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用設備: {device}")

    config_path = "configs/efficientnet_v2l_512_breakthrough.yaml"
    base_dir = Path("outputs/v2l_512_breakthrough")
    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

    # Generate predictions for all folds
    all_fold_preds = []
    filenames = None

    for fold_id in range(5):
        checkpoint_path = base_dir / f"fold{fold_id}" / "best.pt"

        if not checkpoint_path.exists():
            print(f"❌ 檢查點不存在: {checkpoint_path}")
            continue

        fold_filenames, fold_preds = generate_fold_predictions(
            fold_id, config_path, checkpoint_path, device
        )

        # Save individual fold submission
        fold_submission_path = f"data/submission_v2l_512_fold{fold_id}.csv"
        create_submission(fold_filenames, fold_preds, fold_submission_path, class_names)

        all_fold_preds.append(fold_preds)
        if filenames is None:
            filenames = fold_filenames

    # Ensemble: Average all folds
    print(f"\n{'='*70}")
    print("🎯 5-Fold 集成")
    print(f"{'='*70}")

    ensemble_preds = np.mean(all_fold_preds, axis=0)
    ensemble_submission_path = "data/submission_v2l_512_ensemble.csv"

    create_submission(filenames, ensemble_preds, ensemble_submission_path, class_names)

    print(f"\n{'='*70}")
    print("✅ 所有預測生成完成！")
    print(f"{'='*70}")
    print(f"📂 輸出位置:")
    print(f"   • 各 Fold: data/submission_v2l_512_fold{{0-4}}.csv")
    print(f"   • 5-Fold 集成: {ensemble_submission_path}")
    print(f"\n下一步: 與現有最佳模型 (87.574%) 進行混合集成")

if __name__ == "__main__":
    main()
