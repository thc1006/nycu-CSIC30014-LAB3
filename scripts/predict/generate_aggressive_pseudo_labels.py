#!/usr/bin/env python3
"""
生成激進偽標籤 - 方案 B
使用當前最佳模型集成，降低閾值至 0.72，生成 ~2000+ 偽標籤
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import timm

class TestDataset(Dataset):
    """測試集數據集"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.image_files = sorted(list(self.image_dir.glob('*.*')))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_path.name

def load_champion_model(checkpoint_path, model_name, num_classes=4, img_size=384):
    """載入 Champion 模型"""
    # 創建模型
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.3,
        drop_path_rate=0.2
    )

    # 載入權重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.cuda()
    model.eval()

    return model

def predict_with_ensemble(models, dataloader, tta=True):
    """
    使用集成模型進行預測

    Args:
        models: List of (model, weight) tuples
        dataloader: 數據加載器
        tta: 是否使用測試時增強
    """
    all_predictions = []
    all_filenames = []

    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Generating predictions"):
            images = images.cuda()

            # 集成所有模型的預測
            ensemble_logits = None
            total_weight = 0

            for model, weight in models:
                # 基礎預測
                logits = model(images)

                # TTA: 水平翻轉
                if tta:
                    logits_flip = model(torch.flip(images, dims=[3]))
                    logits = (logits + logits_flip) / 2

                # 加權
                if ensemble_logits is None:
                    ensemble_logits = logits * weight
                else:
                    ensemble_logits += logits * weight
                total_weight += weight

            # 歸一化權重
            ensemble_logits = ensemble_logits / total_weight

            # Softmax 獲得概率
            probs = torch.softmax(ensemble_logits, dim=1)

            all_predictions.append(probs.cpu().numpy())
            all_filenames.extend(filenames)

    all_predictions = np.concatenate(all_predictions, axis=0)
    return all_predictions, all_filenames

def generate_tiered_pseudo_labels(predictions, filenames, test_dir):
    """
    生成三層偽標籤

    Tier 1 (0.85+): 高置信度，權重 1.0
    Tier 2 (0.75-0.85): 中等置信度，權重 0.6
    Tier 3 (0.72-0.75): 低置信度，權重 0.3
    """

    max_probs = predictions.max(axis=1)
    pred_labels = predictions.argmax(axis=1)

    # 三層分層
    tier1_mask = max_probs >= 0.85
    tier2_mask = (max_probs >= 0.75) & (max_probs < 0.85)
    tier3_mask = (max_probs >= 0.72) & (max_probs < 0.75)

    results = {
        'tier1': {'filenames': [], 'labels': [], 'confidences': [], 'weight': 1.0},
        'tier2': {'filenames': [], 'labels': [], 'confidences': [], 'weight': 0.6},
        'tier3': {'filenames': [], 'labels': [], 'confidences': [], 'weight': 0.3}
    }

    class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

    # Tier 1
    for i in np.where(tier1_mask)[0]:
        results['tier1']['filenames'].append(filenames[i])
        results['tier1']['labels'].append(class_names[pred_labels[i]])
        results['tier1']['confidences'].append(max_probs[i])

    # Tier 2
    for i in np.where(tier2_mask)[0]:
        results['tier2']['filenames'].append(filenames[i])
        results['tier2']['labels'].append(class_names[pred_labels[i]])
        results['tier2']['confidences'].append(max_probs[i])

    # Tier 3
    for i in np.where(tier3_mask)[0]:
        results['tier3']['filenames'].append(filenames[i])
        results['tier3']['labels'].append(class_names[pred_labels[i]])
        results['tier3']['confidences'].append(max_probs[i])

    return results

def save_pseudo_labels(results, output_dir):
    """保存偽標籤到 CSV"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("📊 激進偽標籤生成報告")
    print("="*60)

    total_samples = 0

    for tier_name, data in results.items():
        tier_num = tier_name[-1]
        n_samples = len(data['filenames'])
        total_samples += n_samples

        if n_samples == 0:
            continue

        # 創建 DataFrame
        df = pd.DataFrame({
            'filename': data['filenames'],
            'label': data['labels'],
            'confidence': data['confidences'],
            'weight': data['weight']
        })

        # 保存
        output_path = output_dir / f'pseudo_labels_{tier_name}.csv'
        df.to_csv(output_path, index=False)

        # 統計
        print(f"\n{tier_name.upper()} (confidence >= {0.85 if tier_num=='1' else 0.75 if tier_num=='2' else 0.72}):")
        print(f"  樣本數: {n_samples}")
        print(f"  權重: {data['weight']}")
        print(f"  平均置信度: {np.mean(data['confidences']):.4f}")
        print(f"  類別分布:")
        for label in ['normal', 'bacteria', 'virus', 'COVID-19']:
            count = df['label'].value_counts().get(label, 0)
            print(f"    {label}: {count} ({count/n_samples*100:.1f}%)")
        print(f"  文件: {output_path}")

    # 合併保存
    all_data = []
    for tier_name, data in results.items():
        if len(data['filenames']) > 0:
            df = pd.DataFrame({
                'filename': data['filenames'],
                'label': data['labels'],
                'confidence': data['confidences'],
                'weight': data['weight'],
                'tier': tier_name
            })
            all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        output_path = output_dir / 'pseudo_labels_combined.csv'
        combined_df.to_csv(output_path, index=False)

        print(f"\n{'='*60}")
        print(f"總計: {total_samples} 個偽標籤")
        print(f"合併文件: {output_path}")
        print(f"{'='*60}\n")

        # 與原始偽標籤對比
        print("📈 相比原始偽標籤:")
        print(f"  原始 (threshold=0.80): 1065 個")
        print(f"  激進 (threshold=0.72): {total_samples} 個")
        print(f"  增加: +{total_samples - 1065} 個 (+{(total_samples-1065)/1065*100:.1f}%)")

def main():
    """主函數"""

    # 配置
    TEST_DIR = Path('data/test')
    OUTPUT_DIR = Path('data/pseudo_labels_aggressive')
    IMG_SIZE = 384
    BATCH_SIZE = 32

    print("🚀 激進偽標籤生成 - 方案 B")
    print("="*60)
    print(f"測試集目錄: {TEST_DIR}")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print(f"圖片大小: {IMG_SIZE}")
    print(f"閾值設定:")
    print(f"  Tier 1: >= 0.85 (權重 1.0)")
    print(f"  Tier 2: 0.75-0.85 (權重 0.6)")
    print(f"  Tier 3: 0.72-0.75 (權重 0.3)")
    print("="*60)

    # 數據轉換
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 創建數據集和加載器
    test_dataset = TestDataset(TEST_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"\n📁 測試集樣本數: {len(test_dataset)}")

    # 載入最佳的 Champion 模型
    print("\n🔧 載入 Champion 模型...")

    models = []
    model_configs = [
        # ConvNeXt-Large (3 folds, weight 1.0 each)
        ('outputs/champion_convnext_large/fold0/best.pt', 'convnext_large', 1.0),
        ('outputs/champion_convnext_large/fold1/best.pt', 'convnext_large', 1.0),
        ('outputs/champion_convnext_large/fold2/best.pt', 'convnext_large', 1.0),
        # ViT-Large (2 folds, weight 1.3 each)
        ('outputs/champion_vit_large/fold0/best.pt', 'vit_large_patch16_384', 1.3),
        ('outputs/champion_vit_large/fold1/best.pt', 'vit_large_patch16_384', 1.3),
        # BEiT-Large (2 folds, weight 1.3 each)
        ('outputs/champion_beit_large/fold0/best.pt', 'beit_large_patch16_384', 1.3),
        ('outputs/champion_beit_large/fold1/best.pt', 'beit_large_patch16_384', 1.3),
    ]

    for checkpoint_path, model_name, weight in model_configs:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            try:
                print(f"  ✓ {model_name} from {checkpoint_path.parent.name} (weight={weight})")
                model = load_champion_model(checkpoint_path, model_name, num_classes=4, img_size=IMG_SIZE)
                models.append((model, weight))
            except Exception as e:
                print(f"  ✗ Failed to load {checkpoint_path}: {e}")
        else:
            print(f"  ✗ Not found: {checkpoint_path}")

    print(f"\n✅ 成功載入 {len(models)} 個模型")

    # 生成預測
    print("\n🔮 生成集成預測 (with TTA)...")
    predictions, filenames = predict_with_ensemble(models, test_loader, tta=True)

    # 生成三層偽標籤
    print("\n📝 生成三層偽標籤...")
    results = generate_tiered_pseudo_labels(predictions, filenames, TEST_DIR)

    # 保存結果
    save_pseudo_labels(results, OUTPUT_DIR)

    print("\n✅ 激進偽標籤生成完成！")
    print(f"下一步: 使用這些偽標籤重新訓練 Top 3 模型")

if __name__ == '__main__':
    main()
