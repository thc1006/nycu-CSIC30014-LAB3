#!/bin/bash
# 🎯 90分突破 - 立即执行脚本
# 预计时间: 10-15小时
# 目标分数: 88.5-90%

set -e

echo "========================================================================"
echo "🚀 90分突破计划 - 开始执行"
echo "========================================================================"
echo "硬件: RTX 4070 Ti SUPER (16GB) + Intel i5-14500 (20 cores)"
echo "当前最佳: 87.574%"
echo "目标: 90.000% (+2.426%)"
echo "========================================================================"
echo ""

# 创建日志目录
mkdir -p logs/breakthrough_90
LOG_DIR="logs/breakthrough_90"

# ============================================================================
# 阶段 1: 快速优化 - DINOv2 TTA (1小时)
# ============================================================================

echo "📊 阶段 1/3: DINOv2 TTA 10-Crop 集成"
echo "预期提升: +0.5-1.0% → 87.2-87.7%"
echo "时间: ~30分钟"
echo "------------------------------------------------------------------------"

python3 << 'EOF'
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm

class TTADataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=518, tta_mode='10crop'):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_size = img_size
        self.tta_mode = tta_mode

        # 基础变换
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(self.img_dir) / row['new_filename']
        img = Image.open(img_path).convert('RGB')

        # 10-crop TTA
        crops = []

        # 原图 5-crop
        five_crop = T.FiveCrop(self.img_size)
        for crop in five_crop(img):
            crops.append(self.normalize(T.ToTensor()(crop)))

        # 水平翻转 5-crop
        img_flip = T.functional.hflip(img)
        for crop in five_crop(img_flip):
            crops.append(self.normalize(T.ToTensor()(crop)))

        return torch.stack(crops), row['new_filename']

print("=" * 70)
print("🔮 DINOv2 TTA 10-Crop Prediction")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# 加载数据
test_dataset = TTADataset('data/test_data_sample.csv', 'data/test_images', img_size=518)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,
                         num_workers=4, pin_memory=True)

print(f"✅ Test dataset: {len(test_dataset)} samples\n")

# 加载5个fold模型并进行TTA预测
all_fold_preds = []

for fold in range(5):
    model_path = f'outputs/dinov2_breakthrough/fold{fold}/best.pt'

    if not Path(model_path).exists():
        print(f"⚠️ Fold {fold} model not found, skipping")
        continue

    print(f"📊 Fold {fold} TTA predicting...")

    # 加载模型
    model = timm.create_model('vit_base_patch14_dinov2', pretrained=False, num_classes=4)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    fold_probs = []
    filenames_list = []

    with torch.no_grad():
        for crops_batch, fnames in tqdm(test_loader, desc=f'Fold {fold} TTA', leave=False):
            # crops_batch: [batch_size, 10, 3, 518, 518]
            batch_size = crops_batch.size(0)
            n_crops = crops_batch.size(1)

            # Reshape: [batch_size * 10, 3, 518, 518]
            crops = crops_batch.view(-1, 3, 518, 518).to(device)

            # 预测
            outputs = model(crops)
            probs = torch.softmax(outputs, dim=1)

            # Reshape back: [batch_size, 10, 4]
            probs = probs.view(batch_size, n_crops, 4)

            # 平均10个crop
            avg_probs = probs.mean(dim=1)  # [batch_size, 4]

            fold_probs.append(avg_probs.cpu().numpy())
            filenames_list.extend(fnames)

    fold_probs = np.concatenate(fold_probs, axis=0)
    all_fold_preds.append(fold_probs)

    print(f"   ✅ Fold {fold} TTA complete\n")

    del model
    torch.cuda.empty_cache()

# 集成5个fold
print(f"🔮 Ensembling {len(all_fold_preds)} folds...")
avg_probs = np.mean(all_fold_preds, axis=0)
final_preds = np.argmax(avg_probs, axis=1)

# 创建提交
class_names = ['normal', 'bacteria', 'virus', 'COVID-19']

# One-hot格式
submission_df = pd.DataFrame({
    'new_filename': filenames_list[:len(final_preds)]
})

for i, cls in enumerate(class_names):
    submission_df[cls] = (final_preds == i).astype(int)

submission_path = 'data/submission_dinov2_tta_10crop.csv'
submission_df.to_csv(submission_path, index=False)

print(f"\n✅ TTA Submission saved: {submission_path}")
print(f"\n📊 Prediction distribution:")
for i, cls in enumerate(class_names):
    count = (final_preds == i).sum()
    print(f"  {cls}: {count} ({count/len(final_preds)*100:.1f}%)")

print("\n" + "=" * 70)
print("✅ DINOv2 TTA Complete!")
print("=" * 70)
EOF

echo ""
echo "✅ 阶段 1 完成！"
echo ""

# 提交TTA结果
echo "📤 提交 TTA 预测到 Kaggle..."
kaggle competitions submit -c cxr-multi-label-classification \
  -f data/submission_dinov2_tta_10crop.csv \
  -m "DINOv2 5-Fold + TTA 10-crop (5 original + 5 flipped) - Target 87.5%+"

echo ""
sleep 5

# 查看提交状态
echo "📊 检查提交状态..."
kaggle competitions submissions -c cxr-multi-label-classification | head -8

echo ""
echo "========================================================================"
echo "✅ 阶段 1 完成: TTA 已提交"
echo "   预期分数: 87.2-87.7% (+0.5-1.0% from 86.7%)"
echo "========================================================================"
echo ""
echo "继续执行阶段 2..."
sleep 3

# ============================================================================
# 阶段 2: 智能超级集成 (30分钟)
# ============================================================================

echo ""
echo "📊 阶段 2/3: 创建智能超级集成"
echo "预期提升: 集成多个高分模型"
echo "目标分数: 88.0-88.5%"
echo "时间: ~30分钟"
echo "------------------------------------------------------------------------"

python3 << 'EOF'
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("🔮 创建智能加权超级集成")
print("=" * 70)

# 可用的高分提交
submissions = {
    'submission_ultra_top3_weighted.csv': {
        'score': 0.87574,
        'weight': 0.35,
        'desc': 'Hybrid Adaptive (最佳)'
    },
    'submission_ultra_majority_vote.csv': {
        'score': 0.86683,
        'weight': 0.25,
        'desc': 'Ultra Majority Vote'
    },
    'submission_dinov2_5fold_onehot.csv': {
        'score': 0.86702,
        'weight': 0.25,
        'desc': 'DINOv2 5-Fold'
    },
    'champion_balanced.csv': {
        'score': 0.84423,
        'weight': 0.15,
        'desc': 'Champion Balanced Stacking'
    },
}

print("\n📋 集成模型列表:")
for fname, info in submissions.items():
    print(f"  {fname:45} {info['score']:.5f} (权重: {info['weight']:.2f}) - {info['desc']}")

print(f"\n总权重: {sum(s['weight'] for s in submissions.values()):.2f}")

# 加载所有预测
all_probs = []
weights = []

for fname, info in submissions.items():
    fpath = Path('data') / fname

    if not fpath.exists():
        fpath = Path('data/champion_submissions') / fname

    if not fpath.exists():
        print(f"⚠️ 文件不存在，跳过: {fname}")
        continue

    df = pd.read_csv(fpath)

    # 检查格式
    if 'label' in df.columns:
        # 转换为one-hot
        class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
        probs = np.zeros((len(df), 4))
        for i, cls in enumerate(class_names):
            probs[df['label'] == cls, i] = 1.0
    else:
        # 已经是one-hot
        class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
        probs = df[class_names].values

    all_probs.append(probs)
    weights.append(info['weight'])
    print(f"✅ 加载: {fname}")

# 加权平均
weights = np.array(weights) / np.sum(weights)  # 归一化
weighted_probs = np.average(all_probs, axis=0, weights=weights)

# 预测
final_preds = np.argmax(weighted_probs, axis=1)

# 创建提交
filenames = pd.read_csv('data/test_data_sample.csv')['new_filename'].values

submission_df = pd.DataFrame({
    'new_filename': filenames
})

class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
for i, cls in enumerate(class_names):
    submission_df[cls] = (final_preds == i).astype(int)

submission_path = 'data/submission_super_ensemble_weighted.csv'
submission_df.to_csv(submission_path, index=False)

print(f"\n✅ Super Ensemble saved: {submission_path}")
print(f"\n📊 Prediction distribution:")
for i, cls in enumerate(class_names):
    count = (final_preds == i).sum()
    print(f"  {cls}: {count} ({count/len(final_preds)*100:.1f}%)")

print("\n" + "=" * 70)
print("✅ Super Ensemble Complete!")
print("预期分数: 88.0-88.5%")
print("=" * 70)
EOF

echo ""
echo "📤 提交超级集成到 Kaggle..."
kaggle competitions submit -c cxr-multi-label-classification \
  -f data/submission_super_ensemble_weighted.csv \
  -m "Super Ensemble: Weighted(87.574×0.35 + 86.7×0.25 + 86.68×0.25 + 84.42×0.15) - Target 88%+"

echo ""
sleep 5

echo "📊 检查提交状态..."
kaggle competitions submissions -c cxr-multi-label-classification | head -8

echo ""
echo "========================================================================"
echo "✅ 阶段 2 完成: 超级集成已提交"
echo "   预期分数: 88.0-88.5%"
echo "========================================================================"
echo ""

# ============================================================================
# 阶段 3: Rank Averaging 终极集成 (15分钟)
# ============================================================================

echo ""
echo "📊 阶段 3/3: Rank Averaging 终极集成"
echo "原理: 将概率转换为排名后再平均，更稳健"
echo "目标分数: 88.5-89.0%"
echo "时间: ~15分钟"
echo "------------------------------------------------------------------------"

python3 << 'EOF'
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import rankdata

print("=" * 70)
print("🏆 Rank Averaging 终极集成")
print("=" * 70)

# 使用所有高分提交
submissions = [
    'submission_ultra_top3_weighted.csv',
    'submission_ultra_majority_vote.csv',
    'submission_dinov2_5fold_onehot.csv',
    'submission_dinov2_tta_10crop.csv',  # 新的TTA
    'champion_balanced.csv',
]

print(f"\n📋 使用 {len(submissions)} 个模型进行 Rank Averaging\n")

all_ranks = []

for fname in submissions:
    fpath = Path('data') / fname

    if not fpath.exists():
        fpath = Path('data/champion_submissions') / fname

    if not fpath.exists():
        print(f"⚠️ 跳过: {fname}")
        continue

    df = pd.read_csv(fpath)

    # 转换为概率
    if 'label' in df.columns:
        class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
        probs = np.zeros((len(df), 4))
        for i, cls in enumerate(class_names):
            probs[df['label'] == cls, i] = 1.0
    else:
        class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
        probs = df[class_names].values

    # 为每个类别计算排名
    ranks = np.zeros_like(probs)
    for i in range(4):
        # 降序排名（概率越高排名越靠前）
        ranks[:, i] = rankdata(-probs[:, i])

    all_ranks.append(ranks)
    print(f"✅ Rank: {fname}")

# 平均排名
avg_ranks = np.mean(all_ranks, axis=0)

# 根据排名预测（排名越小越可能）
final_preds = np.argmin(avg_ranks, axis=1)

# 创建提交
filenames = pd.read_csv('data/test_data_sample.csv')['new_filename'].values

submission_df = pd.DataFrame({
    'new_filename': filenames
})

class_names = ['normal', 'bacteria', 'virus', 'COVID-19']
for i, cls in enumerate(class_names):
    submission_df[cls] = (final_preds == i).astype(int)

submission_path = 'data/submission_rank_averaging.csv'
submission_df.to_csv(submission_path, index=False)

print(f"\n✅ Rank Averaging saved: {submission_path}")
print(f"\n📊 Prediction distribution:")
for i, cls in enumerate(class_names):
    count = (final_preds == i).sum()
    print(f"  {cls}: {count} ({count/len(final_preds)*100:.1f}%)")

print("\n" + "=" * 70)
print("🏆 Rank Averaging Complete!")
print("预期分数: 88.5-89.0%")
print("=" * 70)
EOF

echo ""
echo "📤 提交 Rank Averaging 到 Kaggle..."
kaggle competitions submit -c cxr-multi-label-classification \
  -f data/submission_rank_averaging.csv \
  -m "Rank Averaging: 5 Top Models (87.574% + 86.7% + 86.68% + TTA + Champion) - Target 88.5%+"

echo ""
sleep 5

echo "📊 最终提交状态..."
kaggle competitions submissions -c cxr-multi-label-classification | head -10

echo ""
echo "========================================================================"
echo "🎉 90分突破计划 - 全部完成！"
echo "========================================================================"
echo ""
echo "📊 提交总结:"
echo "   1. DINOv2 TTA 10-crop      → 预期: 87.2-87.7%"
echo "   2. Super Ensemble Weighted → 预期: 88.0-88.5%"
echo "   3. Rank Averaging          → 预期: 88.5-89.0%"
echo ""
echo "⏰ 总耗时: ~1.5小时"
echo ""
echo "🎯 下一步建议:"
echo "   - 等待 Kaggle 评分 (10-15分钟)"
echo "   - 如果达到88%+，可尝试 DINOv2-Large 训练 (8-10小时)"
echo "   - 如果达到89%+，已经非常接近90%目标！"
echo ""
echo "========================================================================"
