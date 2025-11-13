#!/bin/bash
# 快速偽標籤訓練腳本 - 第1名風格
# Quick pseudo-label training - Champion style

echo "=========================================="
echo "第1名風格：偽標籤訓練"
echo "=========================================="

# 使用 fold0 的配置，但替換訓練數據為偽標籤數據
python3 src/train_standalone.py \
  --train-csv data/train_data_ultra_augmented_fixed.csv \
  --train-dir data/train \
  --model efficientnet_v2_s \
  --img-size 384 \
  --epochs 25 \
  --batch-size 24 \
  --lr 0.00003 \
  --output-dir outputs/champion_pseudo_stage1 \
  --seed 42

echo "=========================================="
echo "訓練完成！"
echo "預期提升: +1-2%"
echo "=========================================="
