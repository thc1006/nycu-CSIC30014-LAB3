#!/bin/bash
# 下載 NIH ChestX-ray14 數據集（預處理版本 224x224）
# Download NIH ChestX-ray14 (preprocessed 224x224 version - 7GB)

echo "=========================================="
echo "下載 NIH ChestX-ray14 數據集"
echo "這是第1名最重要的技巧！"
echo "=========================================="

# 創建目錄
mkdir -p data/external/nih_chestxray14

cd data/external/nih_chestxray14

# 使用 Kaggle API 下載預處理版本（224x224, 7GB）
# 這比原始版本（42GB）小得多
echo "下載預處理版本（224x224）..."
kaggle datasets download -d khanfashee/nih-chest-x-ray-14-224x224-resized

# 解壓
echo "解壓縮..."
unzip -q nih-chest-x-ray-14-224x224-resized.zip

echo "=========================================="
echo "下載完成！"
echo "數據集位置: data/external/nih_chestxray14/"
ls -lh
echo "=========================================="
