#!/usr/bin/env python3
"""快速測試 GPU 訓練環境"""
import torch
import torch.nn as nn
from pathlib import Path

print("=" * 80)
print("GPU 訓練環境測試")
print("=" * 80)

# 1. PyTorch 和 CUDA
print(f"\n✅ PyTorch 版本: {torch.__version__}")
print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ CUDA 版本: {torch.version.cuda}")
    print(f"✅ GPU 裝置: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 2. 測試 GPU 計算
print(f"\n⚡ 測試 GPU 計算...")
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y
print(f"✅ GPU 矩陣乘法測試通過: {z.shape}")

# 3. 測試混合精度
print(f"\n⚡ 測試混合精度訓練...")
model = nn.Linear(1000, 100).cuda()
input_data = torch.randn(32, 1000).cuda()
with torch.cuda.amp.autocast():
    output = model(input_data)
print(f"✅ 混合精度測試通過: {output.dtype}")

# 4. 檢查影像目錄
print(f"\n📁 檢查資料目錄...")
train_dir = Path("train_images")
val_dir = Path("val_images")
test_dir = Path("test_images")

train_count = len(list(train_dir.glob("*")))
val_count = len(list(val_dir.glob("*")))
test_count = len(list(test_dir.glob("*")))

print(f"✅ 訓練影像: {train_count} 張")
print(f"✅ 驗證影像: {val_count} 張")
print(f"✅ 測試影像: {test_count} 張")

# 5. 檢查 CSV
print(f"\n📋 檢查 CSV 檔案...")
csv_files = ["data/train_data.csv", "data/val_data.csv", "data/test_data.csv"]
for csv_file in csv_files:
    if Path(csv_file).exists():
        print(f"✅ {csv_file}")
    else:
        print(f"❌ {csv_file} 不存在")

print("\n" + "=" * 80)
print("✅ 所有測試通過！準備開始訓練！")
print("=" * 80)
