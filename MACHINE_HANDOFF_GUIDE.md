# 機器換手完整指南 (Machine Handoff Guide)

**最後更新**: 2025-11-16
**目的**: 從當前機器遷移到新機器或 Google Colab

---

## 📋 快速導航

1. [當前狀態總結](#1-當前狀態總結)
2. [三種換手方案](#2-三種換手方案)
3. [方案 A: Google Colab (推薦)](#方案-a-google-colab-推薦-最快突破-90)
4. [方案 B: 新 GPU 機器](#方案-b-新-gpu-機器-完整遷移)
5. [方案 C: 混合方案](#方案-c-混合方案-並行加速)
6. [依賴性檢查清單](#6-依賴性檢查清單)
7. [故障排查](#7-故障排查)

---

## 1. 當前狀態總結

### 🏆 最佳成績

- **當前最佳**: **87.574%** Macro-F1
- **方法**: Hybrid Adaptive Ensemble (智能偽標籤 + 置信度自適應加權)
- **距離第一名**: 91.085% - 87.574% = **3.511%**
- **目標**: 突破 **90%**

### 🔥 當前訓練中

- **模型**: DINOv2 (vit_base_patch14_dinov2.lvd142m)
- **狀態**: 本地機器背景訓練中（Fold 0-4）
- **預期完成**: 8-10 小時
- **預期分數**: 89.5-90.5% F1

### 📁 已完成的工作

- ✅ 項目代碼清理與重組
- ✅ 最佳 6 個提交結果備份 (`data/submissions/best/`)
- ✅ 完整項目文檔 (`CLAUDE.md`, `README.md`)
- ✅ Git commit 已創建 (e01bb0e)
- ✅ DINOv2 訓練腳本完成
- ✅ Google Colab notebook 已創建

---

## 2. 三種換手方案

### 方案對比

| 方案 | 優勢 | 劣勢 | 適用場景 | 預計時間 |
|------|------|------|----------|----------|
| **A. Google Colab** | 免費 GPU、即時開始、無需設置 | 12-24 小時限制 | 快速驗證、平行訓練 | 2-3 小時啟動 |
| **B. 新 GPU 機器** | 無時間限制、完全控制 | 需要環境設置 | 長期訓練、完整控制 | 1-2 小時設置 |
| **C. 混合方案** | 速度最快、風險分散 | 需要管理多機器 | 加速突破 90% | 立即開始 |

---

## 方案 A: Google Colab (推薦 - 最快突破 90%)

### 為什麼選擇 Colab？

1. **立即開始** - 無需環境設置
2. **免費 GPU** - T4 (16GB) 或 A100 (40GB, Colab Pro)
3. **DINOv2 最佳方案** - 2-3 小時/fold，總計 10-15 小時
4. **預期提升** - +2-4% → 89.5-91.5% F1

### 🚀 快速啟動 (< 5 分鐘)

#### Step 1: 上傳 Notebook

1. 打開 [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook
3. 選擇 `CXR_DINOv2_Breakthrough_90Plus.ipynb`

#### Step 2: 啟用 GPU

1. Runtime → Change runtime type
2. Hardware accelerator → **GPU**
3. (可選) GPU type → **A100** (需要 Colab Pro)

#### Step 3: 準備 Kaggle API

**方式 1: 上傳 kaggle.json** (推薦)
1. 從 Kaggle → Account → Create New API Token 下載 `kaggle.json`
2. 在 Colab 中運行 Cell 3（Kaggle API 設置）
3. 上傳文件

**方式 2: 手動設置**
```python
# 填入你的 Kaggle credentials
kaggle_credentials = {
    "username": "YOUR_KAGGLE_USERNAME",
    "key": "YOUR_KAGGLE_API_KEY"
}
```

#### Step 4: 執行訓練

**選項 A: 快速驗證** (2-3 小時)
- 執行 Cell 7: 訓練 Fold 0
- 檢查 Val F1 ≥ 88%
- 如果成功，繼續完整訓練

**選項 B: 完整訓練** (10-15 小時)
- 執行 Cell 9: 訓練所有 5 Folds
- 需要 Colab Pro 避免中斷

### 📊 預期結果

- **Fold 0 驗證**: 2-3 小時 → Val F1: 88-90%
- **5-Fold 訓練**: 10-15 小時 → Test F1: 89.5-91.5%
- **提交時間**: 即時

### ⚠️ Colab 注意事項

1. **時間限制**:
   - 免費版: 12 小時
   - Colab Pro: 24 小時
   - **建議**: 先跑 Fold 0 驗證，成功後再跑完整訓練

2. **中斷恢復**:
   - 每個 Fold 獨立訓練，中斷後可繼續
   - 模型自動保存到 `outputs/dinov2_colab/fold*/best.pt`

3. **結果保存**:
   - 執行 Cell 12: 保存到 Google Drive
   - 或手動下載模型檢查點

---

## 方案 B: 新 GPU 機器 (完整遷移)

### 環境需求

- **OS**: Ubuntu 22.04+ (Linux)
- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 4070 Ti SUPER, A100, etc.)
- **CUDA**: 12.1+
- **Python**: 3.10+
- **硬碟**: 20 GB (不含數據集)

### 🛠️ 完整設置步驟

#### Step 1: Clone GitHub 倉庫

```bash
# 1. Clone 項目
git clone https://github.com/YOUR_USERNAME/nycu-CSIC30014-LAB3.git
cd nycu-CSIC30014-LAB3

# 2. 檢查最新 commit
git log -1 --oneline
# 應該看到: e01bb0e refactor: Clean up and reorganize project...
```

#### Step 2: 安裝依賴

```bash
# 1. 安裝 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. 安裝其他依賴
pip install timm pandas numpy Pillow tqdm scikit-learn pyyaml

# 3. 驗證安裝
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
```

**預期輸出**:
```
CUDA: True
timm: 0.9.x (或更新版本)
```

#### Step 3: 下載數據集

```bash
# 1. 配置 Kaggle API
# 複製 kaggle.json 到專案根目錄
chmod 600 kaggle.json

# 2. 下載競賽數據 (約 3-4 GB)
kaggle competitions download -c cxr-multi-label-classification
unzip cxr-multi-label-classification.zip -d data/

# 3. 驗證數據
ls data/train_images/ | wc -l  # 應該顯示 2718
ls data/val_images/ | wc -l    # 應該顯示 679
ls data/test_images/ | wc -l   # 應該顯示 1182
```

#### Step 4: 檢查 Fold 數據

```bash
# Fold CSV 應該已經在 Git 倉庫中
ls data/fold_*.csv

# 如果不存在，需要重新生成（但應該已經在倉庫中）
```

#### Step 5: 開始訓練

**選項 1: DINOv2 訓練** (推薦，目標 90%+)
```bash
# 單個 Fold 驗證
python train_dinov2_breakthrough.py \
    --fold 0 \
    --epochs 35 \
    --batch_size 12 \
    --img_size 518 \
    --lr 3e-5 \
    --output_dir outputs/dinov2_new_machine

# 完整 5-Fold 訓練（背景運行）
nohup bash TRAIN_DINOV2_ALL_FOLDS.sh > logs/dinov2_training.log 2>&1 &

# 監控進度
tail -f logs/dinov2_training.log
```

**選項 2: 提交現有最佳結果** (立即可用)
```bash
# 提交當前最佳 (87.574%)
kaggle competitions submit -c cxr-multi-label-classification \
    -f data/submissions/best/01_hybrid_adaptive_87.574.csv \
    -m "Hybrid Adaptive Ensemble - 87.574% (New Machine)"

# 查看結果
kaggle competitions submissions -c cxr-multi-label-classification | head -5
```

---

## 方案 C: 混合方案 (並行加速)

### 策略: 多機器並行

1. **當前機器** (本地 RTX 4070 Ti SUPER)
   - 繼續 DINOv2 訓練 (已在運行中)
   - 預期完成: 8-10 小時

2. **Google Colab** (T4/A100)
   - 同時跑 DINOv2 驗證 (Fold 0)
   - 預期完成: 2-3 小時
   - **優勢**: 快速驗證方案可行性

3. **新 GPU 機器** (如果有)
   - 嘗試其他突破方案 (CAPR, ConvNeXt V2)
   - 或訓練更大的 DINOv2 模型 (Large)

### 執行時間表

| 時間 | 當前機器 | Google Colab | 新機器 (可選) |
|------|----------|--------------|---------------|
| +0h | DINOv2 訓練中 | 啟動 Fold 0 驗證 | Clone 項目 + 設置 |
| +2h | DINOv2 進行中 | Fold 0 完成 ✅ | 開始訓練 |
| +3h | DINOv2 進行中 | 決策: 跑完整訓練? | 訓練進行中 |
| +10h | DINOv2 完成 ✅ | 5-Fold 完成 ✅ | 訓練進行中 |
| +12h | 生成預測 | 提交結果 | 完成並提交 |

### 決策樹

```
開始
 ├─ Colab Fold 0 (2-3h)
 │   ├─ Val F1 ≥ 88% → 立即跑 Colab 完整訓練 (10-15h)
 │   └─ Val F1 < 88% → 等待本地 DINOv2 或嘗試其他方案
 │
 ├─ 本地 DINOv2 (8-10h)
 │   └─ 完成後與 Colab 結果比較，選最佳提交
 │
 └─ 新機器 (可選)
     └─ 訓練備選方案或更大模型
```

---

## 6. 依賴性檢查清單

### Python 套件版本需求

| 套件 | 最低版本 | 推薦版本 | 用途 |
|------|----------|----------|------|
| torch | 2.0.0 | 2.1.0+ | 深度學習框架 |
| torchvision | 0.15.0 | 0.16.0+ | 影像處理 |
| timm | 0.9.0 | 0.9.10+ | DINOv2 模型 |
| pandas | 1.5.0 | 2.0.0+ | 數據處理 |
| numpy | 1.24.0 | 1.24.0+ | 數值計算 |
| Pillow | 9.0.0 | 10.0.0+ | 影像讀取 |
| scikit-learn | 1.2.0 | 1.3.0+ | 評估指標 |
| tqdm | 4.60.0 | 4.66.0+ | 進度條 |
| pyyaml | 6.0 | 6.0.1+ | 配置文件 |

### 自動檢查腳本

```bash
# 創建並運行檢查腳本
cat > check_dependencies.py << 'EOF'
#!/usr/bin/env python3
"""依賴性檢查腳本"""

import sys

def check_package(name, min_version=None):
    try:
        module = __import__(name)
        version = getattr(module, '__version__', 'unknown')
        status = "✅"

        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                status = "⚠️"

        print(f"{status} {name:20} {version}")
        return True
    except ImportError:
        print(f"❌ {name:20} NOT INSTALLED")
        return False

print("🔍 檢查 Python 套件依賴...\n")

packages = [
    ("torch", "2.0.0"),
    ("torchvision", "0.15.0"),
    ("timm", "0.9.0"),
    ("pandas", "1.5.0"),
    ("numpy", "1.24.0"),
    ("PIL", "9.0.0"),
    ("sklearn", "1.2.0"),
    ("tqdm", "4.60.0"),
    ("yaml", "6.0"),
]

all_ok = True
for pkg, min_ver in packages:
    if not check_package(pkg, min_ver):
        all_ok = False

# Check CUDA
print("\n🔍 檢查 CUDA...\n")
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ CUDA not available (CPU only)")
except Exception as e:
    print(f"❌ CUDA check failed: {e}")

print("\n" + "="*50)
if all_ok:
    print("✅ 所有依賴檢查通過！")
    sys.exit(0)
else:
    print("❌ 部分依賴缺失，請安裝")
    sys.exit(1)
EOF

python check_dependencies.py
```

---

## 7. 故障排查

### 問題 1: CUDA out of memory

**症狀**: `RuntimeError: CUDA out of memory`

**解決方案**:
```bash
# 降低 batch size
python train_dinov2_breakthrough.py \
    --batch_size 8  # 從 12 降到 8
    --fold 0 \
    --epochs 35
```

### 問題 2: timm 找不到 DINOv2 模型

**症狀**: `ValueError: Unknown model 'vit_base_patch14_dinov2.lvd142m'`

**解決方案**:
```bash
# 升級 timm 到最新版本
pip install --upgrade timm

# 驗證
python -c "import timm; print([m for m in timm.list_models() if 'dinov2' in m])"
```

### 問題 3: Kaggle API 認證失敗

**症狀**: `OSError: Could not find kaggle.json`

**解決方案**:
```bash
# 確認 kaggle.json 位置和權限
ls -la ~/.kaggle/kaggle.json  # 或專案根目錄的 kaggle.json
chmod 600 kaggle.json

# 測試
kaggle competitions list | head -3
```

### 問題 4: 找不到 Fold 數據

**症狀**: `FileNotFoundError: data/fold0_train.csv`

**解決方案**:
```bash
# 檢查 Fold CSV 是否存在
ls data/fold_*.csv

# 如果不存在，確認 Git 倉庫最新
git pull origin main

# 如果仍然缺失，可能需要重新創建（但應該在倉庫中）
```

### 問題 5: Colab 中斷後如何恢復

**症狀**: 訓練進行到一半，Colab 斷線

**解決方案**:
1. 重新連接 Colab
2. 重新執行環境設置 Cells (1-6)
3. 檢查已完成的 Folds:
   ```python
   import os
   for fold in range(5):
       if os.path.exists(f'outputs/dinov2_colab/fold{fold}/best.pt'):
           print(f\"✅ Fold {fold} 已完成\")
   ```
4. 繼續未完成的 Folds:
   ```python
   # 修改 Cell 9，只訓練未完成的 Folds
   for fold in [2, 3, 4]:  # 假設 0, 1 已完成
       # ... 訓練代碼
   ```

---

## 8. 成功指標

### Colab 快速驗證 (Fold 0)

- ✅ 訓練完成無錯誤
- ✅ Val F1 ≥ 88%
- ✅ Val-Test gap < 2%

### 完整訓練 (5-Fold)

- ✅ 所有 5 個 Folds 訓練完成
- ✅ 平均 Val F1 ≥ 88%
- ✅ Test F1 ≥ 89.5%

### 最終目標

- 🎯 Test F1 ≥ **90%**
- 🏆 進入競賽 Top 5

---

## 9. 聯絡與支援

- **項目倉庫**: (填入你的 GitHub URL)
- **Kaggle 競賽**: https://www.kaggle.com/competitions/cxr-multi-label-classification

---

## 📝 換手檢查清單

### 開始前

- [ ] 閱讀本文件 (MACHINE_HANDOFF_GUIDE.md)
- [ ] 閱讀 README.md 快速啟動指南
- [ ] 閱讀 CLAUDE.md 項目記憶
- [ ] 確認選擇的方案 (Colab / 新機器 / 混合)

### Colab 方案

- [ ] 上傳 Notebook 到 Colab
- [ ] 啟用 GPU Runtime
- [ ] 上傳 kaggle.json
- [ ] 執行 Cell 1-6 (環境設置)
- [ ] 訓練 Fold 0 驗證
- [ ] 決策: 繼續完整訓練或調整

### 新機器方案

- [ ] Clone GitHub 倉庫
- [ ] 安裝 Python 依賴
- [ ] 運行依賴性檢查腳本
- [ ] 下載 Kaggle 數據集
- [ ] 驗證數據完整性
- [ ] 開始訓練或提交現有結果

### 完成後

- [ ] 測試集分數 ≥ 87.574% (當前最佳)
- [ ] 如果 ≥ 90%，慶祝突破！🎉
- [ ] 保存模型檢查點
- [ ] 更新 CLAUDE.md 記錄結果
- [ ] Git commit 並 push

---

**記住**: DINOv2 是最有潛力突破 90% 的方案，先在 Colab 上快速驗證效果！

**預計總時間**:
- Colab Fold 0 驗證: 2-3 小時
- Colab 完整訓練: 10-15 小時
- 新機器設置 + 訓練: 11-17 小時

**祝你成功突破 90%！** 🚀
