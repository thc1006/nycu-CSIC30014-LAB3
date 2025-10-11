# 🔧 Kaggle API Setup & Troubleshooting Guide

## 常見問題與解決方案

### ❌ 錯誤 1: `403 Forbidden`

```
403 Client Error: Forbidden for url: https://www.kaggle.com/api/v1/competitions/data/download-all/chest-xray-pneumonia
```

**可能原因**:

1. **未接受競賽規則** ⚠️ 最常見
2. 使用了錯誤的competition name
3. 這是Dataset而非Competition
4. 權限設置問題

---

## ✅ 解決方案

### 方案 1: 接受競賽規則 (最可能)

**步驟**:

1. 前往競賽頁面: `https://www.kaggle.com/competitions/YOUR-COMPETITION-NAME`
2. 點擊 **"Join Competition"** 或 **"Late Submission"**
3. 閱讀並接受規則
4. 再次運行下載命令

**重要**: 即使競賽已結束，也必須先"加入"競賽才能下載數據！

---

### 方案 2: 使用Dataset而非Competition

如果數據在Kaggle Dataset (不是Competition):

```python
# 在notebook中使用這段代碼替代

# For Kaggle Dataset (not competition)
DATASET_NAME = "paultimothymooney/chest-xray-pneumonia"  # 範例

!kaggle datasets download -d $DATASET_NAME
!unzip -q chest-xray-pneumonia.zip
```

**如何找到正確的dataset name**:
1. 前往Kaggle dataset頁面
2. URL格式: `https://www.kaggle.com/datasets/USERNAME/DATASET-NAME`
3. Dataset name = `USERNAME/DATASET-NAME`

---

### 方案 3: 手動上傳到Google Drive (最穩定)

**步驟**:

#### A. 在本地下載數據

```bash
# 在本地電腦 (已有kaggle.json)
kaggle competitions download -c YOUR-COMPETITION-NAME
# 或
kaggle datasets download -d USERNAME/DATASET-NAME
```

#### B. 上傳到Google Drive

1. 解壓縮下載的zip檔案
2. 上傳到Google Drive
3. 組織成以下結構:

```
MyDrive/chest-xray-data/
  ├── train_images/
  ├── val_images/
  └── test_images/
```

#### C. 使用 `A100_Ultra_Optimized.ipynb`

這個notebook使用Google Drive，不需要Kaggle API。

---

### 方案 4: 直接在Kaggle Notebook訓練 ⭐ 推薦！

**為什麼這是最好的方案**:
- ✅ 數據已經在Kaggle上
- ✅ 不需要下載
- ✅ 免費P100/T4 GPU (或付費A100)
- ✅ 無網路限制

**步驟**:

1. **創建Kaggle Notebook**
   - 前往: https://www.kaggle.com/code
   - 點擊 "New Notebook"

2. **添加數據**
   - 右側 "Add Data"
   - 搜索你的competition/dataset
   - 點擊 "Add"

3. **設定GPU**
   - Settings → Accelerator → GPU P100/T4
   - (付費用戶可選 TPU v3-8)

4. **Clone代碼**
   ```python
   !git clone https://github.com/thc1006/nycu-CSIC30014-LAB3.git
   %cd nycu-CSIC30014-LAB3
   ```

5. **更新路徑**
   ```python
   # Kaggle數據路徑
   import os

   # 查看數據位置
   !ls /kaggle/input/

   # 數據通常在:
   train_path = "/kaggle/input/YOUR-DATASET/train"
   test_path = "/kaggle/input/YOUR-DATASET/test"
   ```

6. **運行訓練**
   ```python
   !python -m src.train_v2 --config configs/model_stage1.yaml
   ```

---

## 🔍 診斷工具

### 檢查Kaggle API設置

```python
# 在Colab或Kaggle Notebook運行

# 1. 檢查kaggle.json
!cat ~/.kaggle/kaggle.json

# 2. 測試API連接
!kaggle competitions list | head -5

# 3. 檢查是否能訪問特定competition
!kaggle competitions list | grep "chest-xray"

# 4. 列出你的datasets
!kaggle datasets list --mine
```

---

## 📋 完整的Notebook修正代碼

複製以下代碼到notebook的"下載數據"部分:

```python
import os
import zipfile

# ============================================================
# 配置區 - 請根據你的情況修改
# ============================================================

# 選項 A: Competition
USE_COMPETITION = True
COMPETITION_NAME = "chest-xray-pneumonia"  # 替換成你的competition名稱

# 選項 B: Dataset (如果不是competition，設置 USE_COMPETITION = False)
DATASET_NAME = "paultimothymooney/chest-xray-pneumonia"  # 範例

# ============================================================
# 自動下載與錯誤處理
# ============================================================

print("=" * 60)
print("開始下載Kaggle數據...")
print("=" * 60)

try:
    if USE_COMPETITION:
        print(f"📥 從Competition下載: {COMPETITION_NAME}")
        print("⚠️  確保你已經:")
        print("   1. 上傳了 kaggle.json")
        print("   2. 訪問競賽頁面並點擊 'Join Competition'")
        print("   3. 接受了競賽規則")
        print()

        # 嘗試下載
        result = !kaggle competitions download -c $COMPETITION_NAME 2>&1

        # 檢查是否成功
        if any("403" in line or "Forbidden" in line for line in result):
            print("❌ 403錯誤 - 未授權訪問")
            print()
            print("解決方案:")
            print(f"1. 訪問: https://www.kaggle.com/competitions/{COMPETITION_NAME}")
            print("2. 點擊 'Join Competition' (即使競賽已結束)")
            print("3. 接受規則後重新運行此cell")
            print()
            print("或者，嘗試以下替代方案:")
            print("- 設置 USE_COMPETITION = False 並使用Dataset")
            print("- 使用 A100_Ultra_Optimized.ipynb (Google Drive版本)")
            print("- 直接在Kaggle Notebook運行訓練")
            raise Exception("需要接受競賽規則")

        # 找到zip文件
        zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]

    else:
        print(f"📥 從Dataset下載: {DATASET_NAME}")
        !kaggle datasets download -d $DATASET_NAME

        # 找到zip文件
        zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]

    print(f"\n✓ 下載完成! 找到 {len(zip_files)} 個zip文件")

    # 解壓縮
    print("\n📦 解壓縮中...")
    for zip_file in zip_files:
        print(f"   處理: {zip_file}")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"   ✓ 完成: {zip_file}")

    print("\n✅ 數據準備完成!")
    print("\n📁 當前目錄結構:")
    !ls -lh

except Exception as e:
    print(f"\n❌ 錯誤: {e}")
    print("\n" + "=" * 60)
    print("替代方案:")
    print("=" * 60)
    print()
    print("方案1: 使用Google Drive")
    print("  → 使用 A100_Ultra_Optimized.ipynb notebook")
    print()
    print("方案2: 在Kaggle Notebook運行")
    print("  → 前往 https://www.kaggle.com/code")
    print("  → 創建新notebook並添加數據")
    print()
    print("方案3: 手動上傳數據")
    print("  → 在本地下載數據")
    print("  → 上傳到Colab Files")
    print()
```

---

## 📊 各方案比較

| 方案 | 優點 | 缺點 | 推薦度 |
|------|------|------|--------|
| **Kaggle Notebook** | 數據已在本地、無下載 | 需熟悉Kaggle介面 | ⭐⭐⭐⭐⭐ |
| **Google Drive** | 穩定、可重複使用 | 需手動上傳 | ⭐⭐⭐⭐ |
| **Kaggle API (Competition)** | 自動化 | 需接受規則 | ⭐⭐⭐ |
| **Kaggle API (Dataset)** | 最簡單 | 僅適用於公開dataset | ⭐⭐⭐⭐ |

---

## 🎯 推薦工作流程

### 對於Kaggle Competition:

```
1. 訪問競賽頁面 → Join Competition
2. 在Kaggle Notebook直接訓練 (推薦)
   或
3. 使用Google Drive + A100_Ultra_Optimized.ipynb
```

### 對於Kaggle Dataset:

```
1. 使用 Kaggle API (dataset模式)
   或
2. 手動下載 → 上傳Google Drive
```

---

## 💡 快速決策樹

```
你的數據在哪裡？
├─ Kaggle Competition
│  ├─ 已經Join Competition?
│  │  ├─ 是 → 使用 A100_Ultra_Optimized_Kaggle.ipynb
│  │  └─ 否 → 先Join，或用方案3
│  └─ 不想Join → 使用Kaggle Notebook直接訓練 ⭐
│
├─ Kaggle Dataset (公開)
│  └─ 使用 dataset下載模式
│
└─ 本地/其他來源
   └─ 上傳到Google Drive → 使用 A100_Ultra_Optimized.ipynb
```

---

## ✅ 測試清單

在開始訓練前，確認:

- [ ] Kaggle credentials已上傳且有效
- [ ] 數據已成功下載或可訪問
- [ ] 數據結構正確 (train/val/test folders)
- [ ] GPU已設為A100
- [ ] 有足夠的Colab/Kaggle使用時間

---

## 🆘 還是不行？

如果以上方案都無法解決，請提供:

1. 完整的錯誤訊息
2. 你的數據來源 (competition名稱或dataset名稱)
3. 是否已經Join competition
4. 使用的notebook版本

---

**最簡單的方案**: 直接在Kaggle Notebook運行，數據已經就緒！🚀
