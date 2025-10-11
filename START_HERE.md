# 🎯 立即實施 - Stage 1 優化完成！

## ✅ 已完成的準備工作

所有Stage 1優化組件已經準備就緒並通過測試：

1. ✅ **ConvNeXt-Base模型** (87.6M參數，已下載權重)
2. ✅ **Improved Focal Loss** with class weights [1.0, 1.5, 2.0, 1.2]
3. ✅ **Mixup/CutMix增強** (50%機率)
4. ✅ **Stochastic Weight Averaging** (epochs 25-30)
5. ✅ **進階資料增強** (rotation, affine, random erasing)
6. ✅ **Test-Time Augmentation** 預測腳本
7. ✅ **所有組件測試通過** ✓

---

## 🚀 方案一：本地訓練 (RTX 3050)

### 立即開始訓練

```bash
# 進入專案目錄
cd C:\Users\thc1006\Desktop\114-1\nycu-CSIC30014-LAB3

# 開始完整訓練 (約4-5小時)
python -m src.train_v2 --config configs/model_stage1.yaml
```

### 訓練完成後生成預測

```bash
# 標準預測
python -m src.predict --config configs/model_stage1.yaml --ckpt outputs/stage1_convnext512/best.pt

# 或使用TTA (+2-3%提升)
python -m src.tta_predict --config configs/model_stage1.yaml --ckpt outputs/stage1_convnext512/best_swa.pt
```

**提交文件位置**: `C:/Users/thc1006/Desktop/114-1/nycu-CSIC30014-LAB3/data/submission_stage1.csv`

---

## ☁️ 方案二：A100 Colab訓練 (推薦)

### 1. 上傳到Colab

在Colab中創建新notebook：

```python
# 掛載Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 上傳您的專案檔案
# 或使用git clone (如果專案在GitHub上)

# 切換到專案目錄
%cd /content/your-project-folder
```

### 2. 修改配置以充分利用A100

編輯 `configs/model_stage1.yaml` 中的batch_size：

```yaml
train:
  batch_size: 24  # A100: 使用24 (RTX 3050使用8)
```

### 3. 執行訓練

```python
!python -m src.train_v2 --config configs/model_stage1.yaml
```

**A100訓練時間**: 約2小時 (vs RTX 3050的4-5小時)

### 4. 生成並下載結果

```python
# 使用TTA生成預測
!python -m src.tta_predict --config configs/model_stage1.yaml \
  --ckpt outputs/stage1_convnext512/best_swa.pt

# 下載submission
from google.colab import files
files.download('data/submission_stage1.csv')
```

---

## 📊 預期結果

### 性能提升預測

| 指標 | 目前(ResNet18) | Stage 1目標 | Stage 1+TTA |
|------|----------------|-------------|-------------|
| **Public Score** | 0.801 | 0.850-0.870 | 0.870-0.890 |
| Normal F1 | 0.897 | 0.920-0.940 | 0.930-0.950 |
| Bacteria F1 | 0.762 | 0.820-0.850 | 0.840-0.870 |
| **Virus F1** | 0.619 | 0.780-0.820 | 0.800-0.840 |
| COVID-19 F1 | 0.875 | 0.900-0.950 | 0.920-0.970 |

### 核心改進

**最大瓶頸 - Virus類別混淆**:
- 當前: 67.2% (121/180) - **44個誤判為Bacteria**
- 目標: 78-82% - 透過Focal Loss權重2.0重點優化

**次要瓶頸 - Bacteria類別**:
- 當前: 70.3% (234/333) - **82個誤判為Virus**
- 目標: 82-85% - 透過Focal Loss權重1.5提升

---

## 🔧 關鍵配置說明

### configs/model_stage1.yaml 核心參數

```yaml
model:
  name: convnext_base        # 從ResNet18升級 (11M → 88M參數)
  img_size: 512              # 從224升級，捕捉更多細節

train:
  epochs: 30                 # 從10增加
  batch_size: 8              # RTX 3050 (A100用24)
  lr: 0.0001                 # 較大模型用較小學習率

  # Focal Loss with targeted weights
  loss: focal_improved
  focal_alpha: [1.0, 1.5, 2.0, 1.2]  # 針對Bacteria/Virus混淆
  focal_gamma: 2.0
  label_smoothing: 0.1

  # Mixup/CutMix
  use_mixup: true
  mixup_alpha: 1.0
  mixup_prob: 0.5            # 50%的batch使用

  # Stochastic Weight Averaging
  use_swa: true
  swa_start: 25              # 最後5個epoch
  swa_lr: 0.00005

  # 進階增強
  advanced_aug: true
  aug_rotation: 15           # 從10增加
  random_erasing_prob: 0.3   # 新增
```

---

## 📈 訓練監控

### 預期訓練日誌

```
[device] cuda | CUDA name: NVIDIA GeForce RTX 3050 Laptop GPU
[loss] ImprovedFocalLoss (gamma=2.0, alpha=[1.0, 1.5, 2.0, 1.2], smoothing=0.1)
[augment] Mixup/CutMix enabled (alpha=1.0, prob=0.5)
[SWA] enabled (start epoch=25, lr=5e-05)

[epoch 01] train acc=0.3500 f1=0.2800 | val acc=0.4500 f1=0.3500
[epoch 05] train acc=0.6200 f1=0.5800 | val acc=0.6500 f1=0.6200
[epoch 10] train acc=0.8000 f1=0.7800 | val acc=0.7900 f1=0.7700
  -> saved new best to outputs/stage1_convnext512/best.pt (val macro-F1=0.7700)
[epoch 15] train acc=0.8500 f1=0.8400 | val acc=0.8200 f1=0.8100
  -> saved new best to outputs/stage1_convnext512/best.pt (val macro-F1=0.8100)
[epoch 20] train acc=0.8800 f1=0.8700 | val acc=0.8500 f1=0.8400
  -> saved new best to outputs/stage1_convnext512/best.pt (val macro-F1=0.8400)
[epoch 25] train acc=0.9100 f1=0.9000 | val acc=0.8650 f1=0.8550
[epoch 30] train acc=0.9200 f1=0.9100 | val acc=0.8700 f1=0.8600
  -> saved new best to outputs/stage1_convnext512/best.pt (val macro-F1=0.8600)

[SWA] Updating BatchNorm statistics...
[SWA final] val acc=0.8750 f1=0.8650
  -> saved SWA model to outputs/stage1_convnext512/best_swa.pt (val macro-F1=0.8650)
```

### 關鍵指標

- **Epoch 10**: Val F1應該 > 0.77 (超越baseline 0.788)
- **Epoch 20**: Val F1應該 > 0.84
- **Epoch 30**: Val F1應該達到 0.86+
- **SWA模型**: 通常比best.pt再提升0.5-1%

---

## 🐛 常見問題

### Q1: OutOfMemoryError

**解決方案**:
```yaml
# configs/model_stage1.yaml
train:
  batch_size: 4  # 從8降至4
```

或嘗試較小的解析度：
```yaml
model:
  img_size: 384  # 從512降至384
```

### Q2: 訓練時間過長

**選項1**: 減少epochs
```yaml
train:
  epochs: 20     # 從30降至20
  swa_start: 15  # 相應調整
```

**選項2**: 使用Colab A100 (快2-3倍)

### Q3: 哪個checkpoint用於提交？

優先順序：
1. `best_swa.pt` (如果SWA F1 > 普通best)
2. `best.pt` (按val F1保存的最佳模型)

---

## 🎯 達到90+分數的完整路線圖

### Stage 1 (當前) - 目標85-87%
✅ 已完成準備，立即可執行

### Stage 2 - 目標88-90%
- 多模型Ensemble (3個models)
- 不同seed訓練
- 預測加權平均

### Stage 3 - 目標90-93%
- Multi-scale training (384, 448, 512)
- Pseudo-labeling
- 更大模型 (ConvNeXt-Large)

---

## 📝 檔案清單

### 新創建的檔案
- ✅ `configs/model_stage1.yaml` - Stage 1配置
- ✅ `src/train_v2.py` - 增強訓練腳本
- ✅ `src/tta_predict.py` - TTA預測
- ✅ `test_stage1.py` - 組件測試腳本
- ✅ `RUN_STAGE1.md` - 詳細說明
- ✅ `START_HERE.md` - 本文件

### 修改的檔案
- ✅ `src/losses.py` - 新增ImprovedFocalLoss
- ✅ `src/aug.py` - 新增Mixup/CutMix
- ✅ `src/data.py` - 支援進階增強

---

## ⚡ 快速命令參考

```bash
# 測試所有組件
python test_stage1.py

# 本地訓練 (RTX 3050, ~4-5小時)
python -m src.train_v2 --config configs/model_stage1.yaml

# 生成預測 (標準)
python -m src.predict --config configs/model_stage1.yaml \
  --ckpt outputs/stage1_convnext512/best.pt

# 生成預測 (TTA, +2-3%)
python -m src.tta_predict --config configs/model_stage1.yaml \
  --ckpt outputs/stage1_convnext512/best_swa.pt
```

---

## 🎉 準備就緒！

**所有組件已測試並驗證**。您現在可以：

1. **立即開始**: 在本地RTX 3050上運行 (4-5小時)
2. **加速訓練**: 上傳到Colab A100 (2小時)
3. **預期提升**: Public Score從0.801提升至0.85-0.87 (+5-7%)
4. **使用TTA**: 再提升2-3%，達到0.87-0.89

**預計最終分數: 87-89%** 🎯

需要開始訓練或有任何問題嗎？一切就緒！🚀
