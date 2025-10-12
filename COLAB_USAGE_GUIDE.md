# Google Colab 使用指南 - 達成 80%+ 最佳成績

## 📋 目標

使用 Google Colab A100 GPU 複現並超越 **80.122%** 的基準成績

---

## 🎯 快速開始（3 步驟）

### Step 1: 準備專案檔案

**選項 A: 使用 Google Drive（推薦）**
```bash
# 1. 將整個專案資料夾壓縮成 zip
# 2. 上傳到 Google Drive
# 3. 在 Colab 中掛載 Drive 並解壓縮
```

**選項 B: 使用 GitHub**
```bash
# 1. Push 專案到 GitHub repo
# 2. 在 Colab 中 clone
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

### Step 2: 上傳到 Colab

1. 打開 Google Colab: https://colab.research.google.com/
2. 上傳 notebook: `Colab_Optimized_80plus.ipynb`
3. 設定 Runtime: **Runtime > Change runtime type > A100 GPU**
4. 運行所有 cells: **Runtime > Run all**

### Step 3: 下載結果並提交

1. Notebook 會自動生成 2 個提交檔案：
   - `submission_baseline.csv` - 標準預測
   - `submission_baseline_tta.csv` - **推薦使用（TTA 增強）**
2. 下載並提交到 Kaggle
3. 預期分數：**80-82%**

---

## 📊 Notebook 架構說明

### Phase 1: 基準模型（80.122% 複現）

**配置（與原始 80.122% 一致）：**
- 模型：ResNet18（預訓練 ImageNet）
- 圖像大小：224 × 224
- Batch Size：32（A100）
- Epochs：12
- 學習率：0.0003
- Optimizer：AdamW
- Scheduler：Cosine warmup
- Loss：CrossEntropy + Label Smoothing (0.05)
- Weighted Sampler：✅ 處理類別不平衡
- AMP：bfloat16（A100 最佳化）

**為什麼這個配置有效？**
1. ✅ **簡單但有效** - 避免過度優化
2. ✅ **Weighted Sampler** - 解決 COVID-19 樣本稀少問題
3. ✅ **Label Smoothing** - 防止過擬合
4. ✅ **預訓練模型** - 利用 ImageNet 知識
5. ✅ **Cosine LR** - 穩定的學習率衰減

### Phase 2: Test-Time Augmentation (TTA)

**增強預測準確性的技術：**

TTA 會對每張測試圖片進行 4 種變換：
1. 原始圖片
2. 水平翻轉
3. 旋轉 +5°
4. 旋轉 -5°

然後平均 4 次預測，提升穩定性。

**預期提升：+0.5% 到 +1.5% F1 score**

### Phase 3: 進階優化（選用）

如果基準模型達到 80%+，可以嘗試：

1. **多模型 Ensemble**
   - ResNet34（更深）
   - EfficientNetV2-S（更高效）
   - 軟投票平均多個模型

2. **更長訓練**
   - 20-30 epochs with early stopping
   - Stochastic Weight Averaging (SWA)

3. **進階增強**
   - CLAHE（對比度增強）
   - Mixup/CutMix

---

## 💡 關鍵發現與建議

### ✅ DO（推薦做）

1. **使用 TTA 提交檔案**
   - `submission_baseline_tta.csv` 通常比標準預測高 1%

2. **保持簡單的配置**
   - 複雜的設定（Mixup, Heavy Aug）可能降低分數
   - 基準 ResNet18 已經很強

3. **監控驗證 F1**
   - 目標：Val F1 ≥ 0.80
   - 如果低於 0.78，延長訓練

4. **使用 A100 GPU**
   - 訓練時間：~15-20 分鐘（12 epochs）
   - 比 T4 快 3 倍

### ❌ DON'T（避免做）

1. **不要過度優化**
   - Exp1-5 使用複雜設定反而表現更差
   - 簡單 > 複雜

2. **不要忽略類別不平衡**
   - 一定要用 Weighted Sampler
   - COVID-19 只有 1% 樣本

3. **不要跳過 Label Smoothing**
   - 防止過擬合的關鍵技術

---

## 📈 預期結果

| 方法 | Val F1 | Public Score | 時間 |
|------|--------|--------------|------|
| Baseline (ResNet18) | 0.80-0.82 | 80-81% | 15 min |
| Baseline + TTA | 0.81-0.83 | 81-82% | 20 min |
| Ensemble (3 models) | 0.82-0.84 | 82-84% | 45 min |
| Ensemble + TTA | 0.83-0.85 | 83-85% | 60 min |

---

## 🔧 故障排除

### 問題 1: Out of Memory (OOM)

**解決方案：**
```python
# 降低 batch size
BASELINE_CONFIG['batch_size'] = 16  # 從 32 降到 16
```

### 問題 2: 訓練太慢

**解決方案：**
```python
# 確認使用 A100（不是 T4）
# Runtime > Change runtime type > A100 GPU

# 確認 TF32 啟用
torch.backends.cuda.matmul.allow_tf32 = True
```

### 問題 3: Val F1 低於 0.78

**解決方案：**
```python
# 1. 增加訓練 epochs
BASELINE_CONFIG['epochs'] = 15

# 2. 降低學習率
BASELINE_CONFIG['lr'] = 0.0002

# 3. 確認 Weighted Sampler 啟用
BASELINE_CONFIG['use_weighted_sampler'] = True
```

### 問題 4: 資料上傳問題

**解決方案：**

**方法 A: 小專案（<500MB）- 直接上傳**
```python
from google.colab import files
uploaded = files.upload()
```

**方法 B: 大專案 - 使用 Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r "/content/drive/MyDrive/nycu-CSIC30014-LAB3" /content/
```

**方法 C: 使用 GitHub**
```bash
!git clone https://github.com/YOUR_USERNAME/nycu-CSIC30014-LAB3.git
%cd nycu-CSIC30014-LAB3
```

---

## 📝 檢查清單

在提交前確認：

- [ ] Colab Runtime 設定為 **A100 GPU**
- [ ] 所有 cells 運行無錯誤
- [ ] Val F1 ≥ 0.80
- [ ] 提交檔案格式正確（1182 rows, 5 columns）
- [ ] 使用 TTA 版本（`submission_baseline_tta.csv`）
- [ ] 下載並保存最佳模型（`baseline_best.pt`）

---

## 🚀 進階：追求 85%+

如果你有額外的時間並想追求更高分數：

### 1. 訓練多個模型

```python
# 在 Section 7 取消註解並訓練：
ADDITIONAL_MODELS = [
    {'name': 'resnet34', 'img_size': 256, 'epochs': 12},
    {'name': 'efficientnet_v2_s', 'img_size': 288, 'epochs': 15},
]
```

### 2. Soft Ensemble（軟投票）

```python
# 平均多個模型的機率
ensemble_probs = (baseline_probs + resnet34_probs + efficientnet_probs) / 3
ensemble_preds = ensemble_probs.argmax(axis=1)
```

### 3. Stochastic Weight Averaging (SWA)

```python
# 使用 src/train_v2.py 的 SWA 功能
!python src/train_v2.py --config configs/model_big_swa.yaml
```

### 4. Pseudo-Labeling

```python
# 1. 用最佳模型預測測試集
# 2. 選擇高信心預測（prob > 0.9）
# 3. 加入訓練集重新訓練
```

---

## 📚 參考資料

- **專案原始配置：** `configs/model_small.yaml`
- **訓練腳本：** `src/train.py`（基礎）、`src/train_v2.py`（進階）
- **資料處理：** `src/data.py`
- **損失函數：** `src/losses.py`

---

## 💬 常見問題（FAQ）

**Q: 為什麼基準模型這麼簡單？**
A: 經過實驗，簡單配置（ResNet18, 224px, 12 epochs）在這個資料集上表現最好。更複雜的模型（ConvNeXt, 進階增強）反而降低了分數。

**Q: TTA 一定會提升分數嗎？**
A: 通常會，但提升幅度約 0.5-1.5%。如果基準模型已經很好，TTA 是低風險的額外提升。

**Q: 需要調整超參數嗎？**
A: 不建議。當前配置已經過優化，隨意調整可能降低分數。除非 Val F1 < 0.78，否則保持預設值。

**Q: 訓練多久？**
A: 在 A100 上，12 epochs 約 15-20 分鐘。如果使用 T4，約 40-60 分鐘。

**Q: 能在本地 GPU 運行嗎？**
A: 可以，但需要調整 batch size（如 RTX 3050 用 batch=12）。訓練時間會較長。

---

## 📞 支援

如果遇到問題：
1. 檢查 **故障排除** 章節
2. 確認所有資料檔案存在
3. 檢查 Colab GPU 配置
4. 查看 notebook 中的錯誤訊息

**祝你取得好成績！🎉**

---

**最後更新：** 2025-10-12
**版本：** v1.0
**相容性：** Google Colab, PyTorch 2.0+, A100/T4 GPU
