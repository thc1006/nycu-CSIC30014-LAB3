# 🎯 達到 90% 完整策略

當前分數: **82.322%**
目標分數: **90%+**

## 📋 執行步驟（按優先順序）

---

### **🥇 Phase 1: ViT 單模型 (預計 87-88%)**

**最快、最簡單的方式！**

#### 在 Colab 執行：

1. **重新載入最新 notebook**
   - 去 https://colab.research.google.com/
   - GitHub → `thc1006/nycu-CSIC30014-LAB3`
   - 打開 `notebooks/Colab_A100_Final.ipynb`

2. **Cell 6 後面添加**（安裝 timm）:
```python
# 安裝 Vision Transformer 支援
!pip install -q timm
```

3. **Cell 14 修改**（使用 ViT 配置）:
```python
# 改為使用 ViT 配置
!python -m src.train_v2 --config configs/colab_vit_90.yaml
```

4. **Cell 20 修改**（TTA 預測）:
```python
# 使用 ViT checkpoint
!python -m src.tta_predict --config configs/colab_vit_90.yaml --ckpt outputs/colab_vit_90/best.pt
```

5. **下載並提交** `submission_tta.csv`

**訓練時間**: 35-40 分鐘 (A100) / 90 分鐘 (T4)
**預期分數**: **87-88%** ✅

#### ViT 配置亮點：
- ✅ Vision Transformer (全局注意力機制)
- ✅ 256px 解析度（捕捉更多細節）
- ✅ Improved Focal Loss (gamma=3.0，專為 COVID-19 不平衡設計)
- ✅ 類別權重 [1.0, 0.57, 1.05, 27.2]
- ✅ Mixup 數據增強
- ✅ 醫學影像專用增強（AutoContrast, Sharpness）
- ✅ 25 epochs（更充分訓練）

---

### **🥈 Phase 2: Ensemble 2-Model (預計 90-91%)**

如果 Phase 1 達到 87%+，用 ensemble 推到 90%！

#### 訓練兩個模型：

**Model 1: ResNet18 (你已有的)**
```python
# 在 Colab Cell 14
!python -m src.train_v2 --config configs/colab_baseline.yaml
!python -m src.tta_predict --config configs/colab_baseline.yaml --ckpt outputs/colab_baseline/best.pt

# 重命名輸出
import shutil
shutil.copy('submission_tta.csv', 'submission_resnet18.csv')
```

**Model 2: ViT (Phase 1 的模型)**
```python
# 已經訓練好，重命名輸出
shutil.copy('submission_tta.csv', 'submission_vit.csv')
```

#### Ensemble 組合：

```python
import pandas as pd
import numpy as np

# 載入兩個預測
pred_resnet = pd.read_csv('submission_resnet18.csv')  # 82.3%
pred_vit = pd.read_csv('submission_vit.csv')          # ~87%

# 加權平均 (ViT 權重較高)
prob_cols = ['normal', 'bacteria', 'virus', 'COVID-19']
weights = [0.30, 0.70]  # ResNet:ViT = 30:70

ensemble = pred_resnet.copy()
ensemble[prob_cols] = (
    weights[0] * pred_resnet[prob_cols].values +
    weights[1] * pred_vit[prob_cols].values
)

# 轉換為 one-hot
predictions = ensemble[prob_cols].values.argmax(axis=1)
one_hot = np.eye(4)[predictions]
ensemble[prob_cols] = one_hot

# 儲存
ensemble.to_csv('submission_ensemble_2.csv', index=False)

# 下載
from google.colab import files
files.download('submission_ensemble_2.csv')
```

**總訓練時間**: ~55 分鐘 (A100)
**預期分數**: **90-91%** ✅✅

---

### **🥉 Phase 3: Ensemble 3-Model (預計 91-92%)**

如果還要更高，訓練第三個模型！

#### Model 3 選項：

**選項 A: ResNet50 (更深的 CNN)**
```python
# 創建配置 configs/colab_resnet50.yaml
# 複製 colab_baseline.yaml，改為:
model:
  name: resnet50
  img_size: 256  # 提高解析度

train:
  batch_size: 16  # ResNet50 需要更小 batch
  epochs: 20
```

**選項 B: EfficientNet-B3**
```python
# 創建配置 configs/colab_effnet.yaml
model:
  name: efficientnet_b3  # 需要修改 train_v2.py 添加支援
  img_size: 300  # EfficientNet 適合更大解析度
```

#### 3-Model Ensemble:

```python
# 載入三個預測
pred1 = pd.read_csv('submission_resnet18.csv')  # 82%
pred2 = pd.read_csv('submission_vit.csv')       # 87%
pred3 = pd.read_csv('submission_resnet50.csv')  # ~85%

# 加權平均
weights = [0.20, 0.50, 0.30]  # ViT 最高權重

ensemble = pred1.copy()
ensemble[prob_cols] = (
    weights[0] * pred1[prob_cols].values +
    weights[1] * pred2[prob_cols].values +
    weights[2] * pred3[prob_cols].values
)

# 轉換為 one-hot
predictions = ensemble[prob_cols].values.argmax(axis=1)
one_hot = np.eye(4)[predictions]
ensemble[prob_cols] = one_hot

ensemble.to_csv('submission_ensemble_3.csv', index=False)
```

**總訓練時間**: ~90 分鐘 (A100)
**預期分數**: **91-92%** ✅✅✅

---

## 📊 預期成績對比

| 方法 | 模型 | 訓練時間 | 預期分數 | 達成難度 |
|------|------|---------|---------|---------|
| **當前** | ResNet18 | 20 min | 82.3% | ✅ 已完成 |
| **Phase 1** | ViT | 40 min | 87-88% | ⭐ 推薦 |
| **Phase 2** | Ensemble (2) | 55 min | 90-91% | ⭐⭐ 達標 |
| **Phase 3** | Ensemble (3) | 90 min | 91-92% | ⭐⭐⭐ 超越 |

---

## 🔑 關鍵成功因素

### 1. **COVID-19 類別是關鍵**
- 只有 37/3780 樣本 (0.98%)
- Focal Loss 的 alpha=[1.0, 0.57, 1.05, **27.2**] 給予最高權重
- 如果 COVID-19 召回率高，整體 F1 會顯著提升

### 2. **多樣性比單個高分更重要**
- Ensemble 要求模型**有差異**
- ResNet (局部特徵) + ViT (全局特徵) = 互補
- 不要 ensemble 3 個相似的模型

### 3. **驗證分數要準確**
- 確保驗證集有 COVID-19 樣本
- 按驗證 F1 加權 ensemble

---

## ⚠️ 常見問題

### Q1: T4 GPU OOM 怎麼辦？
**A**: 降低 batch size
```yaml
# colab_vit_90.yaml
train:
  batch_size: 8  # T4 用 8，A100 用 16
```

### Q2: 訓練太慢？
**A**:
- 優先用 Phase 1（單模型 ViT）
- 如果不夠 90%，再做 Phase 2 ensemble

### Q3: 如何確認 ViT 訓練成功？
**A**: 看訓練輸出
```
[loss] ImprovedFocalLoss (gamma=3.0, alpha=[1.0, 0.57, 1.05, 27.2], smoothing=0.1)
[augment] Mixup/CutMix enabled (alpha=1.0, prob=0.8)

[epoch 01/25] train acc=0.XXX f1=0.XXX | val acc=0.XXX f1=0.XXX
...
[epoch 25/25] train acc=0.XXX f1=0.XXX | val acc=0.XXX f1=0.8XX
```

期待最終 val f1 > 0.80

### Q4: Ensemble 權重怎麼調？
**A**: 根據驗證 F1 分數
```python
# 假設驗證分數
val_f1 = {
    'resnet18': 0.80,
    'vit': 0.87,
    'resnet50': 0.84
}

# 計算權重 (Softmax)
import numpy as np
scores = np.array([0.80, 0.87, 0.84])
weights = np.exp(scores * 10) / np.sum(np.exp(scores * 10))
print(weights)  # 例如: [0.18, 0.56, 0.26]
```

---

## 🚀 快速開始（立即執行）

**最簡單的 90% 路徑**：

```bash
# 1. Colab 安裝 timm
!pip install -q timm

# 2. 訓練 ViT (35 min)
!python -m src.train_v2 --config configs/colab_vit_90.yaml

# 3. TTA 預測
!python -m src.tta_predict --config configs/colab_vit_90.yaml --ckpt outputs/colab_vit_90/best.pt

# 4. 如果 < 90%，ensemble
# 訓練 ResNet18
!python -m src.train_v2 --config configs/colab_baseline.yaml
!python -m src.tta_predict --config configs/colab_baseline.yaml --ckpt outputs/colab_baseline/best.pt

# 5. Ensemble（見上面代碼）
```

---

## 📁 相關文件

- ✅ `configs/colab_vit_90.yaml` - ViT 配置
- ✅ `configs/colab_baseline.yaml` - ResNet18 配置
- ✅ `src/train_v2.py` - 支援 ViT/Focal Loss/Mixup
- ✅ `src/aug.py` - 醫學影像增強
- ✅ `UPGRADE_TO_90_PERCENT.md` - 詳細技術說明

---

## 💡 Pro Tips

1. **監控每類別指標**：
```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred,
    target_names=['Normal', 'Bacteria', 'Virus', 'COVID-19']))
```

2. **保存多個 checkpoint**：
訓練時每 5 個 epoch 保存一次，選最好的

3. **TTA 很重要**：
ViT + TTA 通常能提升 1-2%

4. **驗證集要有代表性**：
確保 COVID-19 類別在驗證集中有樣本

---

## 🎯 最終建議

**如果時間有限**：
→ 只做 **Phase 1 (ViT)** → 預期 87-88%

**如果要確保 90%**：
→ 做 **Phase 1 + Phase 2 (2-model ensemble)** → 預期 90-91%

**如果要衝最高**：
→ 做 **Phase 1 + Phase 2 + Phase 3 (3-model ensemble)** → 預期 91-92%

---

**好運！🍀 相信 ViT 會帶你突破 90%！**

Commit: `b7e08d1`
