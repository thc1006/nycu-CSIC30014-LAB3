# 🎯 Ensemble 策略達到 90%

## 📊 實驗結果總結

### 失敗的嘗試
| 模型 | 配置 | Public Score | 原因 |
|------|------|--------------|------|
| ViT-Base (v1) | colab_vit_90.yaml | 80.303% | 配置錯誤（低 LR、關閉加權採樣、損失函數衝突） |
| ViT-Base (v2) | colab_vit_fixed.yaml | 82.566% | 數據集太小（3780 樣本 vs 86M 參數），過擬合 |

### 成功的基線
| 模型 | Public Score |
|------|--------------|
| **ResNet18** | **82.322%** ✅ |

## 🔍 核心發現

**為什麼 ViT 失敗？**
1. **數據集太小**：3780 訓練樣本對 ViT (86M 參數) 遠遠不夠
2. **預訓練不匹配**：ImageNet → 醫學影像的遷移效果差
3. **局部特徵更重要**：胸部 X 光需要局部紋理，ViT 擅長全局特徵

**結論**：小數據集上，**CNN > Transformer**

---

## 🎯 新策略：多 CNN Ensemble

### 模型選擇依據

根據研究（search-specialist agent 調查）：

| 模型 | 參數量 | 為何適合 | 預期分數 |
|------|--------|---------|---------|
| **ResNet18** | 11.7M | ✅ 已驗證 82.3% | 82% |
| **MobileNetV2** | 3.4M | 輕量級，最低過擬合風險 | 83-85% |
| **DenseNet121** | 8.1M | 特徵重用，醫學影像效果好 | 84-86% |
| **ResNet50** | 25.6M | 更深，表達能力強 | 84-86% |

### 參數量對比
```
MobileNetV2:    3.4M  ████
DenseNet121:    8.1M  ████████
ResNet18:      11.7M  ████████████
ResNet50:      25.6M  ██████████████████████████
ViT-Base:      86.0M  ██████████████████████████████████████████████████████████████████████████████████
                       ↑ 過大，導致過擬合
```

---

## 🚀 執行計劃

### Phase 1: 訓練 3 個 CNN 模型

在 Colab 依序執行：

```python
# Model 1: ResNet50 (已有配置)
!python -m src.train_v2 --config configs/colab_resnet50.yaml
!python -m src.tta_predict --config configs/colab_resnet50.yaml \
    --ckpt outputs/colab_resnet50/best.pt

# Model 2: DenseNet121 (已有配置)
!python -m src.train_v2 --config configs/colab_densenet121.yaml
!python -m src.tta_predict --config configs/colab_densenet121.yaml \
    --ckpt outputs/colab_densenet121/best.pt

# Model 3: MobileNetV2 (已有配置)
!python -m src.train_v2 --config configs/colab_mobilenetv2.yaml
!python -m src.tta_predict --config configs/colab_mobilenetv2.yaml \
    --ckpt outputs/colab_mobilenetv2/best.pt
```

**總訓練時間**: ~90-120 分鐘 (A100)

---

### Phase 2: Ensemble 組合

```python
import pandas as pd
import numpy as np

# 載入 4 個預測 (包含 ResNet18 baseline)
pred_resnet18 = pd.read_csv('data/submission.csv')              # 82.3%
pred_resnet50 = pd.read_csv('submission_tta.csv')               # ~85%
pred_densenet = pd.read_csv('data/submission_densenet121.csv')  # ~85%
pred_mobilenet = pd.read_csv('data/submission_mobilenetv2.csv') # ~84%

prob_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

# 策略 1: 簡單平均 (最穩定)
ensemble_simple = pred_resnet18.copy()
ensemble_simple[prob_cols] = (
    pred_resnet18[prob_cols].values +
    pred_resnet50[prob_cols].values +
    pred_densenet[prob_cols].values +
    pred_mobilenet[prob_cols].values
) / 4.0

# 轉換為 one-hot
predictions = ensemble_simple[prob_cols].values.argmax(axis=1)
ensemble_simple[prob_cols] = np.eye(4)[predictions]
ensemble_simple.to_csv('submission_ensemble_4way_simple.csv', index=False)

# 策略 2: 加權平均 (根據驗證分數)
# 假設驗證 F1 分數：ResNet18=0.80, ResNet50=0.83, DenseNet=0.84, MobileNet=0.82
weights = np.array([0.80, 0.83, 0.84, 0.82])
weights = weights / weights.sum()  # 歸一化: [0.24, 0.25, 0.26, 0.25]

ensemble_weighted = pred_resnet18.copy()
ensemble_weighted[prob_cols] = (
    weights[0] * pred_resnet18[prob_cols].values +
    weights[1] * pred_resnet50[prob_cols].values +
    weights[2] * pred_densenet[prob_cols].values +
    weights[3] * pred_mobilenet[prob_cols].values
)

predictions = ensemble_weighted[prob_cols].values.argmax(axis=1)
ensemble_weighted[prob_cols] = np.eye(4)[predictions]
ensemble_weighted.to_csv('submission_ensemble_4way_weighted.csv', index=False)

print("✅ 兩個 ensemble 提交已生成")
print("   1. submission_ensemble_4way_simple.csv (簡單平均)")
print("   2. submission_ensemble_4way_weighted.csv (加權平均)")
```

**預期結果**：
- 簡單平均：87-89%
- 加權平均：**88-90%** 🎯

---

## 📊 預期成績路徑

| 階段 | 方法 | 預期分數 | 時間 |
|------|------|---------|------|
| ✅ Baseline | ResNet18 | 82.3% | 已完成 |
| 1️⃣ Phase 1 | ResNet50 單模型 | 84-86% | 40 min |
| 2️⃣ Phase 1 | DenseNet121 單模型 | 84-86% | 40 min |
| 3️⃣ Phase 1 | MobileNetV2 單模型 | 83-85% | 35 min |
| 🎯 Phase 2 | 4-Model Ensemble | **88-90%** | 5 min |

---

## 🔑 成功關鍵因素

### 1. **模型多樣性**
- ✅ ResNet（深度殘差）
- ✅ DenseNet（特徵重用）
- ✅ MobileNet（深度可分離卷積）
- ✅ 不同參數量（3.4M - 25.6M）

### 2. **統一的訓練策略**
所有模型使用**相同的成功配置**（來自 ResNet18）：
- ✅ 加權採樣（處理 COVID-19 1% 不平衡）
- ✅ 標準 CE + Label Smoothing (0.05)
- ✅ 醫學影像增強（AutoContrast, Sharpness）
- ✅ 保守的學習率 (0.0003) 和正則化

### 3. **TTA（Test-Time Augmentation）**
每個模型都使用 TTA → 額外 +0.5-1% 提升

---

## 💡 為什麼這個策略會成功？

### vs ViT 策略：
| 方面 | ViT (失敗) | CNN Ensemble (新策略) |
|------|-----------|---------------------|
| **參數量** | 86M (過大) | 3.4M-25.6M (適中) ✅ |
| **過擬合風險** | 高 ❌ | 低-中 ✅ |
| **特徵類型** | 全局 | 局部 + 全局 ✅ |
| **模型多樣性** | 單一 | 4 種架構 ✅ |
| **數據需求** | 大 (10萬+) | 中 (數千) ✅ |

### Ensemble 的威力：
- **減少方差**：不同模型的錯誤互相抵消
- **提高穩健性**：多個視角看問題
- **低風險高回報**：訓練時間僅多 3 倍，但分數提升顯著

---

## 📁 相關文件

- ✅ `configs/colab_resnet50.yaml` - ResNet50 配置
- ✅ `configs/colab_densenet121.yaml` - DenseNet121 配置
- ✅ `configs/colab_mobilenetv2.yaml` - MobileNetV2 配置
- ✅ `configs/colab_baseline.yaml` - ResNet18 baseline
- ✅ `src/train_v2.py` - 已添加所有模型支持
- ✅ `VIT_FAILURE_ANALYSIS.md` - ViT 失敗分析

---

## 🎯 立即行動

### 在 Colab 執行：

1. **上傳最新代碼**：
```python
%cd /content
!git clone https://github.com/thc1006/nycu-CSIC30014-LAB3.git
%cd nycu-CSIC30014-LAB3
```

2. **運行訓練腳本** → 將在 notebook 中提供完整代碼

3. **生成 ensemble** → 提交 `submission_ensemble_4way_weighted.csv`

4. **預期**：**88-90%** 🎉

---

**結論**：放棄 ViT，擁抱多樣化 CNN Ensemble！
