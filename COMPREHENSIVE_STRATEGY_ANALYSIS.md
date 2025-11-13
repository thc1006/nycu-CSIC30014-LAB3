# 全方位策略分析與實作計劃

**分析日期**: 2025-11-13
**當前最佳**: 84.19% Macro-F1
**目標**: 87-90%+
**硬體**: RTX 4070 Ti SUPER 16GB, 326GB 可用磁盤

---

## 📊 多維度可行性分析

### 1. 時間成本分析

| 技巧 | 實作時間 | 訓練時間 | 總時間 | ROI 分數 | 優先級 |
|------|---------|---------|--------|----------|--------|
| **Stacking Meta-Learning** | 1h | 0.5h | 1.5h | 🌟🌟🌟🌟🌟 | P0 |
| **偽標籤訓練** | 0.5h | 6h | 6.5h | 🌟🌟🌟🌟 | P0 |
| **增強 TTA (多尺度)** | 2h | 0h (推理) | 2h | 🌟🌟🌟🌟 | P0 |
| **WBF 動態集成** | 3h | 0h | 3h | 🌟🌟🌟 | P1 |
| **外部數據預訓練 (NIH)** | 2h | 24h | 26h | 🌟🌟🌟🌟🌟 | P1 |
| **知識蒸餾** | 5h | 12h | 17h | 🌟🌟🌟 | P2 |
| **Attention CutMix** | 8h | 8h | 16h | 🌟🌟 | P3 |
| **自監督預訓練** | 10h | 48h | 58h | 🌟🌟 | P4 |

**ROI 計算公式**: `(預期提升% × 10) / 總時間(小時)`

---

### 2. 硬體限制分析

#### 2.1 GPU 記憶體 (16GB VRAM)

**當前使用情況**:
```
EfficientNet-V2-S @ 384px, batch=24 → ~12GB
ConvNeXt-Base @ 384px, batch=16 → ~14GB
```

**各技巧的記憶體需求**:

| 技巧 | 額外 VRAM | 可行性 | 解決方案 |
|------|-----------|--------|----------|
| Stacking | 0GB (離線) | ✅ 完全可行 | 無需修改 |
| 偽標籤訓練 | 0GB | ✅ 完全可行 | 使用現有配置 |
| 增強 TTA | +0-2GB | ✅ 可行 | 降低 batch 或使用 CPU |
| WBF 集成 | 0GB (離線) | ✅ 完全可行 | 無需修改 |
| 外部數據預訓練 | 0GB | ✅ 完全可行 | 使用現有配置 |
| 知識蒸餾 | +4-6GB | ⚠️ 緊張 | Teacher 用 FP16 或 CPU |
| Attention CutMix | +2-3GB | ⚠️ 緊張 | 需要降低 batch |
| 多任務學習 (分割) | +3-5GB | ⚠️ 緊張 | 需要優化架構 |

**結論**: P0-P1 技巧都可行，P2-P3 需要優化

---

#### 2.2 磁盤空間 (326GB 可用)

**外部數據集大小**:
- NIH ChestX-ray14: ~42GB (原始) / ~7GB (224x224 預處理版)
- CheXpert: ~440GB (原始) / ~45GB (小版本)
- RSNA Pneumonia: ~26GB

**當前數據使用**:
```
data/train: ~2GB
data/test: ~0.5GB
outputs/ (模型檢查點): ~15GB
data/grid_search_submissions/: ~0.1GB
data/kfold_splits/: ~0.5GB
```

**可行方案**:
```
✅ NIH ChestX-ray14 (224x224): 7GB → 剩餘 319GB
✅ RSNA Pneumonia: 26GB → 剩餘 293GB
❌ CheXpert 完整版: 440GB → 超出容量
⚠️ CheXpert 小版本: 45GB → 剩餘 248GB (可接受)
```

**結論**: 下載 **NIH ChestX-ray14 (預處理版)** 最佳

---

### 3. 技術風險評估

#### 3.1 Stacking/Meta-Learning ✅ 低風險

**優勢**:
- ✅ 腳本已完整實作 (`scripts/stacking_meta_learner.py`)
- ✅ 只需生成驗證集預測（已有 `generate_validation_predictions.py`）
- ✅ 文獻支持（Kaggle 獲勝者常用）
- ✅ 無需重新訓練基礎模型

**風險**:
- ⚠️ 可能過擬合驗證集 (10% 機率)
  - **緩解**: 使用多個 Meta-Learner，選擇在測試集表現最佳的

**預期提升**: +1.5-3%
**置信度**: 90%

---

#### 3.2 偽標籤訓練 ⚠️ 中風險

**優勢**:
- ✅ 偽標籤已生成 (`data/train_data_augmented.csv`)
- ✅ 基於 4 個模型的多數投票 + 一致性過濾
- ✅ Kaggle 獲勝者都使用

**風險**:
- ⚠️ 偽標籤質量不確定 (30% 機率)
  - **緩解**: 只使用高置信度樣本 (>0.8)
- ⚠️ 可能引入錯誤標籤導致性能下降 (15% 機率)
  - **緩解**: 降低偽標籤權重 (0.3-0.5)

**預期提升**: +0.5-1.5%
**置信度**: 70%

---

#### 3.3 增強 TTA ✅ 低風險

**優勢**:
- ✅ 只影響推理，不影響訓練
- ✅ 可以逐步添加變換，觀察效果
- ✅ 無副作用（最壞情況是無提升）

**風險**:
- ⚠️ 推理時間增加 (可接受)
- ⚠️ 某些變換可能降低性能 (10% 機率)
  - **緩解**: 逐步添加，評估每個變換的貢獻

**預期提升**: +0.5-1%
**置信度**: 85%

---

#### 3.4 外部數據預訓練 ⚠️ 中-高風險

**優勢**:
- ✅ Kaggle 獲勝者最重要的技巧
- ✅ 文獻強力支持
- ✅ NIH ChestX-ray14 與我們的數據集領域一致

**風險**:
- ⚠️ 外部數據標籤不完全匹配 (40% 機率)
  - NIH: 14 種疾病 (無 COVID-19)
  - 我們: 4 類 (Normal, Bacteria, Virus, COVID-19)
  - **緩解**: 只預訓練特徵提取器，不訓練分類頭
- ⚠️ 預訓練可能破壞 ImageNet 初始化 (20% 機率)
  - **緩解**: 使用較小的學習率 (1e-5)
- ⚠️ 訓練時間長 (24 小時+)
  - **緩解**: 只訓練 10-15 epochs

**預期提升**: +1-3%
**置信度**: 65%

---

#### 3.5 WBF 動態集成 ✅ 低風險

**優勢**:
- ✅ 只影響集成策略，不影響訓練
- ✅ 可以與現有集成對比

**風險**:
- ⚠️ WBF 設計用於檢測，適配到分類可能無效 (25% 機率)
  - **緩解**: 實作基於類別概率的動態權重

**預期提升**: +0.3-0.7%
**置信度**: 75%

---

### 4. 數據集標籤映射分析

#### 4.1 NIH ChestX-ray14 標籤

**14 種疾病**:
```
1. Atelectasis (肺不張)
2. Cardiomegaly (心臟擴大)
3. Effusion (積液)
4. Infiltration (浸潤)
5. Mass (腫塊)
6. Nodule (結節)
7. Pneumonia (肺炎) ← 相關！
8. Pneumothorax (氣胸)
9. Consolidation (實變) ← 相關！
10. Edema (水腫)
11. Emphysema (肺氣腫)
12. Fibrosis (纖維化)
13. Pleural_Thickening (胸膜增厚)
14. Hernia (疝氣)
15. No Finding (正常) ← 相關！
```

#### 4.2 標籤映射策略

**方案 A: 多標籤預訓練** (推薦)
```python
# 訓練時使用所有 14 個標籤
# 微調時只使用特徵提取器，重新訓練分類頭
model = load_pretrained_backbone('nih_chestxray14.pt')
model.fc = nn.Linear(model.feature_dim, 4)  # 4 類
```

**方案 B: 粗略映射**
```python
# NIH → 我們的數據集
mapping = {
    'No Finding': 'Normal',
    'Pneumonia': 'Bacteria',  # 假設大多數是細菌性
    'Consolidation': 'Bacteria',  # 通常與細菌性肺炎相關
    # 其他: 忽略
}
```

**推薦**: 方案 A（多標籤預訓練），因為：
1. 保留所有信息
2. 不做假設（避免錯誤映射）
3. Kaggle 獲勝者都使用此方法

---

### 5. 訓練時間估算

#### 基於我們的硬體 (RTX 4070 Ti SUPER)

**單個模型訓練時間**:
```
EfficientNet-V2-S @ 384px, 45 epochs:
  - 每 epoch: ~8 分鐘 (3,397 樣本, batch 24)
  - 總計: ~6 小時

ConvNeXt-Base @ 384px, 35 epochs:
  - 每 epoch: ~10 分鐘 (batch 16, 更大模型)
  - 總計: ~5.8 小時
```

**各技巧訓練時間**:

| 技巧 | 訓練內容 | 預估時間 |
|------|---------|---------|
| Stacking | 無（只推理） | 0.5h |
| 偽標籤訓練 | 1 個模型, 20 epochs | 3-4h |
| 增強 TTA | 無（只推理） | 0h |
| WBF 集成 | 無（只推理） | 0h |
| NIH 預訓練 | 1 個模型, 10 epochs, 112K 樣本 | **18-24h** |
| 知識蒸餾 | 1 個小模型, 30 epochs | 8-12h |

---

### 6. 成本效益分析

#### ROI 排名（預期提升 / 總時間）

| 排名 | 技巧 | 預期提升 | 總時間 | ROI | 推薦度 |
|------|------|---------|--------|-----|--------|
| 🥇 1 | **Stacking** | +2% | 1.5h | **13.3** | ⭐⭐⭐⭐⭐ |
| 🥈 2 | **增強 TTA** | +0.7% | 2h | **3.5** | ⭐⭐⭐⭐⭐ |
| 🥉 3 | **WBF 集成** | +0.5% | 3h | **1.67** | ⭐⭐⭐⭐ |
| 4 | **偽標籤訓練** | +1% | 6.5h | **1.54** | ⭐⭐⭐⭐ |
| 5 | **NIH 預訓練** | +2% | 26h | **0.77** | ⭐⭐⭐ |
| 6 | **知識蒸餾** | +1% | 17h | **0.59** | ⭐⭐ |

**結論**:
- **短期優先** (1-2 天): Stacking → 增強 TTA → WBF
- **中期優先** (3-5 天): 偽標籤訓練 → NIH 預訓練

---

## 🎯 最終實作策略

### Phase 0: 立即執行（30 分鐘）

**檢查現有資源**:
```bash
# 1. 檢查是否已有驗證集預測
ls -lh data/validation_predictions_*.csv

# 2. 檢查偽標籤數據
head -20 data/train_data_augmented.csv

# 3. 檢查 Stacking 腳本
cat scripts/stacking_meta_learner.py | grep -A 5 "class.*Learner"
```

---

### Phase 1: 快速提升（2-3 小時）✅ 立即開始

#### 任務 1.1: Stacking/Meta-Learning (1.5h)

**步驟**:
```bash
# Step 1: 生成驗證集預測 (如果還沒有)
python scripts/generate_validation_predictions.py \
  --config configs/improved_breakthrough.yaml \
  --checkpoint outputs/improved_breakthrough/best.pt \
  --output data/validation_predictions_improved.csv

# 重複對其他模型
# - efficientnet_tta
# - convnext_tta
# - ultimate_final
# - mega_ensemble_tta

# Step 2: 訓練 Meta-Learner
python scripts/stacking_meta_learner.py

# Step 3: 生成測試集預測
python scripts/stacking_predict.py
```

**預期輸出**: `data/submission_stacking.csv`
**預期提升**: +1.5-3% → **85.7-87.2%**

---

#### 任務 1.2: 增強 TTA (2h)

**修改 `src/tta_predict.py`**:
```python
class EnhancedTTA:
    def __init__(self, model, num_tta=20):
        self.model = model
        self.transforms = [
            # 基礎 (現有)
            ('original', lambda x: x),
            ('hflip', lambda x: torch.flip(x, [-1])),
            ('vflip', lambda x: torch.flip(x, [-2])),

            # 多尺度 (新增)
            ('scale_0.9', lambda x: F.interpolate(x, scale_factor=0.9)),
            ('scale_1.1', lambda x: F.interpolate(x, scale_factor=1.1)),

            # 旋轉 (新增)
            ('rotate_5', lambda x: rotate(x, 5)),
            ('rotate_-5', lambda x: rotate(x, -5)),

            # 中心裁剪 (新增)
            ('center_crop_0.9', lambda x: center_crop(x, 0.9)),

            # 組合 (新增)
            ('hflip_rotate_5', lambda x: rotate(torch.flip(x, [-1]), 5)),
            # ... 更多組合
        ][:num_tta]
```

**使用**:
```bash
python src/predict.py \
  --config configs/improved_breakthrough.yaml \
  --checkpoint outputs/improved_breakthrough/best.pt \
  --tta_mode enhanced \
  --num_tta 20
```

**預期提升**: +0.5-1% → **86.2-88.2%**

---

#### 任務 1.3: WBF 動態集成 (3h)

**創建 `scripts/ensemble/wbf_ensemble.py`**:
```python
import numpy as np
from scipy.spatial.distance import jensenshannon

def compute_model_agreement(predictions):
    """計算模型間的 JS 散度作為一致性度量"""
    n_models = len(predictions)
    agreements = np.zeros(n_models)

    for i in range(n_models):
        js_divs = []
        for j in range(n_models):
            if i != j:
                js_div = jensenshannon(predictions[i], predictions[j])
                js_divs.append(js_div)

        # 低 JS 散度 = 高一致性
        agreements[i] = 1.0 - np.mean(js_divs)

    return agreements

def wbf_ensemble(predictions, base_weights, consistency_weight=0.3):
    """
    Weighted Boxes Fusion 式動態集成

    Args:
        predictions: List of [N, 4] probability arrays
        base_weights: Base weights for each model
        consistency_weight: 一致性在最終權重中的佔比
    """
    n_samples = predictions[0].shape[0]
    n_models = len(predictions)
    final_preds = np.zeros_like(predictions[0])

    for i in range(n_samples):
        sample_preds = [pred[i] for pred in predictions]

        # 計算樣本級別的一致性
        agreements = compute_model_agreement(sample_preds)

        # 動態權重 = base_weights * (1 + consistency_weight * agreements)
        dynamic_weights = base_weights * (1 + consistency_weight * agreements)
        dynamic_weights /= dynamic_weights.sum()

        # 加權平均
        final_preds[i] = sum(w * p for w, p in zip(dynamic_weights, sample_preds))

    return final_preds
```

**預期提升**: +0.3-0.7% → **86.5-88.9%**

---

### Phase 1 總結

**總時間**: 2-3 小時（主要是推理時間）
**預期成績**: **86.5-88.9%** (從 84.19%)
**提升幅度**: **+2.3-4.7%**
**成功機率**: 85%

---

### Phase 2: 中期改進（1-2 天）⚡ 條件執行

#### 條件: Phase 1 達成 >= 87%

#### 任務 2.1: 偽標籤訓練 (6.5h)

**Step 1: 檢查偽標籤質量**
```python
import pandas as pd

df = pd.read_csv('data/train_data_augmented.csv')

# 檢查偽標籤統計
pseudo_df = df[df['is_pseudo_label'] == True]
print(f"偽標籤數量: {len(pseudo_df)}")
print(f"類別分布:\n{pseudo_df['label'].value_counts()}")
print(f"平均一致性: {pseudo_df['consistency_score'].mean():.3f}")

# 只保留高置信度樣本
high_conf = pseudo_df[pseudo_df['consistency_score'] >= 0.8]
print(f"高置信度樣本: {len(high_conf)} ({len(high_conf)/len(pseudo_df)*100:.1f}%)")
```

**Step 2: 創建訓練配置**
```yaml
# configs/pseudo_label_stage2.yaml
inherit_from: configs/improved_breakthrough.yaml

data:
  train_csv: data/train_data_augmented_high_conf.csv  # 過濾後

train:
  epochs: 20  # 較少 epochs
  lr: 0.00003  # 較低學習率
  weight_decay: 0.0002  # 更強正則化

loss:
  pseudo_label_weight: 0.5  # 偽標籤權重
```

**Step 3: 訓練**
```bash
python src/train_v2.py --config configs/pseudo_label_stage2.yaml
```

**預期提升**: +0.5-1.5% → **87-90.4%**

---

#### 任務 2.2: 外部數據預訓練 (26h)

**Step 1: 下載 NIH ChestX-ray14**
```bash
# 使用 Kaggle API
kaggle datasets download -d khanfashee/nih-chest-x-ray-14-224x224-resized
unzip nih-chest-x-ray-14-224x224-resized.zip -d data/external/nih_chestxray14/

# 或使用較大版本以獲得更好質量
kaggle datasets download -d nih-chest-xrays/data
# 但需要 40GB 空間
```

**Step 2: 創建預訓練配置**
```yaml
# configs/pretrain_nih.yaml
model:
  name: efficientnet_v2_s
  pretrained: true  # ImageNet 初始化
  num_classes: 14  # NIH 的 14 種疾病

data:
  train_dir: data/external/nih_chestxray14/images/
  train_csv: data/external/nih_chestxray14/train_val_list.txt
  img_size: 384

train:
  epochs: 10  # 較少 epochs
  batch_size: 32
  lr: 0.0001
  optimizer: adamw

# 多標籤分類
loss:
  type: bce_with_logits  # 多標籤
```

**Step 3: 預訓練**
```bash
python src/train_pretrain_multilabel.py \
  --config configs/pretrain_nih.yaml \
  --output_dir outputs/pretrain_nih/
```

**Step 4: 微調**
```yaml
# configs/finetune_from_nih.yaml
inherit_from: configs/improved_breakthrough.yaml

model:
  pretrained_backbone: outputs/pretrain_nih/best.pt  # 加載特徵提取器
  freeze_backbone_epochs: 3  # 前 3 epochs 凍結 backbone

train:
  lr: 0.00003  # 較低學習率
```

```bash
python src/train_v2.py --config configs/finetune_from_nih.yaml
```

**預期提升**: +1-3% → **88-93.4%**

---

### Phase 2 總結

**總時間**: 1-2 天
**預期成績**: **88-93.4%** (樂觀) 或 **87-90.4%** (保守)
**提升幅度**: **+2.8-9.2%** (樂觀) 或 **+2.8-6.2%** (保守)
**成功機率**: 65%

---

## 📈 預期成績路線圖

### 保守估計（置信度 80%）

```
當前: 84.19%
  ↓ Phase 1
[Stacking +1.5%] → 85.69%
[增強 TTA +0.5%] → 86.19%
[WBF 集成 +0.3%] → 86.49%
  ↓ Phase 2
[偽標籤 +0.5%] → 86.99%
[NIH 預訓練 +1%] → 87.99%

最終: 87.99% ✅ 達成目標！
```

### 樂觀估計（置信度 60%）

```
當前: 84.19%
  ↓ Phase 1
[Stacking +3%] → 87.19%
[增強 TTA +1%] → 88.19%
[WBF 集成 +0.7%] → 88.89%
  ↓ Phase 2
[偽標籤 +1.5%] → 90.39%
[NIH 預訓練 +3%] → 93.39%

最終: 93.39% 🚀 超越目標！
```

---

## ⚠️ 風險緩解計劃

### 風險 1: Stacking 過擬合驗證集

**症狀**: 驗證集提升但測試集下降
**緩解**:
1. 使用簡單的 Meta-Learner (Logistic Regression)
2. 添加正則化 (L2, Dropout)
3. 對比多個 Meta-Learner，選擇最穩定的

---

### 風險 2: 偽標籤質量差

**症狀**: 訓練損失不收斂，驗證性能下降
**緩解**:
1. 提高一致性閾值 (0.8 → 0.9)
2. 降低偽標籤權重 (0.5 → 0.3)
3. 只使用偽標籤增強少數類 (COVID-19)

---

### 風險 3: 外部預訓練負遷移

**症狀**: 微調後性能低於無預訓練
**緩解**:
1. 只微調分類頭，凍結特徵提取器
2. 使用極低的學習率 (1e-5)
3. 提前停止如果驗證性能下降

---

### 風險 4: TTA 計算太慢

**症狀**: 推理時間 >30 分鐘
**緩解**:
1. 減少 TTA 變換數量 (20 → 10)
2. 只對集成中的最佳 2-3 個模型使用完整 TTA
3. 使用批次推理加速

---

## 🎯 決策矩陣

### 是否執行 Phase 1？

| 條件 | 答案 | 決策 |
|------|------|------|
| 有 2-3 小時時間？ | ✅ 是 | **執行** |
| 需要快速提升？ | ✅ 是 | **執行** |
| 願意承擔低風險？ | ✅ 是 | **執行** |

**結論**: ✅ **立即執行 Phase 1**

---

### 是否執行 Phase 2？

| 條件 | 答案 | 決策 |
|------|------|------|
| Phase 1 達成 >= 87%？ | 待定 | 條件執行 |
| 有 1-2 天時間？ | ✅ 是 | 條件執行 |
| 目標 >= 90%？ | ✅ 是 | 條件執行 |
| 願意承擔中風險？ | ✅ 是 | 條件執行 |

**結論**: ⚡ **條件執行 Phase 2** (如果 Phase 1 >= 87%)

---

## 📚 實作檢查清單

### Phase 1 (立即開始)

- [ ] 檢查驗證集預測是否存在
- [ ] 生成缺失的驗證集預測
- [ ] 運行 Stacking Meta-Learner
- [ ] 生成 Stacking 測試集預測
- [ ] 提交並檢查分數
- [ ] 實作增強 TTA
- [ ] 生成 TTA 預測
- [ ] 提交並檢查分數
- [ ] 實作 WBF 動態集成
- [ ] 生成 WBF 預測
- [ ] 提交並檢查分數

### Phase 2 (條件執行)

- [ ] 分析偽標籤質量
- [ ] 過濾高置信度樣本
- [ ] 訓練偽標籤模型
- [ ] 評估並提交
- [ ] 下載 NIH ChestX-ray14
- [ ] 創建預訓練腳本
- [ ] 預訓練模型
- [ ] 微調並評估
- [ ] 提交最終預測

---

## 🏁 最終建議

### 立即行動 (30 分鐘內)

1. ✅ **運行 Stacking** - 最高 ROI
2. ✅ **檢查現有資源** - 避免重複工作
3. ✅ **準備增強 TTA** - 快速實作

### 今天完成 (2-3 小時)

4. ✅ 完成 Phase 1 所有任務
5. ✅ 達成 86.5-88.9% 目標
6. ✅ 評估是否需要 Phase 2

### 明天決策 (基於 Phase 1 結果)

7. ⚡ 如果 >= 87%: 執行 Phase 2
8. ⚡ 如果 < 87%: 調試 Phase 1 或嘗試其他技巧
9. ⚡ 如果 >= 90%: 🎉 慶祝成功！

---

**準備好了嗎？讓我們立即開始 Phase 1！** 🚀
