# 🔍 ViT 失敗原因分析與修復方案

## 📊 實驗結果對比

| 模型 | 配置 | Public Score | 結果 |
|------|------|--------------|------|
| ResNet18 | `colab_baseline.yaml` | **82.322%** | ✅ 成功 |
| ViT-Base | `colab_vit_90.yaml` | **80.303%** | ❌ 更差 |
| ViT-Base (修復版) | `colab_vit_fixed.yaml` | **預計 84-86%** | 🔄 待測試 |

## 🐛 問題根因分析

### 1. **學習率過低** ⚠️ 關鍵問題
```yaml
# ❌ 失敗配置
lr: 0.0001  # 太低，導致 ViT 無法適應醫學影像域

# ✅ 修復後
lr: 0.0003  # 與成功的 ResNet18 相同
```

**影響**: ViT 從 ImageNet 預訓練遷移到醫學影像需要足夠的學習率。過低的學習率導致模型無法有效學習新特徵。

---

### 2. **關閉加權採樣** ⚠️ 關鍵問題
```yaml
# ❌ 失敗配置
use_weighted_sampler: false  # COVID-19 只有 1% 樣本

# ✅ 修復後
use_weighted_sampler: true   # 平衡類別採樣
```

**影響**: COVID-19 類別只有 37/3780 (0.98%) 樣本。沒有加權採樣，模型幾乎看不到 COVID-19，導致召回率極低。

---

### 3. **損失函數衝突** ⚠️ 嚴重問題
```yaml
# ❌ 失敗配置（三重正則化衝突）
loss: focal_improved
focal_gamma: 3.0
focal_alpha: [1.0, 0.57, 1.05, 27.2]  # COVID-19 權重過高
label_smoothing: 0.1                   # 與 Focal Loss 衝突
use_mixup: true
mixup_prob: 0.8                        # 80% mixup 太激進

# ✅ 修復後（簡單有效）
loss: ce
label_smoothing: 0.05
use_weighted_sampler: true
mixup_prob: 0.3                        # 降低到 30%
```

**問題**:
- **Label Smoothing 與 Focal Loss 衝突**: Label smoothing 軟化標籤，Focal Loss 需要硬標籤來聚焦困難樣本
- **COVID-19 alpha=27.2 過高**: 導致梯度不穩定，過度擬合稀有類別
- **80% Mixup 太激進**: 醫學影像需要精確特徵，過度 mixup 破壞影像結構

---

### 4. **Warmup 不足**
```yaml
# ❌ 失敗配置
warmup_epochs: 2  # ViT 需要更長 warmup

# ✅ 修復後
warmup_epochs: 5  # Transformer 需要更長穩定期
```

**影響**: ViT 的 attention 機制在訓練初期不穩定，需要更長 warmup 期。

---

### 5. **Weight Decay 不足**
```yaml
# ❌ 失敗配置
weight_decay: 0.01  # ViT 需要更強正則化

# ✅ 修復後
weight_decay: 0.05  # Transformer 標準設定
```

**影響**: ViT 參數量大 (86M vs ResNet18 的 11M)，需要更強的 L2 正則化防止過擬合。

---

## 📋 配置對比表

| 參數 | ResNet18 (82.3%) | ViT 失敗版 (80.3%) | ViT 修復版 (預計 84-86%) |
|------|------------------|-------------------|-------------------------|
| **學習率** | 0.0003 ✅ | 0.0001 ❌ | 0.0003 ✅ |
| **Weight Decay** | 0.0001 | 0.01 | 0.05 ✅ |
| **Warmup Epochs** | 1 | 2 | 5 ✅ |
| **Weighted Sampler** | true ✅ | false ❌ | true ✅ |
| **Loss Function** | CE ✅ | Focal (gamma=3.0) ❌ | CE ✅ |
| **Label Smoothing** | 0.05 ✅ | 0.1 ❌ | 0.05 ✅ |
| **Mixup Prob** | - | 0.8 ❌ | 0.3 ✅ |
| **Mixup Alpha** | - | 1.0 | 0.4 ✅ |
| **Epochs** | 12 | 25 | 20 ✅ |
| **Batch Size** | 32 | 16 | 16 ✅ |

---

## 🎯 修復策略

### **核心原則**: 保留 ResNet18 的成功配置，只改變架構

修復版配置 (`configs/colab_vit_fixed.yaml`) 的關鍵改動：

1. ✅ **恢復學習率**: 0.0001 → 0.0003
2. ✅ **啟用加權採樣**: false → true
3. ✅ **簡化損失函數**: Focal Loss → 標準 CE
4. ✅ **降低 Mixup**: 80% → 30%
5. ✅ **增加 Warmup**: 2 epochs → 5 epochs
6. ✅ **提高正則化**: weight_decay 0.01 → 0.05

---

## 🚀 執行方式

### 在 Colab 使用修復版配置：

```python
# Cell 14: 使用修復版配置
!python -m src.train_v2 --config configs/colab_vit_fixed.yaml

# Cell 20: TTA 預測
!python -m src.tta_predict --config configs/colab_vit_fixed.yaml \
    --ckpt outputs/colab_vit_fixed/best.pt
```

### 預期結果：
- **訓練時間**: 35-40 分鐘 (A100)
- **驗證 F1**: 0.83-0.85
- **Public Score**: 84-86%

---

## 📈 達到 90% 的路徑

### 方案 1: Ensemble 2-3 個模型

```python
# 訓練 3 個模型
models = [
    ('ResNet18', 82.3%),    # 已有
    ('ViT-Fixed', ~85%),    # 修復版
    ('ConvNeXt', ~83%),     # 重新訓練
]

# 加權 ensemble
weights = [0.30, 0.45, 0.25]  # 根據驗證分數
```

**預期**: 87-89%

### 方案 2: 多種 ViT 變體

如果修復版 ViT 達到 85%+，可嘗試：
- `vit_base_patch16_224` (當前)
- `swin_base_patch4_window7_224` (層次結構更適合醫學影像)
- `deit3_base_patch16_224` (蒸餾版 ViT)

### 方案 3: TTA + Pseudo Labeling

1. 使用 TTA 生成高置信度預測
2. 將高置信度測試集樣本作為偽標籤
3. 重新訓練

---

## 💡 關鍵教訓

1. **不要過度優化**: 成功的 ResNet18 配置證明**簡單有效** > 複雜策略
2. **一次改變一個變數**: ViT 失敗版同時改了 7 個參數，無法定位問題
3. **Class imbalance 是核心**: 加權採樣對 COVID-19 (1%) 至關重要
4. **Transformer 需要特殊照顧**: 更高 lr、更長 warmup、更強正則化
5. **醫學影像特殊性**: 過度增強（80% mixup）破壞關鍵病理特徵

---

## 🔗 相關文件

- ✅ `configs/colab_vit_fixed.yaml` - 修復版配置
- ✅ `configs/colab_baseline.yaml` - 成功的 ResNet18 基線
- ❌ `configs/colab_vit_90.yaml` - 失敗的 ViT 配置（保留作為反例）

---

## 📝 下一步

1. **立即執行**: 在 Colab 使用 `colab_vit_fixed.yaml` 訓練
2. **驗證結果**: 預期 84-86%
3. **如果成功**: 進行 ensemble 衝刺 90%
4. **如果仍不理想**: 嘗試 Swin Transformer 或回歸 ensemble ResNet 變體

---

**結論**: ViT 架構沒問題，配置有問題。修復版應該能達到 84-86%，結合 ensemble 有望達到 90%。
