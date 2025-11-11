# 🎯 升級到 90% 指南

當前分數: **82.322%**
目標分數: **90%+**

## 📋 快速開始 (最簡單的方法)

### 方法 1: 使用改進的配置 (預計 87-88%)

1. **安裝 timm 庫**（在 Colab Cell 6 的依賴安裝後添加）:
```python
!pip install -q timm  # 用於 ViT 和其他先進模型
```

2. **修改 `src/train_v2.py` 的 `build_model` 函數**（第 17-50 行）:

在現有代碼後添加:
```python
def build_model(name: str, num_classes: int):
    """Build model with support for ViT and other advanced models"""

    # 原有的 ResNet/EfficientNet 代碼保留...

    # 添加 ViT 支持 (在 elif 鏈末尾添加)
    elif name.startswith('vit_') or name.startswith('swin_'):
        # 使用 timm 庫載入 Vision Transformer
        import timm
        m = timm.create_model(name, pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")

    return m
```

3. **在 Colab 使用新配置**:
- 上傳 `configs/colab_vit_90.yaml`
- 修改 Cell 14 的訓練命令:
```python
!python -m src.train_v2 --config configs/colab_vit_90.yaml
```

### 方法 2: 使用 Ensemble (預計 90-92%)

訓練多個模型並組合預測：

```python
# 1. 訓練 3 個不同模型
models = [
    ('resnet18_224', 'configs/colab_baseline.yaml'),      # 你已有的 82.3%
    ('vit_base_256', 'configs/colab_vit_90.yaml'),        # 新的 ViT ~87%
    ('efficientnet_b3', 'configs/colab_effnet.yaml'),     # EfficientNet ~85%
]

# 2. 生成 3 個預測文件
for model_name, config in models:
    !python -m src.train_v2 --config {config}
    !python -m src.tta_predict --config {config} --ckpt outputs/{model_name}/best.pt

# 3. 組合預測（簡單平均或加權平均）
import pandas as pd
import numpy as np

# 載入 3 個預測
pred1 = pd.read_csv('submission_baseline.csv')  # 82.3%
pred2 = pd.read_csv('submission_vit.csv')       # ~87%
pred3 = pd.read_csv('submission_effnet.csv')    # ~85%

# 加權平均 (根據驗證分數加權)
weights = [0.25, 0.45, 0.30]  # ViT 權重最高
prob_cols = ['normal', 'bacteria', 'virus', 'COVID-19']

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

ensemble.to_csv('submission_ensemble.csv', index=False)
```

---

## 🔧 進階修改 (需要更多開發)

### 1. 實施 Focal Loss

在 `src/train_v2.py` 添加 Focal Loss 類：

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance.
    Especially important for COVID-19 (only 37/3780 samples = 0.98%)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights [1.0, 0.57, 1.05, 27.2]
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, num_classes] logits
            targets: [B] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 在 main() 函數中使用:
if train_cfg.get('loss', 'ce') == 'focal':
    # 計算類別權重
    class_counts = [1009, 1776, 958, 37]  # [Normal, Bacteria, Virus, COVID-19]
    weights = torch.tensor([1.0, 0.57, 1.05, 27.2], device=device)

    loss_fn = FocalLoss(
        alpha=weights,
        gamma=train_cfg.get('focal_gamma', 3.0)
    )
else:
    # 原有的 CrossEntropy
    loss_fn = ...
```

### 2. 實施 Mixup (已有代碼，只需啟用)

在訓練循環中使用:

```python
def train_one_epoch(...):
    for imgs, targets, _ in loader:
        imgs, targets = imgs.to(device), targets.to(device)

        # Apply Mixup if enabled
        if use_mixup and np.random.rand() < mixup_prob:
            from .aug import mixup_data
            imgs, targets_a, targets_b, lam = mixup_data(imgs, targets, alpha=1.0)

            # Forward
            logits = model(imgs)

            # Mixup loss
            loss = lam * loss_fn(logits, targets_a) + (1 - lam) * loss_fn(logits, targets_b)
        else:
            # Standard training
            logits = model(imgs)
            loss = loss_fn(logits, targets)

        # Backward...
```

---

## 📊 預期結果對比

| 方法 | 配置 | 預期分數 | 訓練時間 (A100) |
|------|------|---------|----------------|
| **當前** | ResNet18 + CE | 82.3% | 20 min |
| **方法 1** | ViT + Focal + Medical Aug | 87-88% | 35 min |
| **方法 2** | 3-Model Ensemble | 90-92% | 90 min |

---

## ⚠️ 重要注意事項

### 1. T4 GPU 調整
如果使用 T4 GPU，需要降低 batch size:
```yaml
batch_size: 8  # ViT 在 T4 上需要更小的 batch
```

### 2. 監控 COVID-19 類別
COVID-19 只有 37 個訓練樣本，是最關鍵的類別。訓練時特別注意：
```python
# 在驗證時檢查各類別指標
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=['Normal', 'Bacteria', 'Virus', 'COVID-19']))
```

### 3. 早停 (Early Stopping)
ViT 訓練 25 epochs，使用早停防止過擬合:
```python
patience = 5  # 如果 5 epochs 沒改善就停止
best_f1 = 0
patience_counter = 0

for epoch in range(epochs):
    val_f1 = validate(...)

    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        save_checkpoint(...)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
```

---

## 🚀 最簡單的執行方式

如果只想快速嘗試，在 Colab 中執行:

```python
# Cell: 安裝額外依賴
!pip install -q timm

# Cell: 修改 build_model（複製上面的代碼）
# ... 在 train_v2.py 中添加 ViT 支持

# Cell: 訓練 ViT 模型
!python -m src.train_v2 --config configs/colab_vit_90.yaml

# Cell: TTA 預測
!python -m src.tta_predict --config configs/colab_vit_90.yaml --ckpt outputs/colab_vit_90/best.pt

# Cell: 下載並提交 submission_vit_tta.csv
```

預期結果: **87-88%** (單模型)

如果需要到 90%+，再訓練 2-3 個模型做 ensemble。

---

## 📝 文件清單

已創建的新文件:
- ✅ `configs/colab_vit_90.yaml` - ViT 配置
- ✅ `src/aug.py` - 更新了醫學影像增強
- ✅ 本文件 - 實施指南

需要修改的文件:
- `src/train_v2.py` - 添加 ViT 支持 (約 10 行代碼)
- `notebooks/Colab_A100_Final.ipynb` - Cell 6 添加 timm

---

好運！如果有任何問題，檢查這個文件中的程式碼片段。
