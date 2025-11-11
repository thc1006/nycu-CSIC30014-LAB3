# 🌙 整晚自動訓練 TODO 清單

**目標**: 從 80 分提升到 90 分以上
**GPU**: RTX 3050 (4GB)
**預計總時間**: 約 11-12 小時
**開始時間**: ___:___
**預計完成**: ___:___

---

## 📋 實驗清單

### ✅ 準備工作 (5 分鐘)

- [ ] 確認 GPU 正常：`nvidia-smi`
- [ ] 確認資料集完整：檢查 `train_images/`, `val_images/`, `test_images/`
- [ ] 確認環境：`pip list | grep torch`
- [ ] 備份當前最佳模型：`cp outputs/run1/best.pt outputs/run1/best_backup.pt`

---

### 🔥 實驗 1: ConvNeXt-Tiny + 288px (2.5 hours)

**配置**: `configs/exp1_convnext_tiny.yaml`
**目標分數**: 83-85%
**策略**: 中等模型 + 高解析度 + Improved Focal Loss

#### 執行命令:
```bash
python -m src.train_v2 --config configs/exp1_convnext_tiny.yaml
python -m src.tta_predict --config configs/exp1_convnext_tiny.yaml
```

#### 完成標記:
- [ ] 訓練完成 (25 epochs)
- [ ] Val F1 ≥ 0.83
- [ ] 生成 `submission_exp1.csv`
- [ ] 檢查點: `outputs/exp1_convnext_tiny/best.pt`

**實際 Val F1**: _____
**完成時間**: ___:___

---

### 🚀 實驗 2: EfficientNetV2-S + 320px + SWA (3 hours)

**配置**: `configs/exp2_efficientnetv2.yaml`
**目標分數**: 84-86%
**策略**: 高效架構 + 高解析度 + SWA + 強增強

#### 執行命令:
```bash
python -m src.train_v2 --config configs/exp2_efficientnetv2.yaml
python -m src.tta_predict --config configs/exp2_efficientnetv2.yaml
```

#### 完成標記:
- [ ] 訓練完成 (30 epochs + SWA)
- [ ] Val F1 ≥ 0.84
- [ ] 生成 `submission_exp2.csv`
- [ ] 檢查點: `outputs/exp2_efficientnetv2/best.pt` + `best_swa.pt`

**實際 Val F1**: _____
**完成時間**: ___:___

---

### ⚡ 實驗 3: ResNet34 + 384px + Long (2 hours)

**配置**: `configs/exp3_resnet34_long.yaml`
**目標分數**: 85-87%
**策略**: 中型模型 + 最高解析度 + 極長訓練 + 超強增強

#### 執行命令:
```bash
python -m src.train_v2 --config configs/exp3_resnet34_long.yaml
python -m src.tta_predict --config configs/exp3_resnet34_long.yaml
```

#### 完成標記:
- [ ] 訓練完成 (35 epochs + SWA)
- [ ] Val F1 ≥ 0.85
- [ ] 生成 `submission_exp3.csv`
- [ ] 檢查點: `outputs/exp3_resnet34/best.pt`

**實際 Val F1**: _____
**完成時間**: ___:___

---

### 💫 實驗 4: EfficientNet-B0 + 256px + Ultra Long (2.5 hours)

**配置**: `configs/exp4_efficientnet_b0.yaml`
**目標分數**: 84-86%
**策略**: 輕量模型 + 中解析度 + 極長訓練 (40 epochs)

#### 執行命令:
```bash
python -m src.train_v2 --config configs/exp4_efficientnet_b0.yaml
python -m src.tta_predict --config configs/exp4_efficientnet_b0.yaml
```

#### 完成標記:
- [ ] 訓練完成 (40 epochs + SWA)
- [ ] Val F1 ≥ 0.84
- [ ] 生成 `submission_exp4.csv`
- [ ] 檢查點: `outputs/exp4_efficientnet_b0/best.pt`

**實際 Val F1**: _____
**完成時間**: ___:___

---

### 🌟 實驗 5: ResNet18 + 384px + Ultra Aug (1.5 hours)

**配置**: `configs/exp5_resnet18_ultra.yaml`
**目標分數**: 83-85%
**策略**: 輕量模型 + 高解析度 + 超長訓練 (50 epochs) + 最強增強

#### 執行命令:
```bash
python -m src.train_v2 --config configs/exp5_resnet18_ultra.yaml
python -m src.tta_predict --config configs/exp5_resnet18_ultra.yaml
```

#### 完成標記:
- [ ] 訓練完成 (50 epochs + SWA)
- [ ] Val F1 ≥ 0.83
- [ ] 生成 `submission_exp5.csv`
- [ ] 檢查點: `outputs/exp5_resnet18_ultra/best.pt`

**實際 Val F1**: _____
**完成時間**: ___:___

---

### 🎯 最終步驟: Ensemble (5 分鐘)

**目標分數**: 87-92%
**策略**: 合併所有模型的預測，提升 2-4%

#### 執行命令:
```bash
python ensemble.py
```

#### 完成標記:
- [ ] 生成 `submission_ensemble_soft.csv` (推薦)
- [ ] 生成 `submission_ensemble_hard.csv` (備選)
- [ ] 檢查類別分佈是否合理

**完成時間**: ___:___

---

## 📊 最終結果總結

### 個別模型表現:

| 實驗 | 模型 | Val F1 | 預測檔案 | 狀態 |
|------|------|--------|----------|------|
| Exp 1 | ConvNeXt-Tiny | ____ | `submission_exp1.csv` | [ ] |
| Exp 2 | EfficientNetV2-S | ____ | `submission_exp2.csv` | [ ] |
| Exp 3 | ResNet34 | ____ | `submission_exp3.csv` | [ ] |
| Exp 4 | EfficientNet-B0 | ____ | `submission_exp4.csv` | [ ] |
| Exp 5 | ResNet18 | ____ | `submission_exp5.csv` | [ ] |

### Ensemble 結果:

- **Soft Voting**: `submission_ensemble_soft.csv`
- **Hard Voting**: `submission_ensemble_hard.csv`

### 推薦提交順序:

1. **優先**: `submission_ensemble_soft.csv`
2. **備選 1**: Val F1 最高的個別模型
3. **備選 2**: `submission_ensemble_hard.csv`

---

## 🚀 快速啟動指令

### 方式 1: 自動執行所有實驗 (推薦)

```bash
python run_all_experiments.py
```

### 方式 2: 手動逐個執行

```bash
# 實驗 1
python -m src.train_v2 --config configs/exp1_convnext_tiny.yaml
python -m src.tta_predict --config configs/exp1_convnext_tiny.yaml

# 實驗 2
python -m src.train_v2 --config configs/exp2_efficientnetv2.yaml
python -m src.tta_predict --config configs/exp2_efficientnetv2.yaml

# 實驗 3
python -m src.train_v2 --config configs/exp3_resnet34_long.yaml
python -m src.tta_predict --config configs/exp3_resnet34_long.yaml

# 實驗 4
python -m src.train_v2 --config configs/exp4_efficientnet_b0.yaml
python -m src.tta_predict --config configs/exp4_efficientnet_b0.yaml

# 實驗 5
python -m src.train_v2 --config configs/exp5_resnet18_ultra.yaml
python -m src.tta_predict --config configs/exp5_resnet18_ultra.yaml

# Ensemble
python ensemble.py
```

---

## 🛠️ 故障排除

### 如果訓練中斷:

1. 檢查 GPU 記憶體：`nvidia-smi`
2. 如果 OOM，降低該實驗的 batch_size
3. 重新執行該實驗的訓練指令

### 如果某個實驗失敗:

- 跳過該實驗，繼續下一個
- 至少需要 3 個模型才能做有效的 Ensemble

### 如果 TTA 太慢:

- 可以跳過 TTA，直接用 `src/predict.py` 生成預測
- TTA 通常可以提升 1-2%，但不是必須的

---

## 📝 注意事項

1. **訓練期間不要關閉電腦**
2. **確保電腦不會進入休眠模式**
3. **定期檢查進度** (可以用 `tail -f training_log.txt` 監控)
4. **備份重要檔案** (模型檢查點和提交檔案)

---

## 🎉 完成後

- [ ] 所有實驗執行完畢
- [ ] 生成 Ensemble 預測
- [ ] 備份所有 submission_*.csv 檔案
- [ ] 準備提交到 Kaggle

**預期最終分數**: 87-92%
**實際最終分數**: _____

---

**祝你好運！💪**
