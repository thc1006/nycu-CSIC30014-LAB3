# 🎯 Gen3 自適應策略 - 準備就緒

**創建時間**: 2025-11-15 16:40
**狀態**: ✅ **完全就緒，等待 Gen2 完成**

---

## 📋 Gen3 配置概覽

### 核心策略

**自適應閾值偽標籤**：
- **Normal**: 0.92 閾值（簡單類別，高質量要求）
- **Bacteria**: 0.90 閾值（簡單類別）
- **Virus**: 0.85 閾值（困難類別，放寬閾值）
- **COVID-19**: 0.80 閾值（最困難，最低閾值確保樣本量）

**預期偽標籤數量**: 800-900 個（vs Gen2 的 532 個）

### 訓練增強

**正則化**：
- Dropout: 0.40 (↑ from Gen2 0.35)
- Drop Path: 0.30 (↑ from 0.25)
- Weight Decay: 0.0004 (↑ from 0.0003)
- Label Smoothing: 0.20 (↑ from 0.18)

**學習率**：
- 初始 LR: 0.00007 (↓ from 0.00008) - 更細緻微調
- 最小 LR: 0.0000003 (↓ from 0.0000005)
- Warmup: 5 epochs (↑ from 4)

**數據增強**（略保守以應對偽標籤噪音）：
- Mixup: 0.75 prob, 1.8 alpha (↓ from 0.8, 2.0)
- CutMix: 0.55 prob (↓ from 0.6)
- Rotation: ±22° (↓ from ±25°)

---

## 🚀 執行流程

### 自動化腳本

**完整自動流程** (`AUTO_BREAKTHROUGH_90.sh`):
1. ⏳ 等待 Gen2 完成（自動檢測）
2. 📊 生成 Gen2 5-Fold 集成預測
3. 📤 提交 Gen2 到 Kaggle
4. 🤔 根據分數決定是否執行 Gen3
5. 如果 < 90%：
   - 📦 生成 Gen3 自適應偽標籤
   - 🔥 訓練 Gen3 (7-8 小時)
   - 📊 生成 Gen3 預測
   - 📤 提交 Gen3
6. 🎉 完成！

**使用方法**：
```bash
# 方法 1: 完全自動（推薦）
bash AUTO_BREAKTHROUGH_90.sh

# 方法 2: 手動分步執行
# 步驟 1: 等待 Gen2
watch -n 300 './monitor_gen2.sh'

# 步驟 2: Gen2 完成後
python3 scripts/generate_gen2_predictions.py
kaggle competitions submit -f data/submission_gen2_ensemble.csv ...

# 步驟 3: 如果需要 Gen3
python3 scripts/generate_gen3_adaptive_pseudo_labels.py
bash START_GEN3_TRAINING.sh
```

---

## 📊 預期性能

### Gen2 預期

| 指標 | 保守預估 | 樂觀預估 | 依據 |
|------|----------|----------|------|
| 驗證 F1 | 88.5% | 89.5% | 532 偽標籤 +20.7% 數據 |
| 測試 F1 | 89.0% | 90.0% | 研究證明 +15.95-26.75% |
| Val-Test Gap | 0.5% | 1.5% | 更好泛化 |

### Gen3 預期（如需）

| 指標 | 保守預估 | 樂觀預估 | 依據 |
|------|----------|----------|------|
| 偽標籤數 | 800 | 900 | 自適應閾值 |
| 驗證 F1 | 89.0% | 90.0% | 累積效應 |
| 測試 F1 | **89.5%** | **91.0%** 🎯 | 多輪迭代 |

---

## 🔬 關鍵創新

### 1. 類別特定閾值

**問題**: 固定閾值 0.95 對困難類別太嚴格
**解決**:
- 簡單類別（Normal, Bacteria）: 高閾值 0.90-0.92
- 困難類別（Virus, COVID-19）: 低閾值 0.80-0.85

**優勢**:
- ✅ 獲得更多困難類別樣本
- ✅ 維持簡單類別高質量
- ✅ 平衡類別分布

### 2. 漸進式正則化

**Gen2 → Gen3 正則化梯度**:
```
Dropout:         0.35 → 0.40
Label Smoothing: 0.18 → 0.20
Weight Decay:    0.0003 → 0.0004
```

**理由**: 偽標籤噪音可能隨迭代增加，需要更強正則化

### 3. 保守數據增強

**策略**: Gen3 降低增強強度
- Mixup alpha: 2.0 → 1.8
- Rotation: ±25° → ±22°

**理由**: 避免過度混合引入額外噪音

---

## 📁 文件清單

### 配置文件
- ✅ `configs/efficientnet_v2l_512_gen3.yaml` - Gen3 訓練配置

### 腳本文件
- ✅ `scripts/generate_gen3_adaptive_pseudo_labels.py` - 自適應偽標籤生成
- ✅ `scripts/generate_gen2_predictions.py` - Gen2 預測生成
- ✅ `scripts/generate_gen3_predictions.py` - Gen3 預測生成
- ✅ `START_GEN3_TRAINING.sh` - Gen3 訓練啟動
- ✅ `AUTO_BREAKTHROUGH_90.sh` - 完整自動化流程

### 監控工具
- ✅ `monitor_gen2.sh` - Gen2 訓練監控

---

## ⏰ 時間規劃

### Gen2 完成時間線（預估）

```
16:30 (現在) → Fold 0 Epoch 2
18:00        → Fold 0 完成
18:30        → Fold 1 完成
20:00        → Fold 2 完成
21:30        → Fold 3 完成
23:00        → Fold 4 完成
23:15        → 生成預測
23:30        → 提交測試
```

### 如需 Gen3

```
23:30        → Gen2 結果 < 90%
23:45        → 生成 Gen3 偽標籤
00:00        → 啟動 Gen3 訓練
08:00 (明天) → Gen3 完成
08:15        → 生成預測
08:30        → 提交測試
09:00        → 結果：89.5-91.0% 🎯
```

**最晚完成時間**: 明天早上 9:00

---

## 🎲 風險評估

### 高風險情況

1. **Gen2 < 88%**: 偽標籤質量問題
   - 應對: Gen3 提高閾值至 0.93/0.91/0.87/0.82

2. **Gen3 過擬合**: 偽標籤噪音累積
   - 應對: 已配置更強正則化

3. **訓練失敗**: OOM 或其他錯誤
   - 應對: Batch size 已設為 4（最小）

### 成功率評估

| 場景 | 概率 | 預期結果 |
|------|------|----------|
| Gen2 ≥ 90% | 40% | ✅ 直接成功 |
| Gen2 89-90% + Gen3 | 35% | ✅ Gen3 成功 |
| Gen2 88-89% + Gen3 | 20% | ⚠️ 接近 90% |
| Gen2 < 88% | 5% | ❌ 需要重新策略 |

**總成功率**: ~75% 達到 90%+

---

## 💡 備用方案

如果 Gen3 仍未達 90%:

1. **Gen4 超激進閾值**: 0.85/0.82/0.78/0.75
2. **模型湯集成**: Gen2 + Gen3 權重平均
3. **Stacking**: Gen2/Gen3 作為 Layer 1
4. **溫度縮放**: 校準 Gen3 預測置信度

---

**狀態**: ✅ **一切就緒，等待 Gen2 完成！**

使用 `./monitor_gen2.sh` 查看訓練進度
使用 `bash AUTO_BREAKTHROUGH_90.sh` 啟動自動流程（Gen2 完成後）
