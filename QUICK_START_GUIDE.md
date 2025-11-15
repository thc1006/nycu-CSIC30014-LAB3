# 🚀 快速開始指南 - 突破 90% 自動化流程

**當前狀態**: Gen2 訓練中 (Epoch 4/50, Val F1 47.36%)
**預計完成**: 今晚 23:00-00:00
**目標分數**: 90.0%+

---

## 📍 現在可以做什麼？

### 選項 1: 監控 Gen2 訓練（推薦）

```bash
# 實時監控訓練進度
watch -n 60 './monitor_gen2.sh'

# 或查看詳細日誌
tail -f outputs/v2l_512_gen2/logs/fold0.log
```

**預期時間線**:
- Fold 0: 完成於 ~18:00 (Val F1 預估 88-89%)
- Fold 1-4: 每個 ~90 分鐘
- 全部完成: ~23:00

---

### 選項 2: 等待後自動執行（最簡單）

Gen2 完成後運行：

```bash
bash AUTO_BREAKTHROUGH_90.sh
```

這個腳本會自動：
1. ✅ 檢測 Gen2 完成
2. 📊 生成 5-Fold 集成預測
3. 📤 提交到 Kaggle
4. 🤔 根據分數決定是否執行 Gen3
5. 🎉 自動完成整個流程

---

### 選項 3: 手動分步執行

#### 步驟 1: 等待 Gen2 完成

```bash
# 持續監控
watch -n 300 './monitor_gen2.sh'

# 當所有 fold 完成時繼續
```

#### 步驟 2: 生成 Gen2 預測

```bash
python3 scripts/generate_gen2_predictions.py
```

輸出: `data/submission_gen2_ensemble.csv`

#### 步驟 3: 提交測試

```bash
kaggle competitions submit \
  -c cxr-multi-label-classification \
  -f data/submission_gen2_ensemble.csv \
  -m "Gen2: 532 Pseudo-labels + 5-Fold Ensemble"
```

#### 步驟 4: 查看結果

```bash
kaggle competitions submissions -c cxr-multi-label-classification | head -5
```

#### 步驟 5: 如果 < 90%，執行 Gen3

```bash
# 生成 Gen3 偽標籤
python3 scripts/generate_gen3_adaptive_pseudo_labels.py

# 訓練 Gen3 (7-8 小時)
bash START_GEN3_TRAINING.sh

# Gen3 完成後生成預測
python3 scripts/generate_gen3_predictions.py

# 提交 Gen3
kaggle competitions submit \
  -c cxr-multi-label-classification \
  -f data/submission_gen3_ensemble.csv \
  -m "Gen3: Adaptive Pseudo-labeling (800-900 samples) + 5-Fold"
```

---

## 📊 預期結果

### Gen2 預期

| 場景 | 概率 | 驗證 F1 | 測試 F1 |
|------|------|---------|---------|
| 樂觀 | 30% | 89.5% | 90.0%+ ✅ |
| 基準 | 50% | 88.5% | 89.0-89.5% |
| 保守 | 20% | 87.5% | 88.0-88.5% |

### Gen3 預期（如需）

| 場景 | 概率 | 測試 F1 |
|------|------|---------|
| 樂觀 | 40% | 90.5-91.0% 🎯 |
| 基準 | 50% | 89.5-90.0% ✅ |
| 保守 | 10% | 89.0-89.5% |

**總成功率**: ~75% 達到 90%+

---

## 🔍 如何監控進度

### GPU 使用情況

```bash
nvidia-smi

# 持續監控
watch -n 5 nvidia-smi
```

**正常狀態**:
- 使用率: 95-100%
- 記憶體: ~11 GB / 16 GB
- 溫度: 75-85°C

### 訓練日誌

```bash
# 查看當前 epoch
grep "epoch" outputs/v2l_512_gen2/logs/fold0.log | tail -5

# 查看最佳分數
grep "saved new best" outputs/v2l_512_gen2/logs/fold0.log
```

### 進程狀態

```bash
# 檢查訓練進程
ps aux | grep train_v2.py | grep -v grep

# 進程數量應該是 5 個 (主進程 + 4 workers)
```

---

## ⚠️ 故障排除

### 訓練卡住不動

```bash
# 檢查日誌最後幾行
tail -20 outputs/v2l_512_gen2/logs/fold0.log

# 如果確認卡住，重啟訓練
pkill -f "train_v2.py.*gen2"
bash START_GEN2_TRAINING_NOW.sh > logs/gen2_restart.log 2>&1 &
```

### GPU OOM 錯誤

```bash
# 已配置 batch_size=4（最小）
# 如果仍 OOM，檢查是否有其他進程佔用 GPU
nvidia-smi

# 清理其他進程
pkill -f python
```

### 預測生成失敗

```bash
# 確保所有 fold 模型存在
ls -lh outputs/v2l_512_gen2/fold*/best.pt

# 手動生成單個 fold 預測（修改腳本 fold 範圍）
```

---

## 📁 重要文件位置

### 配置文件
- Gen2: `configs/efficientnet_v2l_512_gen2.yaml`
- Gen3: `configs/efficientnet_v2l_512_gen3.yaml`

### 訓練數據
- Gen2: `data/fold{0-4}_train_gen2.csv` (3,280 樣本/fold)
- Gen3: 將生成 `data/fold{0-4}_train_gen3.csv`

### 模型檢查點
- Gen2: `outputs/v2l_512_gen2/fold{0-4}/best.pt`
- Gen3: `outputs/v2l_512_gen3/fold{0-4}/best.pt`

### 提交文件
- Gen2 集成: `data/submission_gen2_ensemble.csv`
- Gen3 集成: `data/submission_gen3_ensemble.csv`

### 日誌
- 訓練: `outputs/v2l_512_gen2/logs/fold{0-4}.log`
- 執行: `logs/gen2_training_fixed.log`

---

## 💡 小技巧

### 後台運行並斷開連接

```bash
# 使用 screen 或 tmux
screen -S gen2_training
./monitor_gen2.sh
# 按 Ctrl+A, D 斷開

# 重新連接
screen -r gen2_training
```

### 自動通知

```bash
# Gen2 完成時發送通知（需要配置）
bash AUTO_BREAKTHROUGH_90.sh && echo "Gen2 完成！" | mail -s "訓練完成" your@email.com
```

### 保存所有日誌

```bash
# 將監控輸出保存到文件
watch -n 60 './monitor_gen2.sh' | tee -a logs/monitor_history.log
```

---

## 🎯 成功指標

### Gen2 訓練成功標誌

- ✅ 所有 5 個 fold 完成訓練
- ✅ 最佳 Val F1 ≥ 88.0%
- ✅ 每個 fold 模型文件 > 450 MB
- ✅ 無 OOM 或其他錯誤

### Gen2 測試成功標誌

- 🎯 測試 F1 ≥ 90.0% → **成功！**
- ✅ 測試 F1 89.0-89.9% → 執行 Gen3
- ⚠️ 測試 F1 88.0-88.9% → 執行 Gen3 + 調整閾值
- ❌ 測試 F1 < 88.0% → 檢查問題

---

**祝順利突破 90%！** 🚀

如有問題，查看詳細文檔：`GEN3_STRATEGY_READY.md`
