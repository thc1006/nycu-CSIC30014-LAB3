# 🎯 優化策略總結

## 當前狀態
- **Current Score**: 80+ (Public)
- **Current Val F1**: 0.8033
- **Current Model**: ResNet18 @ 224px, 8 epochs
- **GPU**: RTX 3050 (4GB VRAM)

## 目標
- **Target Score**: 90+ (Public)
- **Required Improvement**: +10%

## 五個實驗的策略差異

### 實驗 1: ConvNeXt-Tiny + 288px
**核心策略**: 升級模型架構
- **模型**: ConvNeXt-Tiny (28M 參數 vs ResNet18 11M)
- **解析度**: 288px (提升 29%)
- **Epochs**: 25
- **Loss**: Improved Focal Loss with class weights [1.0, 1.8, 2.2, 3.0]
- **Augmentation**: Medium strength
- **預期提升**: +3-5%

**優勢**: 現代架構，性能提升明顯
**風險**: 記憶體使用較高 (~3.8GB)

---

### 實驗 2: EfficientNetV2-S + 320px + SWA
**核心策略**: 高效率 + 高解析度 + SWA
- **模型**: EfficientNetV2-S (21M 參數)
- **解析度**: 320px (提升 43%)
- **Epochs**: 30 + SWA
- **Loss**: Improved Focal Loss with COVID-19 weight = 4.0
- **Augmentation**: Strong
- **SWA**: 啟用，從 epoch 23 開始
- **預期提升**: +4-6%

**優勢**: 最佳的參數效率，SWA 提升泛化能力
**風險**: 訓練時間最長 (~3 hours)

---

### 實驗 3: ResNet34 + 384px + Long
**核心策略**: 最高解析度 + 極長訓練
- **模型**: ResNet34 (21M 參數)
- **解析度**: 384px (提升 71%！)
- **Epochs**: 35 + SWA
- **Loss**: Focal Loss with extreme COVID-19 weight = 5.0
- **Augmentation**: Very strong
- **預期提升**: +5-7%

**優勢**: 高解析度捕捉細節，長訓練充分學習
**風險**: batch_size 只能是 6，收斂較慢

---

### 實驗 4: EfficientNet-B0 + 256px + Ultra Long
**核心策略**: 輕量模型 + 超長訓練
- **模型**: EfficientNet-B0 (5M 參數，最輕量)
- **解析度**: 256px (提升 14%)
- **Epochs**: 40 (最多)
- **Loss**: Improved Focal Loss
- **Augmentation**: Medium
- **SWA**: 啟用
- **預期提升**: +4-6%

**優勢**: 記憶體最省 (~3.2GB)，batch_size 最大 (14)
**風險**: 模型容量較小

---

### 實驗 5: ResNet18 + 384px + Ultra Aug
**核心策略**: 原始模型 + 高解析度 + 最強增強
- **模型**: ResNet18 (11M 參數，與當前相同)
- **解析度**: 384px (提升 71%)
- **Epochs**: 50 (最多)
- **Loss**: Focal Loss with extreme COVID-19 weight = 6.0
- **Augmentation**: Ultra strong (rotation=35°, scale=0.7-1.3)
- **SWA**: 啟用
- **預期提升**: +3-5%

**優勢**: 訓練時間最快 (~1.5h)，可作為基線對比
**風險**: 模型容量限制

---

## 關鍵優化技術

### 1. Improved Focal Loss
- **原理**: 對難分類樣本賦予更高權重
- **Class weights**: 針對 COVID-19 類別提升權重 (3.0-6.0)
- **Label smoothing**: 0.08-0.15，防止過擬合
- **預期效果**: +2-3%

### 2. Mixup/CutMix
- **原理**: 混合兩個樣本，提升泛化能力
- **Probability**: 0.5-0.65
- **Alpha**: 0.7-1.2
- **預期效果**: +1-2%

### 3. SWA (Stochastic Weight Averaging)
- **原理**: 平均訓練後期的多個模型權重
- **Start epoch**: 最後 5-10 個 epochs
- **預期效果**: +1-2%

### 4. Test-Time Augmentation (TTA)
- **原理**: 對測試圖片做 6 種變換，平均預測
- **Transformations**: original, hflip, vflip, rot90/180/270
- **預期效果**: +1-2%

### 5. Ensemble
- **原理**: 合併多個模型的預測
- **Methods**: Soft voting (平均概率) vs Hard voting (多數投票)
- **預期效果**: +2-4%

---

## 預期結果分析

### 個別模型預期分數:

| 實驗 | Val F1 | Public Score | 關鍵特點 |
|------|--------|--------------|----------|
| Exp 1 | 0.83-0.85 | 83-85 | ConvNeXt 架構優勢 |
| Exp 2 | 0.84-0.86 | 84-86 | 高效率 + SWA |
| Exp 3 | 0.85-0.87 | 85-87 | 最高解析度 |
| Exp 4 | 0.84-0.86 | 84-86 | 極長訓練 |
| Exp 5 | 0.83-0.85 | 83-85 | 超強增強 |

### Ensemble 預期分數:

- **Best Case**: 89-92 (所有模型都成功)
- **Typical Case**: 87-90 (3-4 個模型成功)
- **Worst Case**: 85-87 (僅 2 個模型成功)

---

## 為什麼這個策略會有效？

### 1. 多樣性 (Diversity)
- 5 種不同的模型架構
- 5 種不同的解析度 (256-384px)
- 不同的 random seeds
- 不同的超參數組合

**效果**: Ensemble 時減少相關性，提升泛化能力

### 2. 針對性優化 (Targeted Improvements)
- **解析度提升**: 224px → 256-384px (細節更清晰)
- **COVID-19 權重**: 1.2 → 3.0-6.0 (解決類別不平衡)
- **訓練時長**: 8 → 25-50 epochs (更充分學習)
- **數據增強**: 中等 → 強/超強 (泛化能力)

### 3. 穩定性保障 (Robustness)
- SWA: 平滑模型權重
- TTA: 測試時增強
- Ensemble: 集成多個模型
- Label smoothing: 防止過擬合

---

## 失敗風險與應對

### 風險 1: OOM (Out of Memory)
**症狀**: CUDA out of memory error
**解決**:
- 降低該實驗的 batch_size (configs/*.yaml)
- 跳過該實驗，繼續下一個

### 風險 2: 訓練不收斂
**症狀**: Val F1 沒有提升或震盪
**解決**:
- 降低 learning rate
- 減少數據增強強度
- 增加 warmup epochs

### 風險 3: 過擬合
**症狀**: Train F1 很高，Val F1 很低
**解決**:
- 啟用 SWA
- 增加數據增強
- 增加 label smoothing

### 風險 4: 訓練太慢
**症狀**: 單個 epoch 超過 15 分鐘
**解決**:
- 降低解析度
- 減少 epochs
- 使用更輕量的模型

---

## 下一步優化方向 (如果還不夠 90%)

1. **Pseudo-labeling**: 用訓練好的模型標記測試集，加入訓練
2. **Multi-scale training**: 訓練時隨機改變輸入解析度
3. **CutOut/GridMask**: 更激進的數據增強
4. **Knowledge Distillation**: 用大模型指導小模型
5. **Class-balanced sampling**: 重採樣平衡類別
6. **External data**: 使用額外的胸部 X 光資料集

---

**Good luck! 🚀**
