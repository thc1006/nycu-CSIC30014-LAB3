# 胸部 X 光分類項目 - 快速啟動指南

**專案狀態**: 當前最佳 **87.574%** Macro-F1 | 目標: 突破 **90%+**

**最後更新**: 2025-11-16 (專案已清理重組)

---

## 專案概覽

這是一個深度學習醫學影像分類項目，使用胸部 X 光影像進行 4 分類：
- **Normal** (正常)
- **Bacteria** (細菌性肺炎)
- **Virus** (病毒性肺炎)
- **COVID-19** (新冠肺炎)

### 當前成果

| 排名 | 配置 | Test F1 | 文件路徑 |
|------|------|---------|----------|
| 🥇 | Hybrid Adaptive Ensemble | **87.574%** | `data/submissions/best/01_hybrid_adaptive_87.574.csv` |
| 🥈 | Adaptive Confidence | 86.683% | `data/submissions/best/02_adaptive_confidence_86.683.csv` |
| 🥉 | Class-Specific Weighting | 86.638% | `data/submissions/best/03_class_specific_86.638.csv` |

**詳細歷史**: 見 [`CLAUDE.md`](CLAUDE.md) - 完整的項目記憶和策略分析

---

## 快速啟動 (新機器)

### 1. 環境準備

**系統需求**:
- Ubuntu 22.04+ (Linux)
- CUDA 12.1+ with RTX 4070 Ti SUPER (16GB VRAM)
- Python 3.10+
- 至少 20 GB 硬碟空間 (不含數據集)

**Python 依賴**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm pandas numpy Pillow tqdm scikit-learn pyyaml
```

### 2. 數據準備

**預期數據結構**:
```
data/
├── train_images/          # 訓練影像 (2,718 張)
├── val_images/            # 驗證影像 (679 張)
├── test_images/           # 測試影像 (1,182 張)
├── train.csv              # 訓練標籤
├── val.csv                # 驗證標籤
└── test.csv               # 測試列表
```

**Fold 數據** (5-Fold CV):
- `data/fold_0.csv` ~ `data/fold_4.csv`

### 3. 訓練最佳模型

**單一模型訓練** (EfficientNet-V2-S, 83.90% Test F1):
```bash
python train_breakthrough.py \
    --config configs/best/improved_breakthrough.yaml \
    --output_dir outputs/my_run
```

**DINOv2 訓練** (當前策略, 預期 89-90%+):
```bash
python train_dinov2_breakthrough.py \
    --fold 0 \
    --epochs 35 \
    --batch_size 6 \
    --img_size 518 \
    --output_dir outputs/dinov2_run
```

### 4. 生成預測

**單一模型預測**:
```bash
python src/predict_utils.py \
    --model_path outputs/my_run/best.pt \
    --output data/my_submission.csv
```

**集成預測** (推薦):
```bash
# 從最佳提交 CSV 創建加權集成
python scripts/ensemble/create_voting_ensemble.py
```

---

## 專案結構 (已重組)

```
nycu-CSIC30014-LAB3/
├── CLAUDE.md                          # 📖 專案記憶 (必讀!)
├── README.md                          # 本文件
├── BREAKTHROUGH_STRATEGY_ANALYSIS.md  # 突破策略分析
├── PROJECT_CLEANUP_PLAN.md            # 清理計劃記錄
│
├── data/
│   ├── submissions/best/              # ⭐ 前 6 名提交 CSV
│   ├── train_images/, val_images/, test_images/
│   ├── fold_*.csv                     # 5-Fold 分割
│   └── pseudo_labels/                 # 偽標籤數據
│
├── src/                               # 核心模組
│   ├── data.py                        # 數據加載
│   ├── models.py                      # 模型定義
│   ├── losses.py                      # Loss 函數
│   └── train_utils.py                 # 訓練工具
│
├── configs/
│   ├── best/                          # ✅ 最佳配置 (3 個)
│   ├── dinov2/                        # DINOv2 配置
│   └── archived/                      # 歸檔配置
│
├── scripts/
│   ├── train/                         # 訓練腳本
│   ├── predict/                       # 預測腳本
│   └── ensemble/                      # 集成腳本
│
├── outputs/
│   ├── dinov2_breakthrough/           # 🔥 當前訓練 (DINOv2 5-Fold)
│   └── best_models/                   # 預留最佳模型
│
├── logs/
│   ├── dinov2_full_training.log       # DINOv2 訓練日誌
│   └── dinov2_breakthrough/           # Per-fold 日誌
│
└── archive/                           # 歸檔區 (舊文件)
```

---

## 核心文件說明

### 1. 訓練腳本 (根目錄)

- **`train_breakthrough.py`**: 最佳單一模型訓練 (EfficientNet-V2-S, 83.90%)
- **`train_dinov2_breakthrough.py`**: DINOv2 突破訓練 (目標 90%+)
- **`train_champion_models.py`**: 大型模型集成訓練

### 2. 最佳配置

- **`configs/best/improved_breakthrough.yaml`**: 83.90% 配置
- **`configs/best/breakthrough_training.yaml`**: 原始突破配置
- **`configs/best/efficientnet_v2l_512_breakthrough.yaml`**: V2-L 大型模型

### 3. 數據文件

**頂級提交** (已複製到 `data/submissions/best/`):
1. `01_hybrid_adaptive_87.574.csv` - 智能偽標籤 + 自適應加權
2. `02_adaptive_confidence_86.683.csv` - 置信度動態加權
3. `03_class_specific_86.638.csv` - 類別特定權重
4. `04_champion_arch_85.800.csv` - 10 大模型架構集成
5. `05_champion_balanced_84.423.csv` - 三層 Stacking
6. `06_ensemble_017_84.19.csv` - Grid Search 優化

---

## 當前進行中的工作

### DINOv2 突破訓練 (目標 90%+)

**狀態**: 訓練中 (Fold 0-4, 8-10 小時)

**策略**:
- **模型**: Vision Transformer Base (vit_base_patch14_dinov2)
- **參數**: 86.6M
- **預訓練**: 142M 圖片自監督學習
- **預期提升**: +2-4% → **89.5-90.5%** Test F1

**監控訓練**:
```bash
# 查看主日誌
tail -f logs/dinov2_full_training.log

# 查看當前 fold 詳細日誌
tail -f logs/dinov2_breakthrough/fold*.log

# 檢查訓練進程
ps aux | grep dinov2
```

**訓練完成後**:
```bash
# 生成 5-Fold 集成預測並提交
python scripts/predict/generate_dinov2_predictions.py
```

---

## 下一步策略

### 如果 DINOv2 達到 90%+ ✅
1. 嘗試更大的 DINOv2 模型 (Large, Giant)
2. 結合 DINOv2 與現有最佳模型集成
3. 探索 Test-Time Augmentation (TTA)

### 如果 DINOv2 未達標 (< 89%) ⚠️
**備選方案** (詳見 `BREAKTHROUGH_STRATEGY_ANALYSIS.md`):
1. **CAPR Pseudo-labeling** (+2-3%) - 類別自適應偽標籤
2. **ConvNeXt V2** (+0.5-1.5%) - 新一代 CNN
3. **Contrastive Learning** (+1.5-2.5%) - 自監督對比學習

---

## 重要提醒

### 數據集不平衡
```
Normal:     906 (26.67%)
Bacteria: 1,581 (46.54%)
Virus:      876 (25.79%)
COVID-19:    34 (1.00%)  ⚠️ 極度稀缺
```

**應對策略**:
- Focal Loss with α=[1.0, 1.5, 2.0, 12.0]
- Class-specific ensemble weights
- 偽標籤重點增強 COVID-19 樣本

### 醫學影像特性
- **不要使用過度的醫學預處理** (CLAHE/Unsharp) - 破壞 ImageNet 預訓練特徵
- **保持高解析度** (384px+) - 醫學細節重要
- **TTA 需謹慎** - 水平翻轉會顛倒左右肺

---

## 快速檢查清單

### 新機器上手 (< 10 分鐘)

- [ ] 1. 複製專案到新機器
- [ ] 2. 閱讀本 README.md (5 分鐘)
- [ ] 3. 閱讀 [`CLAUDE.md`](CLAUDE.md) 關鍵部分 (當前狀態、最佳集成)
- [ ] 4. 檢查 `data/submissions/best/` 確認最佳結果
- [ ] 5. 安裝 Python 依賴
- [ ] 6. 檢查 DINOv2 訓練進度 (如果正在運行)
- [ ] 7. 決定下一步策略 (繼續 DINOv2 或啟動備選方案)

---

## 聯絡與支援

**Kaggle 競賽**: [CXR Multi-Label Classification](https://www.kaggle.com/competitions/cxr-multi-label-classification)

**競賽目標**: Top 5 (當前排名視最新提交而定)

**第一名分數**: 91.085% (距離 **3.511%**)

---

**🎯 讓我們一起突破 90%！**
