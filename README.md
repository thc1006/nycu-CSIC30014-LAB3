# 胸部 X 光分類項目 - 快速啟動指南

**專案狀態**: 當前最佳 **88.564%** Macro-F1 (ULTRA_PATTERN_MATCHING.csv)

**最後更新**: 2025-11-21 (專案已大規模清理重組)

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
| 🥇 | **ULTRA Pattern Matching** | **88.564%** | `data/submissions/best/ULTRA_PATTERN_MATCHING.csv` |
| 🥈 | Hybrid Adaptive Ensemble | 87.574% | `data/submissions/best/01_hybrid_adaptive_87.574.csv` |
| 🥉 | Adaptive Confidence | 86.683% | `data/submissions/best/02_adaptive_confidence_86.683.csv` |

**詳細歷史**: 見 [`CLAUDE.md`](CLAUDE.md) - 完整的項目記憶和策略分析

---

## 專案結構 (✨ 全新整理)

```
nycu-CSIC30014-LAB3/
├── 📖 核心文檔
│   ├── README.md                          # 本文件 - 快速啟動指南
│   ├── CLAUDE.md                          # 專案記憶 (必讀!)
│   ├── Lab3.md                            # 作業規格
│   ├── LAB3_REPORT.md                     # 實驗報告 (Markdown)
│   ├── LAB3_110263008_蔡秀吉.pdf           # 實驗報告 (PDF)
│   └── LICENSE                            # 授權條款
│
├── 📊 數據文件
│   ├── train_images/                      # 訓練影像 (2,718 張)
│   ├── val_images/                        # 驗證影像 (679 張)
│   ├── test_images/                       # 測試影像 (1,182 張)
│   └── data/                              # 數據標籤與提交結果
│       ├── submissions/
│       │   ├── best/                      # ⭐ 前3名提交 CSV
│       │   └── archived/                  # 歷史提交記錄
│       ├── train_data.csv                 # 訓練標籤
│       ├── val_data.csv                   # 驗證標籤
│       └── fold_*.csv                     # 5-Fold CV 分割
│
├── 💻 核心代碼
│   ├── src/                               # 核心模組
│   │   ├── data.py                        # 數據加載
│   │   ├── models.py                      # 模型定義
│   │   ├── losses.py                      # Loss 函數
│   │   └── train_utils.py                 # 訓練工具
│   ├── configs/                           # 配置文件
│   │   ├── best/                          # ✅ 最佳配置
│   │   └── archived/                      # 歷史配置
│   ├── scripts/                           # 輔助腳本
│   │   ├── train/                         # 訓練腳本
│   │   ├── predict/                       # 預測腳本
│   │   └── ensemble/                      # 集成腳本
│   └── 🚀 主要訓練腳本 (根目錄)
│       ├── train_breakthrough.py           # 最佳單一模型訓練
│       └── train_dinov2_breakthrough.py    # DINOv2 訓練
│
├── 📦 輸出與工具
│   ├── outputs/                           # 訓練輸出 (模型檢查點)
│   ├── convert_to_pdf_fixed.py            # PDF 生成工具
│   ├── PDF_CONVERSION_INSTRUCTIONS.md    # PDF 轉換指南
│   └── kaggle.json                        # Kaggle API 憑證
│
└── 📂 歸檔區
    └── archive/                           # 舊文件歸檔
        ├── old_docs/                      # 歷史文檔
        ├── old_scripts/                   # 歷史腳本
        └── old_notebooks/                 # Jupyter notebooks
```

### 清理成果

- ✅ **從 146 個文件減少到 22 個核心文件** (85% 減少!)
- ✅ **移除**: 臨時文件、日誌、重複 CSV
- ✅ **歸檔**: 60+ 個舊腳本、20+ 個舊文檔、5 個 Jupyter notebooks
- ✅ **組織**: 提交結果統一到 `data/submissions/`

---

## 快速啟動 (新機器)

### 1. 環境準備

**系統需求**:
- Windows 10/11 或 Ubuntu 22.04+
- CUDA 12.1+ with GPU (建議 RTX 4070 Ti SUPER 16GB)
- Python 3.10+
- 至少 20 GB 硬碟空間

**Python 依賴**:
```bash
pip install -r requirements.txt
# 或手動安裝核心套件:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm pandas numpy Pillow tqdm scikit-learn pyyaml
```

### 2. 數據準備

**影像數據已存在於根目錄**:
- `train_images/` - 訓練影像 (2,718 張)
- `val_images/` - 驗證影像 (679 張)
- `test_images/` - 測試影像 (1,182 張)

**標籤文件位於 `data/` 目錄**:
- `data/train_data.csv`
- `data/val_data.csv`
- `data/fold_*.csv` (5-Fold CV)

### 3. 訓練最佳模型

**單一模型訓練** (EfficientNet-V2-S, 83.90% Test F1):
```bash
python train_breakthrough.py \
    --config configs/best/improved_breakthrough.yaml \
    --output_dir outputs/my_run
```

**訓練時間**: 約 25-30 分鐘 (RTX 4070 Ti SUPER)

### 4. 生成預測

**使用訓練好的模型**:
```bash
python src/predict_utils.py \
    --model_path outputs/my_run/best.pt \
    --output data/my_submission.csv
```

### 5. 提交至 Kaggle

```bash
# 設置 Kaggle API (首次使用)
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 提交最佳結果
kaggle competitions submit -c cxr-multi-label-classification \
    -f data/submissions/best/ULTRA_PATTERN_MATCHING.csv \
    -m "Best submission - 88.564%"
```

---

## 核心文件說明

### 1. 訓練腳本

| 文件 | 說明 | 模型 | Test F1 |
|------|------|------|---------|
| `train_breakthrough.py` | 最佳單一模型訓練 | EfficientNet-V2-S | 83.90% |
| `train_dinov2_breakthrough.py` | DINOv2 突破訓練 | Vision Transformer | 86.70% |

### 2. 最佳配置

| 文件 | 說明 | 關鍵特徵 |
|------|------|----------|
| `configs/best/improved_breakthrough.yaml` | 83.90% 配置 | 移除醫學預處理 + 強化 Mixup |
| `configs/best/breakthrough_training.yaml` | 原始突破配置 | Focal Loss α=12.0 for COVID-19 |

### 3. 最佳提交結果

| 文件 | Test F1 | 關鍵策略 |
|------|---------|----------|
| `data/submissions/best/ULTRA_PATTERN_MATCHING.csv` | **88.564%** | 超強模式匹配集成 |
| `data/submissions/best/01_hybrid_adaptive_87.574.csv` | 87.574% | 智能偽標籤 + 自適應加權 |
| `data/submissions/best/02_adaptive_confidence_86.683.csv` | 86.683% | 置信度動態加權 |

### 4. 實用工具

- **`convert_to_pdf_fixed.py`**: Markdown → PDF 轉換器 (生成實驗報告 PDF)
- **`PDF_CONVERSION_INSTRUCTIONS.md`**: PDF 轉換完整指南
- **`REPORT_COVERAGE_ANALYSIS.md`**: 報告內容完整性分析

---

## 實驗報告

### 生成 PDF 報告

```bash
# 從 Markdown 生成 PDF
python convert_to_pdf_fixed.py
```

**輸出**: `LAB3_110263008_蔡秀吉.pdf` (287 KB)

**報告內容**:
- Introduction (5%): 任務介紹
- Implementation Details (20%): 模型與數據加載
- Strategy Design (50%): 預處理、訓練策略、超參數
- Discussion (20%): 實驗發現與教訓
- Github Link (5%): https://github.com/thc1006/nycu-CSIC30014-LAB3

---

## 重要提醒

### 數據集不平衡

```
Normal:     906 (26.67%)
Bacteria: 1,581 (46.54%)
Virus:      876 (25.79%)
COVID-19:    34 (1.00%)  ⚠️ 極度稀缺 (1:46.5 比例)
```

**應對策略**:
- ✅ Focal Loss with α=[1.0, 1.5, 2.0, 12.0] (COVID-19 加權 12 倍)
- ✅ Weighted Random Sampling (COVID-19 採樣權重 33×)
- ✅ Class-specific ensemble weights (每類獨立優化)
- ✅ 偽標籤重點增強稀缺類別

### 醫學影像特性

- ❌ **不要使用過度的醫學預處理** (CLAHE/Unsharp Masking) - 破壞 ImageNet 預訓練特徵 (-3.29%)
- ✅ **保持高解析度** (384px+) - 醫學細節重要
- ❌ **TTA 需謹慎** - 水平翻轉會顛倒左右肺 (心臟位置錯誤) (-2.48%)

---

## 關鍵成功因素

### 1. Class-Specific Ensemble (+4.48%)

**創新策略**: 為每個類別設定不同的模型權重

```python
class_weights = {
    'normal':    [0.50, 0.50],  # 兩個模型各 50%
    'bacteria':  [0.60, 0.40],  # EfficientNet 為主 (局灶性實變)
    'virus':     [0.40, 0.60],  # Swin-Large 為主 (間質性模式)
    'covid19':   [0.70, 0.30]   # 大幅偏向 EfficientNet (周邊 GGO)
}
```

**效果**: 從 84.09% → **88.564%** (+4.48%)

### 2. 強化 Focal Loss

```yaml
loss: improved_focal
focal_alpha: [1.0, 1.5, 2.0, 12.0]  # COVID-19 權重 12 倍
focal_gamma: 3.5                     # 高 γ 值抑制易分類樣本
label_smoothing: 0.12                # 防止過擬合
```

### 3. 移除有害的醫學預處理

**發現**: CLAHE + Unsharp Masking 破壞 ImageNet 預訓練特徵
**效果**: 移除後提升 +3.29% (80.61% → 83.90%)

### 4. 高解析度訓練

- **384×384** (最佳) - 保留醫學細節
- ~~352×352~~ (損失太多資訊)
- ~~224×224~~ (完全不適合)

---

## 常見問題

### Q: 如何重現最佳結果？

A: 最佳結果 (88.564%) 來自集成學習，需要訓練多個模型:

```bash
# 1. 訓練 EfficientNet-V2-S (83.90%)
python train_breakthrough.py --config configs/best/improved_breakthrough.yaml

# 2. 訓練 DINOv2 (86.70%)
python train_dinov2_breakthrough.py --fold 0 --epochs 35

# 3. 使用 Class-Specific Ensemble (見 CLAUDE.md)
```

### Q: 為什麼醫學預處理會降低性能？

A: 因為模型使用 ImageNet 預訓練權重，期望自然影像的分布。CLAHE 和銳化會過度增強對比度，破壞預訓練特徵的分布，導致性能下降。

### Q: COVID-19 只有 34 張訓練樣本，如何提升？

A: 三層策略:
1. **Focal Loss**: α=12.0 大幅加權
2. **Weighted Sampling**: 33× 採樣權重
3. **Class-Specific Ensemble**: 針對 COVID-19 優化權重

### Q: TTA (測試時增強) 為何有害？

A: 胸部 X 光有解剖學不對稱性 (心臟在左側)，水平翻轉會產生非生理影像 (心臟在右側)，導致模型混淆，性能下降 -2.48%。

---

## 歸檔說明

為保持專案整潔，以下文件已移至 `archive/`:

### archive/old_docs/ (20+ 文檔)
- 歷史突破策略分析
- 實驗計劃文檔
- 狀態報告

### archive/old_scripts/ (60+ 腳本)
- 實驗性訓練腳本
- 分析工具腳本
- Shell 自動化腳本

### archive/old_notebooks/ (5 個 notebooks)
- Colab 訓練 notebooks
- 實驗性 notebooks

**註**: 如需查看歷史實驗細節，請查閱 `archive/` 目錄或 `CLAUDE.md` 文檔。

---

## 聯絡與支援

**Kaggle 競賽**: [CXR Multi-Label Classification](https://www.kaggle.com/competitions/cxr-multi-label-classification)

**Github Repository**: https://github.com/thc1006/nycu-CSIC30014-LAB3

**競賽目標**: 維持 Top 5

**當前排名**: 視最新提交而定

**第一名分數**: 91.085% (距離 **2.521%**)

---

**🎯 專案已完成並提交！最佳成績: 88.564%**
