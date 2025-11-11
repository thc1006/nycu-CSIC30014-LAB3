# 專案目錄整理總結
**日期**: 2025-11-11
**清理完成**: 從 85+ 檔案 → 7 個核心檔案

---

## 整理結果

### 根目錄 (僅 7 個重要檔案)
```
✓ CLAUDE.md                    Claude 對話記錄
✓ GIT_CLEANUP_SUMMARY.md        Git 清理總結 (6K+ 檔案問題)
✓ Lab3.md                       作業說明
✓ PROGRESS_REPORT.md            最新進度報告 (84.190% 最佳成績)
✓ QUICK_REFERENCE.txt           快速參考指令
✓ README.md                     專案說明
✓ requirements.txt              Python 依賴
```

**從 85 → 7 檔案 (減少 91.8%)**

---

## 新建目錄結構

### 📁 scripts/ (11 個當前使用的腳本)

#### scripts/ensemble/ (8 個 ensemble 腳本)
- `mega_ensemble_tta.py` - 12 模型 MEGA 融合 (83.999%)
- `grid_search_ensemble.py` - 網格搜尋權重優化 (84.190% ⭐)
- `create_ultimate_ensemble.py` - 終極融合框架
- `ensemble_diverse_models.py` - 多樣化模型融合
- `simple_ensemble.py` - 簡單融合
- `soft_ensemble.py` - Soft voting 融合
- `ultimate_smart_ensemble.py` - 智能融合
- `quick_ensemble.py` - 快速融合

#### scripts/preprocessing/ (2 個預處理腳本)
- `preprocess_clahe_fast.py` - CLAHE 影像預處理
- `generate_pseudo_labels.py` - 偽標籤生成

#### scripts/ (1 個主腳本)
- `ultimate_auto_91plus.sh` - 91%+ 自動化訓練管線

---

### 📦 archive/ (47 個歸檔檔案)

#### archive/old_docs/ (17 個舊文檔)
- `OVERNIGHT_TODO.md`
- `REACH_90_PERCENT.md`
- `RUN_STAGE1.md`
- `START_HERE.md`
- `QUICK_START.md`
- `COLAB_DEBUG_INSTRUCTIONS.md`
- `ENSEMBLE_STRATEGY_90.md`
- `IMPROVEMENT_STRATEGY.md`
- `FINAL_SUMMARY.md`
- `RESULTS_ANALYSIS.md`
- `COVID19_影像特徵分析報告.md`
- `STATUS_REPORT.md`
- `STRATEGY_SUMMARY.md`
- `UPGRADE_TO_90_PERCENT.md`
- `VIT_FAILURE_ANALYSIS.md`
- `影像資料深度分析報告.md`
- `项目完整分析报告.md`

#### archive/old_analysis/ (11 個舊分析腳本)
- `analyze_dataset.py`
- `analyze_dataset_simple.py`
- `analyze_dataset_visual.py`
- `analyze_images_deep.py`
- `analyze_images_statistics.py`
- `analyze_predictions.py`
- `check_progress.py`
- `comprehensive_medical_analysis.py`
- `fix_data_split.py`
- `fix_csv_quickly.py`
- `run_all_experiments.py`

#### archive/old_scripts/ (19 個舊訓練/融合腳本)
- `ensemble.py`
- `create_ensemble.py`
- `ensemble_predictions.py`
- `ensemble_probabilities.py`
- `convert_to_label.py`
- `convert_to_onehot.py`
- `auto_analyze_and_train.sh`
- `master_pipeline.sh`
- `max_gpu_train.sh`
- `final_train.sh`
- `monitor_training.sh`
- `pseudo_labeling.py`
- `ULTRA_BREAKTHROUGH_PIPELINE.sh`
- `test_gpu_training.py`
- `test_stage1.py`
- `test_training_init.py`
- `train_kfold_cv.py`
- `ultimate_gpu_train.sh`
- `ultra_deep_analysis.py`

---

## 檔案統計

```
整理前: 85+ 檔案混亂分布於根目錄
整理後:
├─ 根目錄:    7 個重要檔案 (91.8% 減少)
├─ scripts/:  11 個當前使用腳本
└─ archive/:  47 個歸檔舊檔案
```

---

## 使用當前腳本

### 最佳 Ensemble 腳本:
```bash
# 網格搜尋最優權重 (當前最佳: 84.190%)
python3 scripts/ensemble/grid_search_ensemble.py

# MEGA 融合 (12 models + TTA)
python3 scripts/ensemble/mega_ensemble_tta.py
```

### 預處理腳本:
```bash
# CLAHE 影像增強
python3 scripts/preprocessing/preprocess_clahe_fast.py

# 偽標籤生成
python3 scripts/preprocessing/generate_pseudo_labels.py
```

### 自動化訓練:
```bash
# 91%+ 全自動訓練管線
bash scripts/ultimate_auto_91plus.sh
```

---

## 查看歸檔檔案

### 舊文檔:
```bash
ls archive/old_docs/
cat archive/old_docs/REACH_90_PERCENT.md  # 查看歷史策略
```

### 舊腳本:
```bash
ls archive/old_analysis/
ls archive/old_scripts/
```

---

## 當前最佳成績

**84.190%** (ensemble_017)
- 配置: 47.6% ultimate_final + 28.6% mega + 19.0% ultimate_smart + 4.8% improved
- 位置: `data/grid_search_submissions/ensemble_017.csv`
- 腳本: `scripts/ensemble/grid_search_ensemble.py`

---

## 目錄維護建議

1. **根目錄只保留核心文檔** - 7 個檔案已足夠
2. **新實驗腳本放 scripts/** - 依功能分類 (ensemble/preprocessing)
3. **舊檔案移 archive/** - 保留但不影響工作目錄
4. **定期清理 outputs/** - 舊訓練記錄和模型可定期刪除

---

**整理完成時間**: 2025-11-11 19:30
**整理效果**: 目錄乾淨清晰，易於導航和管理
