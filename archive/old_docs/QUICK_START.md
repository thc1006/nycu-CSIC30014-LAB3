# ⚡ 快速開始指南

## 🎯 目標
從 **80 分** 提升到 **90 分以上**

## 📦 準備好的內容
- ✅ 5 個優化實驗配置
- ✅ 自動化訓練腳本
- ✅ Ensemble 腳本
- ✅ 詳細 TODO 清單

## 🚀 一鍵啟動（推薦）

### Windows:
```bash
START_OVERNIGHT_TRAINING.bat
```

### Python:
```bash
python run_all_experiments.py
```

## 📋 實驗概覽

| # | 模型 | 解析度 | Epochs | 時間 | 預期分數 |
|---|------|--------|--------|------|----------|
| 1 | ConvNeXt-Tiny | 288px | 25 | 2.5h | 83-85% |
| 2 | EfficientNetV2-S | 320px | 30 | 3h | 84-86% |
| 3 | ResNet34 | 384px | 35 | 2h | 85-87% |
| 4 | EfficientNet-B0 | 256px | 40 | 2.5h | 84-86% |
| 5 | ResNet18 | 384px | 50 | 1.5h | 83-85% |

**總時間**: ~11-12 小時
**Ensemble 預期**: 87-92%

## 📝 詳細文檔
- `OVERNIGHT_TODO.md` - 完整的待辦清單
- `STRATEGY_SUMMARY.md` - 策略詳解

## 🎉 完成後
```bash
python ensemble.py
```

生成檔案：
- `submission_ensemble_soft.csv` (推薦)
- `submission_ensemble_hard.csv` (備選)
