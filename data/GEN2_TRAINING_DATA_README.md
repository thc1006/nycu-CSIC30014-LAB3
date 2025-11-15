
# Gen2 訓練數據說明

## 數據組成
每個 fold 的訓練集包含：
1. 原始訓練數據 (~3,520 張) - 來自 data/train/
2. 測試集高置信度偽標籤 (532 張) - 來自 test_images/

## 總計
- 每個 fold: ~4,052 張影像
- 數據增加: +15.1%

## 訓練配置調整需求
訓練腳本需要支持從兩個目錄載入影像：
- 原始訓練影像: train_images/ (或 data/train/)
- 測試影像: test_images/

可以在 Dataset __getitem__ 中添加邏輯：
```python
if filename in test_image_list:
    path = os.path.join('test_images', filename)
else:
    path = os.path.join('data/train', filename)
```

或者更簡單：先檢查 test_images/，不存在則用 data/train/
