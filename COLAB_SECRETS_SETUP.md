# 🔑 Google Colab Secrets 設置指南

## 為什麼使用 Secrets？

✅ **安全**: API key 不會出現在 notebook 中，不會被推送到公開 repo
✅ **方便**: 只需設置一次，所有授權的 notebook 都可以使用
✅ **簡單**: 不需要每次都上傳 `kaggle.json`

---

## 📋 設置步驟

### 1. 在 Google Colab 中添加 Secret

1. 打開任一 Colab Notebook
2. 點擊左側邊欄的 **🔑 圖標** (Secrets)
3. 點擊 **"+ Add new secret"**
4. 填寫:
   - **Name (名稱)**: `KAGGLE_KEY`
   - **Value (值)**: 你的 Kaggle API key (從 kaggle.json 中複製 `"key"` 的值)
5. 點擊 **"Add secret"**

### 2. 獲取 Kaggle API Key

如果你還沒有 Kaggle API key:

1. 登入 [Kaggle](https://www.kaggle.com/)
2. 點擊右上角頭像 → **Account**
3. 滾動到 **API** 區塊
4. 點擊 **"Create New API Token"**
5. 下載的 `kaggle.json` 格式如下:
   ```json
   {
     "username": "thc1006",
     "key": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   }
   ```
6. 複製 `"key"` 的值 (不包含引號)

---

## 🚀 在 Notebook 中使用

**已經配置好了！** 兩個 Notebook 會自動:

1. 從 Colab Secrets 讀取 `KAGGLE_KEY`
2. 自動創建 `kaggle.json` (username 寫死為 `thc1006`)
3. 配置 Kaggle API

**執行 Cell 時會看到**:
```
🔑 從 Colab Secrets 讀取 Kaggle API...
✅ 成功讀取 KAGGLE_KEY
✅ Kaggle API 配置完成 (username: thc1006)
```

---

## ⚠️ 故障排除

### 錯誤: "無法讀取 KAGGLE_KEY"

**原因**: Colab Secrets 中沒有設置 `KAGGLE_KEY`

**解決方法**:
1. 檢查左側邊欄 🔑 Secrets 中是否有 `KAGGLE_KEY`
2. 確認名稱拼寫正確 (區分大小寫)
3. 確認 notebook 有權限訪問該 secret

### 錯誤: "401 Unauthorized"

**原因**: API key 錯誤或過期

**解決方法**:
1. 在 Kaggle Account 頁面重新生成 API Token
2. 更新 Colab Secrets 中的 `KAGGLE_KEY` 值

---

## 🔒 安全最佳實踐

✅ **DO**:
- 使用 Colab Secrets 存儲敏感資訊
- 定期更新 API key
- 只授權信任的 notebook 訪問 secrets

❌ **DON'T**:
- 將 API key 直接寫在 notebook 程式碼中
- 將 `kaggle.json` 推送到公開 GitHub repo
- 在公開場合分享 API key

---

## 📝 修改內容

### 修改前 (Cell 6):
```python
from google.colab import files

print("📤 請上傳 kaggle.json...")
uploaded = files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### 修改後 (Cell 6):
```python
from google.colab import userdata
import json

print("🔑 從 Colab Secrets 讀取 Kaggle API...")

# 從 Colab Secrets 讀取 API key
kaggle_key = userdata.get('KAGGLE_KEY')

# 創建 kaggle.json
kaggle_config = {
    "username": "thc1006",
    "key": kaggle_key
}

!mkdir -p ~/.kaggle

with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump(kaggle_config, f)

!chmod 600 ~/.kaggle/kaggle.json

print("✅ Kaggle API 配置完成")
```

---

## 📚 相關資源

- [Google Colab Secrets 官方文檔](https://colab.research.google.com/notebooks/secrets.ipynb)
- [Kaggle API 文檔](https://github.com/Kaggle/kaggle-api)
- [項目 GitHub](https://github.com/thc1006/nycu-CSIC30014-LAB3)

---

**最後更新**: 2025-11-16
**適用 Notebook**: Colab_A100_AGGRESSIVE.ipynb, Colab_L4_OPTIMIZED.ipynb
