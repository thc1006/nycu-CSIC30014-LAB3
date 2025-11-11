# Colab 調試指令

## 問題：FileNotFoundError: 'train_images/1643.jpeg'

請在 Colab 的 **Cell 12 之後** 插入新的 code cell，運行以下調試代碼：

```python
import os
import pandas as pd

print("="*80)
print("DEBUG: 檢查文件結構")
print("="*80)

# 1. 檢查當前目錄
print(f"\n當前工作目錄: {os.getcwd()}")

# 2. 列出根目錄內容
print("\n根目錄內容:")
for item in sorted(os.listdir('.')):
    item_type = "DIR" if os.path.isdir(item) else "FILE"
    print(f"  [{item_type}] {item}")

# 3. 檢查 train_images 是否存在
if os.path.exists('train_images'):
    print(f"\n[OK] train_images 存在")

    # 列出前10個文件
    files = sorted([f for f in os.listdir('train_images') if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"train_images 中有 {len(files)} 個圖片")
    print(f"前10個文件: {files[:10]}")

    # 檢查 1643.jpeg 是否存在
    if '1643.jpeg' in files:
        print("\n[OK] 1643.jpeg 存在於 train_images/")
    else:
        print("\n[ERROR] 1643.jpeg 不在 train_images/")
        # 搜尋所有目錄
        print("\n搜尋 1643.jpeg...")
        for root, dirs, files_in_dir in os.walk('.'):
            if '1643.jpeg' in files_in_dir:
                print(f"  找到: {os.path.join(root, '1643.jpeg')}")
else:
    print("\n[ERROR] train_images 目錄不存在！")

# 4. 檢查 CSV 檔案內容
print("\n" + "="*80)
print("檢查 CSV 路徑")
print("="*80)

train_csv = 'data/train_data.csv'
if os.path.exists(train_csv):
    df = pd.read_csv(train_csv)
    print(f"\n[OK] {train_csv} 存在")
    print(f"總共 {len(df)} 筆資料")
    print(f"\nCSV 欄位: {df.columns.tolist()}")
    print(f"\n前5筆資料:")
    print(df.head())

    # 檢查 1643.jpeg 在 CSV 中
    if 'new_filename' in df.columns:
        if '1643.jpeg' in df['new_filename'].values:
            row = df[df['new_filename'] == '1643.jpeg'].iloc[0]
            print(f"\n[OK] 1643.jpeg 在 CSV 中")
            print(f"資料: {row.to_dict()}")
        else:
            print("\n[ERROR] 1643.jpeg 不在 CSV 的 new_filename 欄位中")
            print(f"CSV 中的檔名範例: {df['new_filename'].head(10).tolist()}")

print("\n" + "="*80)
```

## 預期結果

運行後，請提供完整輸出，特別是：
1. 當前工作目錄路徑
2. train_images/ 中的檔案數量和範例
3. 1643.jpeg 是否存在
4. CSV 中的檔名格式

這將幫助我們確定問題所在。

## 可能的問題

1. **圖片不在 train_images/**：可能在其他子目錄
2. **檔名不匹配**：CSV 中是 `1643.jpeg` 但實際檔名可能不同
3. **路徑問題**：相對路徑配置錯誤

提供輸出後我會立即修復！
