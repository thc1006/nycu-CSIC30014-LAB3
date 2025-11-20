# PDF 轉換指南

本文件提供將 `LAB3_REPORT.md` 轉換為 `LAB3_110263008_蔡秀吉.pdf` 的多種方法。

## 方法 1: 使用 Pandoc（推薦）

### 1.1 安裝 Pandoc

**Windows**:
- 下載安裝包: https://pandoc.org/installing.html
- 或使用 Chocolatey: `choco install pandoc`
- 或使用 Scoop: `scoop install pandoc`

**Linux**:
```bash
sudo apt-get install pandoc texlive-xetex
```

**Mac**:
```bash
brew install pandoc basictex
```

### 1.2 轉換為 PDF

```bash
pandoc LAB3_REPORT.md -o LAB3_110263008_蔡秀吉.pdf \
    --pdf-engine=xelatex \
    -V CJKmainfont="Microsoft YaHei" \
    -V geometry:margin=2.5cm \
    -V fontsize=11pt \
    --toc \
    --toc-depth=2 \
    --highlight-style=tango
```

---

## 方法 2: 使用線上工具

### 2.1 Markdown to PDF 線上轉換器

1. 打開 https://www.markdowntopdf.com/ 或 https://md2pdf.netlify.app/
2. 上傳 `LAB3_REPORT.md`
3. 選擇中文字體（Microsoft YaHei 或 SimSun）
4. 下載 PDF
5. 重新命名為 `LAB3_110263008_蔡秀吉.pdf`

### 2.2 其他線上工具

- https://cloudconvert.com/md-to-pdf
- https://www.zamzar.com/convert/md-to-pdf/

---

## 方法 3: 使用 Python 腳本（已提供）

### 3.1 安裝依賴

```bash
pip install reportlab
```

### 3.2 運行腳本

```bash
python create_pdf_reportlab.py
```

**或使用 Pandoc 腳本**:
```bash
python generate_pdf_simple.py
```

---

## 方法 4: 使用 Typora（最簡單）

### 4.1 安裝 Typora

下載地址: https://typora.io/

### 4.2 轉換步驟

1. 用 Typora 打開 `LAB3_REPORT.md`
2. 點擊 `File` → `Export` → `PDF`
3. 選擇輸出路徑並命名為 `LAB3_110263008_蔡秀吉.pdf`
4. 確保選擇中文字體支持

---

## 方法 5: 使用 VS Code + Markdown PDF 擴展

### 5.1 安裝擴展

在 VS Code 中安裝 "Markdown PDF" 擴展

### 5.2 配置（可選）

在 `settings.json` 中添加：
```json
{
    "markdown-pdf.styles": [
        "https://cdn.jsdelivr.net/npm/github-markdown-css/github-markdown.css"
    ],
    "markdown-pdf.headerTemplate": "<div style='font-size:9px; text-align:center;'><span class='pageNumber'></span>/<span class='totalPages'></span></div>",
    "markdown-pdf.margin": {
        "top": "2.5cm",
        "bottom": "2.5cm",
        "left": "2.5cm",
        "right": "2.5cm"
    }
}
```

### 5.3 轉換

1. 打開 `LAB3_REPORT.md`
2. 按 `Ctrl+Shift+P`（Mac: `Cmd+Shift+P`）
3. 輸入 "Markdown PDF: Export (pdf)"
4. 等待轉換完成
5. 重新命名輸出檔案

---

## 方法 6: 使用 Microsoft Word

### 6.1 轉換為 Word

```bash
pandoc LAB3_REPORT.md -o LAB3_REPORT.docx
```

或線上工具: https://www.markdowntoword.com/

### 6.2 Word 轉 PDF

1. 在 Word 中打開 `.docx` 檔案
2. 調整格式和字體
3. `檔案` → `另存為` → 選擇 PDF 格式
4. 命名為 `LAB3_110263008_蔡秀吉.pdf`

---

## 推薦方案

| 方法 | 難度 | 質量 | 推薦指數 |
|------|------|------|----------|
| Pandoc | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Typora | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 線上工具 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Python 腳本 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| VS Code | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Word | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**最快**: Typora 或線上工具（<5 分鐘）
**最佳質量**: Pandoc（需要安裝，但輸出最專業）
**最靈活**: Word（可手動調整格式）

---

## 注意事項

1. **檔名要求**: 必須命名為 `LAB3_110263008_蔡秀吉.pdf`
2. **字體支持**: 確保系統有中文字體（Microsoft YaHei, SimHei, SimSun 等）
3. **頁邊距**: 建議設置為 2.5cm
4. **字體大小**: 11pt 正文，18pt 大標題
5. **目錄**: 如果使用 Pandoc，會自動生成目錄

---

## 故障排除

### Q: Pandoc 顯示中文亂碼
A: 使用 XeLaTeX 引擎並指定中文字體:
```bash
pandoc input.md -o output.pdf --pdf-engine=xelatex -V CJKmainfont="Microsoft YaHei"
```

### Q: Python 腳本無法運行
A: 安裝缺少的依賴:
```bash
pip install reportlab markdown
```

### Q: PDF 檔案太大
A: 使用 Pandoc 時添加 `--dpi=150` 降低 DPI

### Q: 代碼格式不正確
A: 確保使用等寬字體（Courier, Monaco, Consolas）

---

## 聯絡方式

如果遇到問題，請參考:
- Pandoc 文檔: https://pandoc.org/MANUAL.html
- Typora 支持: https://support.typora.io/
