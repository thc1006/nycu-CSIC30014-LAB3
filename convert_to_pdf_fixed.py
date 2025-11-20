#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡單的 Markdown 到 PDF 轉換器
使用 reportlab 生成專業的 PDF 報告
"""
import sys
import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

def register_chinese_font():
    """註冊中文字體"""
    fonts_to_try = [
        ('msyh', r'C:\Windows\Fonts\msyh.ttc'),
        ('simsun', r'C:\Windows\Fonts\simsun.ttc'),
    ]

    for font_name, font_path in fonts_to_try:
        try:
            if Path(font_path).exists():
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                print(f"[OK] Using font: {font_name}")
                return font_name
        except:
            continue

    print("[WARNING] No Chinese font found, using Helvetica")
    return 'Helvetica'

def create_styles(chinese_font):
    """創建樣式"""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Heading1'],
        fontName=chinese_font,
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER,
        leading=30
    ))

    styles.add(ParagraphStyle(
        name='H1',
        parent=styles['Heading1'],
        fontName=chinese_font,
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='H2',
        parent=styles['Heading2'],
        fontName=chinese_font,
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=15,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='H3',
        parent=styles['Heading3'],
        fontName=chinese_font,
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=10,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='Body',
        parent=styles['Normal'],
        fontName=chinese_font,
        fontSize=11,
        leading=16,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    ))

    styles.add(ParagraphStyle(
        name='ListItem',
        parent=styles['Body'],
        leftIndent=20,
        bulletIndent=10
    ))

    styles.add(ParagraphStyle(
        name='CustomCode',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=9,
        leftIndent=20,
        textColor=colors.HexColor('#333333'),
        backColor=colors.HexColor('#f8f8f8'),
        borderPadding=10
    ))

    return styles

def parse_markdown_line(line, styles):
    """解析單行 Markdown"""
    line = line.rstrip()

    if line.startswith('# '):
        if 'LAB3' in line and ('實驗報告' in line or '报告' in line):
            return ('para', Paragraph(line[2:].strip(), styles['MainTitle']))
        return ('h1', Paragraph(line[2:].strip(), styles['H1']))
    elif line.startswith('## '):
        return ('h2', Paragraph(line[3:].strip(), styles['H2']))
    elif line.startswith('### '):
        return ('h3', Paragraph(line[4:].strip(), styles['H3']))
    elif line.startswith('#### '):
        return ('h3', Paragraph(line[5:].strip(), styles['H3']))

    if not line.strip():
        return ('spacer', Spacer(1, 0.2*cm))

    if line.strip().startswith('- ') or line.strip().startswith('* '):
        text = line.strip()[2:]
        return ('para', Paragraph(f'• {text}', styles['ListItem']))

    if line.strip() and line.strip()[0].isdigit() and '. ' in line[:5]:
        return ('para', Paragraph(line.strip(), styles['ListItem']))

    if line.strip() in ['---', '***', '___']:
        return ('spacer', Spacer(1, 0.3*cm))

    if line.strip():
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
        text = re.sub(r'`(.+?)`', r'<font name="Courier" color="#d14">\1</font>', text)
        return ('para', Paragraph(text, styles['Body']))

    return None

def convert_md_to_pdf(md_file, pdf_file):
    """將 Markdown 轉換為 PDF"""
    print(f"[1/4] Reading: {md_file}")

    with open(md_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"[2/4] Creating PDF: {pdf_file}")

    chinese_font = register_chinese_font()
    styles = create_styles(chinese_font)

    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=A4,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm,
        leftMargin=2.5*cm,
        rightMargin=2.5*cm
    )

    story = []
    in_code_block = False
    code_lines = []

    for i, line in enumerate(lines):
        if line.strip().startswith('```'):
            if in_code_block:
                if code_lines:
                    code_text = '\n'.join(code_lines)
                    if len(code_text) > 2000:
                        code_text = code_text[:2000] + '\n... (code truncated)'
                    story.append(Preformatted(code_text, styles['CustomCode']))
                    story.append(Spacer(1, 0.3*cm))
                code_lines = []
            in_code_block = not in_code_block
            continue

        if in_code_block:
            code_lines.append(line.rstrip())
            continue

        result = parse_markdown_line(line, styles)
        if result:
            element_type, element = result

            if element_type == 'h1' and i > 10:
                story.append(PageBreak())

            story.append(element)

    print("[3/4] Building PDF document...")
    doc.build(story)

    pdf_path = Path(pdf_file)
    size_kb = pdf_path.stat().st_size / 1024
    print(f"[4/4] Success! PDF generated")
    print(f"      File: {pdf_file}")
    print(f"      Size: {size_kb:.1f} KB")

    return True

def main():
    print("=" * 60)
    print("LAB3 Report PDF Converter")
    print("=" * 60)
    print()

    input_md = Path("LAB3_REPORT.md")
    output_pdf = Path("LAB3_110263008_蔡秀吉.pdf")

    if not input_md.exists():
        print(f"[ERROR] Cannot find {input_md}")
        return 1

    try:
        convert_md_to_pdf(input_md, output_pdf)
        print()
        print("[DONE] Please check the generated PDF file")
        return 0
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
