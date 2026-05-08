---
title: "Ocr And Documents — 从 PDF/扫描件提取文本（pymupdf、marker-pdf）"
sidebar_label: "Ocr And Documents"
description: "从 PDF/扫描件提取文本（pymupdf、marker-pdf）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Ocr And Documents

从 PDF/扫描件提取文本（pymupdf、marker-pdf）。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/ocr-and-documents` |
| 版本 | `2.3.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `PDF`, `文档`, `研究`, `Arxiv`, `文本提取`, `OCR` |
| 相关技能 | [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是技能激活时代理看到的指令内容。
:::

# PDF 与文档提取

对于 DOCX：使用 `python-docx`（解析实际文档结构，远优于 OCR）。
对于 PPTX：参见 `powerpoint` 技能（使用 `python-pptx` 并完全支持幻灯片/备注）。
此技能涵盖 **PDF 和扫描文档**。

## 步骤 1：是否有远程 URL？

如果文档有 URL，**始终先尝试 `web_extract`**：

```
web_extract(urls=["https://arxiv.org/pdf/2402.03300"])
web_extract(urls=["https://example.com/report.pdf"])
```

这通过 Firecrawl 处理 PDF 转 Markdown，无需本地依赖。

仅在以下情况使用本地提取：文件是本地的、web_extract 失败，或你需要批量处理。

## 步骤 2：选择本地提取器

| 特性 | pymupdf (~25MB) | marker-pdf (~3-5GB) |
|------|-----------------|---------------------|
| **文本型 PDF** | ✅ | ✅ |
| **扫描 PDF (OCR)** | ❌ | ✅（90+ 语言） |
| **表格** | ✅（基本） | ✅（高准确度） |
| **公式 / LaTeX** | ❌ | ✅ |
| **代码块** | ❌ | ✅ |
| **表单** | ❌ | ✅ |
| **页眉/页脚移除** | ❌ | ✅ |
| **阅读顺序检测** | ❌ | ✅ |
| **图像提取** | ✅（嵌入） | ✅（带上下文） |
| **图像 → 文本 (OCR)** | ❌ | ✅ |
| **EPUB** | ✅ | ✅ |
| **Markdown 输出** | ✅（通过 pymupdf4llm） | ✅（原生，更高质量） |
| **安装大小** | ~25MB | ~3-5GB（PyTorch + 模型） |
| **速度** | 即时 | ~1-14s/页（CPU），~0.2s/页（GPU） |

**决策：** 除非你需要 OCR、公式、表单或复杂布局分析，否则使用 pymupdf。

如果用户需要 marker 功能但系统缺少约 5GB 可用磁盘空间：
> "此文档需要 OCR/高级提取（marker-pdf），需要约 5GB 用于 PyTorch 和模型。你的系统有 [X]GB 可用空间。选项：释放空间、提供 URL 以便我使用 web_extract，或者我可以尝试 pymupdf，它适用于文本型 PDF 但不适用于扫描文档或公式。"

---

## pymupdf（轻量级）

```bash
pip install pymupdf pymupdf4llm
```

**通过辅助脚本**：
```bash
python scripts/extract_pymupdf.py document.pdf              # 纯文本
python scripts/extract_pymupdf.py document.pdf --markdown    # Markdown
python scripts/extract_pymupdf.py document.pdf --tables      # 表格
python scripts/extract_pymupdf.py document.pdf --images out/ # 提取图像
python scripts/extract_pymupdf.py document.pdf --metadata    # 标题、作者、页数
python scripts/extract_pymupdf.py document.pdf --pages 0-4   # 特定页面
```

**内联**：
```bash
python3 -c "
import pymupdf
doc = pymupdf.open('document.pdf')
for page in doc:
    print(page.get_text())
"
```

---

## marker-pdf（高质量 OCR）

```bash
# 先检查磁盘空间
python scripts/extract_marker.py --check

pip install marker-pdf
```

**通过辅助脚本**：
```bash
python scripts/extract_marker.py document.pdf                # Markdown
python scripts/extract_marker.py document.pdf --json         # 带元数据的 JSON
python scripts/extract_marker.py document.pdf --output_dir out/  # 保存图像
python scripts/extract_marker.py scanned.pdf                 # 扫描 PDF (OCR)
python scripts/extract_marker.py document.pdf --use_llm      # LLM 增强准确度
```

**CLI**（随 marker-pdf 安装）：
```bash
marker_single document.pdf --output_dir ./output
marker /path/to/folder --workers 4    # 批量处理
```

---

## Arxiv 论文

```
# 仅摘要（快速）
web_extract(urls=["https://arxiv.org/abs/2402.03300"])

# 完整论文
web_extract(urls=["https://arxiv.org/pdf/2402.03300"])

# 搜索
web_search(query="arxiv GRPO reinforcement learning 2026")
```

## 拆分、合并与搜索

pymupdf 原生处理这些操作 — 使用 `execute_code` 或内联 Python：

```python
# 拆分：将第 1-5 页提取到新 PDF
import pymupdf
doc = pymupdf.open("report.pdf")
new = pymupdf.open()
for i in range(5):
    new.insert_pdf(doc, from_page=i, to_page=i)
new.save("pages_1-5.pdf")
```

```python
# 合并多个 PDF
import pymupdf
result = pymupdf.open()
for path in ["a.pdf", "b.pdf", "c.pdf"]:
    result.insert_pdf(pymupdf.open(path))
result.save("merged.pdf")
```

```python
# 搜索所有页面中的文本
import pymupdf
doc = pymupdf.open("report.pdf")
for i, page in enumerate(doc):
    results = page.search_for("revenue")
    if results:
        print(f"第 {i+1} 页：{len(results)} 处匹配")
        print(page.get_text("text"))
```

无需额外依赖 — pymupdf 在一个包中覆盖拆分、合并、搜索和文本提取。

---

## 注意

- `web_extract` 始终是 URL 的首选
- pymupdf 是安全的默认选择 — 即时、无需模型、随处可用
- marker-pdf 用于 OCR、扫描文档、公式、复杂布局 — 仅在需要时安装
- 两个辅助脚本都接受 `--help` 查看完整用法
- marker-pdf 首次使用时下载约 2.5GB 模型到 `~/.cache/huggingface/`
- 对于 Word 文档：`pip install python-docx`（优于 OCR — 解析实际结构）
- 对于 PowerPoint：参见 `powerpoint` 技能（使用 python-pptx）
