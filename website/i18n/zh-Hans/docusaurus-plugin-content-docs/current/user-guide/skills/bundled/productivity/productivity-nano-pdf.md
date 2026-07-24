---
title: "Nano Pdf — 通过自然语言指令编辑现有 PDF 中的文本"
sidebar_label: "Nano Pdf"
description: "通过自然语言指令编辑现有 PDF 中的文本"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Nano Pdf

通过自然语言指令（prompt）编辑现有 PDF 中的文本。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/nano-pdf` |
| 版本 | `1.0.0` |
| 作者 | community |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `PDF`, `Documents`, `Editing`, `NLP`, `Productivity` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# nano-pdf

使用自然语言指令编辑 PDF。指定页面并描述需要修改的内容。

## 前置条件

```bash
# Install with uv (recommended — already available in Hermes)
uv pip install nano-pdf

# Or with pip
pip install nano-pdf
```

## 用法

```bash
nano-pdf edit <file.pdf> <page_number> "<instruction>"
```

## 示例

```bash
# Change a title on page 1
nano-pdf edit deck.pdf 1 "Change the title to 'Q3 Results' and fix the typo in the subtitle"

# Update a date on a specific page
nano-pdf edit report.pdf 3 "Update the date from January to February 2026"

# Fix content
nano-pdf edit contract.pdf 2 "Change the client name from 'Acme Corp' to 'Acme Industries'"
```

## 注意事项

- 页码可能从 0 或 1 开始，具体取决于版本——如果编辑命中了错误的页面，请用 ±1 重试
- 编辑后务必验证输出的 PDF（使用 `read_file` 检查文件大小，或直接打开查看）
- 该工具底层使用 LLM——需要 API 密钥（运行 `nano-pdf --help` 查看配置说明）
- 适合文本内容修改；复杂的版式调整可能需要其他方案