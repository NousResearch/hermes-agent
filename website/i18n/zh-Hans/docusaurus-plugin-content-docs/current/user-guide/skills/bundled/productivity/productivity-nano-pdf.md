---
title: "Nano Pdf — 通过 nano-pdf CLI 编辑 PDF 文本/错别字/标题（自然语言提示）"
sidebar_label: "Nano Pdf"
description: "通过 nano-pdf CLI 编辑 PDF 文本/错别字/标题（自然语言提示）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Nano Pdf

通过 nano-pdf CLI 编辑 PDF 文本/错别字/标题（自然语言提示）。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/productivity/nano-pdf` |
| 版本 | `1.0.0` |
| 作者 | community |
| 许可证 | MIT |
| 标签 | `PDF`, `文档`, `编辑`, `NLP`, `生产力` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是技能激活时代理看到的指令内容。
:::

# nano-pdf

使用自然语言指令编辑 PDF。指向某一页并描述要更改的内容。

## 前提条件

```bash
# 使用 uv 安装（推荐 — Hermes 中已可用）
uv pip install nano-pdf

# 或使用 pip
pip install nano-pdf
```

## 使用

```bash
nano-pdf edit <file.pdf> <page_number> "<instruction>"
```

## 示例

```bash
# 更改第 1 页的标题
nano-pdf edit deck.pdf 1 "将标题改为'Q3 Results'并修复副标题中的错别字"

# 更新特定页面的日期
nano-pdf edit report.pdf 3 "将日期从 2026 年 1 月更新为 2026 年 2 月"

# 修改内容
nano-pdf edit contract.pdf 2 "将客户名称从'Acme Corp'改为'Acme Industries'"
```

## 注意

- 页码可能是从 0 开始或从 1 开始的，取决于版本 — 如果编辑命中了错误的页面，用 ±1 重试
- 编辑后始终验证输出 PDF（使用 `read_file` 检查文件大小，或打开它）
- 该工具底层使用 LLM — 需要 API 密钥（查看 `nano-pdf --help` 了解配置）
- 适用于文本更改；复杂的布局修改可能需要不同的方法
