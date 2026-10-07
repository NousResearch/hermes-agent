---
title: "Duckduckgo Search — 通过 DuckDuckGo 免费搜索网络 — 文本、新闻、图片、视频"
sidebar_label: "Duckduckgo Search"
description: "通过 DuckDuckGo 免费搜索网络 — 文本、新闻、图片、视频"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Duckduckgo Search

通过 DuckDuckGo 免费搜索网络 — 文本、新闻、图片、视频。无需 API 密钥。已安装时优先使用 `ddgs` CLI；仅在确认当前运行时中 `ddgs` 可用后，才使用 Python DDGS 库。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/research/duckduckgo-search` 安装 |
| 路径 | `optional-skills/research/duckduckgo-search` |
| 版本 | `1.3.0` |
| 作者 | gamedevCloudy |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `search`, `duckduckgo`, `web-search`, `free`, `fallback` |
| 相关 skill | [`arxiv`](/user-guide/skills/bundled/research/research-arxiv) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# DuckDuckGo Search

使用 DuckDuckGo 进行免费网络搜索。**无需 API 密钥。**

当 `web_search` 不可用或不适用时（例如未设置 `FIRECRAWL_API_KEY`），优先使用此 skill。也可在明确需要 DuckDuckGo 结果时作为独立搜索路径使用。

## 检测流程

在选择方案前，先检查实际可用的工具：

```bash
# Check CLI availability
command -v ddgs >/dev/null && echo "DDGS_CLI=installed" || echo "DDGS_CLI=missing"
```

决策树：
1. 若 `ddgs` CLI 已安装，优先使用 `terminal` + `ddgs`
2. 若 `ddgs` CLI 未安装，不要假设 `execute_code` 能导入 `ddgs`
3. 若用户明确需要 DuckDuckGo，先在相关环境中安装 `ddgs`
4. 否则回退到内置的 web/browser 工具

重要运行时说明：
- Terminal 与 `execute_code` 是独立的运行时
- shell 中安装成功不代表 `execute_code` 能导入 `ddgs`
- 永远不要假设 `execute_code` 内已预装第三方 Python 包

## 安装

仅在明确需要 DuckDuckGo 搜索且运行时尚未提供时，才安装 `ddgs`。

```bash
# Python package + CLI entrypoint
pip install ddgs

# Verify CLI
ddgs --help
```

若工作流依赖 Python 导入，请在使用 `from ddgs import DDGS` 前，先验证该运行时能否导入 `ddgs`。

## 方法一：CLI 搜索（推荐）

当 `ddgs` 命令存在时，通过 `terminal` 使用它。这是推荐路径，因为它避免了假设 `execute_code` 沙箱中已安装 `ddgs` Python 包。

```bash
# Text search
ddgs text -q "python async programming" -m 5

# News search
ddgs news -q "artificial intelligence" -m 5

# Image search
ddgs images -q "landscape photography" -m 10

# Video search
ddgs videos -q "python tutorial" -m 5

# With region filter
ddgs text -q "best restaurants" -m 5 -r us-en

# Recent results only (d=day, w=week, m=month, y=year)
ddgs text -q "latest AI news" -m 5 -t w

# JSON output for parsing
ddgs text -q "fastapi tutorial" -m 5 -o json
```

### CLI 参数

| 参数 | 说明 | 示例 |
|------|-------------|---------|
| `-q` | 查询词 — **必填** | `-q "search terms"` |
| `-m` | 最大结果数 | `-m 5` |
| `-r` | 地区 | `-r us-en` |
| `-t` | 时间范围 | `-t w`（一周） |
| `-s` | 安全搜索 | `-s off` |
| `-o` | 输出格式 | `-o json` |

## 方法二：Python API（仅在验证后使用）

仅在确认 `ddgs` 已安装于该运行时后，才在 `execute_code` 或其他 Python 运行时中使用 `DDGS` 类。不要默认认为 `execute_code` 包含第三方包。

正确表述：
- "在安装或确认包可用后，在 `execute_code` 中使用 `ddgs`"

避免表述：
- "`execute_code` 包含 `ddgs`"
- "DuckDuckGo 搜索在 `execute_code` 中默认可用"

**重要：** `max_results` 必须始终以**关键字参数**形式传入 — 所有方法中以位置参数传入均会报错。

### 文本搜索

适用场景：通用研究、公司信息、文档查询。

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.text("python async programming", max_results=5):
        print(r["title"])
        print(r["href"])
        print(r.get("body", "")[:200])
        print()
```

返回字段：`title`、`href`、`body`

### 新闻搜索

适用场景：时事动态、突发新闻、最新更新。

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.news("AI regulation 2026", max_results=5):
        print(r["date"], "-", r["title"])
        print(r.get("source", ""), "|", r["url"])
        print(r.get("body", "")[:200])
        print()
```

返回字段：`date`、`title`、`body`、`url`、`image`、`source`

### 图片搜索

适用场景：视觉参考、产品图片、示意图。

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.images("semiconductor chip", max_results=5):
        print(r["title"])
        print(r["image"])
        print(r.get("thumbnail", ""))
        print(r.get("source", ""))
        print()
```

返回字段：`title`、`image`、`thumbnail`、`url`、`height`、`width`、`source`

### 视频搜索

适用场景：教程、演示、讲解视频。

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.videos("FastAPI tutorial", max_results=5):
        print(r["title"])
        print(r.get("content", ""))
        print(r.get("duration", ""))
        print(r.get("provider", ""))
        print(r.get("published", ""))
        print()
```

返回字段：`title`、`content`、`description`、`duration`、`provider`、`published`、`statistics`、`uploader`

### 快速参考

| 方法 | 适用场景 | 关键字段 |
|--------|----------|------------|
| `text()` | 通用研究、公司信息 | title, href, body |
| `news()` | 时事动态、最新更新 | date, title, source, body, url |
| `images()` | 视觉内容、示意图 | title, image, thumbnail, url |
| `videos()` | 教程、演示 | title, content, duration, provider |

## 工作流：先搜索后提取

DuckDuckGo 返回标题、URL 和摘要，而非完整页面内容。如需获取完整页面内容，先搜索，再用 `web_extract`、browser 工具或 curl 提取最相关的 URL。

CLI 示例：

```bash
ddgs text -q "fastapi deployment guide" -m 3 -o json
```

Python 示例，仅在确认该运行时已安装 `ddgs` 后使用：

```python
from ddgs import DDGS

with DDGS() as ddgs:
    results = list(ddgs.text("fastapi deployment guide", max_results=3))
    for r in results:
        print(r["title"], "->", r["href"])
```

然后使用 `web_extract` 或其他内容获取工具提取最佳 URL 的内容。

## 限制

- **频率限制**：大量快速请求后，DuckDuckGo 可能进行限流。如有需要，在多次搜索之间添加短暂延迟。
- **无内容提取**：`ddgs` 返回摘要，而非完整页面内容。如需完整文章/页面，请使用 `web_extract`、browser 工具或 curl。
- **结果质量**：总体良好，但可配置性不如 Firecrawl 的搜索。
- **可用性**：DuckDuckGo 可能屏蔽来自部分云 IP 的请求。若搜索返回空结果，请尝试不同关键词或等待几秒后重试。
- **字段可变性**：不同结果或 `ddgs` 版本间返回字段可能有所不同。对可选字段使用 `.get()` 以避免 `KeyError`。
- **独立运行时**：在 terminal 中成功安装 `ddgs` 不代表 `execute_code` 能自动导入它。

## 故障排查

| 问题 | 可能原因 | 处理方式 |
|---------|--------------|------------|
| `ddgs: command not found` | CLI 未安装在 shell 环境中 | 安装 `ddgs`，或改用内置 web/browser 工具 |
| `ModuleNotFoundError: No module named 'ddgs'` | Python 运行时未安装该包 | 在准备好该运行时之前，不要在其中使用 Python DDGS |
| 搜索无结果 | 临时限流或查询词不佳 | 等待几秒后重试，或调整查询词 |
| CLI 正常但 `execute_code` 导入失败 | Terminal 与 `execute_code` 是不同的运行时 | 继续使用 CLI，或单独准备 Python 运行时 |

## 常见陷阱

- **`max_results` 仅支持关键字参数**：`ddgs.text("query", 5)` 会报错，请使用 `ddgs.text("query", max_results=5)`。
- **不要假设 CLI 已存在**：使用前先检查 `command -v ddgs`。
- **不要假设 `execute_code` 能导入 `ddgs`**：除非该运行时已单独准备，否则 `from ddgs import DDGS` 可能抛出 `ModuleNotFoundError`。
- **包名**：该包名为 `ddgs`（原名 `duckduckgo-search`），使用 `pip install ddgs` 安装。
- **不要混淆 `-q` 和 `-m`**（CLI）：`-q` 用于查询词，`-m` 用于最大结果数。
- **空结果**：若 `ddgs` 返回空结果，可能是被限流。等待几秒后重试。

## 验证版本

已针对 `ddgs==9.11.2` 语义验证示例。Skill 指南现将 CLI 可用性与 Python 导入可用性视为独立问题，以确保文档化的工作流与实际运行时行为一致。