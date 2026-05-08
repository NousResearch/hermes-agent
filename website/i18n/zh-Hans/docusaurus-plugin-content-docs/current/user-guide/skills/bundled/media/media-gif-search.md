---
title: "GIF 搜索 — 通过 curl + jq 从 Tenor 搜索/下载 GIF"
sidebar_label: "GIF 搜索"
description: "通过 curl + jq 从 Tenor 搜索/下载 GIF"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# GIF 搜索

通过 curl + jq 从 Tenor 搜索/下载 GIF。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/media/gif-search` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `GIF`、`媒体`、`搜索`、`Tenor`、`API` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# GIF 搜索（Tenor API）

使用 curl 直接通过 Tenor API 搜索和下载 GIF。无需额外工具。

## 使用场景

用于查找反应 GIF、创建视觉内容和在聊天中发送 GIF。

## 设置

在环境中设置 Tenor API 密钥（添加到 `~/.hermes/.env`）：

```bash
TENOR_API_KEY=your_key_here
```

在 https://developers.google.com/tenor/guides/quickstart 获取免费 API 密钥 — Google Cloud Console 的 Tenor API 密钥是免费的，且具有慷慨的速率限制。

## 前置条件

- `curl` 和 `jq`（macOS/Linux 标准工具）
- `TENOR_API_KEY` 环境变量

## 搜索 GIF

```bash
# 搜索并获取 GIF URL
curl -s "https://tenor.googleapis.com/v2/search?q=thumbs+up&limit=5&key=${TENOR_API_KEY}" | jq -r '.results[].media_formats.gif.url'

# 获取更小/预览版本
curl -s "https://tenor.googleapis.com/v2/search?q=nice+work&limit=3&key=${TENOR_API_KEY}" | jq -r '.results[].media_formats.tinygif.url'
```

## 下载 GIF

```bash
# 搜索并下载第一个结果
URL=$(curl -s "https://tenor.googleapis.com/v2/search?q=celebration&limit=1&key=${TENOR_API_KEY}" | jq -r '.results[0].media_formats.gif.url')
curl -sL "$URL" -o celebration.gif
```

## 获取完整元数据

```bash
curl -s "https://tenor.googleapis.com/v2/search?q=cat&limit=3&key=${TENOR_API_KEY}" | jq '.results[] | {title: .title, url: .media_formats.gif.url, preview: .media_formats.tinygif.url, dimensions: .media_formats.gif.dims}'
```

## API 参数

| 参数 | 描述 |
|---|---|
| `q` | 搜索查询（空格 URL 编码为 `+`） |
| `limit` | 最大结果数（1-50，默认 20） |
| `key` | API 密钥（来自 `$TENOR_API_KEY` 环境变量） |
| `media_filter` | 格式过滤：`gif`、`tinygif`、`mp4`、`tinymp4`、`webm` |
| `contentfilter` | 安全级别：`off`、`low`、`medium`、`high` |
| `locale` | 语言：`en_US`、`es`、`fr` 等 |

## 可用媒体格式

每个结果在 `.media_formats` 下有多个格式：

| 格式 | 用途 |
|---|---|
| `gif` | 完整质量 GIF |
| `tinygif` | 小预览 GIF |
| `mp4` | 视频版本（文件更小） |
| `tinymp4` | 小预览视频 |
| `webm` | WebM 视频 |
| `nanogif` | 微型缩略图 |

## 注意

- URL 编码查询：空格编码为 `+`，特殊字符编码为 `%XX`
- 在聊天中发送时，`tinygif` URL 更轻量
- GIF URL 可以直接在 markdown 中使用：`![alt](url)`
