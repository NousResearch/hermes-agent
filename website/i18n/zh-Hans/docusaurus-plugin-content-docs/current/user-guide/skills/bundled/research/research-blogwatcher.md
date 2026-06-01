---
title: "Blogwatcher — 通过 blogwatcher-cli 工具监控博客和 RSS/Atom 订阅源"
sidebar_label: "Blogwatcher"
description: "通过 blogwatcher-cli 工具监控博客和 RSS/Atom 订阅源"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Blogwatcher

通过 blogwatcher-cli 工具监控博客和 RSS/Atom 订阅源。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/research/blogwatcher` |
| 版本 | `2.0.0` |
| 作者 | JulienTant (fork of Hyaxia/blogwatcher) |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `RSS`, `Blogs`, `Feed-Reader`, `Monitoring` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Blogwatcher

使用 `blogwatcher-cli` 工具追踪博客和 RSS/Atom 订阅源的更新。支持自动订阅源发现、HTML 抓取回退、OPML 导入，以及文章已读/未读管理。

## 安装

选择以下任一方式：

- **Go：** `go install github.com/JulienTant/blogwatcher-cli/cmd/blogwatcher-cli@latest`
- **Docker：** `docker run --rm -v blogwatcher-cli:/data ghcr.io/julientant/blogwatcher-cli`
- **二进制文件（Linux amd64）：** `curl -sL https://github.com/JulienTant/blogwatcher-cli/releases/latest/download/blogwatcher-cli_linux_amd64.tar.gz | tar xz -C /usr/local/bin blogwatcher-cli`
- **二进制文件（Linux arm64）：** `curl -sL https://github.com/JulienTant/blogwatcher-cli/releases/latest/download/blogwatcher-cli_linux_arm64.tar.gz | tar xz -C /usr/local/bin blogwatcher-cli`
- **二进制文件（macOS Apple Silicon）：** `curl -sL https://github.com/JulienTant/blogwatcher-cli/releases/latest/download/blogwatcher-cli_darwin_arm64.tar.gz | tar xz -C /usr/local/bin blogwatcher-cli`
- **二进制文件（macOS Intel）：** `curl -sL https://github.com/JulienTant/blogwatcher-cli/releases/latest/download/blogwatcher-cli_darwin_amd64.tar.gz | tar xz -C /usr/local/bin blogwatcher-cli`

所有发布版本：https://github.com/JulienTant/blogwatcher-cli/releases

### Docker 持久化存储

默认情况下，数据库位于 `~/.blogwatcher-cli/blogwatcher-cli.db`。在 Docker 中，容器重启后数据会丢失。使用 `BLOGWATCHER_DB` 或挂载卷来持久化数据：

```bash
# 命名卷（最简单）
docker run --rm -v blogwatcher-cli:/data -e BLOGWATCHER_DB=/data/blogwatcher-cli.db ghcr.io/julientant/blogwatcher-cli scan

# 主机绑定挂载
docker run --rm -v /path/on/host:/data -e BLOGWATCHER_DB=/data/blogwatcher-cli.db ghcr.io/julientant/blogwatcher-cli scan
```

### 从原版 blogwatcher 迁移

如果从 `Hyaxia/blogwatcher` 升级，请移动数据库文件：

```bash
mv ~/.blogwatcher/blogwatcher.db ~/.blogwatcher-cli/blogwatcher-cli.db
```

二进制文件名已从 `blogwatcher` 更改为 `blogwatcher-cli`。

## 常用命令

### 管理博客

- 添加博客：`blogwatcher-cli add "My Blog" https://example.com`
- 指定订阅源添加：`blogwatcher-cli add "My Blog" https://example.com --feed-url https://example.com/feed.xml`
- 使用 HTML 抓取添加：`blogwatcher-cli add "My Blog" https://example.com --scrape-selector "article h2 a"`
- 列出已追踪博客：`blogwatcher-cli blogs`
- 移除博客：`blogwatcher-cli remove "My Blog" --yes`
- 从 OPML 导入：`blogwatcher-cli import subscriptions.opml`

### 扫描与阅读

- 扫描所有博客：`blogwatcher-cli scan`
- 扫描单个博客：`blogwatcher-cli scan "My Blog"`
- 列出未读文章：`blogwatcher-cli articles`
- 列出所有文章：`blogwatcher-cli articles --all`
- 按博客筛选：`blogwatcher-cli articles --blog "My Blog"`
- 按分类筛选：`blogwatcher-cli articles --category "Engineering"`
- 标记文章为已读：`blogwatcher-cli read 1`
- 标记文章为未读：`blogwatcher-cli unread 1`
- 全部标记为已读：`blogwatcher-cli read-all`
- 标记某博客全部已读：`blogwatcher-cli read-all --blog "My Blog" --yes`

## 环境变量

所有标志均可通过带 `BLOGWATCHER_` 前缀的环境变量设置：

| 变量 | 描述 |
|---|---|
| `BLOGWATCHER_DB` | SQLite 数据库文件路径 |
| `BLOGWATCHER_WORKERS` | 并发扫描 worker 数量（默认：8） |
| `BLOGWATCHER_SILENT` | 扫描时仅输出"scan done" |
| `BLOGWATCHER_YES` | 跳过确认提示 |
| `BLOGWATCHER_CATEGORY` | 按分类筛选文章的默认值 |

## 示例输出

```
$ blogwatcher-cli blogs
Tracked blogs (1):

  xkcd
    URL: https://xkcd.com
    Feed: https://xkcd.com/atom.xml
    Last scanned: 2026-04-03 10:30
```

```
$ blogwatcher-cli scan
Scanning 1 blog(s)...

  xkcd
    Source: RSS | Found: 4 | New: 4

Found 4 new article(s) total!
```

```
$ blogwatcher-cli articles
Unread articles (2):

  [1] [new] Barrel - Part 13
       Blog: xkcd
       URL: https://xkcd.com/3095/
       Published: 2026-04-02
       Categories: Comics, Science

  [2] [new] Volcano Fact
       Blog: xkcd
       URL: https://xkcd.com/3094/
       Published: 2026-04-01
       Categories: Comics
```

## 注意事项

- 未提供 `--feed-url` 时，自动从博客主页发现 RSS/Atom 订阅源。
- 若 RSS 失败且已配置 `--scrape-selector`，则回退至 HTML 抓取。
- RSS/Atom 订阅源中的分类会被存储，可用于筛选文章。
- 支持从 Feedly、Inoreader、NewsBlur 等导出的 OPML 文件批量导入博客。
- 数据库默认存储于 `~/.blogwatcher-cli/blogwatcher-cli.db`（可通过 `--db` 或 `BLOGWATCHER_DB` 覆盖）。
- 使用 `blogwatcher-cli <command> --help` 查看所有标志和选项。