---
title: "Blogwatcher — 通过 blogwatcher-cli 工具监控博客和 RSS/Atom 订阅源"
sidebar_label: "Blogwatcher"
description: "通过 blogwatcher-cli 工具监控博客和 RSS/Atom 订阅源"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Blogwatcher

通过 blogwatcher-cli 工具监控博客和 RSS/Atom 订阅源。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/research/blogwatcher` |
| 版本 | `2.0.0` |
| 作者 | JulienTant（Hyaxia/blogwatcher 的分支） |
| 许可证 | MIT |
| 标签 | `RSS`, `博客`, `订阅阅读器`, `监控` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是技能激活时代理看到的指令内容。
:::

# Blogwatcher

使用 `blogwatcher-cli` 工具追踪博客和 RSS/Atom 订阅源更新。支持自动订阅源发现、HTML 抓取回退、OPML 导入和已读/未读文章管理。

## 安装

选择一种方法：

- **Go：** `go install github.com/JulienTant/blogwatcher-cli/cmd/blogwatcher-cli@latest`
- **Docker：** `docker run --rm -v blogwatcher-cli:/data ghcr.io/julientant/blogwatcher-cli`
- **二进制文件（Linux amd64）：** `curl -sL https://github.com/JulienTant/blogwatcher-cli/releases/latest/download/blogwatcher-cli_linux_amd64.tar.gz | tar xz -C /usr/local/bin blogwatcher-cli`
- **二进制文件（Linux arm64）：** `curl -sL https://github.com/JulienTant/blogwatcher-cli/releases/latest/download/blogwatcher-cli_linux_arm64.tar.gz | tar xz -C /usr/local/bin blogwatcher-cli`
- **二进制文件（macOS Apple Silicon）：** `curl -sL https://github.com/JulienTant/blogwatcher-cli/releases/latest/download/blogwatcher-cli_darwin_arm64.tar.gz | tar xz -C /usr/local/bin blogwatcher-cli`
- **二进制文件（macOS Intel）：** `curl -sL https://github.com/JulienTant/blogwatcher-cli/releases/latest/download/blogwatcher-cli_darwin_amd64.tar.gz | tar xz -C /usr/local/bin blogwatcher-cli`

所有版本：https://github.com/JulienTant/blogwatcher-cli/releases

### 使用持久存储的 Docker

默认情况下，数据库位于 `~/.blogwatcher-cli/blogwatcher-cli.db`。在 Docker 中，这在容器重启时会丢失。使用 `BLOGWATCHER_DB` 或卷挂载来持久化：

```bash
# 命名卷（最简单）
docker run --rm -v blogwatcher-cli:/data -e BLOGWATCHER_DB=/data/blogwatcher-cli.db ghcr.io/julientant/blogwatcher-cli scan

# 主机绑定挂载
docker run --rm -v /path/on/host:/data -e BLOGWATCHER_DB=/data/blogwatcher-cli.db ghcr.io/julientant/blogwatcher-cli scan
```

### 从原始 blogwatcher 迁移

如果从 `Hyaxia/blogwatcher` 升级，移动你的数据库：

```bash
mv ~/.blogwatcher/blogwatcher.db ~/.blogwatcher-cli/blogwatcher-cli.db
```

二进制文件名称从 `blogwatcher` 更改为 `blogwatcher-cli`。

## 常用命令

### 管理博客

- 添加博客：`blogwatcher-cli add "我的博客" https://example.com`
- 添加带显式订阅源：`blogwatcher-cli add "我的博客" https://example.com --feed-url https://example.com/feed.xml`
- 添加带 HTML 抓取：`blogwatcher-cli add "我的博客" https://example.com --scrape-selector "article h2 a"`
- 列出追踪的博客：`blogwatcher-cli blogs`
- 移除博客：`blogwatcher-cli remove "我的博客" --yes`
- 从 OPML 导入：`blogwatcher-cli import subscriptions.opml`

### 扫描和阅读

- 扫描所有博客：`blogwatcher-cli scan`
- 扫描一个博客：`blogwatcher-cli scan "我的博客"`
- 列出未读文章：`blogwatcher-cli articles`
- 列出所有文章：`blogwatcher-cli articles --all`
- 按博客过滤：`blogwatcher-cli articles --blog "我的博客"`
- 按类别过滤：`blogwatcher-cli articles --category "工程"`
- 标记文章已读：`blogwatcher-cli read 1`
- 标记文章未读：`blogwatcher-cli unread 1`
- 全部标记已读：`blogwatcher-cli read-all`
- 按博客全部标记已读：`blogwatcher-cli read-all --blog "我的博客" --yes`

## 环境变量

所有标志可以通过 `BLOGWATCHER_` 前缀的环境变量设置：

| 变量 | 描述 |
|---|---|
| `BLOGWATCHER_DB` | SQLite 数据库文件路径 |
| `BLOGWATCHER_WORKERS` | 并发扫描工作线程数（默认：8） |
| `BLOGWATCHER_SILENT` | 扫描时仅输出"scan done" |
| `BLOGWATCHER_YES` | 跳过确认提示 |
| `BLOGWATCHER_CATEGORY` | 按类别过滤文章的默认值 |

## 示例输出

```
$ blogwatcher-cli blogs
追踪的博客 (1)：

  xkcd
    URL：https://xkcd.com
    订阅源：https://xkcd.com/atom.xml
    上次扫描：2026-04-03 10:30
```

```
$ blogwatcher-cli scan
正在扫描 1 个博客...

  xkcd
    来源：RSS | 找到：4 | 新增：4

共找到 4 篇新文章！
```

```
$ blogwatcher-cli articles
未读文章 (2)：

  [1] [新] Barrel - Part 13
       博客：xkcd
       URL：https://xkcd.com/3095/
       发布：2026-04-02
       类别：漫画、科学

  [2] [新] Volcano Fact
       博客：xkcd
       URL：https://xkcd.com/3094/
       发布：2026-04-01
       类别：漫画
```

## 注意

- 当未提供 `--feed-url` 时，从博客主页自动发现 RSS/Atom 订阅源。
- 如果 RSS 失败且配置了 `--scrape-selector`，则回退到 HTML 抓取。
- RSS/Atom 订阅源中的类别会被存储，可用于过滤文章。
- 支持从 Feedly、Inoreader、NewsBlur 等导出的 OPML 文件批量导入博客。
- 数据库默认存储在 `~/.blogwatcher-cli/blogwatcher-cli.db`（使用 `--db` 或 `BLOGWATCHER_DB` 覆盖）。
- 使用 `blogwatcher-cli <command> --help` 发现所有标志和选项。
