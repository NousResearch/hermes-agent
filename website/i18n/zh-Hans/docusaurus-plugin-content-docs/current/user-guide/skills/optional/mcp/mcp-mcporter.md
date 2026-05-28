---
title: "Mcporter"
sidebar_label: "Mcporter"
description: "使用 mcporter CLI 列出、配置、认证并直接调用 MCP 服务器/工具（HTTP 或 stdio），包括临时服务器、配置编辑及 CLI/类型生成等功能。"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Mcporter

使用 mcporter CLI 列出、配置、认证并直接调用 MCP 服务器/工具（HTTP 或 stdio），包括临时服务器、配置编辑及 CLI/类型生成等功能。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mcp/mcporter` 安装 |
| 路径 | `optional-skills/mcp/mcporter` |
| 版本 | `1.0.0` |
| 作者 | community |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `MCP`, `Tools`, `API`, `Integrations`, `Interop` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# mcporter

使用 `mcporter` 直接从终端发现、调用并管理 [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) 服务器和工具。

## 前置条件

需要 Node.js：
```bash
# 无需安装（通过 npx 运行）
npx mcporter list

# 或全局安装
npm install -g mcporter
```

## 快速开始

```bash
# 列出此机器上已配置的 MCP 服务器
mcporter list

# 列出指定服务器的工具及 schema 详情
mcporter list <server> --schema

# 调用工具
mcporter call <server.tool> key=value
```

## 发现 MCP 服务器

mcporter 会自动发现机器上其他 MCP 客户端（Claude Desktop、Cursor 等）已配置的服务器。如需查找新服务器，可浏览 [mcpfinder.dev](https://mcpfinder.dev) 或 [mcp.so](https://mcp.so) 等注册表，然后以临时方式连接：

```bash
# 通过 URL 连接任意 MCP 服务器（无需配置）
mcporter list --http-url https://some-mcp-server.com --name my_server

# 或临时运行 stdio 服务器
mcporter list --stdio "npx -y @modelcontextprotocol/server-filesystem" --name fs
```

## 调用工具

```bash
# key=value 语法
mcporter call linear.list_issues team=ENG limit:5

# 函数语法
mcporter call "linear.create_issue(title: \"Bug fix needed\")"

# 临时 HTTP 服务器（无需配置）
mcporter call https://api.example.com/mcp.fetch url=https://example.com

# 临时 stdio 服务器
mcporter call --stdio "bun run ./server.ts" scrape url=https://example.com

# JSON 载荷
mcporter call <server.tool> --args '{"limit": 5}'

# 机器可读输出（推荐用于 Hermes）
mcporter call <server.tool> key=value --output json
```

## 认证与配置

```bash
# 对服务器进行 OAuth 登录
mcporter auth <server | url> [--reset]

# 管理配置
mcporter config list
mcporter config get <key>
mcporter config add <server>
mcporter config remove <server>
mcporter config import <path>
```

配置文件位置：`./config/mcporter.json`（可通过 `--config` 覆盖）。

## Daemon（守护进程）

用于持久化服务器连接：
```bash
mcporter daemon start
mcporter daemon status
mcporter daemon stop
mcporter daemon restart
```

## 代码生成

```bash
# 为 MCP 服务器生成 CLI 包装器
mcporter generate-cli --server <name>
mcporter generate-cli --command <url>

# 检查已生成的 CLI
mcporter inspect-cli <path> [--json]

# 生成 TypeScript 类型/客户端
mcporter emit-ts <server> --mode client
mcporter emit-ts <server> --mode types
```

## 注意事项

- 使用 `--output json` 获取结构化输出，便于解析
- 临时服务器（HTTP URL 或 `--stdio` 命令）无需任何配置即可使用，适合一次性调用
- OAuth 认证可能需要交互式浏览器流程 — 如有需要，请使用 `terminal(command="mcporter auth <server>", pty=true)`