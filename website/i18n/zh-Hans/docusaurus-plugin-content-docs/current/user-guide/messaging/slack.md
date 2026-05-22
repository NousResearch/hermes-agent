---
sidebar_position: 4
title: "Slack"
description: "使用 Socket Mode 将 Hermes Agent 作为 Slack 机器人连接"
---

# Slack 设置

将 Hermes Agent 作为 Slack 机器人连接，使用 Socket Mode（WebSocket）进行通信，因此不需要公开的 HTTP 端点——可在防火墙后面、本地笔记本或私有服务器上运行。

:::warning 经典 Slack 应用已弃用
基于 RTM API 的经典 Slack 应用在 2025 年 3 月已被弃用。Hermes 使用现代的 Bolt SDK 与 Socket Mode。如你仍在使用旧应用，请按下面步骤新建应用。 
:::

## 概览

| 组件 | 值 |
|------|-----|
| **库** | `slack-bolt` / `slack_sdk`（Python，Socket Mode） |
| **连接** | WebSocket（无需公网 URL） |
| **需要的凭证** | Bot Token（`xoxb-`）和 App-Level Token（`xapp-`） |
| **用户识别** | Slack Member ID（例如 `U01ABC2DEF3`） |

## 快速步骤（概览）

1. 使用 Hermes 生成的 manifest 创建应用（推荐）或手工创建
2. 在 OAuth & Permissions 中添加必需的 Bot Token Scopes
3. 启用 Socket Mode 并生成 `xapp-` App-Level Token
4. 在 Events 中订阅消息事件（`message.im`、`message.channels` 等）
5. 安装应用到 workspace 并获取 `xoxb-` Bot Token
6. 将 `SLACK_BOT_TOKEN`、`SLACK_APP_TOKEN`、`SLACK_ALLOWED_USERS` 写入 `~/.hermes/.env`
7. 启动网关并邀请机器人到相应频道

文档包含逐步详细说明与示例命令；如需我把每一步完整翻译（包括权限、事件订阅与 manifest 操作），我会继续完成。