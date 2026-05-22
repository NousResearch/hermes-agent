---
sidebar_position: 8
title: "Open WebUI"
description: "将 Open WebUI 连接到 Hermes Agent 的 OpenAI-compatible API 服务器。"
---

# Open WebUI 集成

Open WebUI 可以作为 Hermes Agent 的 Web 前端。Hermes 的 API 服务器会按请求创建 server-side `AIAgent`，因此工具执行发生在 API 服务器所在的机器上。

## 快速开始

1. 启用 API 服务器
2. 启动 gateway
3. 在 Open WebUI 中把 OpenAI base URL 指向 Hermes 的 `/v1`
4. 使用相同的 API key

## 相关文档

- [配置](/user-guide/configuration)
- [个人资料](/user-guide/profiles)
- [API 服务器](/user-guide/features/api-server)

## 多用户配置（Profiles） {#multi-user-setup-with-profiles}

如果你希望在 Open WebUI 里把不同 Hermes profile 当作不同模型入口使用，可以为每个 profile 启动独立 API 连接，并在 Open WebUI 里分别添加。
