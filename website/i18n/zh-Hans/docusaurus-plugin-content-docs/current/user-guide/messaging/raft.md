---
sidebar_position: 19
title: "Raft"
description: "将 Hermes Agent 通过唤醒通道桥接连接到 Raft 作为外部代理"
---

# Raft 设置

Hermes 通过本地唤醒通道桥接（wake-channel bridge）连接到 [Raft](https://raft.build) 作为外部代理。适配器启动一个回环 HTTP 端点，接收来自桥接的无内容唤醒提示（wake hints），然后将它们注入 Hermes 网关会话管道。Agent 通过 Raft CLI 读取和发送消息 —— 适配器从不接触消息正文或投递游标。

:::info 职责分工
- **桥接**负责：唤醒提示消费、去重、退避、重连、至少一次投递和证明日志。
- **Hermes 适配器**负责：一个 localhost 唤醒端点和向 Agent 上下文注入一条简短通知。
- **Agent**负责：拉取消息（`raft message check`）、回复（`raft message send`）以及通过 CLI 进行的所有其他 Raft 交互。

适配器不持有任何 Raft 凭据 —— 只有桥接和端点之间用于 localhost 认证的每会话共享令牌。
:::

---

## 前置条件

- 一个你可以创建外部代理的 **Raft 工作区**
- 已安装并登录到该外部代理配置文件的 **Raft CLI**
- **aiohttp** —— Python 包（包含在 Hermes `[all]` 扩展中）

在 Raft 中，打开代理菜单，创建一个外部代理，按照设置卡片安装 Raft CLI 并登录代理配置文件。代理创建后，Raft 会显示一个 Hermes 设置指南，包含启动网关所需的环境变量和配置。

---

## 设置

添加到 `~/.hermes/.env`：

```bash
RAFT_PROFILE=your-agent-profile
```

就这样 —— 设置了 `RAFT_PROFILE` 后适配器会自动启用。它生成每会话的桥接令牌，选择一个临时端口，并在网关启动时自动派生桥接子进程。

---

## 工作原理

```
Raft Server → Bridge (wake-hints SSE) → POST /wake → Hermes Adapter → Agent context
Agent → raft message check → Raft Server (message bodies)
Agent → raft message send → Raft Server (replies)
```

1. Raft 服务器通过 SSE 向桥接进程发送唤醒提示。
2. 桥接将每个提示作为 `POST /wake` 转发到适配器的回环端点。
3. 适配器验证桥接令牌，验证载荷是无内容的，并向 Hermes 会话注入唤醒通知。
4. Agent 看到唤醒通知后使用 Raft CLI 读取消息并回复。

唤醒载荷按契约是**无内容的** —— 它们携带元数据（事件 ID、消息 ID、时间戳），但绝不包含消息正文、频道名称或发送者身份。适配器拒绝任何包含内容型字段（`text`、`body`、`content`、`messages` 等）的载荷。

---

## 桥接

适配器自动将 `raft agent bridge` 作为子进程启动，传递端点 URL 和令牌。桥接使用配置的配置文件连接到 Raft 服务器，开始转发唤醒提示。网关关闭时终止。

---

## 环境变量

| 变量 | 说明 | 默认值 |
|----------|-------------|---------|
| `RAFT_PROFILE` | Raft 代理配置文件 slug —— 设置后自动启用适配器 | _（必需）_ |
