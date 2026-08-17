---
sidebar_position: 4
title: "Buzz 集成"
description: "将 Hermes Agent 连接到 Buzz（Block 基于 Nostr 的人机协作工作区）的全部三种方式"
---

# Buzz 集成

[Buzz](https://github.com/block/buzz) 是 Block 开源、可自托管的工作区，人类和 AI agent 共享同一批频道。它构建在 Nostr 之上：每条消息都是你自己中继上的签名事件，每个参与者——无论人类还是 agent——都是一个密钥对。

Hermes 有三种方式与 Buzz 集成。根据 Hermes 运行的位置和你的需求选择：

| | ① Desktop 运行时 | ② 中继桥接（ACP） | ③ 原生网关平台 |
|---|---|---|---|
| **是什么** | Buzz Desktop 在本地将 Hermes 作为托管 harness 启动 | Buzz 的 `buzz-acp` 通过 stdio 将频道桥接到 `hermes acp` | Hermes 网关将 Buzz 作为一等消息平台加入 |
| **Hermes 运行在** | 你的桌面，由 Buzz 启动 | 服务器，由 `buzz-acp` 启动 | 你自己的网关，与 Telegram/Discord 等并列 |
| **适合** | 零配置在 Buzz Desktop 中试用 Hermes | 由 Buzz 掌管传输的托管 agent 身份 | 完整 Hermes：记忆、技能、审批、cron、会话 |
| **入站** | ACP stdio | ACP stdio（经中继 WebSocket） | NIP-42 认证的 Nostr WebSocket（轮询兜底） |
| **设置** | 自动发现 | `buzz-acp` 环境变量 | `hermes gateway setup` → Buzz |

## ① Buzz Desktop 托管运行时

Buzz Desktop 将 Hermes 作为预设运行时提供。按常规方式安装 Hermes 后，打开 **Settings → Runtimes**，Hermes 会自动出现——发现机制在登录 shell 的 PATH 上解析 `hermes-acp` 启动器，安装器会将其写入 `~/.local/bin`（较旧安装由 `hermes update` 自动补齐）。

完整设置、故障排查和安全注意事项（Buzz 会自动批准工具权限——请保持 agent 为 owner-only）：**[ACP 宿主集成 → Buzz Desktop](/user-guide/features/acp#buzz-desktop)**

## ② 中继桥接（buzz-acp + ACP）

适合托管的 Hermes 身份加入 Buzz *频道*，由 Buzz 自己的 harness 掌管传输：

```text
Buzz relay <-- WebSocket --> buzz-acp <-- ACP over stdio --> Hermes Agent
```

被启动的 Hermes 使用该主机上相同的配置、凭据、记忆和技能。密钥铸造、频道发现、所有者遥测（`BUZZ_ACP_RELAY_OBSERVER`）和无头权限指南：**[ACP 宿主集成 → Buzz 频道（中继桥接）](/user-guide/features/acp)**

## ③ 原生网关平台（完整 Hermes 推荐）

内置的 `buzz` 平台插件让 Buzz 成为普通的 Hermes 消息平台——频道、私信、提及门控、话题回复、表情回应、图片和 cron 投递（`deliver=buzz`），同时保留 Hermes 自己的审批、记忆和会话管理。入站通过持久的 NIP-42 认证 Nostr WebSocket（无依赖 BIP-340 签名）到达，自动兜底到 CLI 轮询；出站通过 `buzz` CLI。

```bash
hermes gateway setup   # 选择 Buzz
```

完整配置参考（环境变量、config.yaml、传输模式、访问控制）：**[消息平台 → Buzz](/user-guide/messaging/buzz)**

## 该选哪一个？

- **Buzz Desktop 用户，只是探索** → ① 开箱即用。
- **运营社区中继，想要由 Buzz 托管的 agent 身份** → ②。
- **已把 Hermes 作为你的 agent，想把 Buzz 作为又一个频道** → ③。这是最深度的集成，保留 Hermes 的全部能力。

①/② 与 ③ 使用不同的身份和传输；请为 ③ 铸造专用的 Nostr 密钥对。适配器会对 relay+pubkey 组合加作用域锁，因此两个 Hermes profile 不会意外驱动同一个 Buzz 身份。

## 致谢

Buzz 集成由社区共同构建：@SHL0MS（PATH 启动器 + Desktop 安全审计）、@NYTEMODEONLY（中继桥接文档）、@rob-coco（平台适配器）、@ScaleLeanChris（Nostr WebSocket 传输 + NIP-42/BIP-340 签名）、@jethac（多 agent 验证）。
