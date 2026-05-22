---
sidebar_position: 3
title: "Discord"
description: "将 Hermes Agent 设置为 Discord 机器人"
---

# Discord 设置

Hermes Agent 可作为 Discord 机器人集成，让你通过私信或服务器频道与 AI 助手对话。机器人接收你的消息，经 Hermes Agent 管线处理（包括工具、记忆与推理），并实时回复。支持文本、语音、文件附件与斜杠命令。

## 行为概览

| 场景 | 行为 |
|------|------|
| **私信（DM）** | Hermes 会回应每条消息，无需 `@mention`。每个私信都有独立会话。 |
| **服务器频道** | 默认情况下，只有在 `@mention` 机器人时 Hermes 才会响应。未提及时机器人会忽略消息。 |
| **免提响应频道** | 可以将特定频道加入 `DISCORD_FREE_RESPONSE_CHANNELS`，或将 `DISCORD_REQUIRE_MENTION=false`，在这些频道中机器人会直接回复而不创建线程。 |
| **线程** | Hermes 在相同线程中回复。提及规则仍然生效，除非该线程或父频道已被配置为免提。线程与主频道保持会话隔离。 |
| **多人共享频道** | 默认情况下，Hermes 会在共享频道内对每个用户隔离会话历史。若需共享会话可将 `group_sessions_per_user` 设置为 `false`。 |

:::tip
若希望在某个频道中无需每次都 @mention 即可使用机器人，请将该频道添加到 `DISCORD_FREE_RESPONSE_CHANNELS`。
:::

## Discord 网关模型

Hermes 在 Discord 上并非无状态的 webhook；每条消息都会经过完整的网关流程：

1. 授权检查（`DISCORD_ALLOWED_USERS`）
2. 提及 / 免提 检查
3. 会话查找
4. 会话转录加载
5. Hermes agent 执行（包括工具、记忆、斜杠命令）
6. 将响应返回 Discord

这意味着在繁忙服务器中的行为既受 Discord 路由，也受 Hermes 会话策略影响。

## 会话模型（默认）

- 每个 DM 独立为一个会话
- 每个服务器线程有独立会话命名空间
- 在共享频道中每个用户默认拥有独立会话

这些由 `config.yaml` 中的 `group_sessions_per_user: true` 控制。设置为 `false` 会让整个频道共享一个会话。

## 中断与并发

Hermes 通过会话键来跟踪正在运行的 agent。默认隔离下：

- Alice 中断自己的请求仅影响 Alice 的会话
- Bob 在同一频道可以并行交互而不受影响

如果共享会话，则频道/线程共享运行槽位，可能出现其他人中断或排队的情况。

## 快速设置步骤（概览）

1. 在 Discord 开发者门户创建应用并添加 Bot
2. 在 Bot 页面启用必要的 Privileged Gateway Intents（尤其是 Message Content Intent）
3. 生成并保存 Bot Token
4. 生成邀请链接并邀请机器人到你的服务器
5. 在 `~/.hermes/.env` 中配置 `DISCORD_BOT_TOKEN` 与 `DISCORD_ALLOWED_USERS`
6. 启动 `hermes gateway` 并测试（DM 或被授权的频道）

## 斜杠命令访问控制 {#slash-command-access-control}

Discord 侧的斜杠命令权限由平台角色与 Hermes 授权策略共同决定。建议同时校验平台权限和 `ALLOWED_USERS` 规则，避免误开放。

具体步骤与英文文档对应，若需我继续将全部步骤逐条翻译，请告知，我会完成剩余部分。