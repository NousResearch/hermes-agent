---
sidebar_position: 15
title: "教程：多智能体话题群聊"
description: "让多个 Hermes 智能体在相互隔离的话题群聊（美食群、每只股票一个群）中共同讨论，可在单机或多台机器上运行"
---

# 多智能体话题群聊

你希望多个 Hermes 智能体围绕同一件事展开讨论，并按话题彼此隔离：一个美食群、每只股票一个群、一个头脑风暴群。Hermes 已经支持这种用法，不需要单独的“讨论网关”：你运行的网关本身就承载 **群聊（Group Chat）**，群里的成员是 **Bot**（即 Hermes 配置档案 profile）。本教程介绍如何搭建，以及目前的边界在哪里。

| 你想要的 | 对应的 Hermes 功能 |
|---|---|
| 多个各有角色、模型和记忆的智能体 | [Bot](../user-guide/bot-mode.md)，每个对应一个 [profile](../user-guide/profiles.md) |
| 每个话题一个隔离的讨论空间 | 每个话题一个 [群聊](../user-guide/bot-mode.md#groups-and-group-chats) |
| 无人值守时讨论继续进行 | 网关的持久化群聊驱动器 |
| 其他机器上的智能体加入讨论 | [多连接](../user-guide/multi-connection-desktop.md) + 跨机器群聊 |
| 不打开桌面应用，智能体之间互发消息 | [`hermes peer`](../user-guide/bot-mode.md#bot-initiated-dms-across-machines-hermes-peer) / `message_agent` |

## 1. 创建智能体

为每个智能体创建一个 profile，再在 `SOUL.md` 中设定各自的性格。以股票讨论为例：

```bash
hermes profile create bull --description "Argues the bull case with data"
hermes profile create bear --description "Argues the bear case and hunts for risks"
hermes profile create quant --description "Checks claims against prices and filings"
```

编辑每个智能体的 `~/.hermes/profiles/<name>/SOUL.md`：它支持什么观点、接受什么证据、说话有多直接。不同智能体使用不同模型效果更好：在各自 profile 的 `config.yaml` 中设置 `model`，或在 Bot 设置中选择。每个 Bot 拥有独立的记忆和技能，因此量化分析员可以装一个其他成员没有的行情数据技能。

在 [桌面应用](../user-guide/desktop.md) 中，这些 profile 会出现在 **Bots** 标签页里。在终端中 `hermes -p bull chat` 打开的仍是同一个智能体。

## 2. 每个话题开一个群聊

在 Bots 标签页中点击 **New Group Chat**，选择 2–6 个 Bot，并用话题命名群聊：`NVDA`、`TSLA`、`food`、`weekend-brainstorm`。每个话题重复一次即可。一个 Bot 可以加入任意多个群聊。

群聊之间相互隔离：

- 每个成员在 **每个群聊中都有独立的会话**（标题为 `Group: <群聊> · <话题串>`），所以空头分析员在 NVDA 群里说的话不会混进 TSLA 群。
- 群聊记录、成员和头像都属于该群聊，重命名时会一并保留。

## 3. 开始讨论

在群聊中发一条消息，会触发最多 **三轮** 成员发言：

- 不 @ 任何人时，所有成员都可以回复。每个成员只在有新观点时发言，否则跳过；一整轮都没人发言时讨论结束。
- `@bear 这个逻辑哪里会崩？` 只让该成员回答。成员之间可以用 `@名字` 互相点名，遇到需要你判断的问题时用 `@user` 交还给你，群聊会显示 **needs you** 标记。
- 硬性上限（3 轮、每次发送 10 条消息）防止讨论无限循环。想继续辩论就再发一条消息。

分析类群聊的实用模式：先让每个人表态（“各用一段话说说你对 NVDA 财报前的看法”），再发起质询（`@quant 核实一下多头的营收说法`），最后让一位成员总结分歧。

## 4. 让它持续运行

当一个群聊的所有成员都在 **同一个网关** 上时，由该网关自己调度发言轮次。关闭桌面应用不会中断正在进行的讨论，重新打开时应用会从群聊记录中补齐。把网关作为服务运行（`hermes gateway install`），群聊就会一直可用。

## 5. 接入其他机器上的智能体

在 **Settings → Connections** 中登记另一台机器的网关（局域网、Tailscale、SSH 或 Hermes Cloud）。它的 Bot 会出现在你的列表中，并可以加入同一个群聊。每个成员的发言都在 **它自己的机器上** 运行，使用自己的模型、记忆和密钥。NAT 与网络可达性请参阅 [跨机器的 Bot](../user-guide/bot-mode.md#bots-across-machines)。

如果 **完全不使用桌面应用**，只想让智能体之间互发消息，可以把另一个网关注册为 peer（对方需要运行 `api_server` 平台）：

```bash
hermes peer add spark --url http://spark.lan:8377 --key <API_SERVER_KEY>
hermes peer dm spark/bear < question.txt
```

之后每个 Bot 都能用 `message_agent` 工具联系该 peer 上的智能体。这是两个智能体之间的直接对话，而不是共享群聊。

## 目前的限制

- 群聊只能在桌面应用中创建和管理，目前还没有 `hermes room` 命令行或网页仪表盘页面；该需求在 [#89995](https://github.com/NousResearch/hermes-agent/issues/89995) 中跟踪。
- 一个群聊容纳 2–6 个 Bot。需要更大的讨论组时，可以按子话题拆分，或设置一个主持人 Bot，每轮 @ 它需要的成员。
- 跨多个网关的群聊需要桌面应用来调度发言。只有所有成员都在同一个网关上的群聊，才能在关闭应用后继续运行。
- 讨论在有人往群里发消息时开始。[定时任务（Routines）](../user-guide/bot-mode.md#routines) 运行在 Bot 自己的对话中，可以为你准备素材（隔夜新闻、价格摘要）再由你发到群里，但不会自己开启一轮群聊讨论。

## 另请参阅

- [Bot Mode](../user-guide/bot-mode.md)：Bot、群聊和 Bot 间消息的完整说明
- [Profiles](../user-guide/profiles.md)：Bot 的底层是什么
- [委托与并行工作](./delegation-patterns.md)：一个智能体把任务分派给子代理，适合不需要常驻讨论的场景
