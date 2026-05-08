---
sidebar_position: 7
title: "网关内部原理"
description: "消息网关如何启动、授权用户、路由会话和投递消息"
---

# 网关内部原理

消息网关是将 Hermes 连接到 14+ 个外部消息平台的长时间运行进程，通过统一架构。

## 关键文件

| 文件 | 用途 |
|------|------|
| `gateway/run.py` | `GatewayRunner` — 主循环、斜杠命令、消息调度（约 12,000 行） |
| `gateway/session.py` | `SessionStore` — 对话持久化和会话键构建 |
| `gateway/delivery.py` | 到目标平台/频道的出站消息投递 |
| `gateway/pairing.py` | 用户授权的 DM 配对流程 |
| `gateway/channel_directory.py` | 将聊天 ID 映射到 cron 投递的人类可读名称 |
| `gateway/hooks.py` | 钩子发现、加载和生命周期事件调度 |
| `gateway/mirror.py` | `send_message` 的跨会话消息镜像 |
| `gateway/status.py` | 配置文件作用域网关实例的 token 锁管理 |
| `gateway/builtin_hooks/` | 始终注册的钩子的扩展点（未包含） |
| `gateway/platforms/` | 平台适配器（每个消息平台一个） |

## 架构概述

```text
┌─────────────────────────────────────────────────┐
│                  GatewayRunner                    │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Telegram │  │ Discord  │  │  Slack   │       │
│  │ Adapter  │  │ Adapter  │  │ Adapter  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │
│       └─────────────┼─────────────┘             │
│                     ▼                           │
│              _handle_message()                  │
│                     │                           │
│         ┌───────────┼───────────┐               │
│         ▼           ▼           ▼               │
│  Slash command   AIAgent    Queue/BG            │
│    dispatch      creation   sessions             │
│                     │                           │
│                     ▼                           │
│                 SessionStore                    │
│              (SQLite persistence)               │
└───────┴─────────────┴─────────────┴─────────────┘
```

## 消息流

当消息从任何平台到达时：

1. **平台适配器** 接收原始事件，将其规范化为 `MessageEvent`
2. **基适配器** 检查活动会话守卫：
   - 如果代理为此会话正在运行 → 队列消息，设置中断事件
   - 如果是 `/approve`、`/deny`、`/stop` → 绕过守卫（内联调度）
3. **GatewayRunner._handle_message()** 接收事件：
   - 通过 `_session_key_for_source()` 解析会话键（格式：`agent:main:{platform}:{chat_type}:{chat_id}`）
   - 检查授权（见下文）
   - 检查是否是斜杠命令 → 调度到命令处理器
   - 检查代理是否已在运行 → 拦截命令如 `/stop`、`/status`
   - 否则 → 创建 `AIAgent` 实例并运行对话
4. **响应** 通过平台适配器发送回

### 会话键格式

会话键编码完整路由上下文：

```
agent:main:{platform}:{chat_type}:{chat_id}
```

例如：`agent:main:telegram:private:123456789`

支持主题的平台（Telegram 论坛主题、Discord 线程、Slack 线程）可以在 chat_id 部分包含主题 ID。**永远不要手动构造会话键** — 始终使用 `gateway/session.py` 中的 `build_session_key()`。

### 两级消息守卫

当代理正在主动运行时，入站消息通过两个顺序守卫：

1. **第 1 级 — 基适配器**（`gateway/platforms/base.py`）：检查 `_active_sessions`。如果会话处于活动状态，将消息队列到 `_pending_messages` 并设置中断事件。这在消息**到达**网关运行器**之前**捕获它们。

2. **第 2 级 — 网关运行器**（`gateway/run.py`）：检查 `_running_agents`。拦截特定命令（`/stop`、`/new`、`/queue`、`/status`、`/approve`、`/deny`）并适当地路由它们。其他一切触发 `running_agent.interrupt()`。

必须到达运行器而代理被阻止的命令（如 `/approve`）通过 `await self._message_handler(event)` **内联调度** — 它们绕过后台任务系统以避免竞争条件。

## 授权

网关使用多层授权检查，按顺序评估：

1. **每个平台 allow-all 标志**（例如 `TELEGRAM_ALLOW_ALL_USERS`）— 如果设置，该平台上的所有用户都被授权
2. **平台允许列表**（例如 `TELEGRAM_ALLOWED_USERS`）— 逗号分隔的用户 ID
3. **DM 配对** — 经过身份验证的用户可以通过配对码配对新用户
4. **全局 allow-all**（`GATEWAY_ALLOW_ALL_USERS`）— 如果设置，所有平台上的所有用户都被授权
5. **默认：拒绝** — 未授权用户被拒绝

### DM 配对流程

```text
Admin: /pair
Gateway: "Pairing code: ABC123. Share with the user."
New user: ABC123
Gateway: "Paired! You're now authorized."
```

配对状态在 `gateway/pairing.py` 中持久化并在重启后保留。

## 斜杠命令调度

网关中的所有斜杠命令通过相同的解析管道：

1. `resolve_command()` 从 `hermes_cli/commands.py` 将输入映射到规范名称（处理别名、前缀匹配）
2. 规范名称对照 `GATEWAY_KNOWN_COMMANDS` 检查
3. `_handle_message()` 中的处理器根据规范名称调度
4. 某些命令在配置上被门控（`CommandDef` 上的 `gateway_config_gate`）

### 运行代理守卫

在代理处理时**不得**执行的命令被提前拒绝：

```python
if _quick_key in self._running_agents:
    if canonical == "model":
        return "⏳ Agent is running — wait for it to finish or /stop first."
```

绕过命令（`/stop`、`/new`、`/approve`、`/deny`、`/queue`、`/status`）有特殊处理。

## 配置源

网关从多个来源读取配置：

| 源 | 提供内容 |
|--------|-----------------|
| `~/.hermes/.env` | API 密钥、bot token、平台凭据 |
| `~/.hermes/config.yaml` | 模型设置、工具配置、显示选项 |
| 环境变量 | 覆盖上述任何内容 |

与 CLI（使用 `load_cli_config()` 和硬编码默认值）不同，网关直接通过 YAML 加载器读取 `config.yaml`。这意味着存在于 CLI 默认值字典中但不在用户配置文件中的配置键在 CLI 和网关之间可能表现不同。

## 平台适配器

每个消息平台在 `gateway/platforms/` 中有一个适配器：

```text
gateway/platforms/
├── base.py              # BaseAdapter — 所有平台共享的逻辑
├── telegram.py          # Telegram Bot API（长轮询或 webhook）
├── discord.py           # 通过 discord.py 的 Discord bot
├── slack.py             # Slack Socket Mode
├── whatsapp.py          # WhatsApp Business Cloud API
├── signal.py            # 通过 signal-cli REST API 的 Signal
├── matrix.py            # 通过 mautrix 的 Matrix（可选 E2EE）
├── mattermost.py        # Mattermost WebSocket API
├── email.py             # 通过 IMAP/SMTP 的 Email
├── sms.py               # 通过 Twilio 的 SMS
├── dingtalk.py          # DingTalk WebSocket
├── feishu.py            # Feishu/Lark WebSocket 或 webhook
├── wecom.py             # WeCom（WeChat Work）回调
├── weixin.py            # Weixin（个人微信）通过 iLink Bot API
├── bluebubbles.py       # 通过 BlueBubbles macOS 服务器的 Apple iMessage
├── qqbot.py             # 通过官方 API v2 的 QQ Bot（腾讯 QQ）
├── webhook.py           # 入站/出站 webhook 适配器
├── api_server.py        # REST API 服务器适配器
└── homeassistant.py     # Home Assistant 对话集成
```

适配器实现公共接口：
- `connect()` / `disconnect()` — 生命周期管理
- `send_message()` — 出站消息投递
- `on_message()` — 入站消息规范化 → `MessageEvent`

### Token 锁

使用唯一凭据连接的适配器调用 `connect()` 中的 `acquire_scoped_lock()` 和 `disconnect()` 中的 `release_scoped_lock()`。这防止两个配置文件同时使用相同的 bot token。

## 投递路径

出站投递（`gateway/delivery.py`）处理：

- **直接回复** — 将响应发送回原始聊天
- **主频道投递** — 将 cron 作业输出和后台结果路由到配置的主频道
- **显式目标投递** — `send_message` 工具指定 `telegram:-1001234567890`
- **跨平台投递** — 投递到与原始消息不同的平台

Cron 作业投递**不会**镜像到网关会话历史 — 它们仅存在于自己的 cron 会话中。这是有意设计选择，以避免目标聊天中的消息交替违规。

## 钩子

网关钩子是响应生命周期事件的 Python 模块：

### 网关钩子事件

| 事件 | 触发时机 |
|-------|-----------|
| `gateway:startup` | 网关进程启动 |
| `session:start` | 新对话会话开始 |
| `session:end` | 会话完成或超时 |
| `session:reset` | 用户用 `/new` 重置会话 |
| `agent:start` | 代理开始处理消息 |
| `agent:step` | 代理完成一次工具调用迭代 |
| `agent:end` | 代理完成并返回响应 |
| `command:*` | 执行任何斜杠命令 |

钩子从 `gateway/builtin_hooks/`（始终活动）和 `~/.hermes/hooks/`（用户安装）发现。每个钩子是一个带有 `HOOK.yaml` 清单和 `handler.py` 的目录。

## 内存 Provider 集成

当内存 provider 插件（例如 Honcho）被启用时：

1. 网关创建带会话 ID 的每个消息 `AIAgent`
2. `MemoryManager` 使用会话上下文初始化 provider
3. Provider 工具（例如 `honcho_profile`、`viking_search`）通过以下路由：

```text
AIAgent._invoke_tool()
  → self._memory_manager.handle_tool_call(name, args)
    → provider.handle_tool_call(name, args)
```

4. 在会话结束/重置时，`on_session_end()` 触发清理和最终数据刷新

### 内存刷新生命周期

当会话被重置、恢复或过期时：
1. 内置内存被刷新到磁盘
2. 内存 provider 的 `on_session_end()` 钩子触发
3. 临时 `AIAgent` 运行仅内存的对话轮次
4. 然后上下文被丢弃或归档

## 后台维护

网关在消息处理旁边运行周期性维护：

- **Cron 滴答** — 检查作业调度并触发到期作业
- **会话过期** — 清理超时放弃的会话
- **内存刷新** — 在会话过期前主动刷新内存
- **缓存刷新** — 刷新模型列表和 provider 状态

## 进程管理

网关作为长时间存在的进程运行，通过以下方式管理：

- `hermes gateway start` / `hermes gateway stop` — 手动控制
- `systemctl`（Linux）或 `launchctl`（macOS）— 服务管理
- `~/.hermes/gateway.pid` 中的 PID 文件 — 配置文件作用域进程跟踪

**配置文件作用域 vs 全局**：`start_gateway()` 使用配置文件作用域 PID 文件。`hermes gateway stop` 仅停止当前配置文件的网关。`hermes gateway stop --all` 使用全局 `ps aux` 扫描杀死所有网关进程（用于更新期间）。

## 相关文档

- [会话存储](./session-storage.md)
- [Cron 内部原理](./cron-internals.md)
- [ACP 内部原理](./acp-internals.md)
- [Agent 循环内部原理](./agent-loop.md)
- [消息网关（用户指南）](/docs/user-guide/messaging)
