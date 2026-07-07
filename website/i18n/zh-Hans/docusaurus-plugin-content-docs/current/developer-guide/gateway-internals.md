---
sidebar_position: 7
title: "Gateway 内部机制"
description: "消息 gateway 如何启动、授权用户、路由会话以及投递消息"
---

# Gateway 内部机制

消息 gateway 是一个长期运行的进程，通过统一架构将 Hermes 连接到 20 余个外部消息平台。

## 关键文件

| 文件 | 用途 |
|------|---------|
| `gateway/run.py` | `GatewayRunner` — 主循环、斜杠命令、消息分发（大文件；请查看 git 获取当前行数） |
| `gateway/session.py` | `SessionStore` — 会话持久化与会话键构造 |
| `gateway/delivery.py` | 向目标平台/频道投递出站消息 |
| `gateway/pairing.py` | 用于用户授权的 DM 配对流程 |
| `gateway/channel_directory.py` | 将聊天 ID 映射为可读名称，用于 cron 投递 |
| `gateway/hooks.py` | Hook（钩子）发现、加载与生命周期事件分发 |
| `gateway/mirror.py` | 为 `send_message` 提供跨会话消息镜像 |
| `gateway/status.py` | 面向 profile 范围的 gateway 实例的 token 锁管理 |
| `gateway/builtin_hooks/` | 始终注册的 hook 扩展点（当前未内置任何 hook） |
| `gateway/platforms/` | 平台适配器（每个消息平台一个） |

## 架构概览

```text
┌─────────────────────────────────────────────────┐
│                  GatewayRunner                  │
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
│    dispatch      creation   sessions            │
│                     │                           │
│                     ▼                           │
│                 SessionStore                    │
│              (SQLite persistence)               │
└───────┴─────────────┴─────────────┴─────────────┘
```

## 消息流程

当消息从任意平台到达时：

1. **平台适配器**接收原始事件，将其规范化为 `MessageEvent`
2. **基础适配器**检查活跃会话守卫：
   - 若该会话的 agent 正在运行 → 将消息加入队列，设置中断事件
   - 若为 `/approve`、`/deny`、`/stop` → 绕过守卫（内联分发）
3. **GatewayRunner._handle_message()** 接收事件：
   - 通过 `_session_key_for_source()` 解析会话键（格式：`agent:main:{platform}:{chat_type}:{chat_id}`）
   - 检查授权（见下方授权章节）
   - 检查是否为斜杠命令 → 分发至命令处理器
   - 检查 agent 是否已在运行 → 拦截 `/stop`、`/status` 等命令
   - 否则 → 创建 `AIAgent` 实例并运行对话
4. **响应**通过平台适配器回传

### 会话键格式

会话键编码了完整的路由上下文：

```
agent:main:{platform}:{chat_type}:{chat_id}
```

示例：`agent:main:telegram:private:123456789`

支持线程的平台（Telegram 论坛话题、Discord 线程、Slack 线程）可能在 chat_id 部分包含线程 ID。**切勿手动构造会话键** — 请始终使用 `gateway/session.py` 中的 `build_session_key()`。

### 两级消息守卫

当 agent 正在运行时，传入消息会依次经过两级守卫：

1. **第一级 — 基础适配器**（`gateway/platforms/base.py`）：检查 `_active_sessions`。若会话处于活跃状态，将消息加入 `_pending_messages` 队列并设置中断事件。此级在消息到达 gateway runner *之前*进行拦截。

2. **第二级 — Gateway runner**（`gateway/run.py`）：检查 `_running_agents`。拦截特定命令（`/stop`、`/new`、`/queue`、`/status`、`/approve`、`/deny`）并进行相应路由。其余所有消息触发 `running_agent.interrupt()`。

必须在 agent 被阻塞时到达 runner 的命令（如 `/approve`）通过 `await self._message_handler(event)` **内联**分发 — 绕过后台任务系统以避免竞态条件。

## 授权

Gateway 使用多层授权检查，按顺序评估：

1. **平台级全量放行标志**（如 `TELEGRAM_ALLOW_ALL_USERS`）— 若设置，该平台所有用户均被授权
2. **平台白名单**（如 `TELEGRAM_ALLOWED_USERS`）— 逗号分隔的用户 ID
3. **DM 配对** — 已认证用户可通过配对码为新用户授权
4. **全局放行标志**（`GATEWAY_ALLOW_ALL_USERS`）— 若设置，所有平台的所有用户均被授权
5. **默认：拒绝** — 未授权用户被拒绝

### DM 配对流程

```text
Admin: /pair
Gateway: "Pairing code: ABC123. Share with the user."
New user: ABC123
Gateway: "Paired! You're now authorized."
```

配对状态持久化于 `gateway/pairing.py`，重启后仍然有效。

## 斜杠命令分发

Gateway 中所有斜杠命令均经过相同的解析流程：

1. `hermes_cli/commands.py` 中的 `resolve_command()` 将输入映射为规范名称（处理别名、前缀匹配）
2. 规范名称与 `GATEWAY_KNOWN_COMMANDS` 进行比对
3. `_handle_message()` 中的处理器根据规范名称进行分发
4. 部分命令受配置门控（`CommandDef` 上的 `gateway_config_gate`）

### 运行中 Agent 守卫

在 agent 处理消息期间不得执行的命令会被提前拒绝：

```python
if _quick_key in self._running_agents:
    if canonical == "model":
        return "⏳ Agent is running — wait for it to finish or /stop first."
```

绕过命令（`/stop`、`/new`、`/approve`、`/deny`、`/queue`、`/status`）具有特殊处理逻辑。

## 配置来源

Gateway 从多个来源读取配置：

| 来源 | 提供内容 |
|--------|-----------------|
| `~/.hermes/.env` | API 密钥、bot token、平台凭据 |
| `~/.hermes/config.yaml` | 模型设置、工具配置、显示选项 |
| 环境变量 | 覆盖上述任意配置 |

与 CLI（使用带硬编码默认值的 `load_cli_config()`）不同，gateway 通过 YAML 加载器直接读取 `config.yaml`。这意味着存在于 CLI 默认值字典但不在用户配置文件中的配置键，在 CLI 和 gateway 之间可能表现不同。

## 平台适配器

大多数消息平台以插件适配器形式位于 `plugins/platforms/<name>/adapter.py`；少数旧适配器仍直接位于 `gateway/platforms/`。它们都继承 `gateway/platforms/base.py` 中的 `BasePlatformAdapter`：

```text
plugins/platforms/                  # 插件打包的适配器（每个一个目录）
├── telegram/adapter.py     # Telegram Bot API（长轮询或 webhook）
├── discord/adapter.py      # Discord bot（通过 discord.py）
├── slack/adapter.py        # Slack Socket Mode
├── whatsapp/adapter.py     # WhatsApp Business Cloud API
├── matrix/adapter.py       # Matrix（通过 mautrix，可选 E2EE）
├── mattermost/adapter.py   # Mattermost WebSocket API
├── email/adapter.py        # 电子邮件（通过 IMAP/SMTP）
├── sms/adapter.py          # 短信（通过 Twilio）
├── dingtalk/adapter.py     # 钉钉 WebSocket
├── feishu/adapter.py       # 飞书/Lark WebSocket 或 webhook
├── wecom/adapter.py        # 企业微信（WeCom）回调
├── line/adapter.py         # LINE Messaging API
├── teams/adapter.py        # Microsoft Teams
├── irc/adapter.py          # IRC（作用域锁的标准示例）
├── homeassistant/adapter.py # Home Assistant 对话集成
└── …                       # google_chat、ntfy、photon、raft、simplex 等

gateway/platforms/                  # 核心 base 与旧的直接适配器
├── base.py              # BasePlatformAdapter — 所有平台的共享逻辑
├── signal.py            # Signal（通过 signal-cli REST API）
├── weixin.py            # 微信（个人版，通过 iLink Bot API）
├── bluebubbles.py       # Apple iMessage（通过 BlueBubbles macOS 服务端）
├── qqbot/               # QQ Bot（腾讯 QQ，通过官方 API v2，子包）
├── yuanbao.py           # 元宝（腾讯）私信/群组适配器
├── msgraph_webhook.py   # Microsoft Graph 变更通知 webhook（Teams、Outlook 等）
├── webhook.py           # 入站/出站 webhook 适配器
└── api_server.py        # REST API 服务器适配器
```

适配器共享 `BasePlatformAdapter` 契约：
- `connect(*, is_reconnect=False)` / `disconnect()` — 生命周期管理
- `send(chat_id, content, reply_to=None, metadata=None)` — 出站消息投递
- `set_message_handler(handler)` — 安装接收入站 `MessageEvent` 的 gateway 回调

每个适配器的平台专属监听器会把入站平台事件规范化为 `MessageEvent`，再交给 base 的 `handle_message(event)` 流程。

### Token 锁

使用唯一凭据连接的适配器在 `connect()` 中调用 `acquire_scoped_lock()`，在 `disconnect()` 中调用 `release_scoped_lock()`。这可防止两个 profile 同时使用同一 bot token。

## 投递路径

出站投递（`gateway/delivery.py`）处理以下场景：

- **直接回复** — 将响应发回原始聊天
- **主频道投递** — 将 cron 任务输出和后台结果路由至已配置的主频道
- **显式目标投递** — `send_message` 工具指定 `telegram:-1001234567890`，或通过 [`hermes send` CLI](/guides/pipe-script-output) 封装同一工具供 shell 脚本使用
- **跨平台投递** — 投递至与原始消息不同的平台

Cron 任务投递**不会**镜像到 gateway 会话历史中 — 它们仅存在于各自的 cron 会话中。这是有意为之的设计选择，以避免消息交替违规。

## Hooks

Gateway hook 是响应生命周期事件的 Python 模块。

### Gateway Hook 事件

| 事件 | 触发时机 |
|-------|-----------|
| `gateway:startup` | Gateway 进程启动时 |
| `session:start` | 新对话会话开始时 |
| `session:end` | 会话完成或超时时 |
| `session:reset` | 用户通过 `/new` 重置会话时 |
| `agent:start` | Agent 开始处理消息时 |
| `agent:step` | Agent 完成一次工具调用迭代时 |
| `agent:end` | Agent 完成并返回响应时 |
| `command:*` | 任意斜杠命令被执行时 |

Hook 从 `gateway/builtin_hooks/`（扩展点 — 当前发行版中为空；`_register_builtin_hooks()` 是一个空操作存根）和 `~/.hermes/hooks/`（用户安装）中发现。每个 hook 是一个包含 `HOOK.yaml` 清单和 `handler.py` 的目录。

## 内存提供者集成

当内存提供者插件（如 Honcho）启用时：

1. Gateway 为每条消息创建一个带会话 ID 的 `AIAgent`
2. `MemoryManager` 使用会话上下文初始化提供者
3. 提供者工具（如 `honcho_profile`、`viking_search`）通过以下路径路由：

```text
AIAgent._invoke_tool()
  → self._memory_manager.handle_tool_call(name, args)
    → provider.handle_tool_call(name, args)
```

4. 会话结束/重置时，`on_session_end()` 触发以进行清理和最终数据刷写

### 内存刷写生命周期

当会话被重置、恢复或过期时：
1. 内置内存刷写至磁盘
2. 内存提供者的 `on_session_end()` hook 触发
3. 临时 `AIAgent` 运行仅含内存的对话轮次
4. 上下文随后被丢弃或归档

## 后台维护

Gateway 在处理消息的同时运行周期性维护任务：

- **Cron 计时** — 检查任务计划并触发到期任务
- **会话过期** — 超时后清理废弃会话
- **内存刷写** — 在会话过期前主动刷写内存
- **缓存刷新** — 刷新模型列表和提供者状态

## 进程管理

Gateway 作为长期运行进程运行，管理方式如下：

- `hermes gateway start` / `hermes gateway stop` — 手动控制
- `systemctl`（Linux）或 `launchctl`（macOS）— 服务管理
- PID 文件位于 `~/.hermes/gateway.pid` — 面向 profile 的进程追踪

**Profile 范围 vs 全局**：`start_gateway()` 使用 profile 范围的 PID 文件。`hermes gateway stop` 仅停止当前 profile 的 gateway。`hermes gateway stop --all` 使用全局 `ps aux` 扫描来终止所有 gateway 进程（用于更新时）。

## 相关文档

- [会话存储](./session-storage.md)
- [Cron 内部机制](./cron-internals.md)
- [ACP 内部机制](./acp-internals.md)
- [Agent 循环内部机制](./agent-loop.md)
- [消息 Gateway（用户指南）](/user-guide/messaging)
