---
sidebar_position: 7
title: "会话"
description: "会话持久化、恢复、搜索、管理和按平台会话跟踪"
---

# 会话

Hermes Agent 自动将每个对话保存为会话。会话支持对话恢复、跨会话搜索和完整对话历史管理。

## 会话如何工作

每个对话——无论是来自 CLI、Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Teams 还是任何其他消息平台——都作为具有完整消息历史的会话存储。会话在两个互补系统中跟踪：

1. **SQLite 数据库**（`~/.hermes/state.db`）——带 FTS5 全文搜索的结构化会话元数据
2. **JSONL 转录文件**（`~/.hermes/sessions/`）——包括工具调用的原始对话转录（gateway）

SQLite 数据库存储：
- 会话 ID、源平台、用户 ID
- **会话标题**（唯一的、可读的名称）
- 模型名称和配置
- 系统提示快照
- 完整消息历史（角色、内容、工具调用、工具结果）
- Token 计数（输入/输出）
- 时间戳（started_at、ended_at）
- 父会话 ID（用于压缩触发的会话拆分）

### 会话来源

每个会话都标记有其源平台：

| 来源 | 描述 |
|--------|-------------|
| `cli` | 交互式 CLI（`hermes` 或 `hermes chat`） |
| `telegram` | Telegram 信使 |
| `discord` | Discord 服务器/DM |
| `slack` | Slack 工作区 |
| `whatsapp` | WhatsApp 信使 |
| `signal` | Signal 信使 |
| `matrix` | Matrix 房间和 DM |
| `mattermost` | Mattermost 频道 |
| `email` | 电子邮件（IMAP/SMTP） |
| `sms` | 通过 Twilio 的 SMS |
| `dingtalk` | DingTalk 信使 |
| `feishu` | Feishu/Lark 信使 |
| `wecom` | WeCom（WeChat Work） |
| `weixin` | Weixin（个人微信） |
| `bluebubbles` | 通过 BlueBubbles macOS 服务器的 Apple iMessage |
| `qqbot` | QQ Bot（腾讯 QQ）通过官方 API v2 |
| `homeassistant` | Home Assistant 对话 |
| `webhook` | 传入 webhook |
| `api-server` | API 服务器请求 |
| `acp` | ACP 编辑器集成 |
| `cron` | 计划 cron 作业 |
| `batch` | 批处理运行 |

## CLI 会话恢复

使用 `--continue` 或 `--resume` 从 CLI 恢复之前的对话：

### 继续上一个会话

```bash
# 恢复最近的 CLI 会话
hermes --continue
hermes -c

# 或使用 chat 子命令
hermes chat --continue
hermes chat -c
```

这从 SQLite 数据库查找最近的 `cli` 会话并加载其完整对话历史。

### 按名称恢复

如果您给会话指定了标题（见下面的 [会话命名](#session-naming)），您可以按名称恢复它：

```bash
# 按名称恢复会话
hermes -c "my project"

# 如果有谱系变体（my project, my project #2, my project #3），
# 这会自动恢复最新的一个
hermes -c "my project"   # → 恢复 "my project #3"
```

### 恢复特定会话

```bash
# 按 ID 恢复特定会话
hermes --resume 20250305_091523_a1b2c3d4
hermes -r 20250305_091523_a1b2c3d4

# 按标题恢复
hermes --resume "refactoring auth"

# 或使用 chat 子命令
hermes chat --resume 20250305_091523_a1b2c3d4
```

会话 ID 在您退出 CLI 会话时显示，可以使用 `hermes sessions list` 找到。

### 恢复时的对话概述

当您恢复会话时，Hermes 在输入提示之前显示一个样式化面板，其中包含上一个对话的简要概述：

<img className="docs-terminal-figure" src="/img/docs/session-recap.svg" alt="Stylized preview of the Previous Conversation recap panel shown when resuming a Hermes session." />
<p className="docs-figure-caption">恢复模式显示一个简要概述面板，其中包含最近的用户和助手轮次，然后返回到实时提示。</p>

概述：
- 显示**用户消息**（金色 `●`）和**助手响应**（绿色 `◆`）
- **截断**长消息（用户 300 字符，助手 200 字符 / 3 行）
- **折叠**工具调用为带工具名称的计数（例如 `[3 tool calls: terminal, web_search]`）
- **隐藏**系统消息、工具结果和内部推理
- **限制**为最后 10 个交换，并带有 "... N earlier messages ..." 指示器
- 使用**暗淡样式**与活跃对话区分

要禁用概述并保持最小单行行为，请在 `~/.hermes/config.yaml` 中设置：

```yaml
display:
  resume_display: minimal   # 默认：full
```

:::tip
会话 ID 遵循格式 `YYYYMMDD_HHMMSS_<hex>` — CLI/TUI 会话使用 6 字符十六进制后缀（例如 `20250305_091523_a1b2c3`），gateway 会话使用 8 字符后缀（例如 `20250305_091523_a1b2c3d4`）。您可以按 ID（完整或唯一前缀）或按标题恢复——两者都可以与 `-c` 和 `-r` 一起使用。
:::

## 会话命名

为会话指定人类可读的标题，以便您可以轻松找到和恢复它们。

### 自动生成的标题

Hermes 在第一次交换后自动为每个会话生成简短的描述性标题（3-7 个词）。这在后台线程中使用快速辅助模型运行，因此不会增加延迟。当使用 `hermes sessions list` 或 `hermes sessions browse` 浏览会话时，您会看到自动生成的标题。

自动命名每个会话只触发一次，如果您已经手动设置了标题，则跳过。

### 手动设置标题

在任何聊天会话（CLI 或 gateway）中使用 `/title` 斜杠命令：

```
/title my research project
```

标题立即应用。如果会话尚未在数据库中创建（例如您在发送第一条消息之前运行 `/title`），它会被排队并在会话开始时应用。

您也可以从命令行重命名现有会话：

```bash
hermes sessions rename 20250305_091523_a1b2c3d4 "refactoring auth module"
```

### 标题规则

- **唯一** — 没有两个会话可以共享相同的标题
- **最多 100 个字符** — 保持列表输出整洁
- **清理** — 控制字符、零宽字符和 RTL 覆盖会自动剥离
- **正常 Unicode 没问题** — 表情符号、CJK、带重音的字符都可以使用

### 压缩时的自动谱系

当会话的上下文被压缩时（手动通过 `/compress` 或自动），Hermes 会创建一个新的继续会话。如果原始会话有标题，新会话会自动获得带编号的标题：

```
"my project" → "my project #2" → "my project #3"
```

当您按名称恢复时（`hermes -c "my project"`），它会自动选择谱系中最新的会话。

### 消息平台中的 /title

`/title` 命令在所有 gateway 平台（Telegram、Discord、Slack、WhatsApp）中都可用：

- `/title My Research` — 设置会话标题
- `/title` — 显示当前标题

## 会话管理命令

Hermes 通过 `hermes sessions` 提供完整的会话管理命令集：

### 列出会话

```bash
# 列出最近的会话（默认：最后 20 个）
hermes sessions list

# 按平台筛选
hermes sessions list --source telegram

# 显示更多会话
hermes sessions list --limit 50
```

当会话有标题时，输出显示标题、预览和相对时间戳：

```
Title                  Preview                                  Last Active   ID
────────────────────────────────────────────────────────────────────────────────────────────────
refactoring auth       Help me refactor the auth module please   2h ago        20250305_091523_a
my project #3          Can you check the test failures?          yesterday     20250304_143022_e
—                      What's the weather in Las Vegas?          3d ago        20250303_101500_f
```

当没有会话有标题时，使用更简单的格式：

```
Preview                                            Last Active   Src    ID
──────────────────────────────────────────────────────────────────────────────────────
Help me refactor the auth module please             2h ago        cli    20250305_091523_a
What's the weather in Las Vegas?                    3d ago        tele   20250303_101500_f
```

### 导出会话

```bash
# 将所有会话导出到 JSONL 文件
hermes sessions export backup.jsonl

# 从特定平台导出会话
hermes sessions export telegram-history.jsonl --source telegram

# 导出单个会话
hermes sessions export session.jsonl --session-id 20250305_091523_a1b2c3d4
```

导出的文件每行包含一个 JSON 对象，具有完整的会话元数据和所有消息。

### 删除会话

```bash
# 删除特定会话（带确认）
hermes sessions delete 20250305_091523_a1b2c3d4

# 无需确认删除
hermes sessions delete 20250305_091523_a1b2c3d4 --yes
```

### 重命名会话

```bash
# 设置或更改会话的标题
hermes sessions rename 20250305_091523_a1b2c3d4 "debugging auth flow"

# 多词标题在 CLI 中不需要引号
hermes sessions rename 20250305_091523_a1b2c3d4 debugging auth flow
```

如果标题已被另一个会话使用，则显示错误。

### 清理旧会话

```bash
# 删除早于 90 天的已结束会话（默认）
hermes sessions prune

# 自定义年龄阈值
hermes sessions prune --older-than 30

# 仅清理特定平台的会话
hermes sessions prune --source telegram --older-than 60

# 跳过确认
hermes sessions prune --older-than 30 --yes
```

:::info
清理仅删除**已结束**的会话（已明确结束或自动重置的会话）。活跃会话永远不会被清理。
:::

### 会话统计

```bash
hermes sessions stats
```

输出：

```
Total sessions: 142
Total messages: 3847
  cli: 89 sessions
  telegram: 38 sessions
  discord: 15 sessions
Database size: 12.4 MB
```

有关更深入的分析——token 使用、成本估算、工具细分和活动模式——请使用 [`hermes insights`](/docs/reference/cli-commands#hermes-insights)。

## 会话搜索工具

Agent 有一个内置的 `session_search` 工具，使用 SQLite 的 FTS5 引擎对所有过去的对话执行全文搜索。

### 工作原理

1. FTS5 搜索匹配的消息，按相关性排名
2. 按会话对结果分组，获取前 N 个唯一会话（默认 3 个）
3. 加载每个会话的对话，截断为中心于匹配的 ~100K 字符
4. 发送到快速摘要模型进行聚焦摘要
5. 返回每个会话的摘要，包括元数据和周围上下文

### FTS5 查询语法

搜索支持标准 FTS5 查询语法：

- 简单关键字：`docker deployment`
- 短语：`"exact phrase"`
- 布尔值：`docker OR kubernetes`、`python NOT java`
- 前缀：`deploy*`

### 何时使用

Agent 会自动提示使用会话搜索：

> *"当用户引用过去对话中的内容或您怀疑存在相关的先前上下文时，使用 session_search 来回忆它，而不是让他们重复自己。"*

## 按平台会话跟踪

### Gateway 会话

在消息平台上，会话由从消息源构建的确定性会话键标识：

| 聊天类型 | 默认键格式 | 行为 |
|-----------|--------------------|----------|
| Telegram DM | `agent:main:telegram:dm:<chat_id>` | 每个 DM 聊天一个会话 |
| Discord DM | `agent:main:discord:dm:<chat_id>` | 每个 DM 聊天一个会话 |
| WhatsApp DM | `agent:main:whatsapp:dm:<canonical_identifier>` | 每个 DM 用户一个会话（当存在映射时，LID/电话别名折叠为一个身份） |
| 群聊 | `agent:main:<platform>:group:<chat_id>:<user_id>` | 当平台公开用户 ID 时，群组内每个用户一个会话 |
| 群线程/主题 | `agent:main:<platform>:group:<chat_id>:<thread_id>` | 所有线程参与者共享会话（默认）。使用 `thread_sessions_per_user: true` 可实现每个用户一个会话。 |
| 频道 | `agent:main:<platform>:channel:<chat_id>:<user_id>` | 当平台公开用户 ID 时，频道内每个用户一个会话 |

当 Hermes 无法获取共享聊天的参与者标识符时，它会回退到该房间的一个共享会话。

### 共享 vs 隔离的群会话

默认情况下，Hermes 在 `config.yaml` 中使用 `group_sessions_per_user: true`。这意味着：

- Alice 和 Bob 都可以在同一个 Discord 频道中与 Hermes 交谈，而无需共享转录历史
- 一个用户的长时间、工具密集型任务不会污染另一个用户的上下文窗口
- 中断处理也保持按用户，因为 running-agent 键与隔离的会话键匹配

如果您想要一个共享的"房间大脑"，请设置：

```yaml
group_sessions_per_user: false
```

这会将组/频道恢复为每个房间的单一共享会话，保留共享对话上下文，但也会共享 token 成本、中断状态和上下文增长。

### 会话重置策略

Gateway 会话根据可配置的策略自动重置：

- **idle** — 在 N 分钟不活动后重置
- **daily** — 每天在特定时间重置
- **both** — 无论哪个先（空闲或每日）重置
- **none** — 永不自动重置

在会话自动重置之前，会给 agent 一个轮次以从对话中保存任何重要的内存或 skills。

带有**活跃后台进程**的会话永远不会自动重置，无论策略如何。

## 存储位置

| 内容 | 路径 | 描述 |
|------|------|-------------|
| SQLite 数据库 | `~/.hermes/state.db` | 所有会话元数据 + 带 FTS5 的消息 |
| Gateway 转录 | `~/.hermes/sessions/` | 每个会话的 JSONL 转录 + sessions.json 索引 |
| Gateway 索引 | `~/.hermes/sessions/sessions.json` | 将会话键映射到活动会话 ID |

SQLite 数据库使用 WAL 模式进行并发读取和单一写入，这非常适合 gateway 的多平台架构。

### 数据库 Schema

`state.db` 中的关键表：

- **sessions** — 会话元数据（id、source、user_id、model、title、timestamps、token counts）。标题有唯一索引（允许 NULL，仅非 NULL 必须唯一）。
- **messages** — 完整消息历史（role、content、tool_calls、tool_name、token_count）
- **messages_fts** — FTS5 虚拟表，用于消息内容的全文搜索

## 会话过期和清理

### 自动清理

- Gateway 会话根据配置的重置策略自动重置
- 重置前，agent 会从过期会话中保存内存和 skills
- 可选自动清理：当 `sessions.auto_prune` 为 `true` 时，早于 `sessions.retention_days`（默认 90 天）的已结束会话会在 CLI/gateway 启动时清理
- 清理后实际删除行时，会对 `state.db` 执行 `VACUUM` 以回收磁盘空间（SQLite 在普通 DELETE 上不会缩小文件）
- 清理最多每 `sessions.min_interval_hours`（默认 24）运行一次；最后运行时间戳在 `state.db` 本身内部跟踪，因此在同一个 `HERMES_HOME` 中的每个 Hermes 进程共享

默认是**关闭**——会话历史对 `session_search` 召回很有价值，静默删除可能会让用户惊讶。在 `~/.hermes/config.yaml` 中启用：

```yaml
sessions:
  auto_prune: true          # 选择加入——默认是 false
  retention_days: 90        # 保留已结束会话这么多天
  vacuum_after_prune: true  # 清理扫描后回收磁盘空间
  min_interval_hours: 24    # 不要比这更频繁地运行扫描
```

活跃会话永远不会被自动清理，无论年龄如何。

### 手动清理

```bash
# 清理早于 90 天的会话
hermes sessions prune

# 删除特定会话
hermes sessions delete <session_id>

# 清理前导出（备份）
hermes sessions export backup.jsonl
hermes sessions prune --older-than 30 --yes
```

:::tip
数据库增长缓慢（典型：数百个会话 10-15 MB），会话历史支持跨过去对话的 `session_search` 召回，因此自动清理默认禁用。如果您运行大量 gateway/cron 工作负载，其中 `state.db` 明显影响性能，请启用它（观察到的故障模式：~1000 个会话的 384 MB state.db 减慢 FTS5 插入和 `/resume` 列表）。使用 `hermes sessions prune` 进行一次性清理，而无需打开自动扫描。
:::
