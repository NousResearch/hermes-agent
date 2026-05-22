---
sidebar_position: 7
title: "会话"
description: "会话持久化、恢复、搜索、管理以及按平台追踪会话"
---

# 会话

<a id="session-naming"></a>

## 会话命名 {#session-naming}

Hermes 会为每个会话自动生成可读标题；你也可以在恢复后手动重命名，以便后续搜索和归档。

Hermes Agent 自动将每次对话保存为会话。会话支持对话恢复、跨会话搜索和完整的对话历史管理。

## 会话如何工作

每次对话——无论是来自 CLI、Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Teams 还是任何其他消息平台——都会作为带有完整消息历史的会话存储。会话在两个互补的系统中追踪：

1. **SQLite 数据库** (`~/.hermes/state.db`) — 结构化会话元数据，带 FTS5 全文搜索
2. **JSONL 转录** (`~/.hermes/sessions/`) — 原始对话转录，包括工具调用（网关）

SQLite 数据库存储：
- 会话 ID、来源平台、用户 ID
- **会话标题**（唯一、人类可读的名称）
- 模型名称和配置
- 系统提示快照
- 完整消息历史（角色、内容、工具调用、工具结果）
- Token 计数（输入/输出）
- 时间戳（started_at、ended_at）
- 父会话 ID（用于压缩触发的会话拆分）

### 会话来源

每个会话都标记有其来源平台：

| 来源 | 说明 |
|--------|-------------|
| `cli` | 交互式 CLI (`hermes` 或 `hermes chat`) |
| `telegram` | Telegram 消息 |
| `discord` | Discord 服务器/DM |
| `slack` | Slack 工作区 |
| `whatsapp` | WhatsApp 消息 |
| `signal` | Signal 消息 |
| `matrix` | Matrix 房间和 DM |
| `mattermost` | Mattermost 频道 |
| `email` | 邮件（IMAP/SMTP） |
| `sms` | 通过 Twilio 的短信 |
| `dingtalk` | 钉钉消息 |
| `feishu` | 飞书/Lark 消息 |
| `wecom` | 企业微信 |
| `weixin` | 微信（个人微信） |
| `bluebubbles` | 通过 BlueBubbles macOS 服务器的 Apple iMessage |
| `qqbot` | QQ 机器人（腾讯 QQ）通过官方 API v2 |
| `homeassistant` | Home Assistant 对话 |
| `webhook` | 传入 webhook |
| `api-server` | API 服务器请求 |
| `acp` | ACP 编辑器集成 |
| `cron` | 定时 cron 任务 |
| `batch` | 批处理运行 |

## CLI 会话恢复

使用 `--continue` 或 `--resume` 从 CLI 恢复之前的对话：

### 继续上次会话

```bash
# 恢复最近的 CLI 会话
hermes --continue
hermes -c

# 或使用 chat 子命令
hermes chat --continue
hermes chat -c
```

这会从 SQLite 数据库中查找最近的 `cli` 会话并加载其完整对话历史。

### 按名称恢复

如果你为会话设置了标题（见下方的[会话命名](#session-naming)），可以按名称恢复：

```bash
# 按名称恢复会话
hermes -c "my project"

# 如果有血统变体（my project、my project #2、my project #3），
# 这会自动恢复最近的一个
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

会话 ID 在你退出 CLI 会话时显示，也可以通过 `hermes sessions list` 找到。

### 恢复时的对话摘要 {#conversation-recap-on-resume}

恢复会话时，Hermes 会在输入提示前以样式化面板显示之前对话的紧凑摘要：

<img className="docs-terminal-figure" src="/img/docs/session-recap.svg" alt="恢复 Hermes 会话时显示的 Previous Conversation 摘要面板的风格化预览。" />
<p className="docs-figure-caption">恢复模式显示一个紧凑的摘要面板，包含最近的用户和助手回合，然后返回到实时提示。</p>

摘要：
- 显示**用户消息**（金色 `●`）和**助手回复**（绿色 `◆`）
- **截断**长消息（用户 300 字符，助手 200 字符 / 3 行）
- **折叠**工具调用为计数和工具名称（例如 `[3 tool calls: terminal, web_search]`）
- **隐藏**系统消息、工具结果和内部推理
- **上限**为最近 10 次交换，带 "... N earlier messages ..." 指示器
- 使用**暗淡样式**以区别于活跃对话

要禁用摘要并保持最小的一行行为，在 `~/.hermes/config.yaml` 中设置：

```yaml
display:
  resume_display: minimal   # 默认: full
```

:::tip
会话 ID 遵循格式 `YYYYMMDD_HHMMSS_<hex>` — CLI/TUI 会话使用 6 字符十六进制后缀（例如 `20250305_091523_a1b2c3`），网关会话使用 8 字符后缀（例如 `20250305_091523_a1b2c3d4`）。你可以按 ID（完整或唯一前缀）或按标题恢复——两者都适用于 `-c` 和 `-r`。
:::

## 跨平台切换

在 CLI 会话中使用 `/handoff <platform>` 将活跃对话转移到消息平台的主频道。代理会从 CLI 离开的地方精确继续——相同的会话 ID、完整的角色感知转录、工具调用等等。

```bash
# 在 CLI 会话内部
/handoff telegram
```

发生了什么：

1. CLI 验证 `<platform>` 已启用并设置了主频道（从目标聊天中运行一次 `/sethome` 来配置）。
2. CLI 标记会话为待处理并**阻塞轮询网关**。如果代理正在回合中它会拒绝——等待当前响应完成。
3. 网关 watcher 接管切换并向目标适配器请求一个新线程：
   - **Telegram** — 打开一个新论坛主题（如果聊天中启用了 Bot API 9.4+ Topics 模式，则为 DM 主题，或论坛超级组主题）。
   - **Discord** — 在主文本频道下创建一个 1440 分钟自动归档的线程。
   - **Slack** — 发布一条种子消息并使用其 `ts` 作为线程锚点。
   - **WhatsApp / Signal / Matrix / SMS** — 没有原生线程，直接回退到主频道。
4. 网关将目标键重新绑定到你现有的 CLI 会话 ID，然后伪造一个合成用户回合让代理确认并总结。回复会落在新线程中。
5. 当网关确认成功时，CLI 打印一个 `/resume` 提示并干净退出：

   ```
   ↻ Handoff complete. The session is now active on telegram.
     Resume it on this CLI later with: /resume my-session-title
   ```

6. 从那时起，对话存在于平台上。在新线程中回复——该频道中任何授权的人都共享同一会话，之后线程中的任何真实用户消息都会无缝加入，因为线程会话键不带 `user_id`。

**恢复回 CLI：** 当你想回到桌面时，只需运行 `/resume <title>`（或从 shell 运行 `hermes -r "<title>"`），从平台离开的地方继续。

**失败模式：**
- 未配置主频道 → CLI 拒绝并给出 `/sethome` 提示。
- 平台未启用 / 网关未运行 → CLI 在 60 秒后超时并给出明确消息，你的 CLI 会话保持完整。
- 线程创建失败（权限、topics 模式关闭）→ 直接回退到主频道仍然完成；没有线程隔离但切换本身有效。
- `adapter.send` 失败（速率限制、临时 API 错误）→ 切换标记为失败并给出原因；该行会清除以便你可以重试。

**值得了解的限制：** 对于没有线程能力的平台，在多用户群组主频道中，合成回合键入为 DM 风格会话。这对自 DM 主频道（典型设置）有效，但对真正的共享群组聊天并不理想。线程覆盖了 Telegram / Discord / Slack——绝大多数常见情况——所以大多数设置永远不会遇到这个问题。

## 会话命名

为会话赋予人类可读的标题，以便轻松查找和恢复。

### 自动生成标题

Hermes 在第一次交换后自动为每个会话生成一个简短的描述性标题（3–7 个词）。这在后台线程中使用快速的辅助模型运行，因此不会增加延迟。浏览会话时你会看到自动生成的标题，使用 `hermes sessions list` 或 `hermes sessions browse`。

自动标题每个会话只触发一次，如果你已手动设置标题则跳过。

### 手动设置标题

在任何聊天会话（CLI 或网关）中使用 `/title` 斜杠命令：

```
/title my research project
```

标题会立即应用。如果会话尚未在数据库中创建（例如，你在发送第一条消息前运行 `/title`），它会被排队并在会话开始后应用。

你也可以从命令行重命名现有会话：

```bash
hermes sessions rename 20250305_091523_a1b2c3d4 "refactoring auth module"
```

### 标题规则

- **唯一** — 两个会话不能共享相同标题
- **最多 100 字符** — 保持列表输出整洁
- **已清理** — 控制字符、零宽字符和 RTL 覆盖会自动剥离
- **正常 Unicode 没问题** — 表情符号、CJK、带重音符号的字符都可以

### 压缩时的自动血统

当会话的上下文被压缩（手动通过 `/compress` 或自动）时，Hermes 创建一个新的延续会话。如果原始会话有标题，新会话会自动获得一个编号标题：

```
"my project" → "my project #2" → "my project #3"
```

当你按名称恢复时（`hermes -c "my project"`），它会自动选择血统中最近的会话。

### 消息平台中的 /title

`/title` 命令在所有网关平台（Telegram、Discord、Slack、WhatsApp）中都有效：

- `/title My Research` — 设置会话标题
- `/title` — 显示当前标题

## 会话管理命令

Hermes 通过 `hermes sessions` 提供一套完整的会话管理命令：

### 列出会话

```bash
# 列出最近会话（默认：最近 20 个）
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
# 导出所有会话到 JSONL 文件
hermes sessions export backup.jsonl

# 导出特定平台的会话
hermes sessions export telegram-history.jsonl --source telegram

# 导出单个会话
hermes sessions export session.jsonl --session-id 20250305_091523_a1b2c3d4
```

导出文件每行包含一个 JSON 对象，带有完整会话元数据和所有消息。

### 删除会话

```bash
# 删除特定会话（带确认）
hermes sessions delete 20250305_091523_a1b2c3d4

# 无需确认删除
hermes sessions delete 20250305_091523_a1b2c3d4 --yes
```

### 重命名会话

```bash
# 设置或更改会话标题
hermes sessions rename 20250305_091523_a1b2c3d4 "debugging auth flow"

# 多词标题在 CLI 中不需要引号
hermes sessions rename 20250305_091523_a1b2c3d4 debugging auth flow
```

如果标题已被另一个会话使用，会显示错误。

### 清理旧会话

```bash
# 删除 90 天前（默认）的已结束会话
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

对于更深入的分析——token 使用、成本估算、工具分解和活动模式——使用 [`hermes insights`](/reference/cli-commands#hermes-insights)。

## 会话搜索工具

代理有一个内置的 `session_search` 工具，使用 SQLite 的 FTS5 引擎在所有过去的对话中执行全文搜索。

### 工作原理

1. FTS5 搜索按相关性排名的匹配消息
2. 按会话分组结果，取前 N 个唯一会话（默认 3）
3. 加载每个会话的对话，截断为以匹配为中心的约 100K 字符
4. 发送到快速总结模型进行聚焦摘要
5. 返回带元数据和上下文的每会话摘要

### FTS5 查询语法

搜索支持标准 FTS5 查询语法：

- 简单关键词：`docker deployment`
- 短语：`"exact phrase"`
- 布尔：`docker OR kubernetes`，`python NOT java`
- 前缀：`deploy*`

### 何时使用

代理被提示自动使用会话搜索：

> *"当用户引用过去对话中的内容，或你怀疑存在相关的先前上下文时，使用 session_search 来回忆它，然后再要求用户重复。"*

## 按平台追踪会话

### 网关会话

在消息平台上，会话由从消息来源构建的确定性会话键键入：

| 聊天类型 | 默认键格式 | 行为 |
|-----------|--------------------|----------|
| Telegram DM | `agent:main:telegram:dm:<chat_id>` | 每个 DM 聊天一个会话 |
| Discord DM | `agent:main:discord:dm:<chat_id>` | 每个 DM 聊天一个会话 |
| WhatsApp DM | `agent:main:whatsapp:dm:<canonical_identifier>` | 每个 DM 用户一个会话（存在映射时 LID/电话别名折叠为一个身份） |
| 群组聊天 | `agent:main:<platform>:group:<chat_id>:<user_id>` | 平台暴露用户 ID 时群组内按用户 |
| 群组线程/主题 | `agent:main:<platform>:group:<chat_id>:<thread_id>` | 所有线程参与者共享会话（默认）。`thread_sessions_per_user: true` 时按用户。 |
| 频道 | `agent:main:<platform>:channel:<chat_id>:<user_id>` | 平台暴露用户 ID 时频道内按用户 |

当 Hermes 无法获取共享聊天的参与者标识符时，它会回退到该房间的一个共享会话。

### 共享与隔离的群组会话

默认情况下，Hermes 在 `config.yaml` 中使用 `group_sessions_per_user: true`。这意味着：

- Alice 和 Bob 可以在同一个 Discord 频道中与 Hermes 交谈而不共享转录历史
- 一个用户的长工具密集型任务不会污染另一个用户的上下文窗口
- 中断处理也保持按用户，因为运行代理键匹配隔离的会话键

如果你想要一个共享的"房间大脑"，设置：

```yaml
group_sessions_per_user: false
```

这将群组/频道恢复为每个房间一个共享会话，保留共享对话上下文但也共享 token 成本、中断状态和上下文增长。

### 会话重置策略

网关会话根据可配置的策略自动重置：

- **idle** — 不活动 N 分钟后重置
- **daily** — 每天特定时间重置
- **both** — 以先发生者为准（空闲或每日）
- **none** — 从不自动重置

会话自动重置前，代理会获得一个回合来保存对话中任何重要的记忆或技能。

带有**活跃后台进程**的会话永远不会自动重置，无论策略如何。

## 存储位置

| 内容 | 路径 | 说明 |
|------|------|-------------|
| SQLite 数据库 | `~/.hermes/state.db` | 所有会话元数据 + 消息，带 FTS5 |
| 网关转录 | `~/.hermes/sessions/` | 每会话 JSONL 转录 + sessions.json 索引 |
| 网关索引 | `~/.hermes/sessions/sessions.json` | 将会话键映射到活跃会话 ID |

SQLite 数据库使用 WAL 模式以支持并发读取器和单写入器，这非常适合网关的多平台架构。

### 数据库架构

`state.db` 中的关键表：

- **sessions** — 会话元数据（id、source、user_id、model、title、timestamps、token counts）。标题有唯一索引（允许 NULL 标题，仅非 NULL 必须唯一）。
- **messages** — 完整消息历史（role、content、tool_calls、tool_name、token_count）
- **messages_fts** — FTS5 虚拟表，用于跨消息内容的全文搜索

## 会话过期和清理

### 自动清理

- 网关会话根据配置的重置策略自动重置
- 重置前，代理保存过期会话中的记忆和技能
- 可选自动清理：当 `sessions.auto_prune` 为 `true` 时，超过 `sessions.retention_days`（默认 90）天的已结束会话在 CLI/网关启动时被清理
- 实际删除行后，`state.db` 会执行 `VACUUM` 以回收磁盘空间（SQLite 在普通 DELETE 时不会缩小文件）
- 清理最多每 `sessions.min_interval_hours`（默认 24）运行一次；最后运行时间戳在 `state.db` 本身内部追踪，因此同一 `HERMES_HOME` 中的所有 Hermes 进程共享

默认是**关闭**的——会话历史对 `session_search` 回忆很有价值，静默删除可能会让用户惊讶。在 `~/.hermes/config.yaml` 中启用：

```yaml
sessions:
  auto_prune: true          # 选择加入 — 默认 false
  retention_days: 90        # 保留已结束会话这么多天
  vacuum_after_prune: true  # 清理后回收磁盘空间
  min_interval_hours: 24    # 不要比这更频繁地重新运行清理
```

活跃会话永远不会自动清理，无论年龄。

### 手动清理

```bash
# 清理 90 天前的会话
hermes sessions prune

# 删除特定会话
hermes sessions delete <session_id>

# 清理前导出（备份）
hermes sessions export backup.jsonl
hermes sessions prune --older-than 30 --yes
```

:::tip
数据库增长缓慢（典型：数百个会话约 10-15 MB），会话历史为 `session_search` 跨过去对话的回忆提供支持，因此自动清理默认禁用。如果你运行的网关/cron 工作负载很重，`state.db` 确实影响性能（观察到的故障模式：约 1000 个会话的 384 MB state.db 减慢 FTS5 插入和 `/resume` 列表），则启用它。使用 `hermes sessions prune` 进行一次性清理，无需打开自动清理。
:::

