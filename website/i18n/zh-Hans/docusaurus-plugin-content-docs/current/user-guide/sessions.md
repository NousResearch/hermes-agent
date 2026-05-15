---
sidebar_position: 7
title: "会话"
description: "会话持久化、恢复、搜索、管理以及跨平台会话追踪"
---

# 会话

Hermes Agent 会自动将每一次对话保存为会话。会话支持对话恢复、跨会话搜索以及完整的对话历史管理。

## 会话工作原理

无论是来自 CLI、Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Teams，还是其他任何消息平台的对话，都会以会话的形式存储完整的消息历史。会话在两个互补系统中进行追踪：

1. **SQLite 数据库** (`~/.hermes/state.db`) — 结构化的会话元数据，支持 FTS5 全文搜索
2. **JSONL 转录文件** (`~/.hermes/sessions/`) — 包含工具调用（gateway）的原始对话转录

SQLite 数据库存储：
- 会话 ID、来源平台、用户 ID
- **会话标题**（唯一、可读的名称）
- 模型名称和配置
- 系统提示快照
- 完整的消息历史（角色、内容、工具调用、工具结果）
- Token 计数（输入/输出）
- 时间戳（started_at、ended_at）
- 父会话 ID（用于压缩触发的会话拆分）

### 什么计入上下文

Hermes 会存储会话历史以便恢复对话，但不会重新发送它曾经处理过的每一个字节。每一次轮次，模型只会看到：
- 选中的系统提示
- 当前对话窗口
- Hermes 为该轮次显式注入的任何内容

媒体附件作为轮次范围的输入处理：

- 图像可以直接附加到下一个模型调用，或者在模型不支持原生视觉时被预先分析成文字描述。
- 音频在配置了语音转文字时会被转录为文本。
- 文本文档可以提取文本后包含；其他文档类型通常以本地路径和简短说明的形式表示。
- 附件路径以及提取/派生的文本可以出现在转录中，但原始图片、音频或二进制文件的字节不会反复复制到后续提示中。

例如，用户发送一张图片并要求 Hermes 制作表情包，Hermes 可能会使用视觉一次检查图像并运行图像处理脚本。之后的轮次不会自动携带原始 JPEG，而只会携带用户的请求、简短的图像描述、缓存路径或最终助手的回复。

导致上下文增长的最常见原因不是媒体文件本身，而是冗长的文字：粘贴的转录、完整日志、庞大的工具输出、长 diff、重复的状态报告以及详细的证明输出。建议使用摘要、文件路径、聚焦的片段以及基于工具的检索，而不是把大型资产复制进聊天。

:::tip
使用 `/compress` 当会话过长，使用 `/new` 开启新线程，使用 `hermes sessions prune` 仅在想要从存储中删除已结束的旧会话时使用。压缩会减少活动上下文；它并不等同于隐私删除。
:::

### 会话来源

每个会话都带有来源平台标签：

| 来源 | 描述 |
|------|------|
| `cli` | 交互式 CLI（`hermes` 或 `hermes chat`） |
| `telegram` | Telegram 消息 |
| `discord` | Discord 服务器/私信 |
| `slack` | Slack 工作区 |
| `whatsapp` | WhatsApp 消息 |
| `signal` | Signal 消息 |
| `matrix` | Matrix 房间和私信 |
| `mattermost` | Mattermost 频道 |
| `email` | Email（IMAP/SMTP） |
| `sms` | 通过 Twilio 的 SMS |
| `dingtalk` | 钉钉 |
| `feishu` | 飞书/Lark |
| `wecom` | 企业微信 |
| `weixin` | 个人微信 |
| `bluebubbles` | macOS 上的 Apple iMessage (BlueBubbles) |
| `qqbot` | 腾讯 QQ Bot（官方 API v2） |
| `homeassistant` | Home Assistant 对话 |
| `webhook` | 入站 webhook |
| `api-server` | API 服务器请求 |
| `acp` | ACP 编辑器集成 |
| `cron` | 定时任务 |
| `batch` | 批处理运行 |

## CLI 会话恢复

使用 `--continue` 或 `--resume` 从 CLI 恢复先前的对话：

### 恢复最近的会话

```bash
# 恢复最近的 CLI 会话
hermes --continue
hermes -c

# 或使用 chat 子命令
hermes chat --continue
hermes chat -c
```
这会从 SQLite 数据库中查找最近的 `cli` 会话并加载完整的对话历史。

### 按名称恢复

如果为会话设置了标题（参见下面的 **会话命名**），可以按名称恢复：

```bash
# 恢复已命名的会话
hermes -c "my project"

# 当存在系列变体（my project、my project #2、my project #3）时，自动恢复最近的那个
hermes -c "my project"   # → 恢复 "my project #3"
```

### 按会话 ID 恢复

```bash
# 通过会话 ID 恢复
hermes --resume 20250305_091523_a1b2c3d4
hermes -r 20250305_091523_a1b2c3d4

# 按标题恢复
hermes --resume "refactoring auth"

# 使用 chat 子命令
hermes chat --resume 20250305_091523_a1b2c3d4
```
会话 ID 会在退出 CLI 会话时显示，也可以通过 `hermes sessions list` 查看。

### 恢复时的对话概览

恢复会话时，Hermes 会在输入提示前以样式化面板展示上一段对话的简要概览：

![恢复概览示例](/img/docs/session-recap.svg)
<p class="docs-figure-caption">恢复模式会在活跃提示前显示最近用户和助手的交互摘要。</p>

概览特点：
- 显示 **用户消息**（金色 `●`）和 **助手回复**（绿色 `◆`）
- 对长消息进行截断（用户 300 字，助手 200 字/3 行）
- 将工具调用折叠为计数并列出工具名称，例如 `[3 tool calls: terminal, web_search]`
- 隐藏系统消息、工具结果和内部推理
- 最多保留最近 10 轮，对更早的消息用 "... N earlier messages ..." 标记
- 使用 **暗淡** 样式区分于活跃对话

如果想禁用完整概览并保持单行行为，可在 `~/.hermes/config.yaml` 中设置：

```yaml
display:
  resume_display: minimal   # 默认 full
```

:::tip
会话 ID 采用 `YYYYMMDD_HHMMSS_<hex>` 格式——CLI/TUI 会话使用 6 位十六进制后缀（如 `20250305_091523_a1b2c3`），网关会话使用 8 位后缀（如 `20250305_091523_a1b2c3d4`）。可以使用完整 ID、唯一前缀或标题进行恢复，均适用于 `-c` 与 `-r`。
:::

## 跨平台移交

在 CLI 会话中使用 `/handoff <platform>` 将实时对话转移到消息平台的主频道。代理会从同一个会话 ID、完整角色转录以及所有工具调用继续。

```bash
# 在 CLI 会话内部
/handoff telegram
```

移交过程：
1. CLI 验证 `<platform>` 已启用并设置了主频道（在目标聊天中运行 `/sethome` 配置）。
2. CLI 将会话标记为 pending 并 **阻塞轮询** 网关。若代理正处于回复中，会拒绝并提示等待。
3. 网关监视器获取移交并向目标适配器请求新线程：
   - **Telegram**：打开新论坛主题（若 Bot API 9.4+ 开启 Topics）或 DM 主题。
   - **Discord**：在主文字频道下创建 1440 分钟自动归档线程。
   ...（同前文）
4. 网关重新绑定目标键到现有 CLI 会话 ID，并伪造用户回合让代理确认并概括。回复落在新线程中。
5. 网关确认成功后，CLI 打印 `/resume` 提示并优雅退出。
6. 之后对话在平台上继续，任意有权限的成员共享同一会话。线程会话不携带 `user_id`，因此同一线程内的所有用户共享上下文。

**返回 CLI**：运行 `/resume <title>`（或 `hermes -r "<title>"`）即可重新接管会话。

**错误情况**：
- 未配置主频道 → CLI 提示 `/sethome`。
- 平台未启用或网关未运行 → CLI 在 60 秒超时后给出明确信息，会话保持完整。
- 线程创建失败（权限、未开启 topics） → 回退到主频道，仍完成移交。
- `adapter.send` 失败（限流、临时 API 错误） → 标记失败并给出原因，可重试。

## 会话命名

为会话赋予可读标题，便于查找和恢复。

### 自动生成标题

Hermes 会在首次交互后自动为每个会话生成 3–7 个词的简短描述标题。此过程在后台线程使用辅助模型完成，不会增加延迟。当使用 `hermes sessions list` 或 `hermes sessions browse` 浏览会话时会看到自动标题。每个会话仅生成一次，手动设置标题后自动标题会被跳过。

### 手动设置标题

在任意聊天会话（CLI 或网关）中使用 `/title` 命令：

```
/title my research project
```
标题会即时生效。如果会话尚未在数据库创建（例如在首次发送消息前使用 `/title`），则会在会话启动后应用。

也可以在命令行重命名已有会话：

```bash
hermes sessions rename 20250305_091523_a1b2c3d4 "refactoring auth module"
```

### 标题规则
- **唯一**——不同会话不能使用相同标题
- **最长 100 字符**——保持列表输出整洁
- **自动清理**——控制字符、零宽字符和 RTL 覆盖符会被剥除
- **支持普通 Unicode**——emoji、中文、带音调字符均可

### 自动衍生（压缩）

当会话上下文被压缩（手动 `/compress` 或自动）时，Hermes 会创建一个继承会话。若原会话已有标题，新会话会自动得到带序号的标题：

```
"my project" → "my project #2" → "my project #3"
```
使用名称恢复（`hermes -c "my project"`）时会自动选取最新的会话。

### `/title` 在消息平台中的使用

所有网关平台均支持 `/title` 命令：
- `/title My Research` — 设置会话标题
- `/title` — 显示当前标题

## 会话管理命令

Hermes 提供完整的会话管理子命令 `hermes sessions`：

### 列出会话

```bash
# 列出最近的会话（默认 20 条）
hermes sessions list

# 按平台过滤
hermes sessions list --source telegram

# 显示更多
hermes sessions list --limit 50
```
带标题的会话会显示标题、预览和相对时间戳。

### 导出会话

```bash
# 导出全部会话为 JSONL
hermes sessions export backup.jsonl

# 导出特定平台的会话
hermes sessions export telegram-history.jsonl --source telegram

# 导出单个会话
hermes sessions export session.jsonl --session-id 20250305_091523_a1b2c3d4
```
导出的文件每行一个 JSON 对象，包含完整的会话元数据和所有消息。

### 删除会话

```bash
# 删除指定会话（需要确认）
hermes sessions delete 20250305_091523_a1b2c3d4

# 直接删除不确认
hermes sessions delete 20250305_091523_a1b2c3d4 --yes
```

### 重命名会话

```bash
# 设置或更改会话标题
hermes sessions rename 20250305_091523_a1b2c3d4 "debugging auth flow"
```
如果标题已被其他会话使用，则会报错。

### 清理旧会话

```bash
# 删除 90 天前结束的会话（默认）
hermes sessions prune

# 自定义保留天数
hermes sessions prune --older-than 30

# 仅对特定平台执行
hermes sessions prune --source telegram --older-than 60

# 跳过确认
hermes sessions prune --older-than 30 --yes
```
:::info
仅会删除 **已结束** 的会话（显式结束或自动重置的会话）。活跃会话永远不会被清理。
:::

### 会话统计

```bash
hermes sessions stats
```
示例输出：
```
Total sessions: 142
Total messages: 3847
  cli: 89 sessions
  telegram: 38 sessions
  discord: 15 sessions
Database size: 12.4 MB
```
如需更深度分析（Token 使用、费用估算、工具分布、活跃模式）请使用 [`hermes insights`](/docs/reference/cli-commands#hermes-insights)。

## 会话搜索工具

内置的 `session_search` 工具使用 SQLite 的 FTS5 引擎对所有历史对话执行全文搜索。

### 工作原理
1. FTS5 搜索匹配的消息并按相关度排序
2. 按会话分组，取前 N（默认 3）唯一会话
3. 加载每个会话的对话并截取约 100K 字符的上下文
4. 交给快速摘要模型生成精简摘要
5. 返回每个会话的摘要、元数据和上下文片段

### FTS5 查询语法
- 简单关键词：`docker deployment`
- 短语：`"exact phrase"`
- 布尔：`docker OR kubernetes`、`python NOT java`
- 前缀：`deploy*`

### 使用时机
当用户提及过去的对话或你怀疑有相关历史上下文时，模型会自动提示使用 `session_search` 在提问前检索。

## 跨平台会话追踪

### 网关会话
在消息平台上，会话键由以下决定性格式组成：

| 聊天类型 | 默认键格式 | 行为 |
|----------|------------|------|
| Telegram DM | `agent:main:telegram:dm:<chat_id>` | 每个私聊对应一个会话 |
| Discord DM | `agent:main:discord:dm:<chat_id>` | 每个私聊对应一个会话 |
| WhatsApp DM | `agent:main:whatsapp:dm:<canonical_identifier>` | 每个用户对应一个会话（别名合并） |
| 群组聊天 | `agent:main:<platform>:group:<chat_id>:<user_id>` | 在平台提供用户 ID 时每用户独立会话 |
| 群组线程/话题 | `agent:main:<platform>:group:<chat_id>:<thread_id>` | 所有线程参与者共享会话（默认）。`thread_sessions_per_user: true` 时每用户独立 |
| 频道 | `agent:main:<platform>:channel:<chat_id>:<user_id>` | 在平台提供用户 ID 时每用户独立 |

如果无法获取参与者标识，Hermes 会退回到房间级共享会话。

### 共享 vs 独立群组会话
默认 `group_sessions_per_user: true`（在 `config.yaml` 中），这意味着：
- Alice 与 Bob 在同一 Discord 频道内对 Hermes 提问时各自拥有独立会话，互不干扰
- 某用户的长工具任务不会污染他人的上下文窗口
- 中断处理也保持按用户分离，因为运行的代理键匹配独立会话键

如想实现共享 “房间大脑”，将其设为 `false`：

```yaml
group_sessions_per_user: false
```
这会把群组/频道恢复为单一共享会话，便于共享上下文但也会共享 token 成本与中断状态。

### 会话重置策略
网关会话会依据以下可配置策略自动重置：
- **idle** — 在 N 分钟无交互后重置
- **daily** — 每天特定时刻重置
- **both** — 以先到者为准（idle 或 daily）
- **none** — 永不自动重置

在自动重置前，代理会获得一次轮次以保存重要记忆或技能。拥有活跃后台进程的会话永不自动重置，无论策略如何。

## 存储位置

| 项目 | 路径 | 描述 |
|------|------|------|
| SQLite 数据库 | `~/.hermes/state.db` | 所有会话元数据 + 消息，带 FTS5 |
| 网关转录 | `~/.hermes/sessions/` | 每会话 JSONL 转录 + `sessions.json` 索引 |
| 网关索引 | `~/.hermes/sessions/sessions.json` | 映射会话键到活动会话 ID |

SQLite 使用 WAL 模式以支持并发读取，单写者模式，非常适合多平台网关架构。

### 数据库模式
- **sessions** — 会话元数据（id、source、user_id、model、title、时间戳、token 计数）。标题拥有唯一索引（NULL 允许，仅非 NULL 必须唯一）。
- **messages** — 完整消息历史（role、content、tool_calls、tool_name、token_count）
- **messages_fts** — FTS5 虚表，用于全文搜索

## 会话过期与清理

### 自动清理
- 网关会话依据配置的重置策略自动重置
- 重置前，代理会保存记忆与技能
- 可选自动剪枝：`sessions.auto_prune` 为 `true` 时，结束的会话在 `sessions.retention_days`（默认 90 天）之前会被剪枝，在 CLI/网关启动时执行
- 剪枝后若实际删除行，则对 `state.db` 执行 `VACUUM` 以回收磁盘空间（SQLite DELETE 不会收缩文件）
- 剪枝最多每 `sessions.min_interval_hours`（默认 24）运行一次；上次运行时间记录在 `state.db` 中，所有 Hermes 进程共享该信息

默认 **关闭** — 会话历史对 `session_search` 检索非常有价值，自动删除可能令用户意外。若在高负载网关/定时任务环境下 `state.db` 体积影响性能（例如 384 MB、约 1000 会话导致 FTS5 插入缓慢），可在 `~/.hermes/config.yaml` 中开启：

```yaml
sessions:
  auto_prune: true          # 选项——默认 false
  retention_days: 90        # 保留已结束会话的天数
  vacuum_after_prune: true  # 剪枝后回收磁盘空间
  min_interval_hours: 24    # 两次剪枝之间的最小间隔
```
活跃会话永不被自动剪枝。

### 手动清理

```bash
# 剪枝超过 90 天的会话
hermes sessions prune

# 删除特定会话
hermes sessions delete <session_id>

# 在剪枝前导出备份
hermes sessions export backup.jsonl
hermes sessions prune --older-than 30 --yes
```

:::tip
数据库增长缓慢（数百会话通常 10‑15 MB），会话历史为 `session_search` 提供跨会话检索能力。若出现大文件（如 384 MB）导致性能下降，可使用上面的自动或手动剪枝。
:::
