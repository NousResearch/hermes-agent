---
sidebar_position: 4
title: "Slack"
description: "使用 Socket Mode 将 Hermes Agent 设置为 Slack 机器人"
---

# Slack 设置

使用 Socket Mode 将 Hermes Agent 作为机器人连接到 Slack。Socket Mode 使用 WebSocket 而非公共 HTTP 端点，因此你的 Hermes 实例不需要公开访问 — 它在防火墙后、笔记本上或私有服务器上都能工作。

:::warning 经典 Slack Apps 已弃用
经典 Slack apps（使用 RTM API）在 **2025 年 3 月完全弃用**。Hermes 使用现代 Bolt SDK 和 Socket Mode。如果你有旧的经典 app，必须按照以下步骤创建一个新的。
:::

## 概览

| 组件 | 值 |
|-----------|-------|
| **库** | Python 的 `slack-bolt` / `slack_sdk`（Socket Mode） |
| **连接** | WebSocket — 无需公共 URL |
| **所需认证 token** | Bot Token (`xoxb-`) + App-Level Token (`xapp-`) |
| **用户识别** | Slack Member IDs（例如 `U01ABC2DEF3`） |

---

## 步骤 1：创建 Slack App

最快的方法是粘贴 Hermes 为你生成的 manifest。它声明每个内置斜杠命令（`/btw`、`/stop`、`/model` …）、每个所需的 OAuth 范围、每个事件订阅，并启用 Socket Mode — 一次完成。

### 选项 A：从 Hermes 生成的 manifest（推荐）

1. 生成 manifest：
   ```bash
   hermes slack manifest --write
   ```
   这会写入 `~/.hermes/slack-manifest.json` 并打印粘贴说明。
2. 进入 [https://api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From an app manifest**
3. 选择你的 workspace，粘贴 JSON 内容，审核，点击 **Next** → **Create**
4. 跳到 **步骤 6：安装 App 到 Workspace**。Manifest 为你处理了 scopes、events 和斜杠命令。

### 选项 B：从头开始（手动）

1. 进入 [https://api.slack.com/apps](https://api.slack.com/apps)
2. 点击 **Create New App**
3. 选择 **From scratch**
4. 输入 app 名称（例如 "Hermes Agent"）并选择你的 workspace
5. 点击 **Create App**

你会进入 app 的**基本信息**页面。继续下面的步骤 2–6。

---

## 步骤 2：配置 Bot Token Scopes

在侧边栏导航到 **Features → OAuth & Permissions**。滚动到 **Scopes → Bot Token Scopes** 并添加以下权限：

| Scope | 用途 |
|-------|---------|
| `chat:write` | 以机器人身份发送消息 |
| `app_mentions:read` | 检测何时在频道中被 @提及 |
| `channels:history` | 读取机器人所在公共频道的消息 |
| `channels:read` | 列出和获取公共频道信息 |
| `groups:history` | 读取机器人被邀请的私人频道消息 |
| `im:history` | 读取直接消息历史 |
| `im:read` | 查看基本 DM 信息 |
| `im:write` | 打开和管理 DM |
| `users:read` | 查找用户信息 |
| `files:read` | 读取和下载附件，包括语音笔记/音频 |
| `files:write` | 上传文件（图片、音频、文档） |

:::caution 缺失 scopes = 缺失功能
没有 `channels:history` 和 `groups:history`，机器人**将不会在频道中接收消息** — 它只能在 DM 中工作。没有 `files:read`，Hermes 可以聊天但**无法可靠地读取用户上传的附件**。这些是最常遗漏的 scopes。
:::

---

## 步骤 3：启用 Socket Mode

Socket Mode 让机器人通过 WebSocket 连接，而不需要公共 URL。

1. 在侧边栏，进入 **Settings → Socket Mode**
2. 将 **Enable Socket Mode** 切换到 ON
3. 系统会提示你创建一个 **App-Level Token**：
   - 命名类似 `hermes-socket`（名称无所谓）
   - 添加 **`connections:write`** scope
   - 点击 **Generate**
4. **复制 token** — 它以 `xapp-` 开头。这是你的 `SLACK_APP_TOKEN`

---

## 步骤 4：订阅事件

此步骤至关重要 — 它控制机器人可以看到哪些消息。

1. 在侧边栏，进入 **Features → Event Subscriptions**
2. 将 **Enable Events** 切换到 ON
3. 展开 **Subscribe to bot events** 并添加：

| 事件 | 必需？ | 用途 |
|-------|-----------|---------|
| `message.im` | **是** | 机器人接收直接消息 |
| `message.channels` | **是** | 机器人接收其所在的**公共**频道消息 |
| `message.groups` | **推荐** | 机器人接收被邀请的**私人**频道消息 |
| `app_mention` | **是** | 当机器人被 @提及时防止 Bolt SDK 错误 |

4. 点击页面底部的 **Save Changes**

:::danger 缺失事件订阅是 #1 设置问题
如果机器人在 DM 中工作但**在频道中不工作**，你几乎肯定忘记了添加 `message.channels`（公共频道）和/或 `message.groups`（私人频道）。没有这些事件，Slack 根本不会将频道消息投递给机器人。
:::

---

## 步骤 5：启用 Messages Tab

此步骤启用直接消息给机器人。没有它，用户在尝试 DM 机器人时看到 **"Sending messages to this app has been turned off"**。

1. 在侧边栏，进入 **Features → App Home**
2. 滚动到 **Show Tabs**
3. 将 **Messages Tab** 切换到 ON
4. 勾选 **"Allow users to send Slash commands and messages from the messages tab"**

:::danger 没有此步骤，DM 完全被阻止
即使所有 scopes 和事件订阅都正确，Slack 也不允许用户向机器人发送直接消息，除非启用 Messages Tab。这是 Slack 平台要求，不是 Hermes 配置问题。
:::

---

## 步骤 6：安装 App 到 Workspace

1. 在侧边栏，进入 **Settings → Install App**
2. 点击 **Install to Workspace**
3. 审核权限并点击 **Allow**
4. 授权后，你会看到以 `xoxb-` 开头的 **Bot User OAuth Token**
5. **复制此 token** — 这是你的 `SLACK_BOT_TOKEN`

---

## 步骤 7：查找用于白名单的用户 ID

Hermes 使用 Slack **Member IDs**（不是用户名或显示名称）做白名单。

查找 Member ID：
1. 在 Slack 中，点击用户名或头像
2. 点击 **View full profile**
3. 点击 **⋮**（更多）按钮
4. 选择 **Copy member ID**

Member IDs 类似 `U01ABC2DEF3`。你至少需要你自己的 Member ID。

---

## 步骤 8：配置 Hermes

将以下内容添加到你的 `~/.hermes/.env` 文件：

```bash
# 必需
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
SLACK_ALLOWED_USERS=U01ABC2DEF3

# 可选
SLACK_HOME_CHANNEL=C01234567890
```

然后启动 gateway：

```bash
hermes gateway
```

---

## 步骤 9：将机器人邀请到频道

启动 gateway 后，你需要**将机器人邀请**到任何你想让它响应的频道：

```
/invite @Hermes Agent
```

---

## 斜杠命令

每个 Hermes 命令（`/btw`、`/stop`、`/new`、`/model`、`/help` …）都是原生 Slack 斜杠命令。在 Slack 中输入 `/`，自动补全选择器列出每个 Hermes 命令及其描述。

在底层：Hermes 自带生成的 Slack app manifest（见步骤 1 选项 A），它在 [`COMMAND_REGISTRY`](https://github.com/NousResearch/hermes-agent/blob/main/hermes_cli/commands.py) 中将每个命令声明为斜杠命令。在 Socket Mode 中，Slack 通过 WebSocket 路由命令事件，无论 manifest 中的 `url` 字段是什么。

### 更新后刷新斜杠命令

当 Hermes 添加新命令时（例如 `hermes update` 后），重新生成 manifest 并更新你的 Slack app：

```bash
hermes slack manifest --write
```

然后在 Slack 中：
1. 打开 [https://api.slack.com/apps](https://api.slack.com/apps) → 你的 Hermes app
2. **Features → App Manifest → Edit**
3. 粘贴 `~/.hermes/slack-manifest.json` 的新内容
4. **Save**。如果 scope 或斜杠命令有变化，Slack 会提示重新安装 app。

### 仍可使用旧版 `/hermes <子命令>`

为与旧版 manifest 保持向后兼容，你仍可以输入 `/hermes btw run the tests` — Hermes 路由方式与 `/btw run the tests` 相同。自由形式的问题也可以：`/hermes what's the weather?` 被当作普通消息处理。

### 高级：仅导出斜杠命令数组

如果你手动维护 Slack manifest 并只需要斜杠命令列表：

```bash
hermes slack manifest --slashes-only > /tmp/slashes.json
```

将该数组粘贴到现有 manifest 的 `features.slash_commands` 键中。

---

## 机器人如何响应

理解 Hermes 在不同上下文中的行为：

| 上下文 | 行为 |
|---------|---------|
| **DM** | 机器人响应每条消息 — 无需 @提及 |
| **频道** | 机器人**仅在 @提及时**响应。在频道中，Hermes 在附加到那条消息的线程中回复。 |
| **线程** | 如果你在现有线程内 @提及 Hermes，它在同一线程中回复。一旦机器人在线程中有活动会话，**该线程中的后续回复不需要 @提及**。 |

:::tip
在频道中，始终 @提及机器人开始对话。一旦机器人在线程中活动，你可以在该线程中回复而不提及它。线程外，没有 @提及的消息会被忽略，以防止繁忙频道中的噪音。
:::

---

## 配置选项

除了步骤 8 中所需的环境变量外，你还可以通过 `~/.hermes/config.yaml` 自定义 Slack 机器人行为。

### 线程和回复行为

```yaml
platforms:
  slack:
    # 控制多部分响应的线程化方式
    # "off"   — 不将回复线程化到原始消息
    # "first" — 第一块内容线程化到用户消息（默认）
    # "all"   — 所有块都线程化到用户消息
    reply_to_mode: "first"

    extra:
      # 是否在线程中回复（默认：true）
      # 为 false 时，频道消息直接回复到频道而非线程。
      # 现有线程内的消息仍在线程中回复。
      reply_in_thread: true

      # 同时将线程回复发布到主频道
      #（Slack 的 "Also send to channel" 功能）
      # 只有第一块的第一条回复会被广播。
      reply_broadcast: false
```

| 键 | 默认值 | 描述 |
|-----|---------|---------|
| `platforms.slack.reply_to_mode` | `"first"` | 多部分消息的线程化模式：`"off"`、`"first"` 或 `"all"` |
| `platforms.slack.extra.reply_in_thread` | `true` | 为 `false` 时，频道消息直接回复而非线程。现有线程内的消息仍在线程中回复。 |
| `platforms.slack.extra.reply_broadcast` | `false` | 为 `true` 时，线程回复同时发布到主频道。只有第一块会被广播。 |

### 会话隔离

```yaml
# 全局设置 — 适用于 Slack 和所有其他平台
group_sessions_per_user: true
```

为 `true`（默认）时，共享频道中的每个用户获得自己隔离的对话会话。在 `#general` 中与 Hermes 聊天的两个人会有各自独立的历史和上下文。

设为 `false` 如果你想要协作模式，让整个频道共享一个对话会话。注意这意味着用户共享上下文增长和 token 费用，且一个用户的 `/reset` 会清除所有人的会话。

### @提及和触发行为

```yaml
slack:
  # 在频道中要求 @提及（这是默认行为；
  # Slack 适配器在频道中强制执行 @提及门控，
  # 但你可以设置此选项以与其他平台保持一致）
  require_mention: true

  # 防止线程自动参与：仅回复包含明确 @提及的频道消息。
  # 关闭（默认）时，Slack 可以"自动参与" — 记住线程中过去的 @提及，
  # 跟进机器人消息的回复，以及无需新 @提及就恢复活动会话。
  # 开启 strict_mention 后，每个新频道消息都必须 @提及机器人 Hermes 才会响应。
  strict_mention: false

  # 触发机器人的自定义 @提及模式
  #（除了默认的 @提及检测外）
  mention_patterns:
    - "hey hermes"
    - "hermes,"

  # 每个传出消息的前缀文本
  reply_prefix: ""
```

:::tip 何时使用 `strict_mention`
在繁忙的 workspace 中，当 Slack 默认的"机器人记住此线程"行为使用户感到困惑时，将其设为 `true` — 例如，一个长期技术支持线程中机器人在开始时提供了帮助，而你宁愿它在没有被明确 ping 的情况下保持沉默。DM 和活动交互会话不受影响。
:::

:::info
Slack 两种模式都支持：默认需要 `@mention` 开始对话，但你可以通过 `SLACK_FREE_RESPONSE_CHANNELS`（逗号分隔的频道 ID）或 `config.yaml` 中的 `slack.free_response_channels` 选择性地让特定频道退出此规则。一旦机器人在线程中有活动会话，该线程中的后续回复不需要 @提及。DM 中机器人始终响应，无需 @提及。
:::

### 未授权用户处理

```yaml
slack:
  # 当未授权用户（不在 SLACK_ALLOWED_USERS 中）DM 机器人时会发生什么
  # "pair"   — 提示他们输入配对码（默认）
  # "ignore" — 静默丢弃消息
  unauthorized_dm_behavior: "pair"
```

你也可以全局设置：

```yaml
unauthorized_dm_behavior: "pair"
```

`slack:` 下的平台特定设置优先于全局设置。

### 语音转录

```yaml
# 全局设置 — 启用/禁用传入语音消息的自动转录
stt_enabled: true
```

为 `true`（默认）时，传入的音频消息在使用配置的 STT 提供商进行自动转录后再由 agent 处理。

### 完整示例

```yaml
# 全局 gateway 设置
group_sessions_per_user: true
unauthorized_dm_behavior: "pair"
stt_enabled: true

# Slack 特定设置
slack:
  require_mention: true
  unauthorized_dm_behavior: "pair"

# 平台配置
platforms:
  slack:
    reply_to_mode: "first"
    extra:
      reply_in_thread: true
      reply_broadcast: false
```

---

## 家庭频道

设置 `SLACK_HOME_CHANNEL` 为一个频道 ID，Hermes 将在该频道投递计划消息、cron 作业结果和其他主动通知。查找频道 ID：

1. 在 Slack 中右键频道名称
2. 点击 **View channel details**
3. 滚动到底部 — Channel ID 在那里显示

```bash
SLACK_HOME_CHANNEL=C01234567890
```

确保机器人已被**邀请到频道**（`/invite @Hermes Agent`）。

---

## 多工作区支持

Hermes 可以通过单个 gateway 实例同时连接**多个 Slack workspace**。每个 workspace 独立认证，拥有自己的 bot user ID。

### 配置

将多个 bot token 作为**逗号分隔列表**提供：

```bash
# 多个 bot token — 每个 workspace 一个
SLACK_BOT_TOKEN=xoxb-workspace1-token,xoxb-workspace2-token,xoxb-workspace3-token

# 仍使用单个 app-level token 用于 Socket Mode
SLACK_APP_TOKEN=xapp-your-app-token
```

或在 `~/.hermes/config.yaml` 中：

```yaml
platforms:
  slack:
    token: "xoxb-workspace1-token,xoxb-workspace2-token"
```

### OAuth Token 文件

除了环境或配置中的 token，Hermes 还从 OAuth token 文件加载 token：

```
~/.hermes/slack_tokens.json
```

此文件是将 team ID 映射到 token 条目的 JSON 对象：

```json
{
  "T01ABC2DEF3": {
    "token": "xoxb-workspace-token-here",
    "team_name": "My Workspace"
  }
}
```

此文件的 token 与通过 `SLACK_BOT_TOKEN` 指定的 token 合并。重复的 token 自动去重。

### 工作原理

- **第一个 token** 是主 token，用于 Socket Mode 连接（AsyncApp）。
- 每个 token 在启动时通过 `auth.test` 认证。gateway 将每个 `team_id` 映射到其自己的 `WebClient` 和 `bot_user_id`。
- 消息到达时，Hermes 使用正确的 workspace 特定客户端来响应。
- 主 `bot_user_id`（来自第一个 token）用于与期望单一 bot 身份的功能的向后兼容。

---

## 语音消息

Hermes 支持 Slack 语音功能：

- **传入：** 语音/音频消息使用配置的 STT 提供商自动转录：本地 `faster-whisper`、Groq Whisper（`GROQ_API_KEY`）或 OpenAI Whisper（`VOICE_TOOLS_OPENAI_KEY`）
- **传出：** TTS 响应作为音频文件附件发送

---

## 按频道提示

为特定 Slack 频道分配临时系统提示。提示在每次对话轮次时注入运行时 — 从不持久化到 transcript 历史，因此更改立即生效。

```yaml
slack:
  channel_prompts:
    "C01RESEARCH": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "C02ENGINEERING": |
      Code review mode. Be precise about edge cases and
      performance implications.
```

键是 Slack 频道 ID（通过频道详情 → "About" → 滚动到底部查找）。匹配频道中的所有消息都会将提示作为临时系统指令注入。

## 按频道技能绑定

在特定频道或 DM 中开始新会话时自动加载技能。与按频道提示（每轮注入）不同，技能绑定在**会话开始时**将技能内容作为用户消息注入 — 它成为对话历史的一部分，不需要在后续轮次中重新加载。

这适用于有专门用途的 DM 或频道（闪卡、特定领域问答机器人、支持分流频道等），你不希望模型自己的技能选择器在每次短回复时都决定是否加载。

```yaml
slack:
  channel_skill_bindings:
    # DM 频道 — 始终以 "german-flashcards" 模式运行
    - id: "D0ATH9TQ0G6"
      skills:
        - german-flashcards
    # 研究频道 — 按顺序预加载多个技能
    - id: "C01RESEARCH"
      skills:
        - arxiv
        - writing-plans
    # 简写形式：单个技能作为字符串
    - id: "C02SUPPORT"
      skill: hubspot-on-demand
```

注意：
- 绑定通过频道 ID 匹配。在绑定频道中的线程消息继承父频道的绑定。
- 技能仅在会话开始时加载（新会话或自动重置后）。如果更改了绑定，运行 `/new` 或等待会话自动重置以使更改生效。
- 结合 `channel_prompts` 在技能指令的基础上添加按频道语气/约束。

---

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| 机器人不响应 DM | 验证 `message.im` 在你的事件订阅中且 app 已重新安装 |
| 机器人在 DM 中工作但在频道中不工作 | **最常见问题。** 添加 `message.channels` 和 `message.groups` 到事件订阅，重新安装 app，并用 `/invite @Hermes Agent` 将机器人邀请到频道 |
| 机器人在频道中不响应 @提及 | 1）检查 `message.channels` 事件已订阅。2）机器人必须被邀请到频道。3）确保已添加 `channels:history` scope。4）在 scope/事件更改后重新安装 app |
| 机器人在私人频道中忽略消息 | 添加 `message.groups` 事件订阅和 `groups:history` scope，然后重新安装 app 并 `/invite` 机器人 |
| DM 中"Sending messages to this app has been turned off" | 在 App Home 设置中启用 **Messages Tab**（见步骤 5） |
| `not_authed` 或 `invalid_auth` 错误 | 重新生成 Bot Token 和 App Token，更新 `.env` |
| 机器人可以聊天但无法在频道中发帖 | 用 `/invite @Hermes Agent` 将机器人邀请到频道 |
| 机器人能聊天但无法读取上传的图片/文件 | 添加 `files:read`，然后**重新安装** app。当 Slack 返回 scope/auth/权限错误时，Hermes 现在会在聊天中显示附件访问诊断。 |
| `missing_scope` 错误 | 在 OAuth & Permissions 中添加所需的 scope，然后**重新安装** app |
| Socket 频繁断开 | 检查网络；Bolt 会自动重连但不稳定连接会导致延迟 |
| 更改了 scope/事件但没有变化 | 任何 scope 或事件订阅更改后，你**必须重新安装** app 到 workspace |

### 快速检查清单

如果机器人在频道中不工作，请验证以下**所有**项：

1. ✅ `message.channels` 事件已订阅（公共频道）
2. ✅ `message.groups` 事件已订阅（私人频道）
3. ✅ `app_mention` 事件已订阅
4. ✅ `channels:history` scope 已添加（公共频道）
5. ✅ `groups:history` scope 已添加（私人频道）
6. ✅ 添加 scope/事件后**重新安装**了 app
7. ✅ 机器人已被**邀请**到频道（`/invite @Hermes Agent`）
8. ✅ 你在消息中 **@提及**了机器人

---

## 安全

:::warning
**始终设置 `SLACK_ALLOWED_USERS`** 限制谁可以与你的机器人交互。没有此设置，gateway 将**默认拒绝所有消息**作为安全措施。切勿分享你的 bot token — 像密码一样对待它们。
:::

- Token 应存储在 `~/.hermes/.env` 中（文件权限 `600`）
- 定期通过 Slack app 设置轮换 token
- 审计谁有权限访问你的 Hermes 配置目录
- Socket Mode 意味着不会暴露公共端点 — 减少攻击面
