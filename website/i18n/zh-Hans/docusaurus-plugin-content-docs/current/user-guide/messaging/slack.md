---
sidebar_position: 4
title: "Slack"
description: "使用 Socket Mode 将 Hermes Agent 设置为 Slack 机器人"
---

# Slack 设置

使用 Socket Mode 将 Hermes Agent 作为机器人连接到 Slack。Socket Mode 使用 WebSocket 而非公开 HTTP 端点，因此你的 Hermes 实例无需公开访问——它可以在防火墙后、笔记本电脑上或私有服务器上正常运行。

:::warning 经典 Slack 应用已弃用
使用 RTM API 的经典 Slack 应用已于 **2025 年 3 月完全弃用**。Hermes 使用带有 Socket Mode 的现代 Bolt SDK。如果你有旧的经典应用，必须按照以下步骤创建新应用。
:::

## 概述

| 组件 | 值 |
|-----------|-------|
| **库** | Python 的 `slack-bolt` / `slack_sdk`（Socket Mode） |
| **连接方式** | WebSocket——无需公开 URL |
| **所需认证令牌** | Bot Token（`xoxb-`）+ App-Level Token（`xapp-`） |
| **用户标识** | Slack Member ID（例如 `U01ABC2DEF3`） |

---

## 第一步：创建 Slack 应用

最快的方式是粘贴 Hermes 为你生成的 manifest（清单文件）。它会一次性声明所有内置斜杠命令（`/btw`、`/stop`、`/model`……）、所有必需的 OAuth 权限范围、所有事件订阅，并启用 Socket Mode。

### 方式 A：使用 Hermes 生成的 manifest（推荐）

1. 生成 manifest：
   ```bash
   hermes slack manifest --write
   ```
   此命令会将 `~/.hermes/slack-manifest.json` 写入磁盘并打印粘贴说明。
2. 前往 [https://api.slack.com/apps](https://api.slack.com/apps) →
   **Create New App** → **From an app manifest**
3. 选择你的工作区，粘贴 JSON 内容，检查后点击 **Next** → **Create**
4. 直接跳至**第六步：将应用安装到工作区**。manifest 已为你处理好权限范围、事件和斜杠命令。

### 方式 B：从头手动创建

1. 前往 [https://api.slack.com/apps](https://api.slack.com/apps)
2. 点击 **Create New App**
3. 选择 **From scratch**
4. 输入应用名称（例如 "Hermes Agent"）并选择你的工作区
5. 点击 **Create App**

你将进入应用的 **Basic Information** 页面。继续执行下方第 2–6 步。

---

## 第二步：配置 Bot Token 权限范围

在侧边栏导航至 **Features → OAuth & Permissions**。向下滚动至 **Scopes → Bot Token Scopes**，添加以下权限：

| 权限范围 | 用途 |
|-------|---------|
| `chat:write` | 以机器人身份发送消息 |
| `app_mentions:read` | 检测在频道中被 @ 提及的情况 |
| `channels:history` | 读取机器人所在公开频道的消息 |
| `channels:read` | 列出并获取公开频道信息 |
| `groups:history` | 读取机器人被邀请加入的私有频道消息 |
| `im:history` | 读取私信历史记录 |
| `im:read` | 查看基本私信信息 |
| `im:write` | 打开并管理私信 |
| `mpim:history` | 读取群组私信（多人私信）历史记录 |
| `mpim:read` | 查看基本群组私信信息 |
| `users:read` | 查询用户信息 |
| `files:read` | 读取并下载附件文件，包括语音备忘录/音频 |
| `files:write` | 上传文件（图片、音频、文档） |

:::caution 缺少权限范围 = 功能缺失
没有 `channels:history` 和 `groups:history`，机器人**将无法接收频道消息**——它只能在私信中工作。没有 `files:read`，Hermes 可以聊天，但**无法可靠读取用户上传的附件**。这是最常被遗漏的权限范围。
:::

**可选权限范围：**

| 权限范围 | 用途 |
|-------|---------|
| `groups:read` | 列出并获取私有频道信息 |

---

## 第三步：启用 Socket Mode

Socket Mode 让机器人通过 WebSocket 连接，无需公开 URL。

1. 在侧边栏前往 **Settings → Socket Mode**
2. 将 **Enable Socket Mode** 切换为开启
3. 系统会提示你创建一个 **App-Level Token**：
   - 命名为类似 `hermes-socket` 的名称（名称不重要）
   - 添加 **`connections:write`** 权限范围
   - 点击 **Generate**
4. **复制该令牌**——它以 `xapp-` 开头。这就是你的 `SLACK_APP_TOKEN`

:::tip
你随时可以在 **Settings → Basic Information → App-Level Tokens** 下找到或重新生成 App-Level Token。
:::

---

## 第四步：订阅事件

此步骤至关重要——它控制机器人能看到哪些消息。

1. 在侧边栏前往 **Features → Event Subscriptions**
2. 将 **Enable Events** 切换为开启
3. 展开 **Subscribe to bot events** 并添加：

| 事件 | 是否必需 | 用途 |
|-------|-----------|---------|
| `message.im` | **必需** | 机器人接收私信 |
| `message.mpim` | **推荐** | 机器人接收其加入的**群组私信**（多人私信）消息 |
| `message.channels` | **必需** | 机器人接收其加入的**公开**频道消息 |
| `message.groups` | **推荐** | 机器人接收被邀请加入的**私有**频道消息 |
| `app_mention` | **必需** | 防止机器人被 @ 提及时出现 Bolt SDK 错误 |

4. 点击页面底部的 **Save Changes**

:::danger 缺少事件订阅是第一大设置问题
如果机器人在私信中正常工作但**在频道中不响应**，你几乎肯定忘记添加 `message.channels`（公开频道）和/或 `message.groups`（私有频道）。没有这些事件，Slack 根本不会将频道消息传递给机器人。
:::

---

## 第五步：启用 Messages Tab

此步骤启用对机器人的私信功能。没有它，用户在尝试私信机器人时会看到**"向此应用发送消息已被关闭"**的提示。

1. 在侧边栏前往 **Features → App Home**
2. 向下滚动至 **Show Tabs**
3. 将 **Messages Tab** 切换为开启
4. 勾选 **"Allow users to send Slash commands and messages from the messages tab"**

:::danger 没有此步骤，私信将被完全屏蔽
即使拥有所有正确的权限范围和事件订阅，除非启用 Messages Tab，否则 Slack 不允许用户向机器人发送私信。这是 Slack 平台的要求，而非 Hermes 的配置问题。
:::

---

## 第六步：将应用安装到工作区

1. 在侧边栏前往 **Settings → Install App**
2. 点击 **Install to Workspace**
3. 检查权限并点击 **Allow**
4. 授权后，你将看到一个以 `xoxb-` 开头的 **Bot User OAuth Token**
5. **复制此令牌**——这就是你的 `SLACK_BOT_TOKEN`

:::tip
如果你之后更改了权限范围或事件订阅，**必须重新安装应用**才能使更改生效。Install App 页面会显示提示横幅。
:::

---

## 第七步：查找用于白名单的用户 ID

Hermes 使用 Slack **Member ID**（而非用户名或显示名称）作为白名单。

查找 Member ID 的方法：

1. 在 Slack 中点击用户的名称或头像
2. 点击 **View full profile**
3. 点击 **⋮**（更多）按钮
4. 选择 **Copy member ID**

Member ID 格式类似 `U01ABC2DEF3`。你至少需要自己的 Member ID。

---

## 第八步：配置 Hermes

将以下内容添加到你的 `~/.hermes/.env` 文件：

```bash
# 必需
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here
SLACK_ALLOWED_USERS=U01ABC2DEF3              # 逗号分隔的 Member ID

# 可选
SLACK_HOME_CHANNEL=C01234567890              # 定时/计划消息的默认频道
SLACK_HOME_CHANNEL_NAME=general              # 主频道的可读名称（可选）
```

或运行交互式设置：

```bash
hermes gateway setup    # 提示时选择 Slack
```

然后启动 gateway：

```bash
hermes gateway              # 前台运行
hermes gateway install      # 安装为用户服务
sudo hermes gateway install --system   # 仅 Linux：开机启动系统服务
```

---

## 第九步：将机器人邀请到频道

启动 gateway 后，你需要**邀请机器人**加入希望它响应的频道：

```
/invite @Hermes Agent
```

机器人**不会**自动加入频道。你必须逐个频道邀请它。

---

## 斜杠命令

每个 Hermes 命令（`/btw`、`/stop`、`/new`、`/model`、`/help`……）都是原生 Slack 斜杠命令——与它们在 Telegram 和 Discord 上的工作方式完全相同。在 Slack 中输入 `/`，自动补全选择器会列出每个 Hermes 命令及其描述。

底层实现：Hermes 附带一个生成的 Slack 应用 manifest（见第一步，方式 A），它将 [`COMMAND_REGISTRY`](https://github.com/NousResearch/hermes-agent/blob/main/hermes_cli/commands.py) 中的每个命令声明为斜杠命令。在 Socket Mode 下，无论 manifest 的 `url` 字段如何，Slack 都会通过 WebSocket 路由命令事件。

### 更新后刷新斜杠命令

当 Hermes 添加新命令时（例如执行 `hermes update` 后），重新生成 manifest 并更新你的 Slack 应用：

```bash
hermes slack manifest --write
```

然后在 Slack 中：
1. 打开 [https://api.slack.com/apps](https://api.slack.com/apps) →
   你的 Hermes 应用
2. **Features → App Manifest → Edit**
3. 粘贴 `~/.hermes/slack-manifest.json` 的新内容
4. **保存**。如果权限范围或斜杠命令有变化，Slack 会提示重新安装应用。

### 旧版 `/hermes <子命令>` 仍然有效

为了向后兼容旧版 manifest，你仍然可以输入 `/hermes btw run the tests`——Hermes 会以与 `/btw run the tests` 相同的方式路由它。自由形式的问题也有效：`/hermes what's the weather?` 会被当作普通消息处理。

### 在话题（thread）中使用命令（`!cmd` 前缀）

Slack 本身会阻止在话题回复中使用原生斜杠命令——在话题中尝试 `/queue`，Slack 会回复 *"/queue is not supported in threads. Sorry!"*。没有任何应用端设置可以重新启用它们；Slack 从不将它们传递给 Hermes。

作为解决方案，Hermes 识别前导 `!` 作为在话题（以及任何其他地方）中有效的替代命令前缀。在话题回复中输入 `!queue`、`!stop`、`!model gpt-5.4` 等普通回复——Hermes 会以与斜杠形式完全相同的方式处理，并在同一话题中回复。

只有第一个 token（词元）会与已知命令列表进行匹配，因此像 `!nice work` 这样的随意消息会原样传递给 agent。

### 高级：仅输出斜杠命令数组

如果你手动维护 Slack manifest 并只需要斜杠命令列表：

```bash
hermes slack manifest --slashes-only > /tmp/slashes.json
```

将该数组粘贴到现有 manifest 的 `features.slash_commands` 键中。

---

## 机器人的响应方式

了解 Hermes 在不同场景下的行为：

| 场景 | 行为 |
|---------|----------|
| **私信** | 机器人响应每条消息——无需 @ 提及 |
| **频道** | 机器人**仅在被 @ 提及时响应**（例如 `@Hermes Agent what time is it?`）。在频道中，Hermes 在该消息附带的话题中回复。 |
| **话题** | 如果你在现有话题中 @ 提及 Hermes，它会在同一话题中回复。一旦机器人在话题中有活跃会话，**该话题中的后续回复无需 @ 提及**——机器人会自然跟进对话。 |

:::tip
在频道中，始终 @ 提及机器人来开始对话。一旦机器人在话题中活跃，你可以在该话题中回复而无需提及它。话题之外，没有 @ 提及的消息会被忽略，以防止在繁忙频道中产生噪音。
:::

---

## 配置选项

除了第八步中的必需环境变量外，你还可以通过 `~/.hermes/config.yaml` 自定义 Slack 机器人行为。

### 话题与回复行为

```yaml
platforms:
  slack:
    # 控制多部分响应的话题方式
    # "off"   — 永不将回复串入原始消息的话题
    # "first" — 第一个分块串入用户消息（默认）
    # "all"   — 所有分块串入用户消息
    reply_to_mode: "first"

    extra:
      # 是否在话题中回复（默认：true）。
      # 为 false 时，频道消息直接在频道中回复，而非话题。
      # 已在话题中的消息仍在话题中回复。
      reply_in_thread: true

      # 同时将话题回复发布到主频道
      # （Slack 的"同时发送到频道"功能）。
      # 仅广播第一条回复的第一个分块。
      reply_broadcast: false
```

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `platforms.slack.reply_to_mode` | `"first"` | 多部分消息的话题模式：`"off"`、`"first"` 或 `"all"` |
| `platforms.slack.extra.reply_in_thread` | `true` | 为 `false` 时，频道消息直接回复而非话题。已在话题中的消息仍在话题中回复。 |
| `platforms.slack.extra.reply_broadcast` | `false` | 为 `true` 时，话题回复也会发布到主频道。仅广播第一个分块。 |

### 会话隔离

```yaml
# 全局设置——适用于 Slack 和所有其他平台
group_sessions_per_user: true
```

为 `true`（默认值）时，共享频道中的每个用户都有自己独立的对话会话。在 `#general` 中与 Hermes 对话的两个人将有各自独立的历史记录和上下文。

设为 `false` 可启用协作模式，整个频道共享一个对话会话。请注意，这意味着用户共享上下文增长和 token 成本，且一个用户的 `/reset` 会清除所有人的会话。

### 提及与触发行为

```yaml
slack:
  # 在频道中要求 @mention（这是默认行为；
  # Slack 适配器无论如何都会在频道中强制执行 @mention 门控，
  # 但你可以明确设置此项以与其他平台保持一致）
  require_mention: true

  # 防止话题自动参与：仅回复包含明确 @mention 的频道消息。
  # 关闭此项（默认），Slack 可以"自动参与"——记住话题中的过去提及，
  # 跟进机器人消息的回复，并在无需新提及的情况下恢复活跃会话。
  # 开启 strict_mention 后，每条新频道消息都必须 @mention 机器人，
  # Hermes 才会响应。
  strict_mention: false

  # 触发机器人的自定义提及模式
  # （除默认 @mention 检测外）
  mention_patterns:
    - "hey hermes"
    - "hermes,"

  # 每条发出消息前添加的文本
  reply_prefix: ""
```

:::tip 何时使用 `strict_mention`
在繁忙工作区中，如果 Slack 默认的"机器人记住此话题"行为让用户感到意外，请将此项设为 `true`——例如，在一个长技术支持话题中，机器人在开始时提供了帮助，而你希望它保持沉默，除非被明确 @ 提及。私信和活跃的交互会话不受影响。
:::

:::info
Slack 支持两种模式：默认情况下需要 `@mention` 才能开始对话，但你可以通过 `SLACK_FREE_RESPONSE_CHANNELS`（逗号分隔的频道 ID）或 `config.yaml` 中的 `slack.free_response_channels` 为特定频道取消此限制。一旦机器人在话题中有活跃会话，后续话题回复无需提及。在私信中，机器人始终响应，无需提及。
:::

### 频道白名单（`allowed_channels`）

将机器人限制在固定的 Slack 频道集合中——当机器人被邀请到许多频道但只应在少数频道中响应时很有用。设置后，不在此列表中的频道消息将被**静默忽略**，即使机器人被 `@mention`。

**私信不受此过滤器影响**，因此授权用户始终可以通过私信联系机器人。

```yaml
slack:
  allowed_channels:
    - "C0123456789"   # #ops
    - "C0987654321"   # #incident-response
```

或通过环境变量（逗号分隔）：

```bash
SLACK_ALLOWED_CHANNELS="C0123456789,C0987654321"
```

行为说明：

- 空/未设置 → 无限制（完全向后兼容）。
- 非空 → 频道 ID 必须在列表中，否则消息在任何其他门控（提及要求、`free_response_channels` 等）运行之前被丢弃。
- Slack 频道 ID 以 `C`（公开）、`G`（私有）或 `D`（私信）开头。可通过 Slack UI 的"打开频道详情"→"关于"面板或 API 查找。

另见：[管理员/用户斜杠命令分离](../../reference/slash-commands.md#permissions-and-adminuser-split)。

### 未授权用户处理

```yaml
slack:
  # 当未授权用户（不在 SLACK_ALLOWED_USERS 中）私信机器人时的处理方式
  # "pair"   — 提示他们输入配对码（默认）
  # "ignore" — 静默丢弃消息
  unauthorized_dm_behavior: "pair"
```

你也可以为所有平台全局设置：

```yaml
unauthorized_dm_behavior: "pair"
```

`slack:` 下的平台特定设置优先于全局设置。

### 语音转录

```yaml
# 全局设置——启用/禁用传入语音消息的自动转录
stt_enabled: true
```

为 `true`（默认值）时，传入的音频消息会在被 agent 处理之前，使用配置的 STT 提供商自动转录。

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

## 主频道

将 `SLACK_HOME_CHANNEL` 设置为频道 ID，Hermes 将在此频道发送计划消息、定时任务结果和其他主动通知。查找频道 ID 的方法：

1. 在 Slack 中右键点击频道名称
2. 点击 **View channel details**
3. 向下滚动——频道 ID 显示在底部

```bash
SLACK_HOME_CHANNEL=C01234567890
```

确保机器人已被**邀请到该频道**（`/invite @Hermes Agent`）。

---

## 多工作区支持

Hermes 可以使用单个 gateway 实例**同时连接多个 Slack 工作区**。每个工作区使用其自己的机器人用户 ID 独立认证。

### 配置

在 `SLACK_BOT_TOKEN` 中以**逗号分隔列表**的形式提供多个 bot token：

```bash
# 多个 bot token——每个工作区一个
SLACK_BOT_TOKEN=xoxb-workspace1-token,xoxb-workspace2-token,xoxb-workspace3-token

# Socket Mode 仍使用单个 app-level token
SLACK_APP_TOKEN=xapp-your-app-token
```

或在 `~/.hermes/config.yaml` 中：

```yaml
platforms:
  slack:
    token: "xoxb-workspace1-token,xoxb-workspace2-token"
```

### OAuth Token 文件

除了环境变量或配置中的 token 外，Hermes 还会从以下位置的 **OAuth token 文件**加载 token：

```
~/.hermes/slack_tokens.json
```

此文件是一个将团队 ID 映射到 token 条目的 JSON 对象：

```json
{
  "T01ABC2DEF3": {
    "token": "xoxb-workspace-token-here",
    "team_name": "My Workspace"
  }
}
```

此文件中的 token 会与通过 `SLACK_BOT_TOKEN` 指定的 token 合并。重复的 token 会自动去重。

### 工作原理

- 列表中的**第一个 token** 是主 token，用于 Socket Mode 连接（AsyncApp）。
- 每个 token 在启动时通过 `auth.test` 进行认证。gateway 将每个 `team_id` 映射到其自己的 `WebClient` 和 `bot_user_id`。
- 消息到达时，Hermes 使用正确的工作区特定客户端进行响应。
- 主 `bot_user_id`（来自第一个 token）用于向后兼容期望单一机器人身份的功能。

---

## 语音消息

Hermes 支持 Slack 上的语音功能：

- **传入：** 语音/音频消息使用配置的 STT 提供商自动转录：本地 `faster-whisper`、Groq Whisper（`GROQ_API_KEY`）或 OpenAI Whisper（`VOICE_TOOLS_OPENAI_KEY`）
- **传出：** TTS 响应以音频文件附件形式发送

---

## 按频道设置 Prompt

为特定 Slack 频道分配临时系统 prompt（提示词）。该 prompt 在运行时每轮注入——从不持久化到对话历史——因此更改立即生效。

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

键为 Slack 频道 ID（通过频道详情 → "关于" → 滚动到底部查找）。匹配频道中的所有消息都会将该 prompt 作为临时系统指令注入。

## 按频道绑定技能

在特定频道或私信中新会话开始时自动加载技能。与按频道设置 prompt（每轮注入）不同，技能绑定在**会话开始时**将技能内容作为用户消息注入——它成为对话历史的一部分，后续轮次无需重新加载。

这非常适合有专用用途的私信或频道（闪卡、特定领域问答机器人、支持分类频道等），在这些场景中你不希望模型自己的技能选择器在每次简短回复时决定是否加载。

```yaml
slack:
  channel_skill_bindings:
    # 私信频道——始终以"german-flashcards"模式运行
    - id: "D0ATH9TQ0G6"
      skills:
        - german-flashcards
    # 研究频道——按顺序预加载多个技能
    - id: "C01RESEARCH"
      skills:
        - arxiv
        - writing-plans
    # 简写形式：单个技能作为字符串
    - id: "C02SUPPORT"
      skill: hubspot-on-demand
```

注意事项：
- 绑定按频道 ID 匹配。对于绑定频道中的话题消息，话题继承父频道的绑定。
- 技能仅在会话开始时加载（新会话或自动重置后）。如果更改绑定，请运行 `/new` 或等待会话自动重置以使其生效。
- 与 `channel_prompts` 结合使用，可在技能指令之上为每个频道设置语气/约束。

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| 机器人不响应私信 | 验证 `message.im` 在事件订阅中，且应用已重新安装 |
| 机器人在私信中正常但在频道中不响应 | **最常见问题。** 将 `message.channels` 和 `message.groups` 添加到事件订阅，重新安装应用，并用 `/invite @Hermes Agent` 邀请机器人加入频道 |
| 机器人不响应频道中的 @mention | 1) 检查 `message.channels` 事件是否已订阅。2) 机器人必须被邀请到频道。3) 确保已添加 `channels:history` 权限范围。4) 更改权限范围/事件后重新安装应用 |
| 机器人忽略私有频道中的消息 | 添加 `message.groups` 事件订阅和 `groups:history` 权限范围，然后重新安装应用并 `/invite` 机器人 |
| 机器人不响应群组私信（多人私信） | 添加 `message.mpim` 事件订阅和 `mpim:history` 权限范围（以及 `mpim:read`），然后**重新安装**应用。没有 `message.mpim`，即使 1:1 私信正常，Slack 也永远不会向机器人投递群组私信消息。 |
| 私信中出现"向此应用发送消息已被关闭" | 在 App Home 设置中启用 **Messages Tab**（见第五步） |
| "not_authed" 或 "invalid_auth" 错误 | 重新生成 Bot Token 和 App Token，更新 `.env` |
| 机器人响应但无法在频道中发帖 | 用 `/invite @Hermes Agent` 邀请机器人加入频道 |
| 机器人可以聊天但无法读取上传的图片/文件 | 添加 `files:read`，然后**重新安装**应用。当 Slack 返回权限范围/认证/权限失败时，Hermes 现在会在聊天中显示附件访问诊断信息。 |
| `missing_scope` 错误 | 在 OAuth & Permissions 中添加所需权限范围，然后**重新安装**应用 |
| Socket 频繁断开 | 检查你的网络；Bolt 会自动重连，但不稳定的连接会导致延迟 |
| 更改了权限范围/事件但没有任何变化 | 更改任何权限范围或事件订阅后，**必须重新安装**应用到工作区 |

### 快速检查清单

如果机器人在频道中不工作，请验证以下**所有**项目：

1. ✅ 已订阅 `message.channels` 事件（公开频道）
2. ✅ 已订阅 `message.groups` 事件（私有频道）
3. ✅ 已订阅 `app_mention` 事件
4. ✅ 已添加 `channels:history` 权限范围（公开频道）
5. ✅ 已添加 `groups:history` 权限范围（私有频道）
6. ✅ 添加权限范围/事件后已**重新安装**应用
7. ✅ 已**邀请**机器人加入频道（`/invite @Hermes Agent`）
8. ✅ 你在消息中**@mention** 了机器人

---

## 安全

:::warning
**始终设置 `SLACK_ALLOWED_USERS`**，填入授权用户的 Member ID。没有此设置，gateway 默认会**拒绝所有消息**作为安全措施。切勿分享你的 bot token——像密码一样对待它们。
:::

- Token 应存储在 `~/.hermes/.env` 中（文件权限 `600`）
- 定期通过 Slack 应用设置轮换 token
- 审计谁有权访问你的 Hermes 配置目录
- Socket Mode 意味着不暴露公开端点——减少一个攻击面