---
sidebar_position: 3
title: "Discord"
description: "将 Hermes Agent 设置为 Discord 机器人"
---

# Discord 设置

Hermes Agent 集成了 Discord 作为机器人，让你可以通过私信或服务端频道与 AI 助手聊天。机器人接收你的消息，通过 Hermes Agent 管道（包括工具使用、记忆和推理）处理它们，并实时回复。它支持文本、语音消息、文件附件和斜杠命令。

在设置之前，这里是大多数人想了解的内容：Hermes 在你的服务器上后是如何行为的。

## Hermes 的行为方式

| 上下文 | 行为 |
|---------|----------|
| **私信** | Hermes 回复每条消息。不需要 `@mention`。每个私信都有自己的会话。 |
| **服务端频道** | 默认情况下，Hermes 只在你 `@mention` 它时回复。如果你在频道中发帖而没有提及它，Hermes 会忽略该消息。 |
| **自由回复频道** | 你可以使用 `DISCORD_FREE_RESPONSE_CHANNELS` 使特定频道无需提及，或使用 `DISCORD_REQUIRE_MENTION=false` 全局禁用提及。这些频道中的消息会内联回复 —— 跳过自动创建主题，以保持频道是轻量级的聊天。 |
| **主题** | Hermes 在同一主题中回复。提及规则仍然适用，除非该主题或其父频道配置为自由回复。主题的会话历史与父频道隔离。 |
| **多用户共享频道** | 默认情况下，Hermes 在频道内为每个用户隔离会话历史，以确保安全和清晰。同一频道中的两个人默认不会共享一个对话记录，除非你明确禁用该选项。 |
| **提及其他用户的消息** | 当 `DISCORD_IGNORE_NO_MENTION` 为 `true`（默认值）时，如果一条消息 @提及了其他用户但**没有**提及机器人，Hermes 会保持沉默。这防止机器人跳入针对其他人的对话。如果你想让机器人回复所有消息而不管提及了谁，请设置为 `false`。这仅适用于服务端频道，不适用于私信。 |

:::tip
如果你想要一个正常的机器人帮助频道，人们可以在那里与 Hermes 交谈而无需每次都标记它，将该频道添加到 `DISCORD_FREE_RESPONSE_CHANNELS`。
:::

### Discord Gateway 模型

Discord 上的 Hermes 不是一个无状态回复的 webhook。它通过完整的消息 gateway 运行，这意味着每条 incoming 消息都会经过：

1. 授权（`DISCORD_ALLOWED_USERS`）
2. 提及 / 自由回复检查
3. 会话查找
4. 会话记录加载
5. 正常的 Hermes agent 执行，包括工具、记忆和斜杠命令
6. 回复投递回 Discord

这很重要，因为在繁忙服务器中的行为取决于 Discord 路由和 Hermes 会话策略。

### Discord 中的会话模型

默认情况下：

- 每个私信都有自己的会话
- 每个服务端主题都有自己的会话命名空间
- 共享频道中的每个用户在该频道内都有自己的会话

因此，如果 Alice 和 Bob 都在 `#research` 中与 Hermes 交谈，Hermes 默认将这些视为单独的对话，即使它们使用的是同一个可见的 Discord 频道。

这可以通过 `config.yaml` 控制：

```yaml
group_sessions_per_user: true
```

仅当你明确希望整个房间共享一个对话时，才将其设置为 `false`：

```yaml
group_sessions_per_user: false
```

共享会话对于协作房间可能很有用，但它们也意味着：

- 用户共享上下文增长和 token 成本
- 一个人的长时间工具密集型任务可能会使其他人的上下文膨胀
- 一个人的运行中任务可能会中断同一房间中另一个人的后续操作

### 中断和并发

Hermes 通过会话 key 跟踪运行中的 agents。

使用默认的 `group_sessions_per_user: true`：

- Alice 中断她自己的运行中请求只影响该频道中 Alice 的会话
- Bob 可以在同一频道继续交谈，而不会继承 Alice 的历史或中断 Alice 的运行

使用 `group_sessions_per_user: false`：

- 整个房间共享该频道/主题的一个运行中 agent 槽位
- 来自不同人的后续消息可能会相互中断或排队

本指南将引导你完成完整的设置过程 —— 从在 Discord 的开发者门户上创建你的机器人到发送你的第一条消息。

## 步骤 1：创建 Discord 应用

1. 前往 [Discord 开发者门户](https://discord.com/developers/applications) 并使用你的 Discord 账号登录。
2. 点击右上角的 **New Application**。
3. 输入你的应用名称（例如 "Hermes Agent"）并接受开发者服务条款。
4. 点击 **Create**。

你将进入 **General Information** 页面。注意 **Application ID** —— 稍后你需要它来构建邀请 URL。

## 步骤 2：创建机器人

1. 在左侧边栏中，点击 **Bot**。
2. Discord 会自动为你的应用创建一个机器人用户。你将看到机器人的用户名，你可以自定义它。
3. 在 **Authorization Flow** 下：
   - 将 **Public Bot** 设置为 **ON** —— 需要使用 Discord 提供的邀请链接（推荐）。这允许 Installation 标签页生成默认的授权 URL。
   - 将 **Require OAuth2 Code Grant** 保持为 **OFF**。

:::tip
你可以在此页面上为你的机器人设置自定义头像和横幅。这是用户在 Discord 中看到的。
:::

:::info[私有机器人替代方案]
如果你更喜欢保持机器人私有（Public Bot = OFF），你必须使用步骤 5 中的 **Manual URL** 方法而不是 Installation 标签页。Discord 提供的链接需要启用 Public Bot。
:::

## 步骤 3：启用特权 Gateway Intents

这是整个设置中最关键的一步。如果没有启用正确的 intents，你的机器人将连接到 Discord 但**将无法读取消息内容**。

在 **Bot** 页面上，向下滚动到 **Privileged Gateway Intents**。你将看到三个开关：

| Intent | 用途 | 必需？ |
|--------|---------|-----------| 
| **Presence Intent** | 查看用户在线/离线状态 | 可选 |
| **Server Members Intent** | 访问成员列表，解析用户名 | **必需** |
| **Message Content Intent** | 读取消息的文本内容 | **必需** |

**通过切换为 ON 来启用 Server Members Intent 和 Message Content Intent**。

- 没有 **Message Content Intent**，你的机器人会收到消息事件，但消息文本为空 —— 机器人根本无法看到你输入了什么。
- 没有 **Server Members Intent**，机器人无法为允许的用户列表解析用户名，并且可能无法识别谁在给它发消息。

:::warning[这是 Discord 机器人不工作的 #1 原因]
如果你的机器人在线但从不回复消息，**Message Content Intent** 几乎可以肯定被禁用了。返回 [开发者门户](https://discord.com/developers/applications)，选择你的应用 → Bot → Privileged Gateway Intents，并确保 **Message Content Intent** 切换为 ON。点击 **Save Changes**。
:::

**关于服务器数量：**

- 如果你的机器人在**少于 100 个服务器**中，你可以自由地打开和关闭 intents。
- 如果你的机器人在**100 个或更多服务器**中，Discord 要求你提交验证申请才能使用特权 intents。对于个人使用，这不是问题。

点击页面底部的 **Save Changes**。

## 步骤 4：获取机器人 Token

机器人 token 是 Hermes Agent 用于以你的机器人身份登录的凭证。仍在 **Bot** 页面上：

1. 在 **Token** 部分下，点击 **Reset Token**。
2. 如果你的 Discord 账号启用了双因素认证，输入你的 2FA 代码。
3. Discord 将显示你的新 token。**立即复制它。**

:::warning[Token 仅显示一次]
Token 仅显示一次。如果你丢失了它，你需要重置它并生成一个新的。永远不要公开分享你的 token 或将其提交到 Git —— 任何拥有此 token 的人都可以完全控制你的机器人。
:::

将 token 存储在安全的地方（例如密码管理器）。你将在步骤 8 中需要它。

## 步骤 5：生成邀请 URL

你需要一个 OAuth2 URL 来邀请机器人到你的服务器。有两种方法可以做到这一点：

### 选项 A：使用 Installation 标签页（推荐）

:::note[需要 Public Bot]
此方法要求在步骤 2 中将 **Public Bot** 设置为 **ON**。如果你将 Public Bot 设置为 OFF，请使用下面的 Manual URL 方法。
:::

1. 在左侧边栏中，点击 **Installation**。
2. 在 **Installation Contexts** 下，启用 **Guild Install**。
3. 对于 **Install Link**，选择 **Discord Provided Link**。
4. 在 Guild Install 的 **Default Install Settings** 下：
   - **Scopes**：选择 `bot` 和 `applications.commands`
   - **Permissions**：选择下面列出的权限。

### 选项 B：Manual URL

你可以使用以下格式直接构建邀请 URL：

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=274878286912
```

将 `YOUR_APP_ID` 替换为步骤 1 中的 Application ID。

### 必需的权限

这些是你的机器人所需的最低权限：

- **View Channels** —— 查看它可以访问的频道
- **Send Messages** —— 回复你的消息
- **Embed Links** —— 格式化富回复
- **Attach Files** —— 发送图片、音频和文件输出
- **Read Message History** —— 维护对话上下文

### 推荐的附加权限

- **Send Messages in Threads** —— 在主题对话中回复
- **Add Reactions** —— 对消息做出反应以表示确认

### 权限整数

| 级别 | 权限整数 | 包含内容 |
|-------|-------------------|-----------------|
| 最小 | `117760` | 查看频道、发送消息、读取消息历史、附加文件 |
| 推荐 | `274878286912` | 上述所有加上嵌入链接、在主题中发送消息、添加反应 |

## 步骤 6：邀请到你的服务器

1. 在浏览器中打开邀请 URL（从 Installation 标签页或你构建的 manual URL）。
2. 在 **Add to Server** 下拉菜单中，选择你的服务器。
3. 点击 **Continue**，然后 **Authorize**。
4. 如果出现提示，完成 CAPTCHA。

:::info
你需要在 Discord 服务器上拥有 **Manage Server** 权限才能邀请机器人。如果你在下拉菜单中看不到你的服务器，请让服务器管理员改用邀请链接。
:::

授权后，机器人将出现在你的服务器的成员列表中（在你启动 Hermes gateway 之前，它将显示为离线）。

## 步骤 7：找到你的 Discord 用户 ID

Hermes Agent 使用你的 Discord 用户 ID 来控制谁可以与机器人交互。要找到它：

1. 打开 Discord（桌面或 Web 应用）。
2. 转到 **Settings** → **Advanced** → 将 **Developer Mode** 切换为 **ON**。
3. 关闭设置。
4. 右键点击你自己的用户名（在消息中、成员列表或你的个人资料中）→ **Copy User ID**。

你的用户 ID 是一个长数字，如 `284102345871466496`。

:::tip
开发者模式还允许你以相同方式复制**频道 ID** 和**服务器 ID** —— 右键点击频道或服务器名称并选择 Copy ID。如果你想手动设置主页频道，你将需要频道 ID。
:::

## 步骤 8：配置 Hermes Agent

### 选项 A：交互式设置（推荐）

运行引导式设置命令：

```bash
hermes gateway setup
```

当提示时选择 **Discord**，然后在询问时粘贴你的机器人 token 和用户 ID。

### 选项 B：手动配置

将以下内容添加到你的 `~/.hermes/.env` 文件：

```bash
# 必需
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=284102345871466496

# 多个允许的用户（逗号分隔）
# DISCORD_ALLOWED_USERS=284102345871466496,198765432109876543
```

然后启动 gateway：

```bash
hermes gateway
```

机器人应在几秒钟内在 Discord 中上线。向它发送一条消息 —— 私信或它可以看到的频道中 —— 进行测试。

:::tip
你可以在后台运行 `hermes gateway` 或作为 systemd 服务以实现持久运行。详见部署文档。
:::

## 配置参考

Discord 行为通过两个文件控制：**`~/.hermes/.env`** 用于凭证和环境级开关，以及 **`~/.hermes/config.yaml`** 用于结构化设置。当两者都设置时，环境变量始终优先于 config.yaml 值。

### 环境变量（`.env`）

| 变量 | 必需 | 默认值 | 描述 |
|----------|----------|---------|-------------|
| `DISCORD_BOT_TOKEN` | **是** | — | 来自 [Discord 开发者门户](https://discord.com/developers/applications) 的机器人 token。 |
| `DISCORD_ALLOWED_USERS` | **是** | — | 允许与机器人交互的逗号分隔的 Discord 用户 ID 列表。没有这个 **或** `DISCORD_ALLOWED_ROLES`，gateway 拒绝所有用户。 |
| `DISCORD_ALLOWED_ROLES` | 否 | — | 逗号分隔的 Discord 角色 ID。拥有这些角色之一的任何成员都被授权 —— 与 `DISCORD_ALLOWED_USERS` 的 OR 语义。在连接时自动启用 **Server Members Intent**。当审核团队人员流动时有用：新审核人员在角色授予后立即获得访问权限，无需推送配置。 |
| `DISCORD_HOME_CHANNEL` | 否 | — | 机器人发送主动消息（cron 输出、提醒、通知）的频道 ID。 |
| `DISCORD_HOME_CHANNEL_NAME` | 否 | `"Home"` | 日志和状态输出中主页频道的显示名称。 |
| `DISCORD_COMMAND_SYNC_POLICY` | 否 | `"safe"` | 控制原生斜杠命令启动同步。`"safe"` 比较现有的全局命令，只更新更改的内容，在无法通过补丁应用 Discord 元数据时重新创建命令。`"bulk"` 保留旧的 `tree.sync()` 行为。`"off"` 完全跳过启动同步。 |
| `DISCORD_REQUIRE_MENTION` | 否 | `true` | 当 `true` 时，机器人只在 `@mentioned` 时在服务端频道中回复。设置为 `false` 以回复每个频道中的所有消息。 |
| `DISCORD_FREE_RESPONSE_CHANNELS` | 否 | — | 逗号分隔的频道 ID，即使 `DISCORD_REQUIRE_MENTION` 为 `true`，机器人也会在这些频道中无需 `@mention` 即可回复。 |
| `DISCORD_IGNORE_NO_MENTION` | 否 | `true` | 当 `true` 时，如果一条消息 `@mentions` 了其他用户但**没有**提及机器人，机器人会保持沉默。防止机器人跳入针对其他人的对话。仅适用于服务端频道，不适用于私信。 |
| `DISCORD_AUTO_THREAD` | 否 | `true` | 当 `true` 时，自动为文本频道中的每个 `@mention` 创建一个新主题，因此每个对话都是隔离的（类似于 Slack 行为）。主题内或私信中的消息不受影响。 |
| `DISCORD_ALLOW_BOTS` | 否 | `"none"` | 控制机器人如何处理来自其他 Discord 机器人的消息。`"none"` —— 忽略所有其他机器人。`"mentions"` —— 只接受 `@mention` Hermes 的机器人消息。`"all"` —— 接受所有机器人消息。 |
| `DISCORD_REACTIONS` | 否 | `true` | 当 `true` 时，机器人在处理过程中向消息添加表情反应（开始时 👀，成功时 ✅，错误时 ❌）。设置为 `false` 以完全禁用反应。 |
| `DISCORD_IGNORED_CHANNELS` | 否 | — | 逗号分隔的频道 ID，机器人**永远**不会回复，即使 `@mentioned`。优先于所有其他频道设置。 |
| `DISCORD_ALLOWED_CHANNELS` | 否 | — | 逗号分隔的频道 ID。设置时，机器人**只**在这些频道中回复（加上允许的私信）。覆盖 `config.yaml` `discord.allowed_channels`。与 `DISCORD_IGNORED_CHANNELS` 结合使用以表达允许/拒绝规则。 |
| `DISCORD_NO_THREAD_CHANNELS` | 否 | — | 逗号分隔的频道 ID，机器人在这些频道中直接在频道中回复，而不是创建一个主题。仅在 `DISCORD_AUTO_THREAD` 为 `true` 时相关。 |
| `DISCORD_REPLY_TO_MODE` | 否 | `"first"` | 控制回复引用行为：`"off"` —— 永远不回复原始消息，`"first"` —— 仅在第一条消息块上回复引用（默认），`"all"` —— 在每个块上回复引用。 |
| `DISCORD_ALLOW_MENTION_EVERYONE` | 否 | `false` | 当 `false`（默认）时，即使其响应包含这些 token，机器人也无法 ping `@everyone` 或 `@here`。设置为 `true` 以重新加入。请参阅下面的 [Mention Control](#mention-control)。 |
| `DISCORD_ALLOW_MENTION_ROLES` | 否 | `false` | 当 `false`（默认）时，机器人无法 ping `@role` 提及。设置为 `true` 以允许。 |
| `DISCORD_ALLOW_MENTION_USERS` | 否 | `true` | 当 `true`（默认）时，机器人可以通过 ID ping 单个用户。 |
| `DISCORD_ALLOW_MENTION_REPLIED_USER` | 否 | `true` | 当 `true`（默认）时，回复消息会 ping 原始作者。 |
| `DISCORD_PROXY` | 否 | — | Discord 连接的代理 URL（HTTP、WebSocket、REST）。覆盖 `HTTPS_PROXY`/`ALL_PROXY`。支持 `http://`、`https://` 和 `socks5://` 方案。 |
| `HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS` | 否 | `0.6` | 适配器在刷新排队的文本块之前等待的宽限窗口。用于平滑流式输出。 |
| `HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS` | 否 | `2.0` | 当单条消息超过 Discord 的长度限制时，分割块之间的延迟。 |

### 配置文件（`config.yaml`）

`~/.hermes/config.yaml` 中的 `discord` 部分镜像上述环境变量。Config.yaml 设置作为默认值应用 —— 如果等效的环境变量已设置，环境变量优先。

```yaml
# Discord 特定的设置
discord:
  require_mention: true           # 在服务端频道中需要 @mention
  free_response_channels: ""      # 逗号分隔的频道 ID（或 YAML 列表）
  auto_thread: true                 # 在 @mention 时自动创建主题
  reactions: true                 # 在处理过程中添加表情反应
  ignored_channels: []            # 机器人永远不会回复的频道 ID
  no_thread_channels: []          # 机器人无需创建主题即可回复的频道 ID
  channel_prompts: {}             # 每频道临时系统提示
  allow_mentions:                 # 允许机器人 ping 的内容（安全默认值）
    everyone: false               # @everyone / @here pings（默认：false）
    roles: false                  # @role pings（默认：false）
    users: true                   # @user pings（默认：true）
    replied_user: true            # 回复引用 pings 作者（默认：true）

# 会话隔离（适用于所有 gateway 平台，不仅仅是 Discord）
group_sessions_per_user: true     # 在共享频道中按用户隔离会话
```

#### `discord.require_mention`

**类型：** boolean — **默认：** `true`

启用后，机器人只在被直接 `@mentioned` 时在服务端频道中回复。无论此设置如何，私信总是会得到回复。

#### `discord.free_response_channels`

**类型：** string 或 list — **默认：** `""`

机器人无需 `@mention` 即可回复所有消息的频道 ID。接受逗号分隔的字符串或 YAML 列表：

```yaml
# 字符串格式
discord:
  free_response_channels: "1234567890,9876543210"

# 列表格式
discord:
  free_response_channels:
    - 1234567890
    - 9876543210
```

如果主题的父频道在此列表中，该主题也变为无需提及。

自由回复频道还**跳过自动创建主题** —— 机器人内联回复，而不是为每条消息启动一个新主题。这使频道可用作轻量级聊天界面。如果你想要主题行为，不要将频道列为自由回复（改为使用正常的 `@mention` 流程）。

#### `discord.auto_thread`

**类型：** boolean — **默认：** `true`

启用后，常规文本频道中的每个 `@mention` 都会自动为对话创建一个新主题。这使主频道保持清洁，并给每个对话提供自己的隔离会话历史。创建主题后，该主题中的后续消息不需要 `@mention` —— 机器人知道它已经参与了。

在现有主题或私信中发送的消息不受此设置影响。在 `discord.free_response_channels` 或 `discord.no_thread_channels` 中列出的频道也会绕过自动创建主题，改为获得内联回复。

#### `discord.reactions`

**类型：** boolean — **默认：** `true`

控制机器人是否添加表情反应到消息作为视觉反馈：

- 👀 在机器人开始处理你的消息时添加
- ✅ 在响应成功送达时添加
- ❌ 在处理过程中发生错误时添加

如果你觉得反应分散注意力，或者机器人的角色没有**添加反应**权限，请禁用此功能。

#### `discord.ignored_channels`

**类型：** string 或 list — **默认：** `[]`

机器人**永远**不会回复的频道 ID，即使被直接 `@mentioned`。这具有最高优先级 —— 如果频道在此列表中，机器人会静默忽略那里的所有消息，无论 `require_mention`、`free_response_channels` 或任何其他设置。

```yaml
# 字符串格式
discord:
  ignored_channels: "1234567890,9876543210"

# 列表格式
discord:
  ignored_channels:
    - 1234567890
    - 9876543210
```

如果主题的父频道在此列表中，该主题中的消息也会被忽略。

#### `discord.no_thread_channels`

**类型：** string 或 list — **默认：** `[]`

机器人直接在频道中回复而不是自动创建主题的频道 ID。这仅在 `auto_thread` 为 `true`（默认值）时有效。在这些频道中，机器人像普通消息一样内联回复，而不是生成新主题。

```yaml
discord:
  no_thread_channels:
    - 1234567890  # 机器人此处内联回复
```

对于专门用于机器人交互的频道很有用，创建主题会增加不必要的干扰。

#### `discord.channel_prompts`

**类型：** mapping — **默认：** `{}`

每频道临时系统提示，在匹配的 Discord 频道或主题中每一轮都会注入，而不会持久化到记录历史中。

```yaml
discord:
  channel_prompts:
    "1234567890": |
      此频道用于研究任务。首选深度比较、
      引用和简洁的综合。
    "9876543210": |
      此论坛用于心理治疗风格的支持。要温暖、接地气、
      并且不评判。
```

行为：

- 精确的主题/频道 ID 匹配获胜。
- 如果消息在主题或论坛帖子中到达，而该主题没有明确的条目，Hermes 会回退到父频道/论坛 ID。
- 提示在运行时临时应用，因此更改它们会立即影响未来的轮次，而无需重写过去的会话历史。

#### `group_sessions_per_user`

**类型：** boolean — **默认：** `true`

这是一个全局 gateway 设置（不是 Discord 特定的），控制同一频道中的用户是否获得隔离的会话历史。

当 `true` 时：在 `#research` 中交谈的 Alice 和 Bob 各自与 Hermes 有单独的对话。当 `false` 时：整个频道共享一个对话记录和一个运行中 agent 槽位。

```yaml
group_sessions_per_user: true
```

有关每种模式的全部含义，请参阅上面的[会话模型](#discord-中的会话模型)部分。

#### `display.tool_progress`

**类型：** string — **默认：** `"all"` — **值：** `off`、`new`、`all`、`verbose`

控制机器人在处理过程中是否在聊天中发送进度消息（例如"正在读取文件..."、"正在运行终端命令..."）。这是适用于所有平台的全局 gateway 设置。

```yaml
display:
  tool_progress: "all"    # off | new | all | verbose
```

- `off` —— 无进度消息
- `new` —— 每轮只显示第一个工具调用
- `all` —— 显示所有工具调用（在 gateway 消息中截断为 40 个字符）
- `verbose` —— 显示完整的工具调用详细信息（可能会产生长消息）

#### `display.tool_progress_command`

**类型：** boolean — **默认：** `false`

启用后，使 `/verbose` 斜杠命令在 gateway 中可用，让你可以循环切换工具进度模式（`off → new → all → verbose → off`），而无需编辑 config.yaml。

```yaml
display:
  tool_progress_command: true
```

## 交互式模型选择器

在 Discord 频道中发送不带参数的 `/model` 以打开基于下拉菜单的模型选择器：

1. **提供商选择** —— 显示可用提供商的选择下拉菜单（最多 25 个）。
2. **模型选择** —— 第二个下拉菜单，包含所选提供商的模型（最多 25 个）。

选择器在 120 秒后超时。只有授权用户（那些在 `DISCORD_ALLOWED_USERS` 中的用户）可以与之交互。如果你知道模型名称，直接输入 `/model <name>`。

## 技能的原生斜杠命令

Hermes 自动将已安装的技能注册为**原生 Discord 应用命令**。这意味着技能与内置命令一起出现在 Discord 的自动完成 `/` 菜单中。

- 每个技能成为一个 Discord 斜杠命令（例如 `/code-review`、`/ascii-art`）
- 技能接受一个可选的 `args` 字符串参数
- Discord 对每个机器人有 100 个应用命令的限制 —— 如果你的技能多于可用槽位，额外的技能会被跳过，并在日志中发出警告
- 技能在机器人启动时与内置命令（如 `/model`、`/reset` 和 `/background`）一起注册

无需额外配置 —— 通过 `hermes skills install` 安装的任何技能都会在下次 gateway 重启时自动注册为 Discord 斜杠命令。

### 禁用斜杠命令注册

如果你针对同一个 Discord 应用运行多个 Hermes gateway（例如暂存 + 生产），只有其中一个应该拥有全局斜杠命令注册 —— 否则最后一次启动获胜，注册会反复变动。在"跟随者"gateway 上关闭斜杠注册：

```yaml
gateway:
  platforms:
    discord:
      extra:
        slash_commands: false   # 默认：true
```

在"主"gateway 上将此保留为 `true` 保持正常行为 —— 内置和已安装技能的全球 `/` 菜单命令。

## 发送媒体（`send_message` + `MEDIA:` 标签）

Discord 适配器通过 `send_message` 工具和 agent 发出的内联 `MEDIA:/path/to/file` 标签，支持每种常见媒体类型的原生文件上传：

| 类型 | 如何送达 |
|---|---|
| 图片（PNG/JPG/WebP） | 原生 Discord 图片附件，带内联预览 |
| 动画 GIF | `send_animation` 上传为 `animation.gif`，因此 Discord 内联播放它（不是作为静态缩略图） |
| 视频（MP4/MOV） | `send_video` —— 原生视频播放器 |
| 音频 / 语音 | `send_voice` —— 尽可能使用原生语音消息，否则使用文件附件 |
| 文档（PDF/ZIP/docx/等） | `send_document` —— 原生附件，带下载按钮 |

Discord 的每个上传大小限制取决于服务器的 boost 等级（免费 25 MB，最多 500 MB）。如果 Hermes 收到 HTTP 413，适配器会回退到指向本地缓存路径的链接，而不是静默失败。

## 主页频道

你可以指定一个"主页频道"，机器人可以在那里发送主动消息（如 cron 作业输出、提醒和通知）。有两种设置方法：

### 使用斜杠命令

在任何 Hermes 所在的 Discord 频道中输入 `/sethome`。该频道成为主页频道。

### 手动配置

将这些添加到你的 `~/.hermes/.env`：

```bash
DISCORD_HOME_CHANNEL=123456789012345678
DISCORD_HOME_CHANNEL_NAME="#bot-updates"
```

将 ID 替换为实际的频道 ID（右键点击 → 在开发者模式下复制频道 ID）。

## 语音消息

Hermes Agent 支持 Discord 语音消息：

- **incoming 语音消息** 使用配置的 STT 提供商自动转录：本地 `faster-whisper`（无 key）、Groq Whisper (`GROQ_API_KEY`) 或 OpenAI Whisper (`VOICE_TOOLS_OPENAI_KEY`)。
- **文本转语音**：使用 `/voice tts` 让机器人发送语音音频回复以及文本回复。
- **Discord 语音频道**：Hermes 还可以加入语音频道，听用户说话，并在频道中回复。

有关完整的设置和操作指南，请参阅：
- [Voice Mode](/docs/user-guide/features/voice-mode)
- [Use Voice Mode with Hermes](/docs/guides/use-voice-mode-with-hermes)

## 论坛频道

Discord 论坛频道（类型 15）不接受直接消息 —— 论坛中的每个帖子都必须是一个主题。Hermes 自动检测论坛频道，并在需要向那里发送时创建一个新的主题帖子，因此 `send_message`、TTS、图片、语音消息和文件附件都可以正常工作，无需 agent 特殊处理。

- **主题名称** 源自消息的第一行（去除 markdown 标题前缀，上限为 100 个字符）。当消息仅为附件时，文件名被用作回退主题名称。
- **附件** 在新的主题帖子的启动消息上随行 —— 无需单独的上传步骤，无部分发送。
- **一次调用，一个主题**：每个论坛发送都会创建一个新主题。因此，对同一论坛的连续发送将产生单独的主题。
- **检测是三层的**：首先是频道目录缓存，其次是进程本地探测缓存，最后是实时 `GET /channels/{id}` 探测作为最后手段（其结果随后在进程的生命周期内被记忆）。

刷新目录（在暴露它的平台上使用 `/channels refresh`，或 gateway 重启）会用机器人启动后创建的任何论坛频道填充缓存。

## 故障排除

### 机器人在线但不回复消息

**原因**：Message Content Intent 被禁用。

**修复**：前往 [开发者门户](https://discord.com/developers/applications) → 你的应用 → Bot → Privileged Gateway Intents → 启用 **Message Content Intent** → 保存更改。重启 gateway。

### 启动时出现"Disallowed Intents"错误

**原因**：你的代码请求了开发者门户中未启用的 intents。

**修复**：在 Bot 设置中启用所有三个特权 Gateway Intents（Presence、Server Members、Message Content），然后重启。

### 机器人无法看到特定频道中的消息

**原因**：机器人的角色没有查看该频道的权限。

**修复**：在 Discord 中，转到频道的设置 → 权限 → 添加机器人的角色，并启用**查看频道**和**读取消息历史**。

### 403 禁止错误

**原因**：机器人缺少必需的权限。

**修复**：使用步骤 5 中的 URL 重新邀请机器人，并带有正确的权限，或在服务器设置 → 角色中手动调整机器人的角色权限。

### 机器人离线

**原因**：Hermes gateway 未运行，或 token 不正确。

**修复**：检查 `hermes gateway` 是否正在运行。验证 `.env` 文件中的 `DISCORD_BOT_TOKEN`。如果你最近重置了 token，请更新它。

### "User not allowed" / 机器人忽略你

**原因**：你的用户 ID 不在 `DISCORD_ALLOWED_USERS` 中。

**修复**：将你的用户 ID 添加到 `~/.hermes/.env` 中的 `DISCORD_ALLOWED_USERS` 并重启 gateway。

### 同一频道中的人意外共享上下文

**原因**：`group_sessions_per_user` 被禁用，或平台无法为该上下文中的消息提供用户 ID。

**修复**：在 `~/.hermes/config.yaml` 中设置此项并重启 gateway：

```yaml
group_sessions_per_user: true
```

如果你有意想要一个共享的房间对话，请将其保留为关闭 —— 只需预期共享的记录历史和共享的中断行为。

## 安全

:::warning
始终设置 `DISCORD_ALLOWED_USERS`（或 `DISCORD_ALLOWED_ROLES`）以限制谁可以与机器人交互。如果没有任何一个，gateway 默认拒绝所有用户，作为一种安全措施。只授权你信任的人 —— 授权用户拥有对 agent 功能的完全访问权限，包括工具使用和系统访问。
:::

### 基于角色的访问控制

对于通过角色而不是个人用户列表来管理访问的服务器（审核团队、支持人员、内部工具），请使用 `DISCORD_ALLOWED_ROLES` —— 逗号分隔的角色 ID 列表。拥有这些角色之一的任何成员都被授权。

```bash
# ~/.hermes/.env —— 与 DISCORD_ALLOWED_USERS 一起或代替它工作
DISCORD_ALLOWED_ROLES=987654321098765432,876543210987654321
```

语义：

- **与用户许可名单的 OR。** 如果用户的 ID 在 `DISCORD_ALLOWED_USERS` **中或** 他们拥有 `DISCORD_ALLOWED_ROLES` 中的任何角色，则该用户被授权。
- **Server Members Intent 自动启用。** 当设置 `DISCORD_ALLOWED_ROLES` 时，机器人在连接时启用 Members intent —— Discord 需要它才能随成员记录发送角色信息。
- **角色 ID，而非名称。** 从 Discord 获取它们：**User Settings → Advanced → Developer Mode ON**，然后右键点击任何角色 → **Copy Role ID**。
- **私信回退。** 在私信中，角色检查扫描共同公会；在任何共享服务器中拥有允许角色的用户也在私信中被授权。

当审核团队人员流动时，这是首选模式 —— 新审核人员在角色授予的那一刻就获得访问权限，无需 `.env` 编辑或 gateway 重启。

### 提及控制

默认情况下，Hermes 阻止机器人 ping `@everyone`、`@here` 和角色提及，即使其回复包含这些 token。这可以防止措辞不当的提示或回显的用户内容向整个服务器发送垃圾邮件。单个 `@user` ping 和回复引用 ping（小小的"回复给..."芯片）保持启用，因此正常的对话仍然有效。

你可以通过环境变量或 `config.yaml` 放宽这些默认值：

```yaml
# ~/.hermes/config.yaml
discord:
  allow_mentions:
    everyone: false      # 允许机器人 ping @everyone / @here
    roles: false         # 允许机器人 ping @role 提及
    users: true          # 允许机器人 ping 单个 @users
    replied_user: true   # 回复消息时 ping 作者
```

```bash
# ~/.hermes/.env —— 环境变量优先于 config.yaml
DISCORD_ALLOW_MENTION_EVERYONE=false
DISCORD_ALLOW_MENTION_ROLES=false
DISCORD_ALLOW_MENTION_USERS=true
DISCORD_ALLOW_MENTION_REPLIED_USER=true
```

:::tip
除非你确切知道为什么需要它们，否则将 `everyone` 和 `roles` 保留为 `false`。LLM 很容易在看似正常的响应中产生字符串 `@everyone`；如果没有这种保护，那将通知你服务器的每个成员。
:::

有关保护你的 Hermes Agent 部署的更多信息，请参阅 [安全指南](../security.md)。
