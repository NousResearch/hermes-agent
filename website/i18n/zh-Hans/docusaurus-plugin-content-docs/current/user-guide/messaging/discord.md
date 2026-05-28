---
sidebar_position: 3
title: "Discord"
description: "将 Hermes Agent 设置为 Discord 机器人"
---

# Discord 设置

Hermes Agent 以机器人形式与 Discord 集成，让你可以通过私信或服务器频道与 AI 助手对话。机器人接收你的消息，通过 Hermes Agent 管道（包括工具调用、记忆和推理）进行处理，并实时响应。它支持文本、语音消息、文件附件和斜杠命令。

在开始设置之前，先介绍大多数人最想了解的内容：Hermes 进入服务器后的行为方式。

## Hermes 的行为方式

| 上下文 | 行为 |
|---------|----------|
| **私信（DM）** | Hermes 响应每条消息，无需 `@提及`。每个私信有独立的会话。 |
| **服务器频道** | 默认情况下，Hermes 仅在被 `@提及` 时响应。如果你在频道中发帖但未提及它，Hermes 会忽略该消息。 |
| **自由响应频道** | 你可以通过 `DISCORD_FREE_RESPONSE_CHANNELS` 将特定频道设为无需提及，或通过 `DISCORD_REQUIRE_MENTION=false` 全局禁用提及要求。这些频道中的消息会直接回复——自动创建线程功能会被跳过，使频道保持轻量级聊天状态。 |
| **线程（Thread）** | Hermes 在同一线程中回复。提及规则仍然适用，除非该线程或其父频道被配置为自由响应。线程的会话历史与父频道相互隔离。 |
| **多用户共享频道** | 默认情况下，Hermes 为安全和清晰起见，在频道内按用户隔离会话历史。在同一频道中交谈的两个人不会共享同一份对话记录，除非你明确禁用该功能。 |
| **提及其他用户的消息** | 当 `DISCORD_IGNORE_NO_MENTION` 为 `true`（默认值）时，如果消息 @提及了其他用户但**未**提及机器人，Hermes 保持沉默。这可防止机器人介入针对其他人的对话。如果你希望机器人响应所有消息而不管提及了谁，请设置为 `false`。此设置仅适用于服务器频道，不适用于私信。 |

:::tip
如果你想要一个普通的机器人帮助频道，让用户无需每次都 @标记就能与 Hermes 对话，请将该频道添加到 `DISCORD_FREE_RESPONSE_CHANNELS`。
:::

### Discord Gateway（网关）模型

Hermes 在 Discord 上不是无状态回复的 webhook（网络钩子）。它通过完整的消息网关运行，这意味着每条传入消息都会经过：

1. 授权验证（`DISCORD_ALLOWED_USERS`）
2. 提及 / 自由响应检查
3. 会话查找
4. 会话记录加载
5. 正常的 Hermes agent 执行，包括工具、记忆和斜杠命令
6. 将响应发送回 Discord

这一点很重要，因为在繁忙服务器中的行为取决于 Discord 路由和 Hermes 会话策略两者。

### Discord 中的会话模型

默认情况下：

- 每个私信有独立的会话
- 每个服务器线程有独立的会话命名空间
- 共享频道中的每个用户在该频道内有独立的会话

因此，如果 Alice 和 Bob 都在 `#research` 中与 Hermes 对话，即使他们使用的是同一个可见的 Discord 频道，Hermes 默认也会将其视为独立的对话。

这由 `config.yaml` 控制：

```yaml
group_sessions_per_user: true
```

仅当你明确希望整个房间共享一个对话时，才将其设置为 `false`：

```yaml
group_sessions_per_user: false
```

共享会话对协作房间可能有用，但这也意味着：

- 用户共享上下文增长和 token（令牌）成本
- 一个人的长时间重度工具任务会使所有人的上下文膨胀
- 一个人正在进行的运行可能会中断同一房间中另一个人的后续操作

### 中断与并发

Hermes 按会话键跟踪正在运行的 agent。

使用默认的 `group_sessions_per_user: true` 时：

- Alice 中断自己正在进行的请求只影响她在该频道中的会话
- Bob 可以继续在同一频道中交谈，不会继承 Alice 的历史记录或中断 Alice 的运行

使用 `group_sessions_per_user: false` 时：

- 整个房间共享该频道/线程的一个正在运行的 agent 槽位
- 不同人的后续消息可能会相互中断或排队等待

本指南将引导你完成完整的设置流程——从在 Discord 开发者门户创建机器人到发送第一条消息。

## 第一步：创建 Discord 应用

1. 前往 [Discord 开发者门户](https://discord.com/developers/applications) 并使用你的 Discord 账号登录。
2. 点击右上角的 **New Application**。
3. 输入应用名称（例如"Hermes Agent"）并接受开发者服务条款。
4. 点击 **Create**。

你将进入 **General Information** 页面。记下 **Application ID**——稍后构建邀请 URL 时需要用到。

## 第二步：创建机器人

1. 在左侧边栏中，点击 **Bot**。
2. Discord 会自动为你的应用创建一个机器人用户。你会看到机器人的用户名，可以自定义。
3. 在 **Authorization Flow** 下：
   - 将 **Public Bot** 设置为 **ON**——使用 Discord 提供的邀请链接时需要此设置（推荐）。这允许 Installation 标签页生成默认授权 URL。
   - 将 **Require OAuth2 Code Grant** 保持为 **OFF**。

:::tip
你可以在此页面为机器人设置自定义头像和横幅，这是用户在 Discord 中看到的样子。
:::

:::info[私有机器人替代方案]
如果你希望保持机器人私有（Public Bot = OFF），则**必须**在第五步中使用**手动 URL** 方法，而不是 Installation 标签页。Discord 提供的链接需要启用 Public Bot。
:::

## 第三步：启用特权网关 Intent（意图）

这是整个设置过程中最关键的步骤。如果没有启用正确的 intent，你的机器人将连接到 Discord，但**无法读取消息内容**。

在 **Bot** 页面，向下滚动到 **Privileged Gateway Intents**。你会看到三个开关：

| Intent | 用途 | 是否必需？ |
|--------|---------|-----------| 
| **Presence Intent** | 查看用户在线/离线状态 | 可选 |
| **Server Members Intent** | 访问成员列表、解析用户名 | **必需** |
| **Message Content Intent** | 读取消息的文本内容 | **必需** |

**将 Server Members Intent 和 Message Content Intent 都切换为 ON。**

- 没有 **Message Content Intent**，你的机器人会收到消息事件，但消息文本为空——机器人实际上看不到你输入的内容。
- 没有 **Server Members Intent**，机器人无法解析允许用户列表中的用户名，可能无法识别是谁在发消息。

:::warning[这是 Discord 机器人不工作的第一大原因]
如果你的机器人在线但从不响应消息，**Message Content Intent** 几乎可以肯定是被禁用了。返回 [开发者门户](https://discord.com/developers/applications)，选择你的应用 → Bot → Privileged Gateway Intents，确保 **Message Content Intent** 已切换为 ON。点击 **Save Changes**。
:::

**关于服务器数量：**
- 如果你的机器人在**少于 100 个服务器**中，可以自由切换 intent。
- 如果你的机器人在 **100 个或更多服务器**中，Discord 要求你提交验证申请才能使用特权 intent。对于个人使用，这不是问题。

点击页面底部的 **Save Changes**。

## 第四步：获取机器人 Token

机器人 token（令牌）是 Hermes Agent 用于以你的机器人身份登录的凭据。仍在 **Bot** 页面：

1. 在 **Token** 部分，点击 **Reset Token**。
2. 如果你的 Discord 账号启用了双重身份验证，请输入你的 2FA 代码。
3. Discord 将显示你的新 token。**立即复制它。**

:::warning[Token 仅显示一次]
Token 只显示一次。如果丢失，你需要重置并生成新的 token。切勿公开分享你的 token 或将其提交到 Git——任何拥有此 token 的人都可以完全控制你的机器人。
:::

将 token 存储在安全的地方（例如密码管理器）。你将在第八步中用到它。

## 第五步：生成邀请 URL

你需要一个 OAuth2 URL 来将机器人邀请到你的服务器。有两种方式：

### 方式 A：使用 Installation 标签页（推荐）

:::note[需要 Public Bot]
此方法要求在第二步中将 **Public Bot** 设置为 **ON**。如果你将 Public Bot 设置为 OFF，请改用下面的手动 URL 方法。
:::

1. 在左侧边栏中，点击 **Installation**。
2. 在 **Installation Contexts** 下，启用 **Guild Install**。
3. 对于 **Install Link**，选择 **Discord Provided Link**。
4. 在 Guild Install 的 **Default Install Settings** 下：
   - **Scopes**：选择 `bot` 和 `applications.commands`
   - **Permissions**：选择下面列出的权限。

### 方式 B：手动 URL

你可以使用以下格式直接构建邀请 URL：

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=274878286912
```

将 `YOUR_APP_ID` 替换为第一步中的 Application ID。

### 所需权限

以下是机器人所需的最低权限：

- **View Channels** — 查看其有权访问的频道
- **Send Messages** — 响应你的消息
- **Embed Links** — 格式化富文本响应
- **Attach Files** — 发送图片、音频和文件输出
- **Read Message History** — 维护对话上下文

### 推荐的附加权限

- **Send Messages in Threads** — 在线程对话中响应
- **Add Reactions** — 对消息添加反应以示确认

### 权限整数

| 级别 | 权限整数 | 包含内容 |
|-------|-------------------|-----------------|
| 最低 | `117760` | View Channels、Send Messages、Read Message History、Attach Files |
| 推荐 | `274878286912` | 以上所有权限，加上 Embed Links、Send Messages in Threads、Add Reactions |

## 第六步：邀请到你的服务器

1. 在浏览器中打开邀请 URL（来自 Installation 标签页或你构建的手动 URL）。
2. 在 **Add to Server** 下拉菜单中，选择你的服务器。
3. 点击 **Continue**，然后点击 **Authorize**。
4. 如有提示，完成 CAPTCHA 验证。

:::info
你需要在 Discord 服务器上拥有 **Manage Server** 权限才能邀请机器人。如果你在下拉菜单中看不到你的服务器，请让服务器管理员使用邀请链接。
:::

授权后，机器人将出现在你服务器的成员列表中（在你启动 Hermes 网关之前，它会显示为离线）。

## 第七步：找到你的 Discord 用户 ID

Hermes Agent 使用你的 Discord 用户 ID 来控制谁可以与机器人交互。查找方式：

1. 打开 Discord（桌面或网页应用）。
2. 前往 **Settings** → **Advanced** → 将 **Developer Mode** 切换为 **ON**。
3. 关闭设置。
4. 右键点击你自己的用户名（在消息中、成员列表中或你的个人资料中）→ **Copy User ID**。

你的用户 ID 是一个类似 `284102345871466496` 的长数字。

:::tip
开发者模式还允许你以相同方式复制**频道 ID** 和**服务器 ID**——右键点击频道或服务器名称并选择 Copy ID。如果你想手动设置主频道，将需要频道 ID。
:::

## 第八步：配置 Hermes Agent

### 方式 A：交互式设置（推荐）

运行引导式设置命令：

```bash
hermes gateway setup
```

在提示时选择 **Discord**，然后在询问时粘贴你的机器人 token 和用户 ID。

### 方式 B：手动配置

将以下内容添加到你的 `~/.hermes/.env` 文件：

```bash
# 必填
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=284102345871466496

# 多个允许用户（逗号分隔）
# DISCORD_ALLOWED_USERS=284102345871466496,198765432109876543
```

然后启动网关：

```bash
hermes gateway
```

机器人应在几秒钟内在 Discord 中上线。发送一条消息——私信或在它可以看到的频道中——进行测试。

:::tip
你可以在后台运行 `hermes gateway` 或将其作为 systemd 服务以持续运行。详情请参阅部署文档。
:::

## 配置参考

Discord 行为通过两个文件控制：**`~/.hermes/.env`** 用于凭据和环境级开关，**`~/.hermes/config.yaml`** 用于结构化设置。当两者都设置时，环境变量始终优先于 config.yaml 的值。

### 环境变量（`.env`）

| 变量 | 是否必填 | 默认值 | 描述 |
|----------|----------|---------|-------------|
| `DISCORD_BOT_TOKEN` | **是** | — | 来自 [Discord 开发者门户](https://discord.com/developers/applications) 的机器人 token。 |
| `DISCORD_ALLOWED_USERS` | **是** | — | 允许与机器人交互的 Discord 用户 ID，逗号分隔。没有此项**或** `DISCORD_ALLOWED_ROLES`，网关将拒绝所有用户。 |
| `DISCORD_ALLOWED_ROLES` | 否 | — | Discord 角色 ID，逗号分隔。拥有其中任一角色的成员即被授权——与 `DISCORD_ALLOWED_USERS` 为 OR 语义。连接时自动启用 **Server Members Intent**。适用于管理团队频繁变动的场景：新管理员一旦被授予角色即可获得访问权限，无需推送配置。 |
| `DISCORD_HOME_CHANNEL` | 否 | — | 机器人发送主动消息（cron 输出、提醒、通知）的频道 ID。 |
| `DISCORD_HOME_CHANNEL_NAME` | 否 | `"Home"` | 主频道在日志和状态输出中的显示名称。 |
| `DISCORD_COMMAND_SYNC_POLICY` | 否 | `"safe"` | 控制原生斜杠命令启动同步。`"safe"` 对现有全局命令进行差异比较，仅更新已更改的内容，当 Discord 元数据更改无法通过补丁应用时重新创建命令。`"bulk"` 保留旧的 `tree.sync()` 行为。`"off"` 完全跳过启动同步。 |
| `DISCORD_REQUIRE_MENTION` | 否 | `true` | 为 `true` 时，机器人仅在服务器频道中被 `@提及` 时响应。设置为 `false` 可响应每个频道中的所有消息。 |
| `DISCORD_THREAD_REQUIRE_MENTION` | 否 | `false` | 为 `true` 时，禁用线程内的提及快捷方式——线程与频道的门控方式相同，即使机器人已经参与其中，也需要 `@提及`。当多个机器人共享一个线程且你希望每个机器人仅在明确 `@提及` 时触发时使用此设置。 |
| `DISCORD_FREE_RESPONSE_CHANNELS` | 否 | — | 机器人无需 `@提及` 即可响应的频道 ID，逗号分隔，即使 `DISCORD_REQUIRE_MENTION` 为 `true` 也适用。 |
| `DISCORD_IGNORE_NO_MENTION` | 否 | `true` | 为 `true` 时，如果消息 `@提及` 了其他用户但**未**提及机器人，机器人保持沉默。防止机器人介入针对其他人的对话。仅适用于服务器频道，不适用于私信。 |
| `DISCORD_AUTO_THREAD` | 否 | `true` | 为 `true` 时，自动为文本频道中的每次 `@提及` 创建新线程，使每个对话相互隔离（类似 Slack 行为）。已在线程或私信中的消息不受影响。 |
| `DISCORD_ALLOW_BOTS` | 否 | `"none"` | 控制机器人如何处理来自其他 Discord 机器人的消息。`"none"` — 忽略所有其他机器人。`"mentions"` — 仅接受 `@提及` Hermes 的机器人消息。`"all"` — 接受所有机器人消息。 |
| `DISCORD_REACTIONS` | 否 | `true` | 为 `true` 时，机器人在处理过程中为消息添加 emoji 反应（开始时 👀，成功时 ✅，出错时 ❌）。设置为 `false` 可完全禁用反应。 |
| `DISCORD_IGNORED_CHANNELS` | 否 | — | 机器人**永不**响应的频道 ID，逗号分隔，即使被 `@提及` 也不响应。优先于所有其他频道设置。 |
| `DISCORD_ALLOWED_CHANNELS` | 否 | — | 频道 ID，逗号分隔。设置后，机器人**仅**在这些频道（以及允许的私信）中响应。覆盖 `config.yaml` 中的 `discord.allowed_channels`。与 `DISCORD_IGNORED_CHANNELS` 结合使用可表达允许/拒绝规则。 |
| `DISCORD_NO_THREAD_CHANNELS` | 否 | — | 机器人直接在频道中响应而不创建线程的频道 ID，逗号分隔。仅在 `DISCORD_AUTO_THREAD` 为 `true` 时有效。 |
| `DISCORD_HISTORY_BACKFILL` | 否 | `true` | 为 `true` 时，当机器人被提及时，将最近的频道滚动历史（自机器人上次响应以来）前置到用户消息中。恢复机器人在 `require_mention` 模式下会错过的上下文。在私信和自由响应频道中跳过。设置为 `false` 可禁用。 |
| `DISCORD_HISTORY_BACKFILL_LIMIT` | 否 | `50` | 组装回填块时向后扫描的最大消息数。实际上扫描通常会更早停止——在机器人自己在频道中的最后一条消息处。 |
| `DISCORD_REPLY_TO_MODE` | 否 | `"first"` | 控制回复引用行为：`"off"` — 从不回复原始消息，`"first"` — 仅在第一个消息块上添加回复引用（默认），`"all"` — 在每个块上都添加回复引用。 |
| `DISCORD_ALLOW_MENTION_EVERYONE` | 否 | `false` | 为 `false`（默认）时，即使响应中包含这些 token，机器人也无法 ping `@everyone` 或 `@here`。设置为 `true` 可重新启用。参见下方[提及控制](#mention-control)。 |
| `DISCORD_ALLOW_MENTION_ROLES` | 否 | `false` | 为 `false`（默认）时，机器人无法 ping `@role` 提及。设置为 `true` 可允许。 |
| `DISCORD_ALLOW_MENTION_USERS` | 否 | `true` | 为 `true`（默认）时，机器人可以通过 ID ping 单个用户。 |
| `DISCORD_ALLOW_MENTION_REPLIED_USER` | 否 | `true` | 为 `true`（默认）时，回复消息会 ping 原始作者。 |
| `DISCORD_PROXY` | 否 | — | Discord 连接的代理 URL（HTTP、WebSocket、REST）。覆盖 `HTTPS_PROXY`/`ALL_PROXY`。支持 `http://`、`https://` 和 `socks5://` 协议。 |
| `DISCORD_ALLOW_ANY_ATTACHMENT` | 否 | `false` | 为 `true` 时，机器人接受任何文件类型的附件（不仅限于内置的 PDF/文本/zip/office 允许列表）。未知类型会被缓存到磁盘，并以 `application/octet-stream` MIME 类型作为本地路径提供给 agent，以便它可以使用 `terminal` / `read_file` / `ffprobe` 等工具检查。 |
| `DISCORD_MAX_ATTACHMENT_BYTES` | 否 | `33554432` | 网关将下载并缓存的每个附件的最大字节数。默认 32 MiB。设置为 `0` 表示无上限（附件在写入时保存在内存中，因此无限制会带来真实的内存成本）。 |
| `HERMES_DISCORD_TEXT_BATCH_DELAY_SECONDS` | 否 | `0.6` | 适配器在刷新排队文本块之前等待的宽限窗口。用于平滑流式输出。 |
| `HERMES_DISCORD_TEXT_BATCH_SPLIT_DELAY_SECONDS` | 否 | `2.0` | 当单条消息超过 Discord 长度限制时，分割块之间的延迟。 |

### 配置文件（`config.yaml`）

`~/.hermes/config.yaml` 中的 `discord` 部分与上述环境变量对应。config.yaml 设置作为默认值应用——如果已设置等效的环境变量，则环境变量优先。

```yaml
# Discord 特定设置
discord:
  require_mention: true           # 在服务器频道中需要 @提及
  thread_require_mention: false   # 为 true 时，线程中也需要 @提及（多机器人线程）
  free_response_channels: ""      # 逗号分隔的频道 ID（或 YAML 列表）
  auto_thread: true               # 在 @提及 时自动创建线程
  reactions: true                 # 处理过程中添加 emoji 反应
  ignored_channels: []            # 机器人永不响应的频道 ID
  no_thread_channels: []          # 机器人不创建线程直接响应的频道 ID
  history_backfill: true          # 在提及时前置最近的频道滚动历史（默认：true）
  history_backfill_limit: 50      # 向后扫描的最大消息数（默认：50）
  channel_prompts: {}             # 每个频道的临时系统 prompt（提示词）
  allow_mentions:                 # 机器人允许 ping 的内容（安全默认值）
    everyone: false               # @everyone / @here ping（默认：false）
    roles: false                  # @role ping（默认：false）
    users: true                   # @user ping（默认：true）
    replied_user: true            # 回复引用会 ping 作者（默认：true）

# 会话隔离（适用于所有网关平台，不仅限于 Discord）
group_sessions_per_user: true     # 在共享频道中按用户隔离会话
```

#### `discord.require_mention`

**类型：** 布尔值 — **默认值：** `true`

启用后，机器人仅在服务器频道中被直接 `@提及` 时响应。无论此设置如何，私信始终会得到响应。

#### `discord.thread_require_mention`

**类型：** 布尔值 — **默认值：** `false`

默认情况下，一旦机器人参与了某个线程（通过 `@提及` 自动创建或回复过一次），它就会继续响应该线程中的每条后续消息，无需再次 `@提及`。这对于一对一对话来说是正确的默认行为。

在**多机器人线程**中，用户每次只与一个机器人交流，这个默认行为会成为隐患——线程中的每个其他机器人也会对每条消息触发，消耗额度并刷屏。将 `thread_require_mention: true` 设置为禁用线程内快捷方式，使线程与频道的门控方式相同。显式 `@提及` 仍然有效。

```yaml
discord:
  require_mention: true
  thread_require_mention: true    # 多机器人设置
```

#### `discord.free_response_channels`

**类型：** 字符串或列表 — **默认值：** `""`

机器人无需 `@提及` 即可响应所有消息的频道 ID。接受逗号分隔的字符串或 YAML 列表：

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

如果线程的父频道在此列表中，该线程也变为无需提及。

自由响应频道还会**跳过自动创建线程**——机器人直接回复而不是为每条消息创建新线程。这使频道可用作轻量级聊天界面。如果你想要线程行为，不要将频道列为自由响应（改用普通的 `@提及` 流程）。

#### `discord.auto_thread`

**类型：** 布尔值 — **默认值：** `true`

启用后，普通文本频道中的每次 `@提及` 都会自动为对话创建新线程。这保持主频道整洁，并为每个对话提供独立的会话历史。一旦创建线程，该线程中的后续消息不需要 `@提及`——机器人知道它已经在参与其中。对于多机器人设置，将 [`thread_require_mention`](#discordthread_require_mention) 设置为 `true` 可禁用此线程内快捷方式。

在现有线程或私信中发送的消息不受此设置影响。`discord.free_response_channels` 或 `discord.no_thread_channels` 中列出的频道也会绕过自动创建线程，改为直接回复。

#### `discord.reactions`

**类型：** 布尔值 — **默认值：** `true`

控制机器人是否为消息添加 emoji 反应作为视觉反馈：
- 👀 机器人开始处理你的消息时添加
- ✅ 响应成功发送时添加
- ❌ 处理过程中发生错误时添加

如果你觉得反应令人分心，或者机器人的角色没有 **Add Reactions** 权限，请禁用此功能。

#### `discord.ignored_channels`

**类型：** 字符串或列表 — **默认值：** `[]`

机器人**永不**响应的频道 ID，即使被直接 `@提及` 也不响应。这具有最高优先级——如果频道在此列表中，机器人会静默忽略那里的所有消息，无论 `require_mention`、`free_response_channels` 或任何其他设置如何。

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

如果线程的父频道在此列表中，该线程中的消息也会被忽略。

#### `discord.no_thread_channels`

**类型：** 字符串或列表 — **默认值：** `[]`

机器人直接在频道中响应而不自动创建线程的频道 ID。仅在 `auto_thread` 为 `true`（默认值）时有效。在这些频道中，机器人像普通消息一样直接回复，而不是创建新线程。

```yaml
discord:
  no_thread_channels:
    - 1234567890  # 机器人在此处直接回复
```

适用于专门用于机器人交互的频道，在这些频道中线程会增加不必要的噪音。

#### `discord.channel_prompts`

**类型：** 映射 — **默认值：** `{}`

每个频道的临时系统 prompt（提示词），在匹配的 Discord 频道或线程的每次对话轮次中注入，不会持久化到对话记录历史中。

```yaml
discord:
  channel_prompts:
    "1234567890": |
      This channel is for research tasks. Prefer deep comparisons,
      citations, and concise synthesis.
    "9876543210": |
      This forum is for therapy-style support. Be warm, grounded,
      and non-judgmental.
```

行为：
- 精确的线程/频道 ID 匹配优先。
- 如果消息到达线程或论坛帖子内，且该线程没有明确条目，Hermes 会回退到父频道/论坛 ID。
- Prompt 在运行时临时应用，因此更改后立即影响后续轮次，无需重写过去的会话历史。

#### `discord.history_backfill`

**类型：** 布尔值 — **默认值：** `true`

启用后，机器人在每次 `@提及` 时恢复错过的频道消息。当 `require_mention: true` 时，机器人只处理直接标记它的消息——频道中的其他所有内容对会话记录都是不可见的。历史回填在触发时向后扫描最近的频道历史，收集机器人上次响应与当前提及之间的消息，并将其作为上下文包含进来。

按界面的行为：

- **服务器频道**（使用 `require_mention: true`）：回填扫描自机器人上次响应以来的频道。当其他参与者在机器人未被提及时发帖时很有用。
- **线程**：回填仅扫描该线程——Discord 对线程的 `channel.history()` 只返回该线程的消息，不包括父频道。这是正确的范围，因为线程通常是自包含的对话。
- **私信**：跳过。每条私信消息都会触发机器人，因此会话记录已经完整——没有提及间隙需要填补。
- **自由响应频道**和**机器人自动创建的线程**：出于同样的原因跳过——没有提及门控意味着没有间隙。

每用户会话（`group_sessions_per_user: true`，默认值）也受益：用户的会话缺少其他频道参与者发布的上下文以及用户在标记机器人之前自己的消息。回填填补了这两个间隙。

```yaml
discord:
  history_backfill: true   # 默认
```

关闭方式：

```yaml
discord:
  history_backfill: false
```

> **注意：** 机器人处理*过程中*到达的消息（在触发和响应之间）不会被捕获。这是一个可接受的简化——用户可以重新发送或再次标记。

#### `discord.history_backfill_limit`

**类型：** 整数 — **默认值：** `50`

恢复频道上下文时向后扫描的最大消息数。实际上扫描通常会更早停止——在机器人自己在频道中的最后一条消息处，这是轮次之间的自然边界。此限制是冷启动和长间隙（最近历史中不存在先前机器人消息）的安全上限。

```yaml
discord:
  history_backfill: true
  history_backfill_limit: 50
```

#### `group_sessions_per_user`

**类型：** 布尔值 — **默认值：** `true`

这是一个全局网关设置（非 Discord 专用），控制同一频道中的用户是否获得隔离的会话历史。

为 `true` 时：Alice 和 Bob 在 `#research` 中交谈，各自与 Hermes 有独立的对话。为 `false` 时：整个频道共享一份对话记录和一个正在运行的 agent 槽位。

```yaml
group_sessions_per_user: true
```

有关每种模式的完整含义，请参阅上方的[会话模型](#session-model-in-discord)部分。

#### `display.tool_progress`

**类型：** 字符串 — **默认值：** `"all"` — **可选值：** `off`、`new`、`all`、`verbose`

控制机器人在处理过程中是否在聊天中发送进度消息（例如"正在读取文件……"、"正在运行终端命令……"）。这是适用于所有平台的全局网关设置。

```yaml
display:
  tool_progress: "all"    # off | new | all | verbose
```

- `off` — 不发送进度消息
- `new` — 每次轮次只显示第一个工具调用
- `all` — 显示所有工具调用（在网关消息中截断为 40 个字符）
- `verbose` — 显示完整的工具调用详情（可能产生较长的消息）

#### `display.tool_progress_command`

**类型：** 布尔值 — **默认值：** `false`

启用后，在网关中提供 `/verbose` 斜杠命令，让你无需编辑 config.yaml 即可循环切换工具进度模式（`off → new → all → verbose → off`）。

```yaml
display:
  tool_progress_command: true
```

## 斜杠命令访问控制

默认情况下，每个允许的用户都可以运行每个斜杠命令。要将你的允许列表分为**管理员**（完整斜杠命令访问权限）和**普通用户**（仅你明确启用的命令），请在 Discord 平台的 `extra` 块中添加 `allow_admin_from` 和 `user_allowed_commands`：

```yaml
gateway:
  platforms:
    discord:
      extra:
        # 现有用户允许列表（不变）
        allow_from:
          - "123456789012345678"  # 管理员用户 ID
          - "999888777666555444"  # 普通用户 ID

        # 新增 — 管理员可访问所有斜杠命令（内置 + 插件）
        allow_admin_from:
          - "123456789012345678"

        # 新增 — 非管理员允许用户只能运行这些斜杠命令。
        # /help 和 /whoami 始终允许，以便用户查看其访问权限。
        user_allowed_commands:
          - status
          - model
          - history

        # 可选：为服务器频道设置单独的管理员/命令列表
        group_allow_admin_from:
          - "123456789012345678"
        group_user_allowed_commands:
          - status
```

**行为：**

- 在某个范围（私信或服务器频道）的 `allow_admin_from` 中的用户可以通过实时命令注册表运行**每个**已注册的斜杠命令——内置的和插件注册的都包括。
- 不在 `allow_admin_from` 中的用户只能运行 `user_allowed_commands` 中列出的命令，加上始终允许的基础命令：`/help` 和 `/whoami`。
- 普通聊天（非斜杠消息）不受影响。非管理员用户仍然可以正常与 agent 对话；他们只是无法触发任意命令。
- **向后兼容：** 如果某个范围未设置 `allow_admin_from`，则该范围的斜杠命令门控被禁用。现有安装无需任何更改即可继续工作。
- 私信管理员状态不意味着服务器频道管理员状态。每个范围有自己的管理员列表。

使用 `/whoami` 查看当前范围、你的级别（管理员 / 用户 / 无限制）以及你可以运行的斜杠命令。

## 交互式模型选择器

在 Discord 频道中不带参数发送 `/model` 以打开基于下拉菜单的模型选择器：

1. **提供商选择** — 显示可用提供商的 Select 下拉菜单（最多 25 个）。
2. **模型选择** — 显示所选提供商模型的第二个下拉菜单（最多 25 个）。

选择器在 120 秒后超时。只有授权用户（`DISCORD_ALLOWED_USERS` 中的用户）才能与其交互。如果你知道模型名称，可以直接输入 `/model <名称>`。

## 技能的原生斜杠命令

Hermes 自动将已安装的技能注册为**原生 Discord 应用命令**。这意味着技能会出现在 Discord 的自动补全 `/` 菜单中，与内置命令并列。

- 每个技能成为一个 Discord 斜杠命令（例如 `/code-review`、`/ascii-art`）
- 技能接受一个可选的 `args` 字符串参数
- Discord 每个机器人有 100 个应用命令的限制——如果你的技能数量超过可用槽位，多余的技能会被跳过并在日志中显示警告
- 技能在机器人启动时与内置命令（如 `/model`、`/reset` 和 `/background`）一起注册

无需额外配置——通过 `hermes skills install` 安装的任何技能都会在下次网关重启时自动注册为 Discord 斜杠命令。

### 禁用斜杠命令注册

如果你针对同一个 Discord 应用运行多个 Hermes 网关（例如测试环境 + 生产环境），只有其中一个应该拥有全局斜杠命令注册——否则最后启动的那个会覆盖之前的注册，导致注册状态不稳定。在"从属"网关上关闭斜杠注册：

```yaml
gateway:
  platforms:
    discord:
      extra:
        slash_commands: false   # 默认：true
```

在"主"网关上保持 `true` 可维持正常行为——为内置命令和已安装技能提供全局 `/` 菜单命令。

## 发送媒体（`send_message` + `MEDIA:` 标签）

Discord 适配器通过 `send_message` 工具和 agent 发出的内联 `MEDIA:/path/to/file` 标签，支持所有常见媒体类型的原生文件上传：

| 类型 | 发送方式 |
|---|---|
| 图片（PNG/JPG/WebP） | 原生 Discord 图片附件，带内联预览 |
| 动态 GIF | `send_animation` 以 `animation.gif` 上传，使 Discord 内联播放（而非静态缩略图） |
| 视频（MP4/MOV） | `send_video` — 原生视频播放器 |
| 音频 / 语音 | `send_voice` — 尽可能使用原生语音消息，否则使用文件附件 |
| 文档（PDF/ZIP/docx 等） | `send_document` — 带下载按钮的原生附件 |

Discord 的每次上传大小限制取决于服务器的加成等级（免费 25 MB，最高 500 MB）。如果 Hermes 收到 HTTP 413，适配器会回退到指向本地缓存路径的链接，而不是静默失败。

## 接收任意文件类型

默认情况下，机器人缓存与内置允许列表匹配的上传——图片、音频、视频、PDF、文本/markdown/csv/log、JSON/XML/YAML/TOML、zip、docx/xlsx/pptx。其他任何内容（`.wav`、`.bin`、自定义扩展名的转储文件）都会被记录为 `Unsupported document type` 并在 agent 看到之前被丢弃。

要接受任意文件类型，启用 `discord.allow_any_attachment`：

```yaml
discord:
  allow_any_attachment: true
  # 可选 — 提高/禁用每文件大小上限。默认为 32 MiB。
  # 整个文件在缓存时保存在内存中，因此无限制
  # 上传会带来真实的内存成本。
  max_attachment_bytes: 33554432   # 字节；0 = 无限制
```

启用该标志后，任何上传的文件都会被下载、缓存到 `~/.hermes/cache/documents/` 下，并以 `application/octet-stream` MIME 类型的 `DOCUMENT` 类型消息事件提供给 agent。Agent 收到指向本地路径的上下文说明（通过 `to_agent_visible_cache_path` 为 Docker/Modal 沙盒终端自动转换），可以使用 `terminal`（`ffprobe`、`unzip`、`file`、`strings` 等）或 `read_file` 检查文件。文件内容**不会**内联到 prompt 中——只有路径——因此二进制上传不会撑爆上下文窗口。

已在允许列表中的已知文本格式（`.txt`、`.md`、`.log`）继续自动注入最多 100 KiB 的内容；启用该标志后此行为不变。

等效环境变量：`DISCORD_ALLOW_ANY_ATTACHMENT=true` 和 `DISCORD_MAX_ATTACHMENT_BYTES=33554432`（或 `0` 表示无上限）。

:::warning 无限制的内存成本
禁用大小上限（`max_attachment_bytes: 0`）意味着用户可以向机器人上传数 GB 的文件，网关会尽职地在缓存到磁盘时将其缓冲到内存中。仅在受信任的单用户安装中设置此项。对于共享机器人，保持默认的 32 MiB 或保守地提高上限。
:::

## 交互式提示（clarify）

当 agent 调用 `clarify` 工具时——询问你偏好哪种方式、获取任务后反馈或在非平凡决策前确认——Discord 会以**每个选项一个按钮**的形式渲染问题：

> 我应该为仪表板使用哪个框架？
>
> [1. Next.js] [2. Remix] [3. Astro] [其他（输入答案）]

点击编号按钮作答，或点击**其他**输入自由格式的响应（你在该频道中发送的下一条消息将成为答案）。开放式的 `clarify` 调用（没有预设选项）会跳过按钮，直接捕获你的下一条消息。

按钮在做出选择后会自动禁用，防止重复点击导致重复解析提示。通过 `~/.hermes/config.yaml` 中的 `agent.clarify_timeout` 配置响应超时（默认 `600` 秒）。如果你在超时内没有响应，agent 会以一条哨兵消息解除阻塞并自行调整，而不是一直挂起。

## 主频道

你可以指定一个"主频道"，机器人在此发送主动消息（例如 cron 任务输出、提醒和通知）。有两种设置方式：

### 使用斜杠命令

在机器人所在的任意 Discord 频道中输入 `/sethome`。该频道即成为主频道。

### 手动配置

将以下内容添加到你的 `~/.hermes/.env`：

```bash
DISCORD_HOME_CHANNEL=123456789012345678
DISCORD_HOME_CHANNEL_NAME="#bot-updates"
```

将 ID 替换为实际的频道 ID（开启开发者模式后右键点击 → Copy Channel ID）。

## 语音消息

Hermes Agent 支持 Discord 语音消息：

- **传入语音消息**使用配置的 STT 提供商自动转录：本地 `faster-whisper`（无需密钥）、Groq Whisper（`GROQ_API_KEY`）或 OpenAI Whisper（`VOICE_TOOLS_OPENAI_KEY`）。
- **文字转语音**：使用 `/voice tts` 让机器人在文字回复的同时发送语音音频响应。
- **Discord 语音频道**：Hermes 还可以加入语音频道，聆听用户说话，并在频道中回话。

完整的设置和操作指南，请参阅：
- [语音模式](/user-guide/features/voice-mode)
- [与 Hermes 使用语音模式](/guides/use-voice-mode-with-hermes)

## 论坛频道

Discord 论坛频道（类型 15）不接受直接消息——论坛中的每个帖子都必须是线程。Hermes 自动检测论坛频道，并在需要发送消息时创建新的线程帖子，因此 `send_message`、TTS、图片、语音消息和文件附件都无需 agent 进行特殊处理即可正常工作。

- **线程名称**从消息的第一行派生（去除 markdown 标题前缀，上限 100 个字符）。当消息仅包含附件时，文件名用作备用线程名称。
- **附件**随新线程的起始消息一起发送——无需单独上传步骤，不会出现部分发送。
- **一次调用，一个线程**：每次论坛发送都会创建一个新线程。因此，连续向同一论坛发送消息会产生独立的线程。
- **检测分三层**：首先是频道目录缓存，其次是进程本地探测缓存，最后是实时 `GET /channels/{id}` 探测（其结果在进程生命周期内被记忆化）。

刷新目录（在暴露该功能的平台上使用 `/channels refresh`，或重启网关）会将机器人启动后创建的任何论坛频道填充到缓存中。

## 故障排除

### 机器人在线但不响应消息

**原因**：Message Content Intent 被禁用。

**解决方法**：前往[开发者门户](https://discord.com/developers/applications) → 你的应用 → Bot → Privileged Gateway Intents → 启用 **Message Content Intent** → Save Changes。重启网关。

### 启动时出现"Disallowed Intents"错误

**原因**：你的代码请求了开发者门户中未启用的 intent。

**解决方法**：在 Bot 设置中启用所有三个 Privileged Gateway Intents（Presence、Server Members、Message Content），然后重启。

### 机器人看不到特定频道中的消息

**原因**：机器人的角色没有查看该频道的权限。

**解决方法**：在 Discord 中，前往频道设置 → Permissions → 为机器人的角色添加 **View Channel** 和 **Read Message History** 权限。

### 403 Forbidden 错误

**原因**：机器人缺少所需权限。

**解决方法**：使用第五步中的 URL 以正确权限重新邀请机器人，或在 Server Settings → Roles 中手动调整机器人的角色权限。

### 机器人离线

**原因**：Hermes 网关未运行，或 token 不正确。

**解决方法**：检查 `hermes gateway` 是否正在运行。验证 `.env` 文件中的 `DISCORD_BOT_TOKEN`。如果你最近重置了 token，请更新它。

### "User not allowed" / 机器人忽略你

**原因**：你的用户 ID 不在 `DISCORD_ALLOWED_USERS` 中。

**解决方法**：将你的用户 ID 添加到 `~/.hermes/.env` 中的 `DISCORD_ALLOWED_USERS` 并重启网关。

### 同一频道中的用户意外共享上下文

**原因**：`group_sessions_per_user` 被禁用，或平台无法为该上下文中的消息提供用户 ID。

**解决方法**：在 `~/.hermes/config.yaml` 中进行以下设置并重启网关：

```yaml
group_sessions_per_user: true
```

如果你有意想要共享房间对话，则保持关闭——只需预期会有共享的对话记录历史和共享的中断行为。

## 安全

:::warning
始终设置 `DISCORD_ALLOWED_USERS`（或 `DISCORD_ALLOWED_ROLES`）以限制谁可以与机器人交互。没有任何一项，网关默认拒绝所有用户作为安全措施。只授权你信任的人——授权用户对 agent 的功能拥有完全访问权限，包括工具调用和系统访问。
:::

### 基于角色的访问控制

对于通过角色而非个人用户列表管理访问权限的服务器（管理团队、支持人员、内部工具），使用 `DISCORD_ALLOWED_ROLES`——逗号分隔的角色 ID 列表。拥有其中任一角色的成员即被授权。

```bash
# ~/.hermes/.env — 与 DISCORD_ALLOWED_USERS 配合使用或替代使用
DISCORD_ALLOWED_ROLES=987654321098765432,876543210987654321
```

语义：

- **与用户允许列表为 OR 关系。** 如果用户 ID 在 `DISCORD_ALLOWED_USERS` 中**或**拥有 `DISCORD_ALLOWED_ROLES` 中的任一角色，则该用户被授权。
- **自动启用 Server Members Intent。** 设置 `DISCORD_ALLOWED_ROLES` 后，机器人在连接时启用 Members intent——Discord 需要此 intent 才能在成员记录中发送角色信息。
- **角色 ID，不是名称。** 从 Discord 获取：**用户设置 → 高级 → 开启开发者模式**，然后右键点击任意角色 → **Copy Role ID**。
- **私信回退。** 在私信中，角色检查会扫描共同服务器；在任何共享服务器中拥有允许角色的用户在私信中也被授权。

当管理团队频繁变动时，这是首选模式——新管理员一旦被授予角色即可获得访问权限，无需编辑 `.env` 或重启网关。

### 提及控制

默认情况下，Hermes 会阻止机器人 ping `@everyone`、`@here` 和角色提及，即使其回复中包含这些 token 也不例外。这可防止措辞不当的 prompt 或回显的用户内容向整个服务器发送垃圾消息。个人 `@user` ping 和回复引用 ping（"回复……"小标签）保持启用，以便正常对话仍然有效。

你可以通过环境变量或 `config.yaml` 放宽这些默认值：

```yaml
# ~/.hermes/config.yaml
discord:
  allow_mentions:
    everyone: false      # 允许机器人 ping @everyone / @here
    roles: false         # 允许机器人 ping @role 提及
    users: true          # 允许机器人 ping 个人 @user
    replied_user: true   # 回复消息时 ping 原始作者
```

```bash
# ~/.hermes/.env — 环境变量优先于 config.yaml
DISCORD_ALLOW_MENTION_EVERYONE=false
DISCORD_ALLOW_MENTION_ROLES=false
DISCORD_ALLOW_MENTION_USERS=true
DISCORD_ALLOW_MENTION_REPLIED_USER=true
```

:::tip
除非你确切知道为什么需要，否则将 `everyone` 和 `roles` 保持为 `false`。LLM 很容易在看似正常的响应中生成字符串 `@everyone`；没有此保护，这将通知你服务器的每个成员。
:::

有关保护 Hermes Agent 部署的更多信息，请参阅[安全指南](../security.md)。