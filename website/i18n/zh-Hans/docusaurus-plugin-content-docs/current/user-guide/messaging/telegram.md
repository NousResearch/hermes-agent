---
sidebar_position: 1
title: "Telegram"
description: "将 Hermes Agent 配置为 Telegram 机器人"
---

# Telegram 设置

Hermes Agent 与 Telegram 作为功能完整的对话机器人集成。连接后，您可以从任何设备与您的代理聊天，发送自动转录的语音备忘录，接收定时任务结果，以及在群组聊天中使用代理。该集成基于 [python-telegram-bot](https://python-telegram-bot.org/)，支持文本、语音、图片和文件附件。

## 步骤 1：通过 BotFather 创建机器人

每个 Telegram 机器人都需要一个由 [@BotFather](https://t.me/BotFather)（Telegram 官方机器人管理工具）颁发的 API 令牌。

1. 打开 Telegram 并搜索 **@BotFather**，或访问 [t.me/BotFather](https://t.me/BotFather)
2. 发送 `/newbot`
3. 选择一个**显示名称**（例如"Hermes Agent"）——这可以是任何名称
4. 选择一个**用户名**——这必须是唯一的并以 `bot` 结尾（例如 `my_hermes_bot`）
5. BotFather 会回复您的 **API 令牌**。格式如下：

```
123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
```

:::warning
保持您的机器人令牌机密。拥有此令牌的任何人都可以控制您的机器人。如果泄露，立即通过 BotFather 中的 `/revoke` 撤销它。
:::

## 步骤 2：自定义您的机器人（可选）

这些 BotFather 命令可以改善用户体验。向 @BotFather 发送消息并使用：

| 命令 | 用途 |
|---------|---------|
| `/setdescription` | 用户开始聊天前显示的"此机器人能做什么？"文本 |
| `/setabouttext` | 机器人个人资料页面的简短文本 |
| `/setuserpic` | 为您的机器人上传头像 |
| `/setcommands` | 定义命令菜单（聊天中的 `/` 按钮） |
| `/setprivacy` | 控制机器人是否查看所有群组消息（参见步骤 3） |

:::tip
对于 `/setcommands`，一个有用的起始集合：

```
help - 显示帮助信息
new - 开始新对话
sethome - 将此聊天设为主页频道
```
:::

## 步骤 3：隐私模式（群组关键）

Telegram 机器人有一个**隐私模式**，默认情况下**已启用**。这是使用群组机器人时最常见的困惑来源。

**隐私模式开启时**，您的机器人只能看到：
- 以 `/` 命令开头的消息
- 直接回复机器人自己消息的内容
- 服务消息（成员加入/离开、固定消息等）
- 机器人为管理员的频道中的消息

**隐私模式关闭时**，机器人会接收群组中的每条消息。

### 如何禁用隐私模式

1. 向 **@BotFather** 发送消息
2. 发送 `/mybots`
3. 选择您的机器人
4. 进入 **机器人设置 → 群组隐私 → 关闭**

:::warning
**更改隐私设置后，您必须将机器人从任何群组中移除并重新添加。** Telegram 会在机器人加入群组时缓存隐私状态，直到机器人被移除并重新添加后才会更新。
:::

:::tip
禁用隐私模式的替代方案：将机器人提升为**群组管理员**。管理员机器人无论隐私设置如何都会接收所有消息，这样可以避免需要切换全局隐私模式。
:::

## 步骤 4：查找您的用户 ID

Hermes Agent 使用数字 Telegram 用户 ID 来控制访问。您的用户 ID **不是**您的用户名——它是一个像 `123456789` 这样的数字。

**方法 1（推荐）：** 向 [@userinfobot](https://t.me/userinfobot) 发送消息——它会立即回复您的用户 ID。

**方法 2：** 向 [@get_id_bot](https://t.me/get_id_bot) 发送消息——另一个可靠的选择。

保存这个数字；您将在下一步需要它。

## 步骤 5：配置 Hermes

### 选项 A：交互式设置（推荐）

```bash
hermes gateway setup
```

出现提示时选择 **Telegram**。向导会询问您的机器人令牌和允许的用户 ID，然后为您写入配置。

### 选项 B：手动配置

将以下内容添加到 `~/.hermes/.env`：

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_ALLOWED_USERS=123456789    # 多个用户用逗号分隔
```

### 启动网关

```bash
hermes gateway
```

机器人应在几秒内上线。在 Telegram 上向它发送消息以验证。

## 从 Docker 后端终端发送生成的文件

如果您的终端后端是 `docker`，请记住 Telegram 附件是由**网关进程**发送的，而不是从容器内部。这意味着最终的 `MEDIA:/...` 路径必须在运行网关的主机上可读。

常见陷阱：
- 代理在 Docker 内写入文件到 `/workspace/report.txt`
- 模型输出 `MEDIA:/workspace/report.txt`
- Telegram 传递失败，因为 `/workspace/report.txt` 只存在于容器内，不在主机上

推荐模式：

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/.hermes/cache/documents:/output"
```

然后：
- 在 Docker 内将文件写入 `/output/...`
- 在 `MEDIA:` 中发出**主机可见**路径，例如：
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`

如果您已有 `docker_volumes:` 部分，请将新挂载添加到同一列表中。YAML 重复键会静默覆盖前面的。

### 支持的 `MEDIA:` 文件扩展名

网关从代理回复中提取 `MEDIA:/path/to/file` 标签，并将引用的文件作为平台原生附件发送。所有网关平台支持的扩展名：

| 类别 | 扩展名 |
|---|---|
| 图片 | `png`、`jpg`、`jpeg`、`gif`、`webp`、`bmp`、`tiff`、`svg` |
| 音频 | `mp3`、`wav`、`ogg`、`m4a`、`opus`、`flac`、`aac` |
| 视频 | `mp4`、`mov`、`webm`、`mkv`、`avi` |
| **文档** | `pdf`、`txt`、`md`、`csv`、`json`、`xml`、`html`、`yaml`、`yml`、`log` |
| **办公** | `docx`、`xlsx`、`pptx`、`odt`、`ods`、`odp` |
| **压缩包** | `zip`、`rar`、`7z`、`tar`、`gz`、`bz2` |
| **书籍/包** | `epub`、`apk`、`ipa` |

此列表上的任何内容在支持的平台上（Telegram、Discord、Signal、Slack、WhatsApp、Feishu、Matrix 等）作为原生附件传递；在没有原生支持的平台上，它会回退到链接或纯文本指示器。**加粗的类别**是最近几个版本添加的——如果您之前依赖模型说"这里是文件：/path/to/report.docx"，请切换到 `MEDIA:/path/to/report.docx` 以获得原生传递。

## Webhook 模式

默认情况下，Hermes 使用**长轮询**连接到 Telegram——网关向 Telegram 的服务器发出出站请求以获取新更新。这适用于本地和常驻部署。

对于**云部署**（Fly.io、Railway、Render 等），**Webhook 模式**更具成本效益。这些平台可以在入站 HTTP 流量上自动唤醒暂停的机器，但不能在出站连接上。由于轮询是出站的，轮询机器人永远不会休眠。Webhook 模式反转了方向——Telegram 将更新推送到您机器人的 HTTPS URL，实现空闲时休眠的部署。

| | 轮询（默认） | Webhook |
|---|---|---|
| 方向 | 网关 → Telegram（出站） | Telegram → 网关（入站） |
| 适用于 | 本地、常驻服务器 | 具有自动唤醒的云平台 |
| 设置 | 无需额外配置 | 设置 `TELEGRAM_WEBHOOK_URL` |
| 空闲成本 | 机器必须保持运行 | 机器可以在消息之间休眠 |

### 配置

将以下内容添加到 `~/.hermes/.env`：

```bash
TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
TELEGRAM_WEBHOOK_SECRET="$(openssl rand -hex 32)"  # 必需
# TELEGRAM_WEBHOOK_PORT=8443        # 可选，默认 8443
```

| 变量 | 必需 | 描述 |
|----------|----------|-------------|
| `TELEGRAM_WEBHOOK_URL` | 是 | Telegram 将发送更新的公共 HTTPS URL。URL 路径自动提取（例如从上例中提取 `/telegram`）。 |
| `TELEGRAM_WEBHOOK_SECRET` | **是**（当设置 `TELEGRAM_WEBHOOK_URL` 时） | Telegram 在每个 webhook 请求中回显的密钥，用于验证。没有它网关拒绝启动——参见 [GHSA-3vpc-7q5r-276h](https://github.com/NousResearch/hermes-agent/security/advisories/GHSA-3vpc-7q5r-276h)。使用 `openssl rand -hex 32` 生成。 |
| `TELEGRAM_WEBHOOK_PORT` | 否 | Webhook 服务器监听的本地端口（默认：`8443`）。 |

当设置 `TELEGRAM_WEBHOOK_URL` 时，网关启动 HTTP webhook 服务器而不是轮询。未设置时，使用轮询模式——与先前版本没有行为变化。

### 云部署示例（Fly.io）

1. 将环境变量添加到您的 Fly.io 应用密钥：

```bash
fly secrets set TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
fly secrets set TELEGRAM_WEBHOOK_SECRET=$(openssl rand -hex 32)
```

2. 在您的 `fly.toml` 中暴露 webhook 端口：

```toml
[[services]]
  internal_port = 8443
  protocol = "tcp"

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
```

3. 部署：

```bash
fly deploy
```

网关日志应显示：`[telegram] Connected to Telegram (webhook mode)`。

## 代理支持

如果 Telegram 的 API 被阻止或您需要通过代理路由流量，请设置 Telegram 专用的代理 URL。这优先于通用 `HTTPS_PROXY` / `HTTP_PROXY` 环境变量。

**选项 1：config.yaml（推荐）**

```yaml
telegram:
  proxy_url: "socks5://127.0.0.1:1080"
```

**选项 2：环境变量**

```bash
TELEGRAM_PROXY=socks5://127.0.0.1:1080
```

支持的协议：`http://`、`https://`、`socks5://`。

代理同时适用于主 Telegram 连接和备用 IP 传输。如果没有设置 Telegram 专用代理，网关会回退到 `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY`（或 macOS 系统代理自动检测）。

## 主页频道

在任何 Telegram 聊天（私信或群组）中使用 `/sethome` 命令将其指定为**主页频道**。定时任务（cron 作业）会将其结果传递到此频道。

您也可以在 `~/.hermes/.env` 中手动设置：

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="My Notes"
```

:::tip
群组聊天 ID 是负数（例如 `-1001234567890`）。您的个人私信聊天 ID 与您的用户 ID 相同。
:::

## 语音消息

### 接收语音（语音转文字）

您在 Telegram 上发送的语音消息会自动由 Hermes 配置的 STT 提供商转录，并作为文本注入对话中。

- `local` 在运行 Hermes 的机器上使用 `faster-whisper`——无需 API 密钥
- `groq` 使用 Groq Whisper，需要 `GROQ_API_KEY`
- `openai` 使用 OpenAI Whisper，需要 `VOICE_TOOLS_OPENAI_KEY`

### 发送语音（文字转语音）

当代理通过 TTS 生成音频时，它作为原生 Telegram **语音气泡**传递——圆形的、内联可播放的那种。

- **OpenAI 和 ElevenLabs** 原生生成 Opus——无需额外设置
- **Edge TTS**（默认免费提供商）输出 MP3，需要 **ffmpeg** 转换为 Opus：

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

没有 ffmpeg，Edge TTS 音频会作为常规音频文件发送（仍然可播放，但使用矩形播放器而不是语音气泡）。

在 `config.yaml` 中的 `tts.provider` 键下配置 TTS 提供商。

## 群组聊天使用

Hermes Agent 可以在 Telegram 群组聊天中使用，需要注意以下几点：

- **隐私模式**决定机器人可以看到哪些消息（参见[步骤 3](#步骤-3-隐私模式群组关键)）
- `TELEGRAM_ALLOWED_USERS` 仍然适用——即使在群组中，也只有授权用户可以触发机器人
- 您可以使用 `telegram.require_mention: true` 让机器人不响应普通群组闲聊
- 使用 `telegram.require_mention: true` 时，群组消息在以下情况下会被接受：
  - 回复机器人的消息之一
  - `@botusername` 提及
  - `/command@botusername`（Telegram 的包含机器人名称的机器人菜单命令形式）
  - 匹配您在 `telegram.mention_patterns` 中配置的正则表达式唤醒词之一
- 使用 `telegram.ignored_threads` 让 Hermes 在特定的 Telegram 论坛主题中保持沉默，即使该群组原本允许自由响应或提及触发回复
- 如果 `telegram.require_mention` 未设置或为 false，Hermes 保持之前的开放群组行为，响应它能看到的所有普通群组消息

### 故障排除：私信正常但群组不行

如果机器人在私聊中响应但在群组中保持沉默，请按顺序检查这些门槛：

1. **Telegram 传递：**关闭 BotFather 隐私模式，将机器人提升为管理员，或直接提及机器人。Hermes 无法响应从未传递到机器人的群组消息。
2. **更改隐私后重新加入：**更改 BotFather 隐私设置后将机器人从群组中移除并重新添加。Telegram 可能会保留现有成员资格的旧传递行为。
3. **Hermes 授权：**确保发送者在 `TELEGRAM_ALLOWED_USERS` 或 `TELEGRAM_GROUP_ALLOWED_USERS` 中，或使用 `TELEGRAM_GROUP_ALLOWED_CHATS` 允许群组聊天。
4. **提及过滤器：**如果设置了 `telegram.require_mention: true`，普通群组闲聊会被忽略，除非消息是斜杠命令、回复机器人、`@botusername` 提及或配置的 `mention_patterns` 匹配。

Telegram 群组和超级群的负聊天 ID 是正常的。如果您使用聊天范围的授权，请将这些 ID 放在 `TELEGRAM_GROUP_ALLOWED_CHATS` 中，而不是发送者用户白名单中。

### 群组触发配置示例

将此添加到 `~/.hermes/config.yaml`：

```yaml
telegram:
  require_mention: true
  mention_patterns:
    - "^\\s*chompy\\b"
  ignored_threads:
    - 31
    - "42"
```

此示例允许所有常规直接触发以及以 `chompy` 开头的消息，即使它们不使用 `@` 提及。
Telegram 主题 `31` 和 `42` 中的消息总是在提及和自由响应检查之前被忽略。

### 关于 `mention_patterns` 的注意事项

- 模式使用 Python 正则表达式
- 匹配不区分大小写
- 模式同时针对文本消息和媒体标题进行检查
- 无效的正则表达式模式会被忽略，并在网关日志中发出警告，而不是让机器人崩溃
- 如果您希望模式仅在消息开头匹配，请用 `^` 锚定它

## 私信主题（Bot API 9.4）

Telegram Bot API 9.4（2026 年 2 月）引入了**私信主题**——机器人可以直接在 1 对 1 私信聊天中创建论坛风格的主题线程，无需超级群组。这允许您在现有的与 Hermes 的私信中运行多个隔离的工作区。

### 使用场景

如果您处理多个长期运行的项目，主题可以保持它们的上下文分开：

- **"网站"主题** — 处理您的生产 Web 服务
- **"研究"主题** — 文献综述和论文探索
- **"综合"主题** — 杂项任务和快速问题

每个主题都有自己独立的对话会话、历史记录和上下文——完全与其他主题隔离。

### 配置

:::caution 前置条件
在将主题添加到配置之前，用户必须在与机器人的私信聊天中**启用主题模式**：

1. 在 Telegram 中打开与 Hermes 机器人的私信聊天
2. 点击顶部的机器人名称打开聊天信息
3. 启用**主题**（将聊天转换为论坛的开关）

没有这个，Hermes 会在启动时记录 `The chat is not a forum` 并跳过主题创建。这是一个 Telegram 客户端设置——机器人无法以编程方式启用它。
:::

在 `~/.hermes/config.yaml` 中的 `platforms.telegram.extra.dm_topics` 下添加主题：

```yaml
platforms:
  telegram:
    extra:
      dm_topics:
      - chat_id: 123456789        # 您的 Telegram 用户 ID
        topics:
        - name: 综合
          icon_color: 7322096
        - name: 网站
          icon_color: 9367192
        - name: 研究
          icon_color: 16766590
          skill: arxiv              # 在此主题中自动加载技能
```

**字段：**

| 字段 | 必需 | 描述 |
|-------|----------|-------------|
| `name` | 是 | 主题显示名称 |
| `icon_color` | 否 | Telegram 图标颜色代码（整数） |
| `icon_custom_emoji_id` | 否 | 主题图标的自定义表情符号 ID |
| `skill` | 否 | 在此主题的新会话中自动加载的技能 |
| `thread_id` | 否 | 主题创建后自动填充——不要手动设置 |

### 工作原理

1. 在网关启动时，Hermes 为每个还没有 `thread_id` 的主题调用 `createForumTopic`
2. `thread_id` 自动保存回 `config.yaml`——后续重启跳过 API 调用
3. 每个主题映射到一个隔离的会话键：`agent:main:telegram:dm:{chat_id}:{thread_id}`
4. 每个主题中的消息都有自己独立的对话历史、内存刷新和上下文窗口

### 技能绑定

具有 `skill` 字段的主题在该主题中新会话开始时自动加载该技能。这与在对话开始时输入 `/skill-name` 的效果完全相同——技能内容被注入第一条消息，随后的消息在对话历史中看到它。

例如，具有 `skill: arxiv` 的主题将在其会话重置时（由于空闲超时、每日重置或手动 `/reset`）预加载 arxiv 技能。

:::tip
在配置之外创建的主题（例如通过手动调用 Telegram API）在 `forum_topic_created` 服务消息到达时会被自动发现。您也可以在网关运行时将主题添加到配置——它们会在下一次缓存未命中时被拾取。
:::

## 多会话私信模式（`/topic`）

ChatGPT 风格的多会话私信——一个机器人，多个并行对话。与上面的运营商策划的 `extra.dm_topics` 不同，此模式是**用户驱动**的：无需配置、无需预声明主题名称。终端用户通过 `/topic` 开启它，然后点击 Telegram **+** 按钮创建任意数量的主题，每个主题都是一个完全独立的 Hermes 会话。

### `/topic` 子命令

| 形式 | 上下文 | 效果 |
|------|---------|--------|
| `/topic` | 根私信，尚未启用 | 检查 BotFather 功能，启用多会话模式，创建固定系统主题 |
| `/topic` | 根私信，已启用 | 显示状态：可供恢复的未链接会话 |
| `/topic` | 在主题内 | 显示当前主题的会话绑定 |
| `/topic help` | 任意位置 | 内联使用说明 |
| `/topic off` | 根私信 | 禁用多会话模式并清除此聊天的所有主题绑定 |
| `/topic <session-id>` | 在主题内 | 将先前的 Telegram 会话恢复到当前主题 |

只有授权用户（通过 `TELEGRAM_ALLOWED_USERS` / 平台 auth 配置的白名单）可以运行 `/topic`。未经授权的发送者会收到拒绝而不是激活。

### 私信主题 vs 多会话私信模式

| | `extra.dm_topics`（配置驱动） | `/topic`（用户驱动） |
|---|---|---|
| 谁激活 | 运营商，在 `config.yaml` 中 | 终端用户，通过发送 `/topic` |
| 主题列表 | 在配置中声明的固定集合 | 用户自由创建/删除主题 |
| 主题名称 | 由运营商选择 | 由用户选择；自动重命名以匹配 Hermes 会话标题 |
| 根私信行为 | 不变——正常聊天 | 成为系统大厅（拒绝非命令消息） |
| 主要用例 | 带有可选技能绑定的永久工作区 | 临时并行会话 |
| 持久性 | 配置中的 `extra.dm_topics` | `telegram_dm_topic_mode` + `telegram_dm_topic_bindings` SQLite 表 |

这两个功能可以在同一机器人上共存——您可以从用户的私信运行 `/topic`，而 `extra.dm_topics` 继续为其他聊天管理运营商声明的主题。

### 前置条件

在 **@BotFather** 中，打开您的机器人 → **机器人设置 → 线程设置**：

1. 开启**线程模式**（启用 `has_topics_enabled`）
2. **不要**禁止用户创建主题（保持 `allows_users_to_create_topics` 开启）

当用户首次运行 `/topic` 时，Hermes 调用 `getMe` 来验证这两个标志。如果任一标志关闭，Hermes 会发送 BotFather 线程设置页面的截图并解释要切换什么——在满足前置条件之前不会发生激活。

### 激活流程

从根私信发送：

```
/topic
```

Hermes 将：

1. 检查 `getMe().has_topics_enabled` 和 `allows_users_to_create_topics`
2. 如果两者都为真，为此私信启用多会话主题模式
3. 创建并固定一个**系统**主题用于状态/命令（尽力而为）
4. 回复用户可以恢复的先前未链接 Telegram 会话列表

激活后，**根私信是一个大厅**：普通提示会被拒绝，并显示指向**所有消息**的指导。系统命令（`/status`、`/sessions`、`/usage`、`/help` 等）仍在根目录中工作。

### 创建新主题（终端用户流程）

1. 在 Telegram 中打开机器人私信
2. 点击机器人界面顶部的**所有消息**，然后发送任何消息
3. Telegram 为该消息创建一个新主题
4. Hermes 在该主题中响应——该主题现在是一个独立会话

每个主题都有自己独立的对话历史、模型状态、工具执行和会话 ID。隔离键是 `agent:main:telegram:dm:{chat_id}:{thread_id}`——与配置驱动的私信主题隔离相同。

### 自动重命名主题

当 Hermes 为主题生成会话标题时（通过自动标题管道，在第一次交换之后），Telegram 主题本身会被重命名以匹配——例如"新主题"变成"数据库迁移计划"。重命名是尽力而为的：失败会被记录但不会破坏会话。

### 主题内的 `/new`

重置当前主题的会话（新会话 ID、全新历史）而不影响其他主题。Hermes 回复一条提醒，指出对于并行工作，创建另一个主题（通过**所有消息**）通常才是您想要的。

### 恢复先前的会话

在主题内发送：

```
/topic <session-id>
```

这会将当前主题绑定到现有 Hermes 会话而不是重新开始。可用于继续在启用主题模式之前开始的对话。限制：

- 目标会话必须属于同一 Telegram 用户
- 目标会话不能已经绑定到另一个主题

Hermes 用会话标题确认并重放最后的助手消息以提供上下文。

要发现会话 ID，在根私信中发送 `/topic`（无参数）——Hermes 列出用户的未链接 Telegram 会话。

### 主题内的 `/topic`（无参数）

显示当前主题的绑定：会话标题、会话 ID，以及关于 `/new` 与创建另一个主题的提示。

### 底层实现

- 激活持久化到 `state.db` 中的 `telegram_dm_topic_mode(chat_id, user_id, enabled, ...)`
- 每个主题绑定持久化到 `telegram_dm_topic_bindings(chat_id, thread_id, session_id, ...)`，其中 `session_id` 上有 `ON DELETE CASCADE`——修剪会话会自动清除其主题绑定
- 主题模式 SQLite 迁移是**可选加入**的：它在第一次 `/topic` 调用时运行，从不在网关启动时运行。在用户在此配置文件中运行 `/topic` 之前，`state.db` 保持不变
- 每个入站私信消息查找其 `(chat_id, thread_id)` 绑定。如果存在，查找通过 `SessionStore.switch_session()` 将消息路由到绑定会话，以便会话键到会话 ID 的映射在磁盘上保持一致
- 主题内的 `/new` 重写绑定行以指向新会话 ID，因此下一条消息保持在全新会话上
- 在 `extra.dm_topics` 中声明的主题**永远不会自动重命名**——即使启用了多会话模式，运营商选择的名称也会保留
- 在支持论坛的私信中，常规（固定在顶部）主题被视为根大厅，无论 Telegram 是否使用 `message_thread_id=1` 或没有 thread_id 传递其消息
- 根大厅提醒被限制为每 30 秒每个聊天一条消息——忘记开启主题模式的用户在根目录输入十个提示不会收到十个回复
- BotFather 设置截图被限制为每 5 分钟每个聊天发送一条——在禁用线程设置时重复 `/topic` 尝试不会重新上传同一张图片
- 在主题内开始的 `/background <prompt>` 将其结果传递回同一主题；后台会话不会触发所属主题的自动重命名
- `/topic` 本身受机器人用户授权检查限制——未经授权的私信收到拒绝而不是激活

### 禁用多会话模式

在根私信中发送 `/topic off`。Hermes 关闭该行，清除聊天的 `(thread_id → session_id)` 绑定，根私信恢复为正常 Hermes 聊天。Telegram 中的现有主题不会被删除——它们只是停止作为独立会话被控制。稍后重新运行 `/topic` 可以重新开启。

如果您需要手动清理（例如跨多个聊天进行批量重置），请直接删除行：

```bash
sqlite3 ~/.hermes/state.db \
  "UPDATE telegram_dm_topic_mode SET enabled = 0 WHERE chat_id = '<your_chat_id>'; \
   DELETE FROM telegram_dm_topic_bindings WHERE chat_id = '<your_chat_id>';"
```

### 降级 Hermes

如果您降级到早于 `/topic` 的 Hermes 版本，该功能只是停止工作——`telegram_dm_topic_mode` 和 `telegram_dm_topic_bindings` 表保留在 `state.db` 中但被旧代码忽略。私信恢复为原生线程隔离（每个 `message_thread_id` 仍然通过 `build_session_key` 获得自己的会话），因此您现有的 Telegram 主题继续保持作为并行会话工作。根私信不再是大厅——消息像以前一样进入代理。重新升级会重新激活多会话模式，恢复到之前的状态。

## 群组论坛主题技能绑定

启用了**主题模式**的超级群组（也称为"论坛主题"）已经获得每个主题的会话隔离——每个 `thread_id` 映射到自己的对话。但您可能希望在消息到达特定群组主题时**自动加载技能**，就像私信主题技能绑定的工作方式一样。

### 使用场景

一个带有不同工作流论坛主题的团队超级群组：

- **工程**主题 → 自动加载 `software-development` 技能
- **研究**主题 → 自动加载 `arxiv` 技能
- **综合**主题 → 无技能，通用助手

### 配置

在 `~/.hermes/config.yaml` 中的 `platforms.telegram.extra.group_topics` 下添加主题绑定：

```yaml
platforms:
  telegram:
    extra:
      group_topics:
      - chat_id: -1001234567890       # 超级群组 ID
        topics:
        - name: 工程
          thread_id: 5
          skill: software-development
        - name: 研究
          thread_id: 12
          skill: arxiv
        - name: 综合
          thread_id: 1
          # 无技能——通用目的
```

**字段：**

| 字段 | 必需 | 描述 |
|-------|----------|-------------|
| `chat_id` | 是 | 超级群组的数字 ID（以 `-100` 开头的负数） |
| `name` | 否 | 人类可读的主题标签（仅供信息参考） |
| `thread_id` | 是 | Telegram 论坛主题 ID——在 `t.me/c/<group_id>/<thread_id>` 链接中可见 |
| `skill` | 否 | 在此主题新会话中自动加载的技能 |

### 工作原理

1. 当消息到达映射的群组主题时，Hermes 在 `group_topics` 配置中查找 `chat_id` 和 `thread_id`
2. 如果匹配条目具有 `skill` 字段，则该技能为会话自动加载——与私信主题技能绑定相同
3. 没有 `skill` 键的主题仅获得会话隔离（现有行为，不变）
4. 未映射的 `thread_id` 或 `chat_id` 值会静默通过——无错误，无技能

### 与私信主题的区别

| | 私信主题 | 群组主题 |
|---|---|---|
| 配置键 | `extra.dm_topics` | `extra.group_topics` |
| 主题创建 | 如果缺少 `thread_id`，Hermes 通过 API 创建主题 | 管理员在 Telegram UI 中创建主题 |
| `thread_id` | 创建后自动填充 | 必须手动设置 |
| `icon_color` / `icon_custom_emoji_id` | 支持 | 不适用（管理员控制外观） |
| 技能绑定 | ✓ | ✓ |
| 会话隔离 | ✓ | ✓（论坛主题已内置） |

:::tip
要查找主题的 `thread_id`，在 Telegram Web 或桌面版中打开该主题并查看 URL：`https://t.me/c/1234567890/5` — 最后一个数字（`5`）是 `thread_id`。超级群组的 `chat_id` 是带 `-100` 前缀的群组 ID（例如群组 `1234567890` 变成 `-1001234567890`）。
:::

## 最近的 Bot API 功能

- **Bot API 9.4（2026 年 2 月）：** 私信主题 — 机器人可以通过 `createForumTopic` 在 1 对 1 私信聊天中创建论坛主题。Hermes 将此用于两个不同的功能：运营商策划的[私信主题](#私信主题-bot-api-94)（配置驱动，固定主题列表）和用户驱动的[多会话私信模式](#多会话私信模式-topic)（通过 `/topic` 激活，无限用户创建主题）。
- **隐私政策：** Telegram 现在要求机器人有隐私政策。通过 BotFather 设置一个，使用 `/setprivacy_policy`，否则 Telegram 可能会自动生成占位符。如果您的机器人面向公众，这尤为重要。
- **消息流式传输：** Bot API 9.x 添加了对流式传输长回复的支持，这可以改善冗长代理回复的感知延迟。

## 渲染：表格和链接预览

Telegram 的 MarkdownV2 没有原生表格语法——管道表格如果通过原始方式传递会呈现为反斜杠转义的乱码。Hermes 自动规范化 markdown 表格：

- **小表格**被展平为**行组项目符号** — 每行成为列标题下的可读项目符号列表。适用于 2-4 列和短单元格。
- **较大或较宽的表格**回退到**带栅格的代码块**，对齐列以防止任何折叠。添加一行提示，以便代理知道在 Telegram 上更喜欢散文后续而不是更多表格。

无需配置——适配器根据每条消息选择正确的回退。如果您想要旧的"始终代码块"行为，请在 `config.yaml` 中设置 `telegram.pretty_tables: false`（默认：`true`）。

**链接预览。** Telegram 自动为机器人消息中的 URL 生成链接预览。如果您宁愿抑制这些（长的 `/tools` 输出、提及十个链接的代理回复等）：

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_link_previews: true
```

启用后，Hermes 将 Telegram 的 `LinkPreviewOptions(is_disabled=True)` 附加到每条出站消息，并在较旧的 `python-telegram-bot` 版本上回退到旧版 `disable_web_page_preview` 参数。

## 群组白名单

Telegram 群组和论坛聊天有两个正交门槛可以配置：

- **发送者用户 ID**（`group_allow_from` / `TELEGRAM_GROUP_ALLOWED_USERS`） — 仅适用于群组/论坛消息的发送者范围白名单。当您希望特定用户能够在群组中调用机器人而不将他们添加到 `TELEGRAM_ALLOWED_USERS`（这也会给他们私信访问权限）时使用此功能。
- **聊天 ID**（`group_allowed_chats` / `TELEGRAM_GROUP_ALLOWED_CHATS`） — 聊天范围白名单。这些群组/论坛的任何成员都可以与机器人交互。适用于团队/支持机器人，其中群组成员资格本身就是访问信号。

```yaml
gateway:
  platforms:
    telegram:
      extra:
        # 全局访问（私信 + 群组）。这里的用户始终可以调用机器人。
        allow_from:
          - "123456789"
        # 允许在群组/论坛中发送的发送者 ID。不授予私信访问权限。
        group_allow_from:
          - "987654321"
        # 整个群组/论坛 — 任何成员都是授权的。
        group_allowed_chats:
          - "-1001234567890"
```

等效的环境变量：

```bash
TELEGRAM_ALLOWED_USERS="123456789"
TELEGRAM_GROUP_ALLOWED_USERS="987654321"
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

行为：

- `TELEGRAM_ALLOWED_USERS` 涵盖所有聊天类型（私信、群组、论坛）。
- `TELEGRAM_GROUP_ALLOWED_USERS` 仅在群组/论坛中授权列出的发送者。除非列在 `TELEGRAM_ALLOWED_USERS` 中，否则他们仍然无法私信机器人。
- `TELEGRAM_GROUP_ALLOWED_CHATS` 中的聊天授权该聊天的每个成员，无论发送者如何。
- 在任何一个中使用 `*` 以允许任何发送者/聊天。
- 这叠加在现有的提及/模式触发器和 `group_topics` + `ignored_threads` 之上。

### 从 PR #17686 之前迁移

在此拆分之前，`TELEGRAM_GROUP_ALLOWED_USERS` 是唯一的旋钮，用户将**聊天 ID** 放入其中。为了向后兼容，`TELEGRAM_GROUP_ALLOWED_USERS` 中以 `-` 开头的聊天 ID 形状的值仍然被认可为聊天 ID，并记录一次弃用警告。迁移：

```bash
# 旧的（仍然有效，但已弃用）
TELEGRAM_GROUP_ALLOWED_USERS="-1001234567890"

# 新的
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

## 交互式模型选择器

当您在 Telegram 聊天中发送不带参数的 `/model` 时，Hermes 显示一个用于切换模型的交互式内联键盘：

1. **提供商选择** — 显示每个可用提供商及其模型计数的按钮（例如"OpenAI (15)"，"✓ Anthropic (12)"表示当前提供商）。
2. **模型选择** — 带 **上一页**/**下一页** 导航的分页模型列表，一个**返回**按钮返回提供商，以及**取消**。

当前模型和提供商显示在顶部。所有导航都通过在同一消息中编辑来进行（不会弄乱聊天）。

:::tip
如果您知道确切模型名称，直接输入 `/model <name>` 跳过选择器。您也可以输入 `/model <name> --global` 以跨会话保持更改。
:::

## DNS-over-HTTPS 备用 IP

在某些受限网络中，`api.telegram.org` 可能解析为无法访问的 IP。Telegram 适配器包括一个**备用 IP** 机制，在保持正确的 TLS 主机名和 SNI 的同时，透明地重试针对备用 IP 的连接。

### 工作原理

1. 如果设置了 `TELEGRAM_FALLBACK_IPS`，则直接使用这些 IP。
2. 否则，适配器自动通过 **Google DNS** 和 **Cloudflare DNS** 查询 DNS-over-HTTPS (DoH) 以发现 `api.telegram.org` 的备用 IP。
3. DoH 返回的与系统 DNS 结果不同的 IP 被用作备用。
4. 如果 DoH 也被阻止，则使用硬编码的种子 IP（`149.154.167.220`）作为最后手段。
5. 一旦备用 IP 成功，它就变得"粘性" — 后续请求直接使用它，而不是先重试主路径。

### 配置

```bash
# 显式备用 IP（逗号分隔）
TELEGRAM_FALLBACK_IPS=149.154.167.220,149.154.167.221
```

或在 `~/.hermes/config.yaml` 中：

```yaml
platforms:
  telegram:
    extra:
      fallback_ips:
        - "149.154.167.220"
```

:::tip
您通常不需要手动配置这个。自动发现通过 DoH 处理大多数受限网络场景。只有在 DoH 在您的网络上也被阻止时，才需要 `TELEGRAM_FALLBACK_IPS` 环境变量。
:::

## 代理支持

如果您的网络需要 HTTP 代理才能访问互联网（企业环境中常见），Telegram 适配器自动读取标准代理环境变量并通过代理路由所有连接。

### 支持的变量

适配器按顺序检查这些环境变量，使用第一个设置的：

1. `HTTPS_PROXY`
2. `HTTP_PROXY`
3. `ALL_PROXY`
4. `https_proxy` / `http_proxy` / `all_proxy`（小写变体）

### 配置

在启动网关之前在您的环境中设置代理：

```bash
export HTTPS_PROXY=http://proxy.example.com:8080
hermes gateway
```

或将其添加到 `~/.hermes/.env`：

```bash
HTTPS_PROXY=http://proxy.example.com:8080
```

代理同时适用于主传输和所有备用 IP 传输。无需额外的 Hermes 配置 — 如果设置了环境变量，它会自动使用。

:::note
这涵盖了 Hermes 用于 Telegram 连接的自定义备用传输层。其他地方使用的标准 `httpx` 客户端已经原生尊重代理环境变量。
:::

## 消息反应

机器人可以向消息添加表情符号反应作为视觉处理反馈：

- 👀 当机器人开始处理您的消息时
- ✅ 当响应成功传递时
- ❌ 如果处理过程中发生错误

反应**默认禁用**。在 `config.yaml` 中启用：

```yaml
telegram:
  reactions: true
```

或通过环境变量：

```bash
TELEGRAM_REACTIONS=true
```

:::note
与 Discord（反应是附加的）不同，Telegram 的 Bot API 在一次调用中替换所有机器人反应。从 👀 到 ✅/❌ 的转换是原子性的 — 您不会同时看到两者。
:::

:::tip
如果机器人没有在群组中添加反应的权限，反应调用会静默失败，消息处理继续正常进行。
:::

## 每频道提示

为特定的 Telegram 群组或论坛主题分配临时系统提示。提示在每个回合运行时注入 — 永不持久化到记录历史中，因此更改会立即生效。

```yaml
telegram:
  channel_prompts:
    "-1001234567890": |
      你是一个研究助理。专注于学术来源，
      引用和简洁的综合。
    "42":  |
      此主题用于创意写作反馈。要温暖且
      具有建设性。
```

键是聊天 ID（群组/超级群组）或论坛主题 ID。对于论坛群组，主题级提示覆盖群组级提示：

- 在群组 `-1001234567890` 中的主题 `42` 内的消息 → 使用主题 `42` 的提示
- 在主题 `99`（无明确条目）中 → 回退到群组 `-1001234567890` 的提示
- 在没有条目的群组中的消息 → 不应用频道提示

数字 YAML 键自动规范化为字符串。

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| 机器人完全不响应 | 验证 `TELEGRAM_BOT_TOKEN` 正确。检查 `hermes gateway` 日志是否有错误。 |
| 机器人返回"未授权" | 您的用户 ID 不在 `TELEGRAM_ALLOWED_USERS` 中。使用 @userinfobot 仔细检查。 |
| 机器人忽略群组消息 | 隐私模式可能已开启。禁用它（步骤 3）或将机器人设为群组管理员。**更改隐私后记得移除并重新添加机器人。** |
| 语音消息未转录 | 验证 STT 可用：安装 `faster-whisper` 用于本地转录，或在 `~/.hermes/.env` 中设置 `GROQ_API_KEY` / `VOICE_TOOLS_OPENAI_KEY`。 |
| 语音回复是文件而不是气泡 | 安装 `ffmpeg`（Edge TTS Opus 转换需要）。 |
| 机器人令牌被撤销/无效 | 通过 BotFather 中的 `/revoke` 然后 `/newbot` 或 `/token` 生成新令牌。更新您的 `.env` 文件。 |
| Webhook 未收到更新 | 验证 `TELEGRAM_WEBHOOK_URL` 公开可访问（用 `curl` 测试）。确保您的平台/反向代理将来自 URL 端口的入站 HTTPS 流量路由到 `TELEGRAM_WEBHOOK_PORT` 配置的本地监听端口（它们不需要是相同的数字）。确保 SSL/TLS 处于活动状态 — Telegram 只发送到 HTTPS URL。检查防火墙规则。 |

## 执行审批

当代理尝试运行潜在危险的命令时，它会在聊天中请求您的批准：

> ⚠️ 此命令可能危险（递归删除）。回复"是"以批准。

回复"yes"/"y"以批准或"no"/"n"以拒绝。

## 安全性

:::warning
始终设置 `TELEGRAM_ALLOWED_USERS` 以限制谁可以与您的机器人交互。没有它，网关默认拒绝所有用户作为安全措施。
:::

切勿公开分享您的机器人令牌。如果泄露，立即通过 BotFather 的 `/revoke` 命令撤销它。

有关更多详细信息，请参阅[安全性文档](/user-guide/security)。您也可以使用 [DM 配对](/user-guide/messaging#dm-pairing-alternative-to-allowlists) 来获得更动态的用户授权方法。
