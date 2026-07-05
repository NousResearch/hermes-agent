---
sidebar_position: 1
title: "Telegram"
description: "将 Hermes Agent 设置为 Telegram 机器人"
---

# Telegram 设置

Hermes Agent 与 Telegram 集成，作为功能完整的对话机器人。连接后，你可以从任何设备与 Agent 聊天、发送自动转录的语音备忘录、接收定时任务结果，并在群聊中使用 Agent。该集成基于 [python-telegram-bot](https://python-telegram-bot.org/) 构建，支持文本、语音、图片和文件附件。

## 第一步：通过 BotFather 创建机器人

每个 Telegram 机器人都需要由 [@BotFather](https://t.me/BotFather)（Telegram 官方机器人管理工具）颁发的 API token（令牌）。

1. 打开 Telegram，搜索 **@BotFather**，或访问 [t.me/BotFather](https://t.me/BotFather)
2. 发送 `/newbot`
3. 选择一个**显示名称**（例如 "Hermes Agent"）——可以是任意名称
4. 选择一个**用户名**——必须唯一且以 `bot` 结尾（例如 `my_hermes_bot`）
5. BotFather 会回复你的 **API token**，格式如下：

```
123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
```

:::warning
请妥善保管你的机器人 token。任何持有该 token 的人都可以控制你的机器人。如果泄露，请立即通过 BotFather 的 `/revoke` 命令撤销。
:::

## 第二步：自定义机器人（可选）

以下 BotFather 命令可改善用户体验。向 @BotFather 发送：

| 命令 | 用途 |
|---------|---------|
| `/setdescription` | 用户开始聊天前显示的"这个机器人能做什么？"文本 |
| `/setabouttext` | 机器人个人资料页面上的简短文字 |
| `/setuserpic` | 为机器人上传头像 |
| `/setcommands` | 定义命令菜单（聊天中的 `/` 按钮） |
| `/setprivacy` | 控制机器人是否能看到所有群消息（见第三步） |

:::tip
对于 `/setcommands`，一个实用的初始命令集：

```
help - Show help information
new - Start a new conversation
sethome - Set this chat as the home channel
```
:::

## 第三步：隐私模式（群组关键设置）

Telegram 机器人有一个**隐私模式**，**默认启用**。这是在群组中使用机器人时最常见的困惑来源。

**隐私模式开启时**，机器人只能看到：
- 以 `/` 命令开头的消息
- 直接回复机器人自身消息的内容
- 服务消息（成员加入/离开、置顶消息等）
- 机器人是管理员的频道中的消息

**隐私模式关闭时**，机器人接收群组中的每条消息。

### 如何关闭隐私模式

1. 向 **@BotFather** 发送消息
2. 发送 `/mybots`
3. 选择你的机器人
4. 进入 **Bot Settings → Group Privacy → Turn off**

:::warning
**更改隐私设置后，必须将机器人从所有群组中移除并重新添加。** Telegram 在机器人加入群组时会缓存隐私状态，在机器人被移除并重新添加之前不会更新。
:::

:::tip
禁用隐私模式的替代方案：将机器人提升为**群组管理员**。管理员机器人无论隐私设置如何都能接收所有消息，这样就无需切换全局隐私模式。
:::

### 观察群组消息但不自动回复

对于 OpenClaw/Yuanbao 风格的群组行为，可配置 Telegram 使机器人能**看到**普通群组消息，但只在被直接触发时**响应**：

```yaml
telegram:
  allowed_chats:
    - "-1001234567890"
  group_allowed_chats:
    - "-1001234567890"
  require_mention: true
  observe_unmentioned_group_messages: true
```

启用此模式后，来自明确白名单聊天/话题的未提及群组消息会作为观察上下文追加到共享聊天/话题会话记录中，但不会触发 Agent。`allowed_chats` 控制机器人在哪里响应；`group_allowed_chats` 授权用于观察上下文的共享群组会话，因此在此模式下使用相同的聊天 ID。同一白名单聊天/话题中后续的 `@botname` 提及、对机器人的回复或配置的提及模式可以使用该观察上下文。触发消息还会标记 `[nickname|user_id]`，并获得每轮安全 prompt（提示词），使模型将之前观察到的内容视为上下文而非发给机器人的指令。

等效环境变量：

```bash
TELEGRAM_ALLOWED_CHATS=-1001234567890
TELEGRAM_GROUP_ALLOWED_CHATS=-1001234567890
TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES=true
```

这需要 Telegram 将普通群组消息传递给 gateway，因此请按上述说明禁用 BotFather 隐私模式或将机器人提升为群组管理员。

## 第四步：获取你的用户 ID

Hermes Agent 使用 Telegram 数字用户 ID 来控制访问权限。你的用户 ID **不是**你的用户名——它是一个类似 `123456789` 的数字。

**方法一（推荐）：** 向 [@userinfobot](https://t.me/userinfobot) 发送消息——它会立即回复你的用户 ID。

**方法二：** 向 [@get_id_bot](https://t.me/get_id_bot) 发送消息——另一个可靠的选项。

保存这个数字，下一步会用到。

## 第五步：配置 Hermes

### 方式 A：交互式设置（推荐）

```bash
hermes gateway setup
```

在提示时选择 **Telegram**。向导会询问你的机器人 token 和允许的用户 ID，然后为你写入配置。

### 方式 B：手动配置

将以下内容添加到 `~/.hermes/.env`：

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_ALLOWED_USERS=123456789    # 多个用户用逗号分隔
```

### 启动 Gateway

```bash
hermes gateway
```

机器人应在几秒内上线。在 Telegram 上向它发送消息以验证。

## 从 Docker 后端终端发送生成的文件

如果你的终端后端是 `docker`，请注意 Telegram 附件由 **gateway 进程**发送，而非从容器内部发送。这意味着最终的 `MEDIA:/...` 路径必须在运行 gateway 的宿主机上可读。

常见问题：

- Agent 在 Docker 内将文件写入 `/workspace/report.txt`
- 模型发出 `MEDIA:/workspace/report.txt`
- Telegram 投递失败，因为 `/workspace/report.txt` 只存在于容器内，而非宿主机上

推荐模式：

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/.hermes/cache/documents:/output"
```

然后：

- 在 Docker 内将文件写入 `/output/...`
- 在 `MEDIA:` 中使用**宿主机可见**的路径，例如：
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`

如果你已有 `docker_volumes:` 部分，将新挂载添加到同一列表中。YAML 重复键会静默覆盖之前的值。

### 支持的 `MEDIA:` 文件扩展名

gateway 从 Agent 回复中提取 `MEDIA:/path/to/file` 标签，并将引用的文件作为平台原生附件发送。所有 gateway 平台支持的扩展名：

| 类别 | 扩展名 |
|---|---|
| 图片 | `png`, `jpg`, `jpeg`, `gif`, `webp`, `bmp`, `tiff`, `svg` |
| 音频 | `mp3`, `wav`, `ogg`, `m4a`, `opus`, `flac`, `aac` |
| 视频 | `mp4`, `mov`, `webm`, `mkv`, `avi` |
| **文档** | `pdf`, `txt`, `md`, `csv`, `json`, `xml`, `html`, `yaml`, `yml`, `log` |
| **Office** | `docx`, `xlsx`, `pptx`, `odt`, `ods`, `odp` |
| **压缩包** | `zip`, `rar`, `7z`, `tar`, `gz`, `bz2` |
| **书籍/安装包** | `epub`, `apk`, `ipa` |

此列表中的任何内容都会在支持原生附件的平台（Telegram、Discord、Signal、Slack、WhatsApp、飞书、Matrix 等）上作为原生附件投递；在不支持原生附件的平台上，会回退为链接或纯文本指示。**加粗**类别是最近几个版本新增的——如果你之前依赖模型输出 `here is the file: /path/to/report.docx`，请改用 `MEDIA:/path/to/report.docx` 以实现原生投递。

## Webhook 模式

默认情况下，Hermes 使用**长轮询**连接 Telegram——gateway 向 Telegram 服务器发出出站请求以获取新更新。这对本地和常驻部署效果良好。

对于**云部署**（Fly.io、Railway、Render 等），**webhook 模式**更具成本效益。这些平台可以在入站 HTTP 流量时自动唤醒休眠的机器，但无法通过出站连接唤醒。由于轮询是出站的，轮询机器人永远无法休眠。Webhook 模式反转了方向——Telegram 将更新推送到你的机器人 HTTPS URL，从而实现空闲时休眠的部署。

| | 轮询（默认） | Webhook |
|---|---|---|
| 方向 | Gateway → Telegram（出站） | Telegram → Gateway（入站） |
| 适用场景 | 本地、常驻服务器 | 支持自动唤醒的云平台 |
| 设置 | 无需额外配置 | 设置 `TELEGRAM_WEBHOOK_URL` |
| 空闲成本 | 机器必须保持运行 | 机器可在消息间隙休眠 |

### 配置

将以下内容添加到 `~/.hermes/.env`：

```bash
TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
TELEGRAM_WEBHOOK_SECRET="$(openssl rand -hex 32)"  # 必填
# TELEGRAM_WEBHOOK_PORT=8443        # 可选，默认 8443
```

| 变量 | 是否必填 | 说明 |
|----------|----------|-------------|
| `TELEGRAM_WEBHOOK_URL` | 是 | Telegram 发送更新的公开 HTTPS URL。URL 路径会自动提取（例如上例中的 `/telegram`）。 |
| `TELEGRAM_WEBHOOK_SECRET` | **是**（设置 `TELEGRAM_WEBHOOK_URL` 时） | Telegram 在每个 webhook 请求中回显的密钥 token，用于验证。gateway 在没有该密钥时拒绝启动——参见 [GHSA-3vpc-7q5r-276h](https://github.com/NousResearch/hermes-agent/security/advisories/GHSA-3vpc-7q5r-276h)。使用 `openssl rand -hex 32` 生成。 |
| `TELEGRAM_WEBHOOK_PORT` | 否 | webhook 服务器监听的本地端口（默认：`8443`）。 |

设置 `TELEGRAM_WEBHOOK_URL` 后，gateway 会启动 HTTP webhook 服务器而非轮询。未设置时使用轮询模式——与之前版本行为无变化。

### 云部署示例（Fly.io）

1. 将环境变量添加到 Fly.io 应用密钥：

```bash
fly secrets set TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
fly secrets set TELEGRAM_WEBHOOK_SECRET=$(openssl rand -hex 32)
```

2. 在 `fly.toml` 中暴露 webhook 端口：

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

gateway 日志应显示：`[telegram] Connected to Telegram (webhook mode)`。

## 代理支持

如果 Telegram 的 API 被封锁，或你需要通过代理路由流量，可设置 Telegram 专用代理 URL。此设置优先于通用的 `HTTPS_PROXY` / `HTTP_PROXY` 环境变量。

**方式一：config.yaml（推荐）**

```yaml
telegram:
  proxy_url: "socks5://127.0.0.1:1080"
```

**方式二：环境变量**

```bash
TELEGRAM_PROXY=socks5://127.0.0.1:1080
```

支持的协议：`http://`、`https://`、`socks5://`。

代理同时适用于主 Telegram 连接和备用 IP 传输。如果未设置 Telegram 专用代理，gateway 会回退到 `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY`（或 macOS 系统代理自动检测）。

## 主频道

在任意 Telegram 聊天（私聊或群组）中使用 `/sethome` 命令，将其指定为**主频道**。定时任务（cron 任务）的结果会投递到此频道。

也可以在 `~/.hermes/.env` 中手动设置：

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="My Notes"
```

:::tip
群聊 ID 是负数（例如 `-1001234567890`）。你的个人私聊 ID 与你的用户 ID 相同。
:::

### 话题模式下的 Cron 投递

如果你在机器人私聊中启用了话题模式，投递到根聊天的 cron 消息会落入仅限系统的大厅——在那里回复不会开启会话，你会看到"主聊天保留给系统命令"的提示。创建一个专用论坛话题（例如 `Cron`）并设置：

```bash
TELEGRAM_CRON_THREAD_ID=<topic_thread_id>
```

`TELEGRAM_CRON_THREAD_ID` 仅针对 cron 投递覆盖 `TELEGRAM_HOME_CHANNEL_THREAD_ID`。在该话题中的回复会继续该话题的现有会话。

## 语音消息

### 接收语音（语音转文字）

你在 Telegram 上发送的语音消息会由 Hermes 配置的 STT（语音转文字）提供商自动转录，并作为文本注入对话。

- `local` 在运行 Hermes 的机器上使用 `faster-whisper`——无需 API 密钥
- `groq` 使用 Groq Whisper，需要 `GROQ_API_KEY`
- `openai` 使用 OpenAI Whisper，需要 `VOICE_TOOLS_OPENAI_KEY`

#### 跳过 STT：将原始音频文件传递给 Agent

如果你希望由 **Agent 本身**处理音频——用于说话人分离、自定义转录工具或仅存档录音——请在 `~/.hermes/config.yaml` 中设置 `stt.enabled: false`：

```yaml
stt:
  enabled: false
```

禁用 STT 后，gateway 仍会将语音/音频附件下载到 Hermes 的音频缓存中，但**不进行转录**。Agent 收到的消息带有如下标记：

```
[The user sent a voice message: /home/<user>/.hermes/cache/audio/<hash>.ogg]
```

你的工具或技能可以直接读取该路径（例如，将其传递给本地说话人分离管道、更丰富的转录模型，或上传到长期存储）。文件扩展名反映 Telegram 投递的原始格式（语音备忘录为 `.ogg`，音频附件为 `.mp3`/`.m4a` 等）。

这与下方的[本地 Bot API 服务器](#large-files-20mb--via-local-bot-api-server)部分配合使用效果极佳，该功能将 Telegram 的 20MB `getFile` 上限提升至 2GB——当你需要处理超过几分钟的录音时非常有用。

### 发送语音（文字转语音）

当 Agent 通过 TTS 生成音频时，它会作为 Telegram 原生**语音气泡**投递——即圆形、可内联播放的那种。

- **OpenAI 和 ElevenLabs** 原生生成 Opus——无需额外设置
- **Edge TTS**（默认免费提供商）输出 MP3，需要 **ffmpeg** 转换为 Opus：

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

没有 ffmpeg，Edge TTS 音频会作为普通音频文件发送（仍可播放，但使用矩形播放器而非语音气泡）。

在 `config.yaml` 的 `tts.provider` 键下配置 TTS 提供商。

## 通过本地 Bot API 服务器处理大文件（>20MB）

Telegram 的**公共** Bot API 将 `getFile` 下载限制为 **20 MB**，因此任何超过该大小的语音备忘录、音频文件、视频或文档都会被 Hermes 静默拒绝并回复"文件过大"。官方解决方案是运行本地 [telegram-bot-api](https://github.com/tdlib/telegram-bot-api) 守护进程——与 Telegram 使用的相同服务器软件，但运行在你的网络上。本地服务器将文件上限提升至 **2 GB**，Hermes 在检测到自定义 `base_url` 配置时会自动解除自身内部限制。

这解锁了以下工作流：

- 向机器人发送长语音备忘录（45 分钟会议、播客）
- 上传大型视频供视觉工具处理
- 存档原始音频用于离线管道，如说话人分离、对齐或训练数据

### 第一步：获取 Telegram API 凭据

本地服务器直接与 Telegram 的 MTProto 层通信（而非公共 Bot API），因此需要 **MTProto 凭据**：

1. 访问 [my.telegram.org/apps](https://my.telegram.org/apps) 并用你的 Telegram 账号登录。
2. 创建一个新应用（任意名称和简短描述均可）。
3. 复制 `api_id` 和 `api_hash`——两者都是必需的。

### 第二步：运行 telegram-bot-api 服务器

社区维护的 [`aiogram/telegram-bot-api`](https://hub.docker.com/r/aiogram/telegram-bot-api) Docker 镜像是最简便的方式。一个最小化的 `docker-compose.yaml`（使用 `--local` 模式启用更高限制）：

```yaml
services:
  tg-bot-api:
    image: aiogram/telegram-bot-api:latest
    container_name: tg-bot-api
    restart: unless-stopped
    ports:
      - "127.0.0.1:8081:8081"   # 仅绑定到回环地址；见安全说明
    environment:
      TELEGRAM_API_ID: "12345"           # 第一步中的 api_id
      TELEGRAM_API_HASH: "abcdef..."     # 第一步中的 api_hash
      TELEGRAM_LOCAL: "1"                # 启用 --local 模式（将 20MB 提升至 2GB）
    volumes:
      - ./tg-bot-api-data:/var/lib/telegram-bot-api
```

启动：

```bash
docker compose up -d tg-bot-api
docker logs --tail 20 tg-bot-api
```

:::warning 安全
本地 Bot API 服务器在 URL 路径中接受你的机器人 token（例如 `/bot<TOKEN>/getMe`），**无额外认证**。任何能访问该端口的人都可以完全控制你的机器人——读取它能看到的每条消息、以它的身份发送消息等。将容器绑定到 `127.0.0.1`，并/或在私有网络上用反向代理保护。**切勿将 8081 端口暴露到公网。**
:::

### 第三步：将机器人从公共 API 登出（一次性操作）

一个机器人在同一时间只能在**一个** Bot API 服务器上活跃。如果你的机器人之前已在 `api.telegram.org` 上运行（几乎可以肯定），你必须先在那里明确登出，本地服务器才会接受它：

```bash
curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/logOut"
# 预期响应：{"ok":true,"result":true}
```

这是一次性迁移步骤——不需要在每次重启时重复。`logOut` 后收到的消息会通过新服务器投递。

验证本地服务器能代表机器人与 Telegram 通信：

```bash
curl "http://127.0.0.1:8081/bot<YOUR_BOT_TOKEN>/getMe"
# 预期响应：{"ok":true,"result":{"id":...,"is_bot":true,...}}
```

### 第四步：将 Hermes 指向本地服务器

在 `~/.hermes/config.yaml` 的 `platforms.telegram.extra` 下添加 URL：

```yaml
platforms:
  telegram:
    extra:
      base_url: "http://127.0.0.1:8081/bot"
      base_file_url: "http://127.0.0.1:8081/file/bot"
      local_mode: true        # 见下方第五步——仅在机器人数据目录
                              # 对 Hermes 进程可读时设置此项
```

:::caution 使用 `platforms.telegram.extra`，而非 `telegram.extra`
目前只有 `platforms.<name>.extra` 形式会深度合并到平台配置中。直接放在顶层 `telegram.extra` 块下的键会被静默丢弃。
:::

设置 `base_url` 后，Hermes 会：

- 基于本地服务器构建 python-telegram-bot 客户端
- 自动将内部文档/音频大小上限从 20 MB 提升至 2 GB
- 在"文件过大"错误消息中报告当前限制（`Maximum: 2048 MB.`），以便清楚了解所处模式

重启 gateway 并查找确认日志行：

```bash
hermes gateway restart
grep -E "Using custom Telegram base_url|Using Telegram local_mode" ~/.hermes/logs/gateway.log | tail
```

### 第五步：`local_mode`——磁盘上的文件访问

本地服务器有**两种**投递文件的方式：

1. **不使用 `--local`**（默认）：文件通过 HTTP 在 `/file/bot<TOKEN>/<path>` 提供，与公共 Bot API 相同。20MB 上限仍然有效。仅作为网络修复使用（例如 `api.telegram.org` 不可达但你可以自托管）；这不是你想要的大小提升方式。
2. **使用 `--local`**（通过上方的 `TELEGRAM_LOCAL=1` 设置）：文件写入服务器文件系统，`getFile` 响应返回**绝对路径**而非 HTTP URL。20MB 上限被解除。Hermes 必须**从磁盘**读取字节，而非通过 HTTP。

要使磁盘读取路径正常工作，请在上方配置中设置 `local_mode: true`，**并**确保 Hermes 进程能读取服务器返回的路径。两种场景：

- **同一台机器**——telegram-bot-api 和 Hermes 运行在同一宿主机上。将数据卷绑定挂载到 Hermes 可读的目录（例如 `/var/lib/telegram-bot-api`），并确保文件所有权匹配。容器会降权到其内部的 `telegram-bot-api` 用户（uid 因镜像而异）；最简单的解决方法是在 compose 服务中添加 `user: "<UID>:<GID>"`，使文件归 Hermes 已运行的 uid 所有。
- **不同机器**——机器人服务器运行在一台主机上（例如 NAS、独立虚拟机），Hermes 运行在另一台上。服务器的数据目录必须以服务器报告的**相同绝对路径**（通常为 `/var/lib/telegram-bot-api`）共享给 Hermes 机器。NFS 效果良好；如果你不想在文件系统级别处理 uid 不匹配问题，带 `uid=` 挂载重映射的 CIFS/SMB 更友好。

如果设置了 `local_mode: true` 但 Hermes 无法 `stat` 返回的文件路径（权限问题或挂载错误），python-telegram-bot 会静默回退到对本地服务器的 HTTP `getFile`——在 `--local` 模式下会响应 `404 Not Found`。症状在 `gateway.log` 中表现为：

```
[Telegram] Failed to cache voice: Not Found
telegram.error.InvalidToken: Not Found
```

如果你看到这个，说明大小提升正在工作，但文件共享没有。以 gateway 运行用户的身份从 Hermes 宿主机执行 `ls -la /var/lib/telegram-bot-api/<TOKEN>/voice/`，并确认单个文件可以 `cat` 而不出现权限错误。

### 第六步：测试

向机器人发送一个超过 20 MB 的语音备忘录或音频文件。查看 gateway 日志：

```bash
tail -f ~/.hermes/logs/gateway.log | grep -iE "telegram|cache"
```

你应该看到 `[Telegram] Cached user voice at /home/<user>/.hermes/cache/audio/...` 行，且**没有**"文件过大"拒绝。结合上方的 `stt.enabled: false`，原始音频文件的路径会出现在 Agent 的入站消息中，供下游处理使用。

## 群聊使用

Hermes Agent 在 Telegram 群聊中工作时有几点注意事项：

- **隐私模式**决定机器人能看到哪些消息（见[第三步](#step-3-privacy-mode-critical-for-groups)）
- `TELEGRAM_ALLOWED_USERS` 仍然适用——即使在群组中，也只有授权用户才能触发机器人
- 你可以通过 `telegram.require_mention: true` 阻止机器人响应普通群组消息
- 设置 `telegram.require_mention: true` 时，以下情况的群组消息会被接受：
  - 回复机器人消息的内容
  - `@botusername` 提及
  - `/command@botusername`（包含机器人名称的 Telegram 机器人菜单命令形式）
  - 与 `telegram.mention_patterns` 中配置的正则唤醒词匹配的内容
- 在有多个 Hermes 机器人的群组中，`telegram.exclusive_bot_mentions` 使路由具有确定性。当消息明确提及一个或多个 Telegram 机器人用户名时，只有被提及的机器人配置文件处理该消息；其他 Hermes 机器人在回复和唤醒词回退运行之前忽略它。此功能默认启用。
- 使用 `telegram.ignored_threads` 使 Hermes 在特定 Telegram 论坛话题中保持沉默，即使群组本来允许自由响应或提及触发的回复
- 如果 `telegram.require_mention` 未设置或为 false，Hermes 保持之前的开放群组行为，响应它能看到的普通群组消息

### 同一群组中的多个 Hermes 机器人

如果你在同一个 Telegram 群组中运行多个 Hermes 配置文件，请为每个配置文件创建一个 Telegram 机器人 token，并为每个配置文件启动一个 gateway。不要在多个运行中的 gateway 中重用同一个机器人 token；Telegram 会拒绝对同一 token 的并发轮询。

推荐的群组配置：

```yaml
telegram:
  require_mention: true
  exclusive_bot_mentions: true
  mention_patterns: []
```

使用此设置，群组消息如 `@research_bot @ops_bot summarize this` 只由 `research_bot` 和 `ops_bot` 处理。群组中的其他 Hermes 机器人保持沉默，即使该消息是对其早期消息的回复或与共享唤醒词匹配。

仅在旧版群组中（明确提及不应覆盖回复和唤醒词触发）才将 `exclusive_bot_mentions: false`。

要运行多个配置文件，每个配置文件运行一次 gateway 命令。例如：

```bash
# 默认配置文件
hermes gateway start
hermes gateway status
hermes gateway stop

# 命名配置文件
hermes -p research gateway start
hermes -p research gateway status
hermes -p research gateway stop
```

对于小型固定机器人集群，使用 shell 循环或脚本，对默认配置文件调用 `hermes gateway <action>`，对每个命名配置文件调用 `hermes -p <profile> gateway <action>`。这比假设单个进程级命令在每个服务管理器上控制所有命名配置文件更可靠。

### 故障排除：私聊正常但群组无响应

如果机器人在私聊中响应但在群组中保持沉默，请按顺序检查以下关卡：

1. **Telegram 投递：** 关闭 BotFather 隐私模式、将机器人提升为管理员，或直接提及机器人。Hermes 无法响应 Telegram 从未投递给机器人的群组消息。
2. **更改隐私后重新加入：** 更改 BotFather 隐私设置后，将机器人从群组中移除并重新添加。Telegram 可能对现有成员保留旧的投递行为。
3. **Hermes 授权：** 确保发送者在 `TELEGRAM_ALLOWED_USERS` 或 `TELEGRAM_GROUP_ALLOWED_USERS` 中，或通过 `TELEGRAM_GROUP_ALLOWED_CHATS` 允许该群聊。
4. **提及过滤器：** 如果设置了 `telegram.require_mention: true`，普通群组消息会被忽略，除非消息是斜杠命令、对机器人的回复、`@botusername` 提及或配置的 `mention_patterns` 匹配。
5. **多机器人路由：** 如果群组包含多个机器人，确保每个 Hermes 配置文件使用唯一的机器人 token，并保持 `exclusive_bot_mentions` 启用，除非你有意使用旧版共享触发行为。

Telegram 群组和超级群组的负数聊天 ID 是正常的。如果你使用聊天范围的授权，请将这些 ID 放在 `TELEGRAM_GROUP_ALLOWED_CHATS` 中，而非发送者用户白名单中。

### 群组触发配置示例

将以下内容添加到 `~/.hermes/config.yaml`：

```yaml
telegram:
  require_mention: true
  exclusive_bot_mentions: true
  mention_patterns:
    - "^\\s*chompy\\b"
  ignored_threads:
    - 31
    - "42"
```

此示例允许所有常规直接触发，以及以 `chompy` 开头的消息，即使它们不使用 `@mention`。
Telegram 话题 `31` 和 `42` 中的消息在提及和自由响应检查运行之前始终被忽略。

### `mention_patterns` 说明

- 模式使用 Python 正则表达式
- 匹配不区分大小写
- 模式同时检查文本消息和媒体说明
- 无效的正则表达式模式会在 gateway 日志中记录警告并被忽略，而不会导致机器人崩溃
- 如果你希望模式仅在消息开头匹配，请用 `^` 锚定

## 私聊话题（Bot API 9.4）

Telegram Bot API 9.4（2026 年 2 月）引入了**私聊话题**——机器人可以直接在一对一私聊中创建论坛风格的话题线程，无需超级群组。这让你可以在与 Hermes 的现有私聊中运行多个隔离的工作区。

### 使用场景

如果你同时处理多个长期项目，话题可以保持各自上下文独立：

- **话题"Website"** — 处理你的生产 Web 服务
- **话题"Research"** — 文献综述和论文探索
- **话题"General"** — 杂项任务和快速问题

每个话题都有自己的对话会话、历史记录和上下文——完全相互隔离。

### 配置

:::caution 前提条件
在配置中添加话题之前，用户必须在与机器人的私聊中**启用话题模式**：

1. 在 Telegram 中打开与 Hermes 机器人的私聊
2. 点击顶部的机器人名称打开聊天信息
3. 启用**话题**（将聊天转换为论坛的开关）

没有此设置，Hermes 会在启动时记录 `The chat is not a forum` 并跳过话题创建。这是 Telegram 客户端设置——机器人无法以编程方式启用它。
:::

在 `~/.hermes/config.yaml` 的 `platforms.telegram.extra.dm_topics` 下添加话题：

```yaml
platforms:
  telegram:
    extra:
      dm_topics:
      - chat_id: 123456789        # 你的 Telegram 用户 ID
        topics:
        - name: General
          icon_color: 7322096
        - name: Website
          icon_color: 9367192
        - name: Research
          icon_color: 16766590
          skill: arxiv              # 在此话题中自动加载技能
```

**字段：**

| 字段 | 是否必填 | 说明 |
|-------|----------|-------------|
| `name` | 是 | 话题显示名称 |
| `icon_color` | 否 | Telegram 图标颜色代码（整数） |
| `icon_custom_emoji_id` | 否 | 话题图标的自定义 emoji ID |
| `skill` | 否 | 在此话题的新会话中自动加载的技能 |
| `thread_id` | 否 | 话题创建后自动填充——请勿手动设置 |

### 工作原理

1. gateway 启动时，Hermes 为每个尚未有 `thread_id` 的话题调用 `createForumTopic`
2. `thread_id` 会自动保存回 `config.yaml`——后续重启会跳过 API 调用
3. 每个话题映射到一个隔离的会话键：`agent:main:telegram:dm:{chat_id}:{thread_id}`
4. 每个话题中的消息都有自己的对话历史、内存刷新和上下文窗口

### 根私聊处理

默认情况下，发送到根私聊（任何话题之外）的消息会正常处理。设置 `ignore_root_dm: true` 可将根私聊变为大厅——对于已配置私聊话题的用户，普通消息会被静默忽略，而系统命令（`/start`、`/help`、`/status` 等）仍然有效。

```yaml
platforms:
  telegram:
    extra:
      ignore_root_dm: true
      dm_topics:
        - chat_id: 123456789
          topics:
            - name: General
```

该检查是**按聊天**进行的：只有在 `dm_topics` 中至少有一个条目的用户的根私聊才会受到影响。没有配置话题的用户不受影响。

### 技能绑定

带有 `skill` 字段的话题会在该话题中新会话开始时自动加载该技能。这与在对话开始时输入 `/skill-name` 完全相同——技能内容会注入到第一条消息中，后续消息在对话历史中可以看到它。

例如，带有 `skill: arxiv` 的话题会在其会话重置时（由于空闲超时、每日重置或手动 `/reset`）预加载 arxiv 技能。

:::tip
在配置之外创建的话题（例如通过手动调用 Telegram API）会在 `forum_topic_created` 服务消息到达时自动被发现。你也可以在 gateway 运行时向配置中添加话题——它们会在下次缓存未命中时被拾取。
:::

## 多会话私聊模式（`/topic`）

ChatGPT 风格的多会话私聊——一个机器人，多个并行对话。与上方运营商策划的 `extra.dm_topics` 不同，此模式是**用户驱动**的：无需配置，无需预先声明话题名称。终端用户通过 `/topic` 开启，然后点击 Telegram 的 **+** 按钮创建任意数量的话题，每个话题都是完全独立的 Hermes 会话。

### `/topic` 子命令

| 形式 | 上下文 | 效果 |
|------|---------|--------|
| `/topic` | 根私聊，尚未启用 | 检查 BotFather 功能，启用多会话模式，创建置顶 System 话题 |
| `/topic` | 根私聊，已启用 | 显示状态：可供恢复的未链接会话 |
| `/topic` | 话题内部 | 显示当前话题的会话绑定 |
| `/topic help` | 任意位置 | 内联使用说明 |
| `/topic off` | 根私聊 | 禁用多会话模式并清除此聊天的所有话题绑定 |
| `/topic <session-id>` | 话题内部 | 将之前的 Telegram 会话恢复到当前话题 |

只有授权用户（通过 `TELEGRAM_ALLOWED_USERS` / 平台认证配置的白名单）才能运行 `/topic`。未授权的发送者会收到拒绝而非激活。

### 私聊话题 vs 多会话私聊模式

| | `extra.dm_topics`（配置驱动） | `/topic`（用户驱动） |
|---|---|---|
| 谁激活 | 运营商，在 `config.yaml` 中 | 终端用户，通过发送 `/topic` |
| 话题列表 | 配置中声明的固定集合 | 用户自由创建/删除话题 |
| 话题名称 | 由运营商选择 | 由用户选择；自动重命名以匹配 Hermes 会话标题 |
| 根私聊行为 | 正常聊天（若 `ignore_root_dm: true` 则为大厅） | 变为系统大厅（非命令消息被拒绝） |
| 主要使用场景 | 带可选技能绑定的永久工作区 | 临时并行会话 |
| 持久化 | 配置中的 `extra.dm_topics` | `telegram_dm_topic_mode` + `telegram_dm_topic_bindings` SQLite 表 |

两个功能可以在同一个机器人上共存——你可以从用户的私聊运行 `/topic`，而 `extra.dm_topics` 继续为其他聊天管理运营商声明的话题。

### 前提条件

在 **@BotFather** 中，打开你的机器人 → **Bot Settings → Threads Settings**：

1. 开启 **Threaded Mode**（启用 `has_topics_enabled`）
2. **不要**禁用用户创建话题（保持 `allows_users_to_create_topics` 开启）

当用户首次运行 `/topic` 时，Hermes 调用 `getMe` 验证两个标志。如果任一标志关闭，Hermes 会发送 BotFather Threads Settings 页面的截图并说明需要切换什么——在满足前提条件之前不会激活。

### 激活流程

从根私聊发送：

```
/topic
```

Hermes 将：

1. 检查 `getMe().has_topics_enabled` 和 `allows_users_to_create_topics`
2. 如果两者都为 true，为此私聊启用多会话话题模式
3. 创建并置顶一个 **System** 话题用于状态/命令（尽力而为）
4. 回复用户可以恢复的之前未链接 Telegram 会话列表

激活后，**根私聊变为大厅**：普通 prompt 会被拒绝，并引导用户前往 **All Messages**。系统命令（`/status`、`/sessions`、`/usage`、`/help` 等）在根目录仍然有效。

### 创建新话题（终端用户流程）

1. 在 Telegram 中打开机器人私聊
2. 点击机器人界面顶部的 **All Messages**，然后发送任意消息
3. Telegram 为该消息创建一个新话题
4. Hermes 在该话题内响应——该话题现在是一个独立会话

每个话题都有自己的对话历史、模型状态、工具执行和会话 ID。隔离键为 `agent:main:telegram:dm:{chat_id}:{thread_id}`——与配置驱动的私聊话题隔离相同。

### 自动重命名话题

当 Hermes 为话题生成会话标题时（通过自动标题管道，在第一次交换后），Telegram 话题本身会被重命名以匹配——例如"New Topic"变为"Database migration plan"。重命名是尽力而为的：失败会被记录但不会中断会话。

要禁用此功能并保留你手动选择的话题名称，请设置：

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_topic_auto_rename: true
```

启用此标志后，Hermes 仍会生成内部会话标题（供 `hermes sessions`、TUI 等使用），但永远不会编辑 Telegram 话题名称。当你在 BotFather Threaded Mode 下手动整理话题，且不希望每次第一次回复都覆盖标题时，此功能很有用。

### 话题内的 `/new`

重置当前话题的会话（新会话 ID，全新历史记录），而不影响其他话题。Hermes 回复提醒，对于并行工作，创建另一个话题（通过 **All Messages**）通常才是你想要的。

### 恢复之前的会话

在话题内发送：

```
/topic <session-id>
```

这会将当前话题绑定到现有 Hermes 会话，而非重新开始。适用于继续在启用话题模式之前开始的对话。限制：

- 目标会话必须属于同一 Telegram 用户
- 目标会话不能已绑定到另一个话题

Hermes 会确认会话标题，并重放最后一条助手消息以提供上下文。

要发现会话 ID，在根私聊发送 `/topic`（无参数）——Hermes 会列出用户未链接的 Telegram 会话。

### 话题内的 `/topic`（无参数）

显示当前话题的绑定：会话标题、会话 ID，以及 `/new` 与创建另一个话题的提示。

### 底层实现

- 激活持久化到 `state.db` 中的 `telegram_dm_topic_mode(chat_id, user_id, enabled, ...)`
- 每个话题绑定持久化到 `telegram_dm_topic_bindings(chat_id, thread_id, session_id, ...)` 中，`session_id` 上有 `ON DELETE CASCADE`——删除会话会自动清除其话题绑定
- 话题模式 SQLite 迁移是**按需**的：它在第一次 `/topic` 调用时运行，而非在 gateway 启动时。在用户在此配置文件中运行 `/topic` 之前，`state.db` 保持不变
- 每条入站私聊消息都会查找其 `(chat_id, thread_id)` 绑定。如果存在，查找会通过 `SessionStore.switch_session()` 将消息路由到绑定的会话，以保持磁盘上会话键到会话 ID 映射的一致性
- 话题内的 `/new` 会重写绑定行以指向新会话 ID，因此下一条消息保持在新会话上
- `extra.dm_topics` 中声明的话题**永远不会自动重命名**——即使启用了多会话模式，运营商选择的名称也会被保留
- 设置 `extra.disable_topic_auto_rename: true` 可关闭聊天中**所有**话题的自动重命名（包括通过 Threaded Mode 创建的临时话题）
- 论坛启用私聊中的 General（置顶顶部）话题被视为根大厅，无论 Telegram 是以 `message_thread_id=1` 还是无 thread_id 投递其消息
- 根大厅提醒每个聊天每 30 秒限速一条——忘记话题模式已开启并在根目录输入十条 prompt 的用户不会收到十条回复
- BotFather 设置截图每个聊天每 5 分钟限速一次发送——在 Threads Settings 仍然禁用时重复尝试 `/topic` 不会重复上传同一张图片
- 在话题内启动的 `/background <prompt>` 会将结果投递回同一话题；后台会话不会触发所属话题的自动重命名
- `/topic` 本身受机器人用户授权检查限制——未授权的私聊会收到拒绝而非激活

### 禁用多会话模式

在根私聊发送 `/topic off`。Hermes 将该行翻转为关闭，清除聊天的 `(thread_id → session_id)` 绑定，根私聊恢复为正常 Hermes 聊天。Telegram 中现有的话题不会被删除——它们只是不再作为独立会话被管控。之后重新运行 `/topic` 可重新开启。

如果你需要手动清理（例如跨多个聊天的批量重置），直接删除行：

```bash
sqlite3 ~/.hermes/state.db \
  "UPDATE telegram_dm_topic_mode SET enabled = 0 WHERE chat_id = '<your_chat_id>'; \
   DELETE FROM telegram_dm_topic_bindings WHERE chat_id = '<your_chat_id>';"
```

### 降级 Hermes

如果你降级到早于 `/topic` 的 Hermes 版本，该功能会停止工作——`telegram_dm_topic_mode` 和 `telegram_dm_topic_bindings` 表保留在 `state.db` 中，但被旧代码忽略。私聊恢复为原生的每线程隔离（每个 `message_thread_id` 仍通过 `build_session_key` 获得自己的会话），因此你现有的 Telegram 话题继续作为并行会话工作。根私聊不再是大厅——消息像以前一样进入 Agent。重新升级会在原来的位置精确恢复多会话模式。

## 群组论坛话题技能绑定

启用了**话题模式**（也称为"论坛话题"）的超级群组已经按话题进行会话隔离——每个 `thread_id` 映射到自己的对话。但你可能希望在特定群组话题中有消息到达时**自动加载技能**，就像私聊话题技能绑定的工作方式一样。

### 使用场景

一个有不同工作流论坛话题的团队超级群组：

- **Engineering** 话题 → 自动加载 `software-development` 技能
- **Research** 话题 → 自动加载 `arxiv` 技能
- **General** 话题 → 无技能，通用助手

### 配置

在 `~/.hermes/config.yaml` 的 `platforms.telegram.extra.group_topics` 下添加话题绑定：

```yaml
platforms:
  telegram:
    extra:
      group_topics:
      - chat_id: -1001234567890       # 超级群组 ID
        topics:
        - name: Engineering
          thread_id: 5
          skill: software-development
        - name: Research
          thread_id: 12
          skill: arxiv
        - name: General
          thread_id: 1
          # 无技能——通用用途
```

**字段：**

| 字段 | 是否必填 | 说明 |
|-------|----------|-------------|
| `chat_id` | 是 | 超级群组的数字 ID（以 `-100` 开头的负数） |
| `name` | 否 | 话题的人类可读标签（仅供参考） |
| `thread_id` | 是 | Telegram 论坛话题 ID——在 `t.me/c/<group_id>/<thread_id>` 链接中可见 |
| `skill` | 否 | 在此话题的新会话中自动加载的技能 |

### 工作原理

1. 当消息到达已映射的群组话题时，Hermes 在 `group_topics` 配置中查找 `chat_id` 和 `thread_id`
2. 如果匹配条目有 `skill` 字段，该技能会为会话自动加载——与私聊话题技能绑定完全相同
3. 没有 `skill` 键的话题只获得会话隔离（现有行为，不变）
4. 未映射的 `thread_id` 值或 `chat_id` 值会静默通过——无错误，无技能

### 与私聊话题的区别

| | 私聊话题 | 群组话题 |
|---|---|---|
| 配置键 | `extra.dm_topics` | `extra.group_topics` |
| 话题创建 | 如果缺少 `thread_id`，Hermes 通过 API 创建话题 | 管理员在 Telegram UI 中创建话题 |
| `thread_id` | 创建后自动填充 | 必须手动设置 |
| `icon_color` / `icon_custom_emoji_id` | 支持 | 不适用（管理员控制外观） |
| 技能绑定 | ✓ | ✓ |
| 会话隔离 | ✓ | ✓（论坛话题已内置） |

:::tip
要找到话题的 `thread_id`，在 Telegram Web 或桌面版中打开该话题并查看 URL：`https://t.me/c/1234567890/5`——最后一个数字（`5`）就是 `thread_id`。超级群组的 `chat_id` 是群组 ID 加上 `-100` 前缀（例如，群组 `1234567890` 变为 `-1001234567890`）。
:::

## 近期 Bot API 功能

- **Bot API 9.4（2026 年 2 月）：** 私聊话题——机器人可以通过 `createForumTopic` 在一对一私聊中创建论坛话题。Hermes 将此用于两个不同功能：运营商策划的[私聊话题](#private-chat-topics-bot-api-94)（配置驱动，固定话题列表）和用户驱动的[多会话私聊模式](#multi-session-dm-mode-topic)（通过 `/topic` 激活，用户创建的无限话题）。
- **隐私政策：** Telegram 现在要求机器人有隐私政策。通过 BotFather 的 `/setprivacy_policy` 设置，或 Telegram 可能自动生成占位符。如果你的机器人面向公众，这一点尤为重要。
- **Bot API 9.5（2026 年 3 月）：通过 `sendMessageDraft` 实现原生流式传输。** Hermes 支持 Telegram 的原生流式草稿 API，作为私聊的可选传输方式。默认仍使用旧版 `editMessageText` 路径，因为草稿预览在某些 Telegram 客户端上可能出现明显的折叠和重新渲染。

### 流式传输（`gateway.streaming.transport`）

启用流式传输（`gateway.streaming.enabled: true`）时，Hermes 从四种传输方式中选择一种：

| 值 | 行为 |
|---|---|
| `auto`（默认） | 在支持的聊天（目前为 Telegram 私聊）上使用原生草稿流式传输；否则使用旧版基于编辑的路径。如果草稿帧失败，会优雅回退。 |
| `draft` | 强制使用原生草稿。如果聊天不支持草稿（例如群组/话题），记录降级日志并回退到编辑方式。 |
| `edit` | 对所有聊天类型使用旧版渐进式 `editMessageText` 轮询。 |
| `off` | 完全禁用流式传输（仅最终回复，无渐进更新）。 |

在 `~/.hermes/config.yaml` 中：

```yaml
gateway:
  streaming:
    enabled: true
    transport: auto    # auto | draft | edit | off
```

**使用 `edit` 传输时私聊中的效果** — gateway 发送一条普通预览消息，并通过 `editMessageText` 渐进更新，避免 Telegram 草稿预览折叠/回滚效果。

**使用 `auto` 或 `draft` 时私聊中的效果** — Telegram 显示逐 token 更新的动画草稿预览。回复完成后，它作为普通消息投递，草稿预览在客户端自然清除。草稿没有消息 ID，因此最终答案才是保留在聊天历史中的内容。

**群组、超级群组、论坛话题怎么办？** Telegram 将 `sendMessageDraft` 限制为私聊（私信）。gateway 对其他所有内容透明地回退到基于编辑的路径——与之前的用户体验相同。

**如果草稿帧失败怎么办？** 任何失败（瞬时网络错误、服务器端拒绝、旧版 python-telegram-bot 安装）都会将该响应的剩余流切换回基于编辑的路径。下一个响应会重新尝试。

## 渲染：富消息、表格和链接预览

**富消息（Bot API 10.1）。** 最终回复中那些会被旧版 MarkdownV2 路径降级的结构——表格、任务列表、可折叠的 `<details>` 以及块级数学公式——会通过 Telegram 原生的 [`sendRichMessage`](https://core.telegram.org/bots/api#sendrichmessage) 发送，使用 Agent 的**原始 markdown**，从而原生渲染、无需客户端展平。在流式传输过程中，最终答案通过 `editMessageText` 的 `rich_message` 参数**就地编辑现有预览**来交付——不发第二条消息、不删除，因此一轮结束时不会出现重复投递的闪烁。在私聊中，实时流式预览也使用 `sendRichMessageDraft`，因此动画草稿与最终的富消息保持一致。普通回复（纯文本、粗体/斜体、简单列表）仍走 MarkdownV2 路径，以在各客户端保持一致的字重和间距。

当内容超过 32,768 字符的富文本上限时，富消息路径会自动跳过；Telegram 的任何拒绝（较旧 `python-telegram-bot` 不支持该端点、解析错误、块/列过多）都会**透明回退**到 MarkdownV2 路径——消息绝不会丢失。瞬时/网络错误**不会**被静默重发（不会产生重复的最终消息）。

**MarkdownV2 回退。** 当某条消息无法使用富消息路径时，Hermes 会将 markdown 转换为 MarkdownV2。由于 MarkdownV2 没有原生表格语法，管道表格会被规范化：

- **小表格**被展平为**行组项目符号**——每行在列标题下变为可读的项目符号列表。适合 2-4 列和短单元格。
- **较大或较宽的表格**回退为带对齐列的**围栏代码块**，以防内容折叠。

富消息现在是**选择启用**。默认保持旧版 MarkdownV2 路径，因为当前 Telegram 客户端可能让 Bot API 富消息难以作为纯文本复制，这对命令片段和移动端交接尤其麻烦。若要为表格、任务列表、折叠 `<details>` 和块级数学启用原生渲染：

```yaml
gateway:
  platforms:
    telegram:
      extra:
        rich_messages: true
```

这个设置用于客户端渲染/复制兼容性；当 Telegram 拒绝富消息 API 调用时，Hermes 已经会自动回退。如果你只是想在保持富消息启用的同时恢复旧版「始终使用代码块」表格行为，可在 `config.yaml` 中设置 `telegram.pretty_tables: false` 禁用表格规范化（默认：`true`）。

**链接预览。** Telegram 会为机器人消息中的 URL 自动生成链接预览。如果你希望抑制这些预览（长 `/tools` 输出、提及十个链接的 Agent 回复等）：

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_link_previews: true
```

启用后，Hermes 为每条出站消息附加 Telegram 的 `LinkPreviewOptions(is_disabled=True)`，并在旧版 `python-telegram-bot` 版本上回退到旧版 `disable_web_page_preview` 参数。

## 群组白名单

Telegram 群组和论坛聊天有两个可配置的正交关卡：

- **发送者用户 ID**（`group_allow_from` / `TELEGRAM_GROUP_ALLOWED_USERS`）——仅适用于群组/论坛消息的发送者范围白名单。当你希望特定用户能在群组中调用机器人，而不将其添加到 `TELEGRAM_ALLOWED_USERS`（这也会给予他们私聊访问权限）时使用。
- **聊天 ID**（`group_allowed_chats` / `TELEGRAM_GROUP_ALLOWED_CHATS`）——聊天范围白名单。这些群组/论坛的任何成员都可以与机器人交互。适用于群组成员身份本身就是访问信号的团队/支持机器人。

```yaml
gateway:
  platforms:
    telegram:
      extra:
        # 全局访问（私聊 + 群组）。此处的用户始终可以调用机器人。
        allow_from:
          - "123456789"
        # 仅在群组/论坛中允许的发送者 ID。不授予私聊访问权限。
        group_allow_from:
          - "987654321"
        # 整个群组/论坛——任何成员都被授权。
        group_allowed_chats:
          - "-1001234567890"
```

等效环境变量：

```bash
TELEGRAM_ALLOWED_USERS="123456789"
TELEGRAM_GROUP_ALLOWED_USERS="987654321"
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

行为：

- `TELEGRAM_ALLOWED_USERS` 覆盖所有聊天类型（私聊、群组、论坛）。
- `TELEGRAM_GROUP_ALLOWED_USERS` 仅在群组/论坛中授权列出的发送者。除非在 `TELEGRAM_ALLOWED_USERS` 中列出，否则他们仍然无法私聊机器人。
- `TELEGRAM_GROUP_ALLOWED_CHATS` 中的聊天授权该聊天的每个成员，无论发送者是谁。
- 在任何这些中使用 `*` 允许任何发送者/聊天。
- 这叠加在现有的提及/模式触发器之上，以及 `group_topics` + `ignored_threads` 之上。

### 从 PR #17686 之前迁移

在此拆分之前，`TELEGRAM_GROUP_ALLOWED_USERS` 是唯一的控制项，用户将**聊天 ID** 放入其中。为了向后兼容，`TELEGRAM_GROUP_ALLOWED_USERS` 中形如聊天 ID 的值（以 `-` 开头）仍被视为聊天 ID，并记录一次弃用警告。迁移方式：

```bash
# 旧版（仍然有效，但已弃用）
TELEGRAM_GROUP_ALLOWED_USERS="-1001234567890"

# 新版
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

### 访客 @mention 绕过（`guest_mode`）

在典型设置中，`group_allowed_chats` 是硬性关卡：来自列表之外群组的消息会被静默丢弃，即使成员明确 @mention 了机器人。这是支持/团队机器人的正确默认值。

对于更随意的设置——朋友群聊，你希望机器人**大部分时间保持沉默**，但**在被明确 ping 时偶尔可用**——启用 `guest_mode`：

```yaml
gateway:
  platforms:
    telegram:
      extra:
        group_allowed_chats:
          - "-1001234567890"   # 你的主要白名单群组
        guest_mode: true       # 非白名单群组：仅在 @mention 时允许
```

等效环境变量：

```bash
TELEGRAM_GUEST_MODE=true
```

默认：`false`。

启用 `guest_mode: true` 后，来自非白名单群组的消息**仅在**明确 @mention 机器人时才被处理。每轮都需要提及——访客交互没有会话粘性，因此机器人永远不会在未被 ping 的朋友群组线程中自动参与。

私聊和白名单群组的行为与之前完全相同。

## 斜杠命令访问控制

默认情况下，每个允许的用户都可以运行每个斜杠命令。要将你的白名单分为**管理员**（完整斜杠命令访问）和**普通用户**（仅你明确启用的命令），请在平台的 `extra` 块中添加 `allow_admin_from` 和 `user_allowed_commands`：

```yaml
gateway:
  platforms:
    telegram:
      extra:
        # 现有白名单（不变）
        allow_from:
          - "123456789"     # 管理员
          - "555555555"     # 普通用户
          - "777777777"     # 普通用户

        # 新增——管理员可使用所有斜杠命令（内置 + 插件）
        allow_admin_from:
          - "123456789"

        # 新增——非管理员允许用户只能运行这些斜杠命令。
        # /help 和 /whoami 始终允许，以便用户查看其访问权限。
        user_allowed_commands:
          - status
          - model
          - history

        # 可选：群组的独立管理员/命令列表
        group_allow_admin_from:
          - "123456789"
        group_user_allowed_commands:
          - status
```

**行为：**

- 在某个范围（私聊或群组）的 `allow_admin_from` 中列出的用户可以运行**每个**已注册的斜杠命令——内置命令和插件注册的命令——通过实时注册表。
- 在 `allow_from` 中但**不在** `allow_admin_from` 中的用户只能运行 `user_allowed_commands` 中列出的命令，加上始终允许的底线：`/help` 和 `/whoami`。
- 普通聊天（非斜杠消息）不受影响。非管理员用户仍然可以正常与 Agent 对话，只是无法触发任意命令。
- **向后兼容：** 如果某个范围未设置 `allow_admin_from`，该范围的斜杠命令限制被禁用。现有安装无需任何更改即可继续工作。
- 私聊管理员状态不意味着群组管理员状态。每个范围都有自己的管理员列表。
- 如果只设置了 `group_allow_admin_from`，私聊范围保持不受限制（向后兼容）模式。

使用 `/whoami` 查看当前范围、你的级别（管理员/用户/不受限制）以及你可以运行的斜杠命令。

## 交互式模型选择器

在 Telegram 聊天中不带参数发送 `/model` 时，Hermes 会显示用于切换模型的交互式内联键盘：

1. **提供商选择** — 显示每个可用提供商及模型数量的按钮（例如，"OpenAI (15)"、"✓ Anthropic (12)"表示当前提供商）。
2. **模型选择** — 带 **Prev**/**Next** 导航的分页模型列表，**Back** 按钮返回提供商，以及 **Cancel**。

当前模型和提供商显示在顶部。所有导航都通过就地编辑同一条消息进行（不会产生聊天杂乱）。

:::tip
如果你知道确切的模型名称，直接输入 `/model <name>` 跳过选择器。你也可以输入 `/model <name> --global` 跨会话持久化更改。
:::

## DNS-over-HTTPS 备用 IP

在某些受限网络中，`api.telegram.org` 可能解析到无法访问的 IP。Telegram 适配器包含一个**备用 IP** 机制，在保留正确 TLS 主机名和 SNI 的同时，透明地对备用 IP 重试连接。

### 工作原理

1. 如果设置了 `TELEGRAM_FALLBACK_IPS`，直接使用这些 IP。
2. 否则，适配器自动通过 DNS-over-HTTPS（DoH）查询 **Google DNS** 和 **Cloudflare DNS**，以发现 `api.telegram.org` 的备用 IP。
3. DoH 返回的与系统 DNS 结果不同的 IP 被用作备用。
4. 如果 DoH 也被封锁，使用硬编码的种子 IP（`149.154.167.220`）作为最后手段。
5. 一旦备用 IP 成功，它就变得"粘性"——后续请求直接使用它，而不先重试主路径。

### 配置

```bash
# 明确的备用 IP（逗号分隔）
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
通常不需要手动配置此项。通过 DoH 的自动发现可以处理大多数受限网络场景。`TELEGRAM_FALLBACK_IPS` 环境变量仅在你的网络上 DoH 也被封锁时才需要。
:::

## 代理支持

如果你的网络需要 HTTP 代理才能访问互联网（企业环境中常见），Telegram 适配器会自动读取标准代理环境变量并通过代理路由所有连接。

### 支持的变量

适配器按顺序检查这些环境变量，使用第一个已设置的：

1. `HTTPS_PROXY`
2. `HTTP_PROXY`
3. `ALL_PROXY`
4. `https_proxy` / `http_proxy` / `all_proxy`（小写变体）

### 配置

在启动 gateway 之前在你的环境中设置代理：

```bash
export HTTPS_PROXY=http://proxy.example.com:8080
hermes gateway
```

或添加到 `~/.hermes/.env`：

```bash
HTTPS_PROXY=http://proxy.example.com:8080
```

代理同时适用于主传输和所有备用 IP 传输。无需额外的 Hermes 配置——如果设置了环境变量，它会自动被使用。

:::note
这涵盖了 Hermes 用于 Telegram 连接的自定义备用传输层。其他地方使用的标准 `httpx` 客户端已经原生支持代理环境变量。
:::

## 消息反应

机器人可以为消息添加 emoji 反应作为视觉处理反馈：

- 👀 当机器人开始处理你的消息时
- ✅ 当响应成功投递时
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
与 Discord（反应是累加的）不同，Telegram 的 Bot API 在单次调用中替换所有机器人反应。从 👀 到 ✅/❌ 的转换是原子性的——你不会同时看到两者。
:::

:::tip
如果机器人在群组中没有添加反应的权限，反应调用会静默失败，消息处理正常继续。
:::

## 按频道 Prompt

为特定 Telegram 群组或论坛话题分配临时系统 prompt。该 prompt 在每轮运行时注入——永远不会持久化到对话历史——因此更改立即生效。

```yaml
telegram:
  channel_prompts:
    "-1001234567890": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "42":  |
      This topic is for creative writing feedback. Be warm and
      constructive.
```

键是聊天 ID（群组/超级群组）或论坛话题 ID。对于论坛群组，话题级 prompt 覆盖群组级 prompt：

- `-1001234567890` 群组内话题 `42` 中的消息 → 使用话题 `42` 的 prompt
- 话题 `99` 中的消息（无明确条目）→ 回退到群组 `-1001234567890` 的 prompt
- 无条目群组中的消息 → 不应用频道 prompt

数字 YAML 键会自动规范化为字符串。

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| 机器人完全不响应 | 验证 `TELEGRAM_BOT_TOKEN` 是否正确。检查 `hermes gateway` 日志中的错误。 |
| 机器人回复"unauthorized" | 你的用户 ID 不在 `TELEGRAM_ALLOWED_USERS` 中。用 @userinfobot 再次确认。 |
| 机器人忽略群组消息 | 隐私模式可能已开启。禁用它（第三步）或将机器人设为群组管理员。**记住更改隐私设置后要移除并重新添加机器人。** |
| 语音消息未转录 | 验证 STT 是否可用：安装 `faster-whisper` 进行本地转录，或在 `~/.hermes/.env` 中设置 `GROQ_API_KEY` / `VOICE_TOOLS_OPENAI_KEY`。 |
| 语音回复是文件而非气泡 | 安装 `ffmpeg`（Edge TTS Opus 转换所需）。 |
| 机器人 token 被撤销/无效 | 通过 BotFather 的 `/revoke` 然后 `/newbot` 或 `/token` 生成新 token。更新你的 `.env` 文件。 |
| Webhook 未接收更新 | 验证 `TELEGRAM_WEBHOOK_URL` 是否可公开访问（用 `curl` 测试）。确保你的平台/反向代理将来自 URL 端口的入站 HTTPS 流量路由到 `TELEGRAM_WEBHOOK_PORT` 配置的本地监听端口（两者不需要是相同的数字）。确保 SSL/TLS 已激活——Telegram 只向 HTTPS URL 发送。检查防火墙规则。 |

## 执行审批

当 Agent 尝试运行潜在危险的命令时，它会在聊天中请求你的审批：

> ⚠️ This command is potentially dangerous (recursive delete). Reply "yes" to approve.

回复"yes"/"y"批准或"no"/"n"拒绝。

## 交互式 Prompt（clarify）

当 Agent 调用 `clarify` 工具时——询问你偏好哪种方式、获取任务后反馈，或在非平凡决策前确认——Telegram 会用**内联键盘按钮**渲染问题：

> ❓ Which framework should I use for the dashboard?
>
> [1. Next.js] [2. Remix] [3. Astro]
> [✏️ Other (type answer)]

点击按钮回答，或点击 **Other** 输入自由形式的回复（你发送的下一条消息成为答案）。开放式 `clarify` 调用（无预设选项）跳过按钮，直接捕获你的下一条消息。

通过 `~/.hermes/config.yaml` 中的 `agent.clarify_timeout` 配置响应超时（默认 `600` 秒）。如果你在超时内没有响应，Agent 会以哨兵消息解除阻塞并适应，而不是挂起。

## 推送通知音量

Telegram 对机器人发送的每条消息都会触发推送通知。对于发出工具进度气泡、流式更新和状态回调的长 Agent 轮次，这很快就会变得嘈杂。Telegram 适配器有两种通知模式：

| 模式 | 行为 |
|------|----------|
| `important`（默认） | 只有**最终响应**、**审批 prompt** 和**斜杠命令确认**会响铃。工具进度、流式块和状态消息以 `disable_notification=true` 投递。 |
| `all` | 每条出站消息都触发推送通知。旧版行为；如果你确实想听到每次工具调用，请选择此项。 |

在 `~/.hermes/config.yaml` 中配置：

```yaml
display:
  platforms:
    telegram:
      notifications: important   # 或 "all"
```

环境变量覆盖（便于快速 A/B 测试）：

```bash
HERMES_TELEGRAM_NOTIFICATIONS=all
```

未知值会记录警告并回退到 `important`。

## 安全

:::warning
始终设置 `TELEGRAM_ALLOWED_USERS` 以限制谁可以与你的机器人交互。没有此设置，gateway 默认拒绝所有用户作为安全措施。
:::

切勿公开分享你的机器人 token。如果泄露，请立即通过 BotFather 的 `/revoke` 命令撤销。

更多详情，请参阅[安全文档](/user-guide/security)。你也可以使用 [DM 配对](/user-guide/messaging#dm-pairing-alternative-to-allowlists) 进行更动态的用户授权方式。