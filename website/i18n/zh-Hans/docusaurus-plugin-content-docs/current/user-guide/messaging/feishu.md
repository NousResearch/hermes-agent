---
sidebar_position: 11
title: "Feishu / Lark"
description: "将 Hermes Agent 设置为飞书或 Lark 机器人"
---

# 飞书 / Lark 设置

Hermes Agent 集成了飞书和 Lark 作为功能完整的机器人。连接后，你可以在私信或群聊中与 agent 聊天，在主聊天中接收 cron 作业结果，并通过正常的 gateway 流程发送文本、图片、音频和文件附件。

该集成支持两种连接模式：

- `websocket` — 推荐；Hermes 打开出站连接，你不需要公网 webhook 端点
- `webhook` — 当你想要飞书/Lark 通过 HTTP 将事件推送到你的 gateway 时很有用

## Hermes 的行为方式

| 上下文 | 行为 |
|---------|----------|
| 私信 | Hermes 回复每条消息。 |
| 群聊 | Hermes 仅在聊天中 @提及机器人时回复。 |
| 共享群聊 | 默认情况下，会话历史在共享聊天中按用户隔离。 |

这种共享聊天行为由 `config.yaml` 控制：

```yaml
group_sessions_per_user: true
```

仅当你明确希望每个聊天有一个共享对话时，才将其设置为 `false`。

## 步骤 1：创建飞书 / Lark 应用

### 推荐：扫码创建（一个命令）

```bash
hermes gateway setup
```

选择 **Feishu / Lark** 并使用你的飞书或 Lark 手机 App 扫描二维码。Hermes 将自动创建具有正确权限的机器人应用并保存凭证。

### 替代方案：手动设置

如果扫码创建不可用，向导会回退到手动输入：

1. 打开飞书或 Lark 开发者控制台：
   - 飞书：[https://open.feishu.cn/](https://open.feishu.cn/)
   - Lark：[https://open.larksuite.com/](https://open.larksuite.com/)
2. 创建一个新应用。
3. 在 **凭证与基础信息** 中，复制 **App ID** 和 **App Secret**。
4. 为应用启用**机器人**能力。
5. 运行 `hermes gateway setup`，选择 **Feishu / Lark**，并在提示时输入凭证。

:::warning
保持 App Secret 私密。任何拥有它的人都可以冒充你的应用。
:::

## 步骤 2：选择连接模式

### 推荐：WebSocket 模式

当 Hermes 运行在你的笔记本电脑、工作站或私有服务器上时，请使用 WebSocket 模式。不需要公网 URL。官方 Lark SDK 会打开并维护一个持久的出站 WebSocket 连接，并自动重新连接。

```bash
FEISHU_CONNECTION_MODE=websocket
```

**要求：** 必须安装 `websockets` Python 包。SDK 在内部处理连接生命周期、心跳和自动重新连接。

**工作原理：** 适配器在后台执行器线程中运行 Lark SDK 的 WebSocket 客户端。入站事件（消息、反应、卡片操作）被分派到主 asyncio 循环。断开连接时，SDK 将尝试自动重新连接。

### 可选：Webhook 模式

仅当你已经在可访问的 HTTP 端点后面运行 Hermes 时，才使用 webhook 模式。

```bash
FEISHU_CONNECTION_MODE=webhook
```

在 webhook 模式下，Hermes 启动一个 HTTP 服务器（通过 `aiohttp`），并在以下位置提供飞书端点：

```text
/feishu/webhook
```

**要求：** 必须安装 `aiohttp` Python 包。

你可以自定义 webhook 服务器绑定地址和路径：

```bash
FEISHU_WEBHOOK_HOST=127.0.0.1   # 默认：127.0.0.1
FEISHU_WEBHOOK_PORT=8765         # 默认：8765
FEISHU_WEBHOOK_PATH=/feishu/webhook  # 默认：/feishu/webhook
```

当飞书发送 URL 验证质询（`type: url_verification`）时，webhook 会自动响应，以便你可以在飞书开发者控制台中完成订阅设置。

## 步骤 3：配置 Hermes

### 选项 A：交互式设置

```bash
hermes gateway setup
```

选择 **Feishu / Lark** 并填写提示。

### 选项 B：手动配置

将以下内容添加到 `~/.hermes/.env`：

```bash
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=secret_xxx
FEISHU_DOMAIN=feishu
FEISHU_CONNECTION_MODE=websocket

# 可选但强烈推荐
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
FEISHU_HOME_CHANNEL=oc_xxx
```

`FEISHU_DOMAIN` 接受：

- `feishu` 用于飞书中国
- `lark` 用于 Lark 国际版

## 步骤 4：启动 Gateway

```bash
hermes gateway
```

然后从飞书/Lark 向机器人发送消息，以确认连接已激活。

## 主聊天

在飞书/Lark 聊天中使用 `/set-home` 将其标记为主频道，用于 cron 作业结果和跨平台通知。

你也可以预配置它：

```bash
FEISHU_HOME_CHANNEL=oc_xxx
```

## 安全

### 用户白名单

对于生产使用，请设置飞书 Open ID 的白名单：

```bash
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
```

如果你将白名单留空，任何可以联系到机器人的人都可以使用它。在群聊中，会在处理消息之前根据发送者的 open_id 检查白名单。

### Webhook 加密密钥

在 webhook 模式下运行时，请设置加密密钥以启用入站 webhook 负载的签名验证：

```bash
FEISHU_ENCRYPT_KEY=your-encrypt-key
```

此密钥在飞书应用配置的 **事件订阅** 部分中找到。设置后，适配器使用签名算法验证每个 webhook 请求：

```
SHA256(timestamp + nonce + encrypt_key + body)
```

计算出的哈希值使用时序安全比较与 `x-lark-signature` 标头进行比较。签名无效或缺失的请求会被拒绝，并返回 HTTP 401。

:::tip
在 WebSocket 模式下，签名验证由 SDK 本身处理，因此 `FEISHU_ENCRYPT_KEY` 是可选的。在 webhook 模式下，强烈推荐用于生产环境。
:::

### 验证 Token

另一层身份验证，检查 webhook 负载内的 `token` 字段：

```bash
FEISHU_VERIFICATION_TOKEN=your-verification-token
```

此 token 也在飞书应用的 **事件订阅** 部分中找到。设置后，每个入站 webhook 负载必须在其 `header` 对象中包含匹配的 `token`。token 不匹配会被拒绝，并返回 HTTP 401。

`FEISHU_ENCRYPT_KEY` 和 `FEISHU_VERIFICATION_TOKEN` 都可以一起使用，以实现深度防御。

## 群消息策略

`FEISHU_GROUP_POLICY` 环境变量控制 Hermes 是否在群聊中回复以及如何回复：

```bash
FEISHU_GROUP_POLICY=allowlist   # 默认
```

| 值 | 行为 |
|-------|----------|
| `open` | Hermes 回复来自任何群中任何用户的 @提及。 |
| `allowlist` | Hermes 仅回复来自 `FEISHU_ALLOWED_USERS` 中列出的用户的 @提及。 |
| `disabled` | Hermes 完全忽略所有群消息。 |

在所有模式中，在处理消息之前，必须明确 @提及机器人（或 @all）。私信始终绕过此门控。

设置 `FEISHU_REQUIRE_MENTION=false` 以让 Hermes 读取所有群流量而无需 @提及：

```bash
FEISHU_REQUIRE_MENTION=false
```

对于每聊天的控制，请在 `group_rules` 条目上设置 `require_mention` —— 请参阅下面的 [每群访问控制](#每群访问控制)。

### 机器人身份

Hermes 在启动时自动检测机器人的 `open_id` 和显示名称。仅当自动检测无法访问飞书 API 时，或者当你的应用使用租户范围的用户 ID 时，才需要手动设置这些：

```bash
FEISHU_BOT_OPEN_ID=ou_xxx     # 仅当自动检测失败时
FEISHU_BOT_USER_ID=xxx        # 如果你的应用使用 sender_id_type=user_id，则必需
FEISHU_BOT_NAME=MyBot         # 仅当自动检测失败时
```

## 机器人间消息传递

默认情况下，Hermes 忽略其他机器人发送的消息。当你希望 Hermes 参与 A2A 编排或从同一群中的其他机器人接收通知时，启用机器人间消息传递。

```bash
FEISHU_ALLOW_BOTS=mentions   # 默认：none
```

| 值 | 行为 |
|-------|----------|
| `none` | 忽略来自其他机器人的所有消息（默认）。 |
| `mentions` | 仅当对等机器人 @提及 Hermes 时才接受。 |
| `all` | 接受每个对等机器人的消息。 |

也可在 `config.yaml` 中配置为 `feishu.allow_bots`（两者都设置时环境变量优先）。

对等机器人不需要添加到 `FEISHU_ALLOWED_USERS` —— 该白名单仅适用于人类发送者。

授予 `application:bot.basic_info:read` 作用域以显示对等机器人名称；没有它，对等机器人仍然可以正确路由，但显示为其 `open_id`。

## 交互式卡片操作

当用户点击机器人发送的交互式卡片上的按钮或与交互式卡片交互时，适配器会将这些操作作为合成的 `/card` 命令事件进行路由：

- 按钮点击变为：`/card button {"key": "value", ...}`
- 卡片定义中的操作的 `value` 负载作为 JSON 包含在内。
- 卡片操作在 15 分钟窗口内去重，以防止双重处理。

卡片操作事件随 `MessageType.COMMAND` 一起分派，因此它们通过正常的命令处理管道流动。

这也是**命令批准**的工作方式 —— 当 agent 需要运行危险命令时，它会发送一个带有"允许一次"/"会话"/"始终"/"拒绝"按钮的交互式卡片。用户点击按钮，卡片操作回调将批准决定传递回 agent。

### 必需的飞书应用配置

交互式卡片需要在飞书开发者控制台中完成**三**个配置步骤。缺少任何一个都会导致用户点击卡片按钮时出现错误 **200340**。

1. **订阅卡片操作事件：**
   在 **事件订阅** 中，将 `card.action.trigger` 添加到你的订阅事件。

2. **启用交互式卡片能力：**
   在 **应用功能 > 机器人** 中，确保 **交互式卡片** 开关已启用。这告诉飞书你的应用可以接收卡片操作回调。

3. **配置卡片请求 URL（仅 webhook 模式）：**
   在 **应用功能 > 机器人 > 消息卡片请求 URL** 中，将 URL 设置为与你的事件 webhook 相同的端点（例如 `https://your-server:8765/feishu/webhook`）。在 WebSocket 模式下，这由 SDK 自动处理。

:::warning
如果没有全部三个步骤，飞书将成功*发送*交互式卡片（发送仅需要 `im:message:send` 权限），但点击任何按钮都将返回错误 200340。卡片看起来可以工作 —— 错误仅在用户与之交互时才会出现。
:::

## 文档评论智能回复

除了聊天，适配器还可以回答留在**飞书/Lark 文档**上的 `@` 提及。当用户对文档（本地文本选择或整篇文档评论）发表评论并 @提及机器人时，Hermes 会读取文档以及周围的评论线程，并在线程上内联发布 LLM 回复。

由 `drive.notice.comment_add_v1` 事件驱动，处理程序：

- 并行获取文档内容和评论时间线（整篇文档线程 20 条消息，本地选择线程 12 条）。
- 使用限定在该单个评论会话的 `feishu_doc` + `feishu_drive` 工具集运行 agent。
- 在 4000 个字符处截断回复，并将其作为线程回复发布回去。
- 在每个文档会话中缓存 1 小时，上限为 50 条消息，因此同一文档上的后续评论会保留上下文。

### 3 层访问控制

文档评论回复是**显式授予仅** —— 没有隐式允许所有模式。权限按此顺序解析（先匹配获胜，每个字段）：

1. **精确文档** —— 限定到特定文档 token 的规则。
2. **通配符** —— 匹配文档模式的规则。
3. **顶层** —— 工作区的默认规则。

每个规则有两个可用策略：

- **`allowlist`** —— 用户/租户的静态列表。
- **`pairing`** —— 静态列表 ∪ 运行时批准的存储。对于审核人员可以实时授予访问权限的推广很有用。

规则位于 `~/.hermes/feishu_comment_rules.json`（`~/.hermes/feishu_comment_pairing.json` 中的配对授予），带有 mtime 缓存的热重载 —— 编辑在下一个评论事件时生效，无需重启 gateway。

CLI：

```bash
# 检查当前规则和配对状态
python -m gateway.platforms.feishu_comment_rules status

# 模拟对特定文档 + 用户的访问检查
python -m gateway.platforms.feishu_comment_rules check <fileType:fileToken> <user_open_id>

# 在运行时管理配对授予
python -m gateway.platforms.feishu_comment_rules pairing list
python -m gateway.platforms.feishu_comment_rules pairing add <user_open_id>
python -m gateway.platforms.feishu_comment_rules pairing remove <user_open_id>
```

### 必需的飞书应用配置

除了已授予的聊天/卡片权限之外，还要添加驱动器评论事件：

- 在 **事件订阅** 中订阅 `drive.notice.comment_add_v1`。
- 授予 `docs:doc:readonly` 和 `drive:drive:readonly` 作用域，以便处理程序可以读取文档内容。

## 媒体支持

### 入站（接收）

适配器接收并缓存来自用户的以下媒体类型：

| 类型 | 扩展名 | 处理方式 |
|------|-----------|-------------------|
| **图片** | .jpg、.jpeg、.png、.gif、.webp、.bmp | 通过飞书 API 下载并在本地缓存 |
| **音频** | .ogg、.mp3、.wav、.m4a、.aac、.flac、.opus、.webm | 下载并缓存；小型文本文件会自动提取 |
| **视频** | .mp4、.mov、.avi、.mkv、.webm、.m4v、.3gp | 下载并作为文档缓存 |
| **文件** | .pdf、.doc、.docx、.xls、.xlsx、.ppt、.pptx 等 | 下载并作为文档缓存 |

来自富文本（帖子）消息的媒体，包括内联图片和文件附件，也会被提取和缓存。

对于小型基于文本的文档（.txt、.md），文件内容会自动注入到消息文本中，以便 agent 可以直接读取它，而无需工具。

### 出站（发送）

| 方法 | 发送内容 |
|--------|--------------|
| `send` | 文本或富帖子消息（根据 markdown 内容自动检测） |
| `send_image` / `send_image_file` | 上传图片到飞书，然后作为原生图片气泡发送（带有可选标题） |
| `send_document` | 上传文件到飞书 API，然后作为文件附件发送 |
| `send_voice` | 将音频文件作为飞书文件附件上传 |
| `send_video` | 上传视频并作为原生媒体消息发送 |
| `send_animation` | GIF 会降级为文件附件（飞书没有原生 GIF 气泡） |

文件上传路由根据扩展名自动进行：

- `.ogg`、`.opus` → 作为 `opus` 音频上传
- `.mp4`、`.mov`、`.avi`、`.m4v` → 作为 `mp4` 媒体上传
- `.pdf`、`.doc(x)`、`.xls(x)`、`.ppt(x)` → 使用其文档类型上传
- 其他所有内容 → 作为通用流文件上传

## Markdown 渲染和帖子回退

当出站文本包含 markdown 格式（标题、粗体、列表、代码块、链接等）时，适配器会自动将其作为带有嵌入式 `md` 标签的飞书**帖子**消息发送，而不是纯文本。这样可以在飞书客户端中实现富渲染。

如果飞书 API 拒绝帖子负载（例如，由于不支持的 markdown 构造），适配器会自动回退为以去除 markdown 的纯文本发送。这种两阶段回退确保消息始终被送达。

纯文本消息（未检测到 markdown）作为简单的 `text` 消息类型发送。

## 处理状态反应

当 agent 正在工作时，机器人会在你的消息上显示 `Typing` 反应。它会在回复到达时被清除，如果处理失败，则替换为 `CrossMark`。

设置 `FEISHU_REACTIONS=false` 以关闭它。

## 突发保护和批处理

适配器包括对快速消息突发的去抖，以避免压垮 agent：

### 文本批处理

当用户快速连续发送多条文本消息时，它们会在被分派之前合并为单个事件：

| 设置 | 环境变量 | 默认 |
|---------|---------|---------|
| 静默期 | `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | 0.6 秒 |
| 每批最大消息数 | `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | 8 |
| 每批最大字符数 | `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | 4000 |

### 媒体批处理

快速连续发送的多个媒体附件（例如，拖动几张图片）会合并为单个事件：

| 设置 | 环境变量 | 默认 |
|---------|---------|---------|
| 静默期 | `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | 0.8 秒 |

### 每聊天序列化

同一聊天中的消息被串行处理（一次一个），以保持对话连贯性。每个聊天都有自己的锁，因此不同聊天中的消息会并发处理。

## 速率限制（Webhook 模式）

在 webhook 模式下，适配器强制执行每 IP 速率限制以防止滥用：

- **窗口：** 60 秒滑动窗口
- **限制：** 每个窗口每个（app_id、路径、IP）三元组 120 个请求
- **跟踪上限：** 最多跟踪 4096 个唯一键（防止无限制的内存增长）

超过限制的请求会收到 HTTP 429（请求过多）。

### Webhook 异常跟踪

适配器跟踪每个 IP 地址的连续错误响应。在同一 6 小时窗口内来自同一 IP 的 25 个连续错误后，会记录警告。这有助于检测配置错误的客户端或探测尝试。

其他 webhook 保护：

- **正文大小限制：** 最大 1 MB
- **正文读取超时：** 30 秒
- **内容类型强制：** 仅接受 `application/json`

## WebSocket 调优

使用 `websocket` 模式时，你可以自定义重新连接和 ping 行为：

```yaml
platforms:
  feishu:
    extra:
      ws_reconnect_interval: 120   # 重新连接尝试之间的秒数（默认：120）
      ws_ping_interval: 30         # WebSocket ping 之间的秒数（可选；如果未设置，则为 SDK 默认）
```

| 设置 | 配置键 | 默认 | 描述 |
|---------|-----------|---------|-------------|
| 重新连接间隔 | `ws_reconnect_interval` | 120 秒 | 重新连接尝试之间等待的时间 |
| Ping 间隔 | `ws_ping_interval` | _(SDK 默认)_ | WebSocket 保活 ping 的频率 |

## 每群访问控制

除了全局 `FEISHU_GROUP_POLICY` 之外，你还可以使用 `config.yaml` 中的 `group_rules` 为每个群聊天设置细粒度规则：

```yaml
platforms:
  feishu:
    extra:
      default_group_policy: "open"     # 不在 group_rules 中的群的默认设置
      admins:                          # 可以管理机器人设置的用户
        - "ou_admin_open_id"
      group_rules:
        "oc_group_chat_id_1":
          policy: "allowlist"          # open | allowlist | blacklist | admin_only | disabled
          allowlist:
            - "ou_user_open_id_1"
            - "ou_user_open_id_2"
        "oc_group_chat_id_2":
          policy: "admin_only"
        "oc_group_chat_id_3":
          policy: "blacklist"
          blacklist:
            - "ou_blocked_user"
        "oc_free_chat":
          policy: "open"
          require_mention: false       # 为此聊天覆盖 FEISHU_REQUIRE_MENTION
```

| 策略 | 描述 |
|--------|-------------|
| `open` | 群中的任何人都可以使用机器人 |
| `allowlist` | 只有群中 `allowlist` 中的用户可以使用机器人 |
| `blacklist` | 除了群中 `blacklist` 中的用户外，任何人都可以使用机器人 |
| `admin_only` | 只有全局 `admins` 列表中的用户可以在此群中使用机器人 |
| `disabled` | 机器人忽略此群中的所有消息 |

在 `group_rules` 条目上设置 `require_mention: false` 以跳过该特定聊天的 @提及要求。如果省略，聊天会继承全局 `FEISHU_REQUIRE_MENTION` 值。

未在 `group_rules` 中列出的群会回退到 `default_group_policy`（默认为 `FEISHU_GROUP_POLICY` 的值）。

## 去重

入站消息使用消息 ID 和 24 小时 TTL 进行去重。去重状态会在重启后持久化到 `~/.hermes/feishu_seen_message_ids.json`。

| 设置 | 环境变量 | 默认 |
|---------|---------|---------|
| 缓存大小 | `HERMES_FEISHU_DEDUP_CACHE_SIZE` | 2048 个条目 |

## 所有环境变量

| 变量 | 必需 | 默认 | 描述 |
|----------|----------|---------|-------------|
| `FEISHU_APP_ID` | ✅ | — | 飞书/Lark 应用 ID |
| `FEISHU_APP_SECRET` | ✅ | — | 飞书/Lark 应用 Secret |
| `FEISHU_DOMAIN` | — | `feishu` | `feishu`（中国）或 `lark`（国际） |
| `FEISHU_CONNECTION_MODE` | — | `websocket` | `websocket` 或 `webhook` |
| `FEISHU_ALLOWED_USERS` | — | _(空)_ | 用于用户白名单的逗号分隔的 open_id 列表 |
| `FEISHU_ALLOW_BOTS` | — | `none` | 接受来自其他机器人的消息：`none`、`mentions` 或 `all` |
| `FEISHU_REQUIRE_MENTION` | — | `true` | 群消息是否必须 @提及机器人 |
| `FEISHU_HOME_CHANNEL` | — | — | 用于 cron/通知输出的聊天 ID |
| `FEISHU_ENCRYPT_KEY` | — | _(空)_ | 用于 webhook 签名验证的加密密钥 |
| `FEISHU_VERIFICATION_TOKEN` | — | _(空)_ | 用于 webhook 负载身份验证的验证 token |
| `FEISHU_GROUP_POLICY` | — | `allowlist` | 群消息策略：`open`、`allowlist`、`disabled` |
| `FEISHU_BOT_OPEN_ID` | — | _(空)_ | 机器人的 open_id（用于 @提及检测） |
| `FEISHU_BOT_USER_ID` | — | _(空)_ | 机器人的 user_id（用于 @提及检测） |
| `FEISHU_BOT_NAME` | — | _(空)_ | 机器人的显示名称（用于 @提及检测） |
| `FEISHU_WEBHOOK_HOST` | — | `127.0.0.1` | Webhook 服务器绑定地址 |
| `FEISHU_WEBHOOK_PORT` | — | `8765` | Webhook 服务器端口 |
| `FEISHU_WEBHOOK_PATH` | — | `/feishu/webhook` | Webhook 端点路径 |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | — | `2048` | 要跟踪的最大去重消息 ID 数 |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | — | `0.6` | 文本突发去抖静默期 |
| `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | — | `8` | 每个文本批次合并的最大消息数 |
| `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | — | `4000` | 每个文本批次合并的最大字符数 |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | — | `0.8` | 媒体突发去抖静默期 |

WebSocket 和每群 ACL 设置通过 `config.yaml` 在 `platforms.feishu.extra` 下配置（请参阅上面的 [WebSocket 调优](#websocket-调优) 和 [每群访问控制](#每群访问控制)）。

## 故障排除

| 问题 | 修复 |
|---------|-----|
| `lark-oapi not installed` | 安装 SDK：`pip install lark-oapi` |
| `websockets not installed; websocket mode unavailable` | 安装 websockets：`pip install websockets` |
| `aiohttp not installed; webhook mode unavailable` | 安装 aiohttp：`pip install aiohttp` |
| `FEISHU_APP_ID or FEISHU_APP_SECRET not set` | 设置两个环境变量或通过 `hermes gateway setup` 配置 |
| `Another local Hermes gateway is already using this Feishu app_id` | 一次只有一个 Hermes 实例可以使用相同的 app_id。首先停止其他 gateway。 |
| 机器人在群中无响应 | 确保机器人被 @提及，检查 `FEISHU_GROUP_POLICY`，并验证发送者在策略为 `allowlist` 时是否在 `FEISHU_ALLOWED_USERS` 中 |
| `Webhook rejected: invalid verification token` | 确保 `FEISHU_VERIFICATION_TOKEN` 与你的飞书应用的事件订阅配置中的 token 匹配 |
| `Webhook rejected: invalid signature` | 确保 `FEISHU_ENCRYPT_KEY` 与你的飞书应用配置中的加密密钥匹配 |
| 帖子消息显示为纯文本 | 飞书 API 拒绝了帖子负载；这是正常的回退行为。查看日志了解详情。 |
| 机器人未收到图片/文件 | 授予 `im:message` 和 `im:resource` 权限作用域到你的飞书应用 |
| 机器人身份未自动检测 | 通常是访问飞书机器人信息端点的暂时性网络问题。将 `FEISHU_BOT_OPEN_ID` 和 `FEISHU_BOT_NAME` 手动设置为解决方法。 |
| 启用 `FEISHU_ALLOW_BOTS` 后仍忽略对等机器人消息 | Hermes 还无法识别自己 —— 设置 `FEISHU_BOT_OPEN_ID`（如果你的应用使用 `sender_id_type=user_id`，则还需要 `FEISHU_BOT_USER_ID`）。 |
| 对等机器人显示为 `ou_xxxxxx` 而不是名称 | 授予 `application:bot.basic_info:read` 权限作用域。 |
| 点击批准按钮时出现错误 200340 | 在飞书开发者控制台中启用**交互式卡片**能力并配置**卡片请求 URL**。请参阅上面的 [必需的飞书应用配置](#必需的飞书应用配置)。 |
| `Webhook rate limit exceeded` | 来自同一 IP 的每分钟超过 120 个请求。这通常是配置错误或循环。 |

## 工具集

飞书 / Lark 使用 `hermes-feishu` 平台预设，其中包括与 Telegram 和其他基于 gateway 的消息传递平台相同的核心工具。
