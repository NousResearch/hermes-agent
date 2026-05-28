---
sidebar_position: 11
title: "飞书 / Lark"
description: "将 Hermes Agent 配置为飞书或 Lark 机器人"
---

# 飞书 / Lark 配置

Hermes Agent 可作为全功能机器人与飞书和 Lark 集成。连接后，你可以在私信或群聊中与 Agent 对话，在 home chat 中接收 cron job 结果，并通过标准 gateway 流程发送文本、图片、音频和文件附件。

该集成支持两种连接模式：

- `websocket` — 推荐；Hermes 主动建立出站连接，无需公开 webhook 端点
- `webhook` — 适用于已将 Hermes 部署在可访问 HTTP 端点后的场景

## Hermes 的行为方式

| 场景 | 行为 |
|---------|----------|
| 私信 | Hermes 回复每一条消息。 |
| 群聊 | Hermes 仅在被 @提及 时回复。 |
| 共享群聊 | 默认情况下，每位用户在共享群聊中的会话历史相互隔离。 |

共享群聊行为由 `config.yaml` 控制：

```yaml
group_sessions_per_user: true
```

仅当你明确希望每个群聊共享同一个对话时，才将其设为 `false`。

## 第一步：创建飞书 / Lark 应用

### 推荐：扫码创建（一条命令）

```bash
hermes gateway setup
```

选择 **飞书 / Lark**，用飞书或 Lark 手机端扫描二维码。Hermes 将自动创建具有正确权限的机器人应用并保存凭据。

### 备选：手动配置

如果扫码创建不可用，向导将回退到手动输入：

1. 打开飞书或 Lark 开发者控制台：
   - 飞书：[https://open.feishu.cn/](https://open.feishu.cn/)
   - Lark：[https://open.larksuite.com/](https://open.larksuite.com/)
2. 创建新应用。
3. 在 **凭证与基础信息** 中，复制 **App ID** 和 **App Secret**。
4. 为应用开启 **机器人** 能力。
5. 运行 `hermes gateway setup`，选择 **飞书 / Lark**，并在提示时输入凭据。

:::warning
请妥善保管 App Secret。任何持有它的人都可以冒充你的应用。
:::

## 第二步：选择连接模式

### 推荐：WebSocket 模式

当 Hermes 运行在你的笔记本、工作站或私有服务器上时，使用 WebSocket 模式。无需公开 URL。官方 Lark SDK 会建立并维护一个持久的出站 WebSocket 连接，并支持自动重连。

```bash
FEISHU_CONNECTION_MODE=websocket
```

**依赖：** 必须安装 `websockets` Python 包。SDK 在内部处理连接生命周期、心跳和自动重连。

**工作原理：** 适配器在后台 executor 线程中运行 Lark SDK 的 WebSocket 客户端。入站事件（消息、表情回应、卡片操作）被分发到主 asyncio 循环。断开连接时，SDK 将自动尝试重连。

### 可选：Webhook 模式

仅当 Hermes 已部署在可访问的 HTTP 端点后时，才使用 webhook 模式。

```bash
FEISHU_CONNECTION_MODE=webhook
```

在 webhook 模式下，Hermes 启动一个 HTTP 服务器（通过 `aiohttp`），并在以下路径提供飞书端点：

```text
/feishu/webhook
```

**依赖：** 必须安装 `aiohttp` Python 包。

你可以自定义 webhook 服务器的绑定地址和路径：

```bash
FEISHU_WEBHOOK_HOST=127.0.0.1   # 默认：127.0.0.1
FEISHU_WEBHOOK_PORT=8765         # 默认：8765
FEISHU_WEBHOOK_PATH=/feishu/webhook  # 默认：/feishu/webhook
```

当飞书发送 URL 验证挑战（`type: url_verification`）时，webhook 会自动响应，以便你在飞书开发者控制台完成订阅配置。当设置了 `FEISHU_VERIFICATION_TOKEN` 时，挑战响应会进行 token 校验——token 缺失或不匹配的挑战请求将被拒绝，防止未经认证的远端通过回显攻击者控制的挑战数据来证明端点控制权。

## 第三步：配置 Hermes

### 方式 A：交互式配置

```bash
hermes gateway setup
```

选择 **飞书 / Lark** 并填写提示信息。

### 方式 B：手动配置

在 `~/.hermes/.env` 中添加以下内容：

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

- `feishu` 对应飞书（中国）
- `lark` 对应 Lark（国际版）

## 第四步：启动 Gateway

```bash
hermes gateway
```

然后从飞书/Lark 向机器人发送消息，确认连接已建立。

## Home Chat

在飞书/Lark 聊天中使用 `/set-home` 将其标记为 cron job 结果和跨平台通知的 home channel。

也可以预先配置：

```bash
FEISHU_HOME_CHANNEL=oc_xxx
```

## 安全

### 用户白名单

在生产环境中，请设置飞书 Open ID 白名单：

```bash
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
```

如果白名单为空，任何能访问机器人的人都可能使用它。在群聊中，消息处理前会根据发送者的 open_id 检查白名单。

如需允许特定群聊，但不开放陌生用户私信，可设置群聊 ID 白名单：

```bash
FEISHU_GROUP_ALLOWED_CHATS=oc_xxx,oc_yyy
```

这会允许来自所列群聊的消息；私信访问仍由 `FEISHU_ALLOWED_USERS`、配对机制或 `FEISHU_ALLOW_ALL_USERS` 控制。

### Webhook 加密密钥

在 webhook 模式下运行时，设置加密密钥以启用入站 webhook payload 的签名验证：

```bash
FEISHU_ENCRYPT_KEY=your-encrypt-key
```

该密钥可在飞书应用配置的 **事件订阅** 部分找到。设置后，适配器使用以下签名算法验证每个 webhook 请求：

```
SHA256(timestamp + nonce + encrypt_key + body)
```

计算出的哈希值与 `x-lark-signature` 请求头进行时序安全比较。签名无效或缺失的请求将被拒绝，返回 HTTP 401。

:::tip
在 WebSocket 模式下，签名验证由 SDK 自身处理，因此 `FEISHU_ENCRYPT_KEY` 是可选的。在 webhook 模式下，生产环境强烈推荐设置。
:::

### 验证 Token

对 webhook payload 中 `token` 字段进行检查的额外认证层：

```bash
FEISHU_VERIFICATION_TOKEN=your-verification-token
```

该 token 同样可在飞书应用的 **事件订阅** 部分找到。设置后，每个入站 webhook payload 的 `header` 对象中必须包含匹配的 `token`。token 不匹配的请求将被拒绝，返回 HTTP 401。

`FEISHU_ENCRYPT_KEY` 和 `FEISHU_VERIFICATION_TOKEN` 可同时使用，实现纵深防御。

## 群消息策略

`FEISHU_GROUP_POLICY` 环境变量控制 Hermes 是否以及如何在群聊中响应：

```bash
FEISHU_GROUP_POLICY=allowlist   # 默认
```

| 值 | 行为 |
|-------|----------|
| `open` | Hermes 响应任意群中任意用户的 @提及。 |
| `allowlist` | Hermes 仅响应 `FEISHU_ALLOWED_USERS` 中列出的用户的 @提及。 |
| `disabled` | Hermes 完全忽略所有群消息。 |

在所有模式下，消息处理前机器人必须被明确 @提及（或 @all）。私信始终绕过此限制。

设置 `FEISHU_REQUIRE_MENTION=false` 可让 Hermes 读取所有群消息而无需 @提及：

```bash
FEISHU_REQUIRE_MENTION=false
```

如需按群控制，在 `group_rules` 条目中设置 `require_mention`——参见下方[按群访问控制](#per-group-access-control)。

### 机器人身份

Hermes 在启动时自动检测机器人的 `open_id` 和显示名称。仅当自动检测无法访问飞书 API，或你的应用使用租户范围用户 ID 时，才需要手动设置：

```bash
FEISHU_BOT_OPEN_ID=ou_xxx     # 仅在自动检测失败时使用
FEISHU_BOT_USER_ID=xxx        # 若应用使用 sender_id_type=user_id 则必填
FEISHU_BOT_NAME=MyBot         # 仅在自动检测失败时使用
```

## 机器人间消息传递

默认情况下，Hermes 忽略其他机器人发送的消息。当你希望 Hermes 参与 A2A 编排或接收同一群中其他机器人的通知时，可启用机器人间消息传递。

```bash
FEISHU_ALLOW_BOTS=mentions   # 默认：none
```

| 值 | 行为 |
|-------|----------|
| `none` | 忽略所有其他机器人的消息（默认）。 |
| `mentions` | 仅当对端机器人 @提及 Hermes 时接受。 |
| `all` | 接受所有对端机器人消息。 |

也可在 `config.yaml` 中配置为 `feishu.allow_bots`（两者同时设置时，环境变量优先）。

对端机器人无需加入 `FEISHU_ALLOWED_USERS`——该白名单仅适用于人类发送者。

授予 `application:bot.basic_info:read` 权限范围可显示对端机器人名称；未授权时，对端机器人仍可正常路由，但显示为其 `open_id`。

## 交互式卡片操作

当用户点击机器人发送的交互式卡片上的按钮或与其交互时，适配器将这些操作路由为合成的 `/card` 命令事件：

- 按钮点击变为：`/card button {"key": "value", ...}`
- 卡片定义中操作的 `value` payload 以 JSON 形式包含在内。
- 卡片操作在 15 分钟窗口内去重，防止重复处理。

Gateway 驱动的更新提示使用原生飞书 `Yes` / `No` 卡片，而非回退到纯文本回复。当 `hermes update --gateway` 需要确认时，适配器将所选答案记录到 Hermes 的 `.update_response` 文件中，并将卡片内联替换为已解决状态。

卡片操作事件以 `MessageType.COMMAND` 分发，因此流经标准命令处理管道。

**命令审批**也通过此机制实现——当 Agent 需要执行危险命令时，会发送一张带有「允许一次 / 本次会话 / 始终允许 / 拒绝」按钮的交互式卡片。用户点击按钮后，卡片操作回调将审批决定传回 Agent。

### 飞书应用所需配置

交互式卡片需要在飞书开发者控制台完成**三项**配置。缺少任何一项，用户点击卡片按钮时将出现错误 **200340**。

1. **订阅卡片操作事件：**
   在 **事件订阅** 中，将 `card.action.trigger` 添加到已订阅事件。

2. **启用交互式卡片能力：**
   在 **应用功能 > 机器人** 中，确保 **交互式卡片** 开关已启用。这告知飞书你的应用可以接收卡片操作回调。

3. **配置卡片请求 URL（仅 webhook 模式）：**
   在 **应用功能 > 机器人 > 消息卡片请求网址** 中，将 URL 设置为与事件 webhook 相同的端点（例如 `https://your-server:8765/feishu/webhook`）。WebSocket 模式下，SDK 会自动处理此项。

:::warning
缺少以上任意一步，飞书将成功*发送*交互式卡片（发送仅需 `im:message:send` 权限），但点击任意按钮将返回错误 200340。卡片看起来正常——错误仅在用户与其交互时才会出现。
:::

## 文档评论智能回复

除聊天外，适配器还可以回复**飞书/Lark 文档**中的 `@` 提及。当用户在文档中评论（局部文本选区或全文评论）并 @提及机器人时，Hermes 读取文档内容及周围的评论线程，并在线程中内联发布 LLM 回复。

由 `drive.notice.comment_add_v1` 事件驱动，处理器：

- 并行获取文档内容和评论时间线（全文线程取 20 条消息，局部选区线程取 12 条）。
- 以 `feishu_doc` + `feishu_drive` 工具集运行 Agent，范围限定于该单次评论会话。
- 每 4000 字符分块，以线程回复形式发布。
- 按文档缓存会话，有效期 1 小时，上限 50 条消息，使同一文档的后续评论保持上下文。

### 三级访问控制

文档评论回复为**显式授权模式**——不存在隐式全员允许模式。权限按以下顺序解析（每个字段取第一个匹配项）：

1. **精确文档** — 限定于特定文档 token 的规则。
2. **通配符** — 匹配文档模式的规则。
3. **顶层** — 工作区的默认规则。

每条规则支持两种策略：

- **`allowlist`** — 静态用户/租户列表。
- **`pairing`** — 静态列表 ∪ 运行时审批存储。适用于管理员可实时授权的灰度发布场景。

规则存储在 `~/.hermes/feishu_comment_rules.json`（pairing 授权存储在 `~/.hermes/feishu_comment_pairing.json`），支持基于 mtime 缓存的热重载——编辑后无需重启 gateway，下一个评论事件即生效。

CLI：

```bash
# 查看当前规则和 pairing 状态
python -m gateway.platforms.feishu_comment_rules status

# 模拟特定文档 + 用户的访问检查
python -m gateway.platforms.feishu_comment_rules check <fileType:fileToken> <user_open_id>

# 运行时管理 pairing 授权
python -m gateway.platforms.feishu_comment_rules pairing list
python -m gateway.platforms.feishu_comment_rules pairing add <user_open_id>
python -m gateway.platforms.feishu_comment_rules pairing remove <user_open_id>
```

### 飞书应用所需配置

在已授予的聊天/卡片权限基础上，添加文档评论事件：

- 在 **事件订阅** 中订阅 `drive.notice.comment_add_v1`。
- 授予 `docs:doc:readonly` 和 `drive:drive:readonly` 权限范围，以便处理器读取文档内容。

## 媒体支持

### 入站（接收）

适配器接收并缓存以下来自用户的媒体类型：

| 类型 | 扩展名 | 处理方式 |
|------|-----------|-------------------|
| **图片** | .jpg, .jpeg, .png, .gif, .webp, .bmp | 通过飞书 API 下载并本地缓存 |
| **音频** | .ogg, .mp3, .wav, .m4a, .aac, .flac, .opus, .webm | 下载并缓存；小型文本文件自动提取内容 |
| **视频** | .mp4, .mov, .avi, .mkv, .webm, .m4v, .3gp | 下载并作为文档缓存 |
| **文件** | .pdf, .doc, .docx, .xls, .xlsx, .ppt, .pptx 等 | 下载并作为文档缓存 |

富文本（post）消息中的媒体，包括内联图片和文件附件，也会被提取并缓存。

对于小型文本文档（.txt, .md），文件内容会自动注入消息文本，使 Agent 无需工具即可直接读取。

### 出站（发送）

| 方法 | 发送内容 |
|--------|--------------|
| `send` | 文本或富文本 post 消息（根据 markdown 内容自动检测） |
| `send_image` / `send_image_file` | 上传图片到飞书，然后以原生图片气泡发送（可附带说明文字） |
| `send_document` | 上传文件到飞书 API，然后以文件附件发送 |
| `send_voice` | 以飞书文件附件形式上传音频文件 |
| `send_video` | 上传视频并以原生媒体消息发送 |
| `send_animation` | GIF 降级为文件附件（飞书不支持原生 GIF 气泡） |

文件上传路由根据扩展名自动判断：

- `.ogg`, `.opus` → 以 `opus` 音频上传
- `.mp4`, `.mov`, `.avi`, `.m4v` → 以 `mp4` 媒体上传
- `.pdf`, `.doc(x)`, `.xls(x)`, `.ppt(x)` → 以对应文档类型上传
- 其他所有格式 → 以通用流文件上传

## Markdown 渲染与 Post 回退

当出站文本包含 markdown 格式（标题、加粗、列表、代码块、链接等）时，适配器自动将其以飞书 **post** 消息形式发送，并嵌入 `md` 标签，而非纯文本。这使飞书客户端能够富文本渲染。

如果飞书 API 拒绝 post payload（例如因不支持的 markdown 语法），适配器自动回退为发送去除 markdown 的纯文本。这种两阶段回退确保消息始终能送达。

纯文本消息（未检测到 markdown）以简单的 `text` 消息类型发送。

## 处理状态表情回应

Agent 工作期间，机器人会在你的消息上显示 `Typing` 表情回应。回复到达后清除，处理失败则替换为 `CrossMark`。

设置 `FEISHU_REACTIONS=false` 可关闭此功能。

## 突发保护与批处理

适配器对快速消息突发进行防抖处理，避免压垮 Agent：

### 文本批处理

当用户快速连续发送多条文本消息时，它们会在分发前合并为单个事件：

| 设置 | 环境变量 | 默认值 |
|---------|---------|---------|
| 静默期 | `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | 0.6s |
| 每批最大消息数 | `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | 8 |
| 每批最大字符数 | `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | 4000 |

### 媒体批处理

快速连续发送的多个媒体附件（例如拖拽多张图片）会合并为单个事件：

| 设置 | 环境变量 | 默认值 |
|---------|---------|---------|
| 静默期 | `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | 0.8s |

### 按聊天串行化

同一聊天中的消息串行处理（每次一条），以保持对话连贯性。每个聊天有独立的锁，不同聊天的消息并发处理。

## 速率限制（Webhook 模式）

在 webhook 模式下，适配器对每个 IP 强制执行速率限制，防止滥用：

- **窗口：** 60 秒滑动窗口
- **限制：** 每个（app_id, path, IP）三元组每窗口 120 次请求
- **追踪上限：** 最多追踪 4096 个唯一键（防止内存无限增长）

超出限制的请求将收到 HTTP 429（请求过多）。

### Webhook 异常追踪

适配器追踪每个 IP 地址的连续错误响应。同一 IP 在 6 小时窗口内连续出现 25 次错误后，将记录警告日志。这有助于检测配置错误的客户端或探测行为。

额外的 webhook 保护措施：
- **请求体大小限制：** 最大 1 MB
- **请求体读取超时：** 30 秒
- **Content-Type 强制：** 仅接受 `application/json`

## WebSocket 调优

使用 `websocket` 模式时，可自定义重连和 ping 行为：

```yaml
platforms:
  feishu:
    extra:
      ws_reconnect_interval: 120   # 重连尝试间隔秒数（默认：120）
      ws_ping_interval: 30         # WebSocket ping 间隔秒数（可选；未设置时使用 SDK 默认值）
```

| 设置 | 配置键 | 默认值 | 说明 |
|---------|-----------|---------|-------------|
| 重连间隔 | `ws_reconnect_interval` | 120s | 两次重连尝试之间的等待时间 |
| Ping 间隔 | `ws_ping_interval` | _（SDK 默认）_ | WebSocket 保活 ping 的频率 |

## 按群访问控制

除全局 `FEISHU_GROUP_POLICY` 外，还可在 config.yaml 的 `group_rules` 中为每个群聊设置细粒度规则：

```yaml
platforms:
  feishu:
    extra:
      default_group_policy: "open"     # 未在 group_rules 中列出的群的默认策略
      admins:                          # 可管理机器人设置的用户
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
          require_mention: false       # 覆盖此聊天的 FEISHU_REQUIRE_MENTION
```

| 策略 | 说明 |
|--------|-------------|
| `open` | 群内任何人均可使用机器人 |
| `allowlist` | 仅群 `allowlist` 中的用户可使用机器人 |
| `blacklist` | 除群 `blacklist` 中的用户外，所有人均可使用机器人 |
| `admin_only` | 仅全局 `admins` 列表中的用户可在此群使用机器人 |
| `disabled` | 机器人忽略此群的所有消息 |

在 `group_rules` 条目中设置 `require_mention: false` 可跳过该特定聊天的 @提及要求。省略时，该聊天继承全局 `FEISHU_REQUIRE_MENTION` 值。

未在 `group_rules` 中列出的群回退到 `default_group_policy`（默认为 `FEISHU_GROUP_POLICY` 的值）。

## 去重

入站消息使用消息 ID 去重，TTL 为 24 小时。去重状态持久化到 `~/.hermes/feishu_seen_message_ids.json`，重启后仍有效。

| 设置 | 环境变量 | 默认值 |
|---------|---------|---------|
| 缓存大小 | `HERMES_FEISHU_DEDUP_CACHE_SIZE` | 2048 条 |

## 所有环境变量

| 变量 | 必填 | 默认值 | 说明 |
|----------|----------|---------|-------------|
| `FEISHU_APP_ID` | ✅ | — | 飞书/Lark App ID |
| `FEISHU_APP_SECRET` | ✅ | — | 飞书/Lark App Secret |
| `FEISHU_DOMAIN` | — | `feishu` | `feishu`（中国）或 `lark`（国际版） |
| `FEISHU_CONNECTION_MODE` | — | `websocket` | `websocket` 或 `webhook` |
| `FEISHU_ALLOWED_USERS` | — | _（空）_ | 用户白名单的逗号分隔 open_id 列表 |
| `FEISHU_GROUP_ALLOWED_CHATS` | — | _（空）_ | 飞书群聊 ID 白名单，逗号分隔 |
| `FEISHU_ALLOW_BOTS` | — | `none` | 接受其他机器人消息：`none`、`mentions` 或 `all` |
| `FEISHU_REQUIRE_MENTION` | — | `true` | 群消息是否必须 @提及 机器人 |
| `FEISHU_HOME_CHANNEL` | — | — | cron/通知输出的聊天 ID |
| `FEISHU_ENCRYPT_KEY` | — | _（空）_ | webhook 签名验证的加密密钥 |
| `FEISHU_VERIFICATION_TOKEN` | — | _（空）_ | webhook payload 认证的验证 token |
| `FEISHU_GROUP_POLICY` | — | `allowlist` | 群消息策略：`open`、`allowlist`、`disabled` |
| `FEISHU_BOT_OPEN_ID` | — | _（空）_ | 机器人的 open_id（用于 @提及 检测） |
| `FEISHU_BOT_USER_ID` | — | _（空）_ | 机器人的 user_id（用于 @提及 检测） |
| `FEISHU_BOT_NAME` | — | _（空）_ | 机器人的显示名称（用于 @提及 检测） |
| `FEISHU_WEBHOOK_HOST` | — | `127.0.0.1` | Webhook 服务器绑定地址 |
| `FEISHU_WEBHOOK_PORT` | — | `8765` | Webhook 服务器端口 |
| `FEISHU_WEBHOOK_PATH` | — | `/feishu/webhook` | Webhook 端点路径 |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | — | `2048` | 最大去重消息 ID 追踪数量 |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | — | `0.6` | 文本突发防抖静默期 |
| `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | — | `8` | 每批文本合并的最大消息数 |
| `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | — | `4000` | 每批文本合并的最大字符数 |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | — | `0.8` | 媒体突发防抖静默期 |

WebSocket 和按群 ACL 设置通过 `config.yaml` 的 `platforms.feishu.extra` 配置（参见上方 [WebSocket 调优](#websocket-tuning) 和[按群访问控制](#per-group-access-control)）。

## 故障排查

| 问题 | 解决方法 |
|---------|-----|
| `lark-oapi not installed` | 安装 SDK：`pip install lark-oapi` |
| `websockets not installed; websocket mode unavailable` | 安装 websockets：`pip install websockets` |
| `aiohttp not installed; webhook mode unavailable` | 安装 aiohttp：`pip install aiohttp` |
| `FEISHU_APP_ID or FEISHU_APP_SECRET not set` | 设置两个环境变量，或通过 `hermes gateway setup` 配置 |
| `Another local Hermes gateway is already using this Feishu app_id` | 同一时间只能有一个 Hermes 实例使用相同的 app_id。请先停止另一个 gateway。 |
| 机器人在群聊中不响应 | 确保机器人被 @提及，检查 `FEISHU_GROUP_POLICY`，若策略为 `allowlist` 则验证发送者是否在 `FEISHU_ALLOWED_USERS` 中 |
| `Webhook rejected: invalid verification token` | 确保 `FEISHU_VERIFICATION_TOKEN` 与飞书应用事件订阅配置中的 token 一致 |
| `Webhook rejected: invalid signature` | 确保 `FEISHU_ENCRYPT_KEY` 与飞书应用配置中的加密密钥一致 |
| Post 消息显示为纯文本 | 飞书 API 拒绝了 post payload；这是正常的回退行为。查看日志了解详情。 |
| 机器人未收到图片/文件 | 为飞书应用授予 `im:message` 和 `im:resource` 权限范围 |
| 机器人身份未自动检测 | 通常是访问飞书机器人信息端点时的瞬时网络问题。可手动设置 `FEISHU_BOT_OPEN_ID` 和 `FEISHU_BOT_NAME` 作为临时解决方案。 |
| 启用 `FEISHU_ALLOW_BOTS` 后对端机器人消息仍被忽略 | Hermes 尚无法识别自身——请设置 `FEISHU_BOT_OPEN_ID`（若应用使用 `sender_id_type=user_id` 则同时设置 `FEISHU_BOT_USER_ID`）。 |
| 对端机器人显示为 `ou_xxxxxx` 而非名称 | 授予 `application:bot.basic_info:read` 权限范围。 |
| 点击审批按钮时出现错误 200340 | 在飞书开发者控制台启用**交互式卡片**能力并配置**卡片请求 URL**。参见上方[飞书应用所需配置](#required-feishu-app-configuration)。 |
| `Webhook rate limit exceeded` | 同一 IP 每分钟请求超过 120 次。通常是配置错误或循环导致。 |

## 工具集

飞书 / Lark 使用 `hermes-feishu` 平台预设，包含与 Telegram 及其他基于 gateway 的消息平台相同的核心工具。