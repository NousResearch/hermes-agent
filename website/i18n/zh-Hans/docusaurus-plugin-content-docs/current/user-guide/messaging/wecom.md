---
sidebar_position: 14
title: "企业微信（WeCom）"
description: "通过 AI 机器人 WebSocket 网关将 Hermes Agent 连接到企业微信"
---

# 企业微信（WeCom）

将 Hermes 连接到[企业微信](https://work.weixin.qq.com/)（WeCom），腾讯的企业消息平台。适配器使用企业微信的 AI 机器人 WebSocket 网关进行实时双向通信——无需公共端点或 Webhook。

## 前置条件

- 企业微信组织账户
- 在企业微信管理后台创建的 AI 机器人
- 来自机器人凭证页面的机器人 ID 和 Secret
- Python 包：`aiohttp` 和 `httpx`

## 设置

### 步骤 1：创建 AI 机器人

#### 推荐：扫码创建（一键）

```bash
hermes gateway setup
```

选择 **企业微信** 并用企业微信手机应用扫描二维码。Hermes 将自动创建具有正确权限的机器人应用并保存凭证。

设置向导将：
1. 在终端中显示二维码
2. 等待您用企业微信手机应用扫描
3. 自动检索机器人 ID 和 Secret
4. 引导您完成访问控制配置

#### 备选：手动设置

如果扫码创建不可用，向导会回退到手动输入：

1. 登录[企业微信管理后台](https://work.weixin.qq.com/wework_admin/frame)
2. 导航到 **应用管理** → **创建应用** → **AI 机器人**
3. 配置机器人名称和描述
4. 从凭证页面复制**机器人 ID** 和 **Secret**
5. 运行 `hermes gateway setup`，选择 **企业微信**，并在提示时输入凭证

:::warning
保持机器人 Secret 私密。拥有它的人可以冒充您的机器人。
:::

### 步骤 2：配置 Hermes

#### 选项 A：交互式设置（推荐）

```bash
hermes gateway setup
```

选择 **企业微信** 并按照提示操作。向导将引导您完成：
- 机器人凭证（通过二维码扫描或手动输入）
- 访问控制设置（白名单、配对模式或开放访问）
- 用于通知的主页频道

#### 选项 B：手动配置

将以下内容添加到 `~/.hermes/.env`：

```bash
WECOM_BOT_ID=your-bot-id
WECOM_SECRET=your-secret

# 可选：限制访问
WECOM_ALLOWED_USERS=user_id_1,user_id_2

# 可选：用于 cron/通知的主页频道
WECOM_HOME_CHANNEL=chat_id
```

### 步骤 3：启动网关

```bash
hermes gateway
```

## 功能

- **WebSocket 传输** — 持久连接，无需公共端点
- **私信和群消息** — 可配置的访问策略
- **按群组发件人白名单** — 精细控制谁可以在每个群组中交互
- **媒体支持** — 图片、文件、语音、视频上传和下载
- **AES 加密媒体** — 入站附件自动解密
- **引用上下文** — 保留回复线程
- **Markdown 渲染** — 富文本响应
- **回复模式流式传输** — 将响应关联到入站消息上下文
- **自动重连** — 连接断开时指数退避

## 配置选项

在 `platforms.wecom.extra` 下的 `config.yaml` 中设置：

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `bot_id` | — | 企业微信 AI 机器人 ID（必需） |
| `secret` | — | 企业微信 AI 机器人 Secret（必需） |
| `websocket_url` | `wss://openws.work.weixin.qq.com` | WebSocket 网关 URL |
| `dm_policy` | `open` | 私信访问：`open`、`allowlist`、`disabled`、`pairing` |
| `group_policy` | `open` | 群组访问：`open`、`allowlist`、`disabled` |
| `allow_from` | `[]` | 允许私信的用户 ID（当 dm_policy=allowlist 时） |
| `group_allow_from` | `[]` | 允许的群组 ID（当 group_policy=allowlist 时） |
| `groups` | `{}` | 按群组配置（见下文） |

## 访问策略

### 私信策略

控制谁可以向机器人发送私信：

| 值 | 行为 |
|-------|----------|
| `open` | 任何人都可以向机器人发私信（默认） |
| `allowlist` | 只有 `allow_from` 中的用户 ID 可以发私信 |
| `disabled` | 所有私信都被忽略 |
| `pairing` | 配对模式（用于初始设置） |

```bash
WECOM_DM_POLICY=allowlist
```

### 群组策略

控制机器人在哪些群组中响应：

| 值 | 行为 |
|-------|----------|
| `open` | 机器人在所有群组中响应（默认） |
| `allowlist` | 机器人只在 `group_allow_from` 中列出的群组 ID 中响应 |
| `disabled` | 所有群组消息都被忽略 |

```bash
WECOM_GROUP_POLICY=allowlist
```

### 按群组发件人白名单

为了更精细的控制，您可以限制允许在特定群组内与机器人交互的用户。这在 `config.yaml` 中配置：

```yaml
platforms:
  wecom:
    enabled: true
    extra:
      bot_id: "your-bot-id"
      secret: "your-secret"
      group_policy: "allowlist"
      group_allow_from:
        - "group_id_1"
        - "group_id_2"
      groups:
        group_id_1:
          allow_from:
            - "user_alice"
            - "user_bob"
        group_id_2:
          allow_from:
            - "user_charlie"
        "*":
          allow_from:
            - "user_admin"
```

**工作原理：**

1. `group_policy` 和 `group_allow_from` 控制确定群组是否被允许。
2. 如果群组通过顶层检查，`groups.<group_id>.allow_from` 列表（如果存在）进一步限制该群组内哪些发送者可以与机器人交互。
3. 通配符 `"*"` 条目作为未明确列出的群组的默认值。
4. 白名单条目支持 `*` 通配符以允许所有用户，且条目不区分大小写。
5. 条目可以可选使用 `wecom:user:` 或 `wecom:group:` 前缀格式——前缀会被自动剥离。

如果未为群组配置 `allow_from`，则该群组中的所有用户都被允许（假设群组本身通过顶层策略检查）。

## 媒体支持

### 入站（接收）

适配器从用户接收媒体附件并将其缓存在本地供代理处理：

| 类型 | 处理方式 |
|------|-----------------|
| **图片** | 下载并缓存在本地。支持基于 URL 和 base64 编码的图片。 |
| **文件** | 下载并缓存。文件名从原始消息中保留。 |
| **语音** | 如果有可用的语音消息文本转录，则提取。 |
| **混合消息** | 解析企业微信混合类型消息（文本 + 图片）并提取所有组件。 |

**引用消息：** 引用（回复）消息中的媒体也会被提取，因此代理可以了解用户正在回复什么。

### AES 加密媒体解密

企业微信使用 AES-256-CBC 加密某些入站媒体附件。适配器自动处理：

- 当入站媒体项包含 `aeskey` 字段时，适配器下载加密字节并使用 PKCS#7 填充的 AES-256-CBC 进行解密。
- AES 密钥是 `aeskey` 字段的 base64 解码值（必须正好是 32 字节）。
- IV 从密钥的前 16 字节派生。
- 这需要 `cryptography` Python 包（`pip install cryptography`）。

无需配置——当接收到加密媒体时，解密透明发生。

### 出站（发送）

| 方法 | 发送内容 | 大小限制 |
|--------|--------------|------------|
| `send` | Markdown 文本消息 | 4000 字符 |
| `send_image` / `send_image_file` | 原生图片消息 | 10 MB |
| `send_document` | 文件附件 | 20 MB |
| `send_voice` | 语音消息（仅原生语音支持 AMR 格式） | 2 MB |
| `send_video` | 视频消息 | 10 MB |

**分块上传：** 文件通过三步协议以 512 KB 块上传（初始化 → 块 → 完成）。适配器自动处理。

**自动降级：** 当媒体超过原生类型的大小限制但在绝对 20 MB 文件限制内时，它会自动作为通用文件附件发送：

- 图片 > 10 MB → 作为文件发送
- 视频 > 10 MB → 作为文件发送
- 语音 > 2 MB → 作为文件发送
- 非 AMR 音频 → 作为文件发送（企业微信原生语音仅支持 AMR）

超过绝对 20 MB 限制的文件会被拒绝，并向聊天发送信息性消息。

## 回复模式流式响应

当机器人通过企业微信回调收到消息时，适配器会记住入站请求 ID。如果在请求上下文仍然激活时发送响应，适配器使用企业微信的回复模式（`aibot_respond_msg`）与流式传输将响应直接关联到入站消息。这在企业微信客户端中提供了更自然的对话体验。

如果入站请求上下文已过期或不可用，适配器回退到通过 `aibot_send_msg` 主动发送消息。

回复模式也适用于媒体：上传的媒体可以作为对原始消息的回复发送。

## 连接和重连

适配器维护到企业微信网关 `wss://openws.work.weixin.qq.com` 的持久 WebSocket 连接。

### 连接生命周期

1. **连接：** 打开 WebSocket 连接并发送带有 bot_id 和 secret 的 `aibot_subscribe` 认证帧。
2. **心跳：** 每 30 秒发送应用级 ping 帧以保持连接活跃。
3. **监听：** 持续读取入站帧并分派消息回调。

### 重连行为

连接丢失时，适配器使用指数退避重连：

| 尝试 | 延迟 |
|---------|-------|
| 第一次重试 | 2 秒 |
| 第二次重试 | 5 秒 |
| 第三次重试 | 10 秒 |
| 第四次重试 | 30 秒 |
| 第五次及以上 | 60 秒 |

每次成功重连后，退避计数器重置为零。断开连接时所有待处理请求 future 都失败，这样调用者不会无限期挂起。

### 去重

入站消息使用消息 ID 进行去重，窗口为 5 分钟，缓存最多 1000 条。这可以防止重连或网络故障期间消息被重复处理。

## 所有环境变量

| 变量 | 必需 | 默认值 | 描述 |
|----------|----------|---------|-------------|
| `WECOM_BOT_ID` | ✅ | — | 企业微信 AI 机器人 ID |
| `WECOM_SECRET` | ✅ | — | 企业微信 AI 机器人 Secret |
| `WECOM_ALLOWED_USERS` | — | _(空)_ | 网关级白名单的逗号分隔用户 ID |
| `WECOM_HOME_CHANNEL` | — | — | cron/通知输出的聊天 ID |
| `WECOM_WEBSOCKET_URL` | — | `wss://openws.work.weixin.qq.com` | WebSocket 网关 URL |
| `WECOM_DM_POLICY` | — | `open` | 私信访问策略 |
| `WECOM_GROUP_POLICY` | — | `open` | 群组访问策略 |

## 故障排除

| 问题 | 修复 |
|---------|-----|
| `WECOM_BOT_ID and WECOM_SECRET are required` | 设置两个环境变量或在设置向导中配置 |
| `WeCom startup failed: aiohttp not installed` | 安装 aiohttp：`pip install aiohttp` |
| `WeCom startup failed: httpx not installed` | 安装 httpx：`pip install httpx` |
| `invalid secret (errcode=40013)` | 验证密钥与您的机器人凭证匹配 |
| `Timed out waiting for subscribe acknowledgement` | 检查到 `openws.work.weixin.qq.com` 的网络连接 |
| 机器人在群组中不响应 | 检查 `group_policy` 设置并确保群组 ID 在 `group_allow_from` 中 |
| 机器人在群组中忽略某些用户 | 检查 `groups` 配置部分中每个群组的 `allow_from` 列表 |
| 媒体解密失败 | 安装 `cryptography`：`pip install cryptography` |
| `cryptography is required for WeCom media decryption` | 入站媒体已加密。安装：`pip install cryptography` |
| 语音消息作为文件发送 | 企业微信原生语音仅支持 AMR 格式。其他格式会自动降级为文件。 |
| `File too large` 错误 | 企业微信对所有文件上传有 20 MB 绝对限制。压缩或分割文件。 |
| 图片作为文件发送 | 图片 > 10 MB 超过原生图片限制，会自动降级为文件附件。 |
| `Timeout sending message to WeCom` | WebSocket 可能已断开。检查日志中的重连消息。 |
| `WeCom websocket closed during authentication` | 网络问题或凭证错误。验证 bot_id 和 secret。 |
