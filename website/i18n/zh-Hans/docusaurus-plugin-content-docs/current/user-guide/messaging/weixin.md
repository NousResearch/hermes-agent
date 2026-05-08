---
sidebar_position: 15
title: "微信（Weixin）"
description: "通过 iLink 机器人 API 将 Hermes Agent 连接到个人微信账户"
---

# 微信（Weixin）

将 Hermes 连接到[微信](https://weixin.qq.com/)（Weixin），腾讯的个人消息平台。适配器使用腾讯的 **iLink 机器人 API** 连接个人微信账户——这与企业微信（WeCom）不同。消息通过长轮询传递，因此无需公共端点或 Webhook。

:::info
此适配器适用于**个人微信账户**（微信）。如果您需要企业/公司微信，请参阅[企业微信适配器](./wecom.md)。
:::

:::warning iLink 机器人身份 — 普通微信群可能无法工作
二维码登录将 Hermes 连接到 **iLink 机器人身份**（例如 `a5ace6fd482e@im.bot`），**而不是**完全可编程的普通个人微信账户。后果：

- iLink 机器人身份通常**无法像普通联系人那样被邀请加入普通微信群**。
- iLink 通常**不为企业类型账户传递普通微信群事件**（包括对用于扫描二维码的个人账户的 `@` 提及）到网关。
- `@` 提及用于扫描二维码的个人微信账户**与** `@` 提及 iLink 机器人**不同**——机器人是一个独立的身份。
- 下面的 `WEIXIN_GROUP_POLICY` / `WEIXIN_GROUP_ALLOWED_USERS` 设置仅在 iLink 实际为您的账户类型返回群组事件时生效。如果不返回，无论策略如何，群组消息永远不会到达 Hermes。

实际上，大多数部署只能可靠地使 iLink 机器人的私信工作。如果配置后群组传递不工作，限制在于 iLink 端，而不是 Hermes。当 `WEIXIN_GROUP_POLICY` 设置为 `disabled` 以外的任何值时，网关会在启动时记录 `WARNING`。
:::

## 前置条件

- 个人微信账户
- Python 包：`aiohttp` 和 `cryptography`
- 使用 `messaging` extra 安装 Hermes 时包含终端二维码渲染

安装所需的依赖：

```bash
pip install aiohttp cryptography
# 可选：用于终端二维码显示
pip install hermes-agent[messaging]
```

## 设置

### 1. 运行设置向导

连接您的微信账户最简单的方式是通过交互式设置：

```bash
hermes gateway setup
```

出现提示时选择 **微信**。向导将：

1. 从 iLink 机器人 API 请求二维码
2. 在终端显示二维码（或提供 URL）
3. 等待您用微信手机应用扫描二维码
4. 提示您在手机上确认登录
5. 自动将账户凭证保存到 `~/.hermes/weixin/accounts/`

确认后，您会看到如下消息：

```
微信连接成功，account_id=your-account-id
```

向导存储 `account_id`、`token` 和 `base_url`，因此您无需手动配置它们。

### 2. 配置环境变量

完成初始二维码登录后，在 `~/.hermes/.env` 中至少设置账户 ID：

```bash
WEIXIN_ACCOUNT_ID=your-account-id

# 可选：覆盖 token（通常从二维码登录自动保存）
# WEIXIN_TOKEN=your-bot-token

# 可选：限制访问
WEIXIN_DM_POLICY=open
WEIXIN_ALLOWED_USERS=user_id_1,user_id_2

# 可选：恢复旧版多行分割行为
# WEIXIN_SPLIT_MULTILINE_MESSAGES=true

# 可选：用于 cron/通知的主页频道
WEIXIN_HOME_CHANNEL=chat_id
WEIXIN_HOME_CHANNEL_NAME=Home
```

### 3. 启动网关

```bash
hermes gateway
```

适配器将恢复保存的凭证，连接到 iLink API，并开始长轮询消息。

## 功能

- **长轮询传输** — 无需公共端点、Webhook 或 WebSocket
- **二维码登录** — 通过 `hermes gateway setup` 扫码连接设置
- **私信消息** — 可配置的访问策略；群组消息取决于 iLink 实际为连接身份传递群组事件（对于 iLink 机器人账户通常不是这种情况——见上方警告）
- **媒体支持** — 图片、视频、文件和语音消息
- **AES-128-ECB 加密 CDN** — 所有媒体传输自动加密/解密
- **上下文令牌持久化** — 磁盘备份的回复连续性，跨重启
- **Markdown 格式化** — 保留 Markdown，包括标题、表格和代码块，因此支持 Markdown 的微信客户端可以原生渲染
- **智能消息分割** — 消息在限制内时保持为单个气泡；仅超大小有效载荷在逻辑边界分割
- **打字指示器** — 代理处理时在微信客户端显示"正在输入…"状态
- **SSRF 保护** — 下载前验证出站媒体 URL
- **消息去重** — 5 分钟滑动窗口防止重复处理
- **带退避的自动重试** — 从临时 API 错误恢复

## 配置选项

在 `platforms.weixin.extra` 下的 `config.yaml` 中设置：

| 键 | 默认值 | 描述 |
|-----|---------|-------------|
| `account_id` | — | iLink 机器人账户 ID（必需） |
| `token` | — | iLink 机器人令牌（必需，从二维码登录自动保存） |
| `base_url` | `https://ilinkai.weixin.qq.com` | iLink API 基础 URL |
| `cdn_base_url` | `https://novac2c.cdn.weixin.qq.com/c2c` | 媒体传输的 CDN 基础 URL |
| `dm_policy` | `open` | 私信访问：`open`、`allowlist`、`disabled`、`pairing` |
| `group_policy` | `disabled` | 群组访问：`open`、`allowlist`、`disabled` |
| `allow_from` | `[]` | 允许私信的用户 ID（当 dm_policy=allowlist 时） |
| `group_allow_from` | `[]` | 允许的群组 ID（当 group_policy=allowlist 时） |
| `split_multiline_messages` | `false` | 为 `true` 时，将多行回复分割成多条聊天消息（旧版行为）。为 `false` 时，保持多行回复为一条消息，除非超过长度限制。 |

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
WEIXIN_DM_POLICY=allowlist
WEIXIN_ALLOWED_USERS=user_id_1,user_id_2
```

### 群组策略

控制机器人在**当 iLink 为连接身份传递群组事件时**在哪些群组中响应。对于二维码登录的 iLink 机器人身份（例如 `...@im.bot`），群组事件通常根本不传递，因此此策略可能无效——见上方 iLink 机器人限制警告。

| 值 | 行为 |
|-------|----------|
| `open` | 机器人在所有群组中响应（如果事件被传递） |
| `allowlist` | 机器人只在 `group_allow_from` 中列出的群组 ID 中响应（如果事件被传递） |
| `disabled` | 所有群组消息都被忽略（默认） |

```bash
WEIXIN_GROUP_POLICY=allowlist
# 注意：这是逗号分隔的群组聊天 ID 列表，而不是成员用户 ID，
# 尽管变量名包含"USERS"。配置时注意这一点。
WEIXIN_GROUP_ALLOWED_USERS=group_id_1,group_id_2
```

:::note
微信的默认群组策略是 `disabled`（与企业微信默认 `open` 不同）。这是故意的——个人微信账户可能在许多群组中，而且 iLink 机器人身份通常根本无法接收普通微信群消息。如果将 `WEIXIN_GROUP_POLICY` 设置为 `disabled` 以外的任何值，网关会在启动时记录 `WARNING`。
:::

## 媒体支持

### 入站（接收）

适配器从用户接收媒体附件，从微信 CDN 下载，解密并缓存在本地供代理处理：

| 类型 | 处理方式 |
|------|-----------------| 
| **图片** | 下载、AES 解密并缓存为 JPEG。 |
| **视频** | 下载、AES 解密并缓存为 MP4。 |
| **文件** | 下载、AES 解密并缓存。保留原始文件名。 |
| **语音** | 如果有文本转录可用，则提取为文本。否则下载并缓存音频（SILK 格式）。 |

**引用消息：** 引用（回复）消息中的媒体也会被提取，因此代理可以了解用户正在回复什么。

### AES-128-ECB 加密 CDN

微信媒体文件通过加密 CDN 传输。适配器透明处理：

- **入站：** 使用 `encrypted_query_param` URL 从 CDN 下载加密媒体，然后使用消息负载中提供的每文件密钥通过 AES-128-ECB 解密。
- **出站：** 文件在本地使用随机 AES-128 密钥加密，上传到 CDN，出站消息中包含加密引用。
- AES 密钥是 16 字节（128 位）。密钥可能以原始 base64 或十六进制编码到达——适配器处理两种格式。
- 这需要 `cryptography` Python 包。

无需配置——加密和解密自动发生。

### 出站（发送）

| 方法 | 发送内容 |
|--------|--------------|
| `send` | 带 Markdown 格式的文本消息 | 
| `send_image` / `send_image_file` | 原生图片消息（通过 CDN 上传） |
| `send_document` | 文件附件（通过 CDN 上传） |
| `send_video` | 视频消息（通过 CDN 上传） |

所有出站媒体都通过加密 CDN 上传流程：

1. 生成随机 AES-128 密钥
2. 使用 AES-128-ECB + PKCS#7 填充加密文件
3. 从 iLink API 请求上传 URL（`getuploadurl`）
4. 将密文上传到 CDN
5. 发送带有加密媒体引用的消息

## 上下文令牌持久化

iLink 机器人 API 要求 `context_token` 与每个出站消息一起回显到给定对话方。适配器维护磁盘备份的上下文令牌存储：

- 令牌按账户+对话方保存到 `~/.hermes/weixin/accounts/<account_id>.context-tokens.json`
- 启动时恢复先前保存的令牌
- 每条入站消息更新该发送者的存储令牌
- 出站消息自动包含最新的上下文令牌

这确保即使网关重启后回复连续性。

## Markdown 格式化

通过 iLink 机器人 API 连接的微信客户端可以直接渲染 Markdown，因此适配器保留 Markdown 而不是重写它：

- **标题**保持为 Markdown 标题（`#`、`##`、…）
- **表格**保持为 Markdown 表格
- **代码栅格**保持为带栅格的代码块
- **过多的空行**在带栅格代码块外折叠为双换行符

## 消息分割

只要消息适合平台限制，消息就作为单个聊天消息传递。只有超大小有效载荷被分割传递：

- 最大消息长度：**4000 字符**
- 限制下的消息保持完整，即使包含多个段落或换行符
- 超大消息在逻辑边界（段落、空行、代码栅格）分割
- 代码栅格尽可能保持完整（除非栅格本身超过限制，否则不会在块中间分割）
- 超大单独块回退到基础适配器的截断逻辑
- 发送多个块时，0.3 秒的块间延迟可防止微信速率限制下降

## 打字指示器

适配器在微信客户端显示打字状态：

1. 当消息到达时，适配器通过 `getconfig` API 获取 `typing_ticket`
2. 打字票每个用户缓存 10 分钟
3. `send_typing` 发送打字开始信号；`stop_typing` 发送打字停止信号
4. 代理处理消息时，网关自动触发打字指示器

## 长轮询连接

适配器使用 HTTP 长轮询（不是 WebSocket）接收消息：

### 工作原理

1. **连接：** 验证凭证并启动轮询循环
2. **轮询：** 调用 `getupdates`，超时时间为 35 秒；服务器保持请求直到消息到达或超时过期
3. **分派：** 入站消息通过 `asyncio.create_task` 并发分派
4. **同步缓冲区：** 持久同步游标（`get_updates_buf`）保存到磁盘，以便适配器在重启后从正确的位置恢复

### 重试行为

API 错误时，适配器使用简单的重试策略：

| 条件 | 行为 |
|-----------|----------|
| 临时错误（第 1-2 次） | 2 秒后重试 |
| 重复错误（第 3+ 次） | 退避 30 秒，然后重置计数器 |
| 会话过期（`errcode=-14`） | 暂停 10 分钟（可能需要重新登录） |
| 超时 | 立即重新轮询（正常长轮询行为） |

### 去重

入站消息使用消息 ID 进行去重，窗口为 5 分钟。这可以防止网络故障或重叠轮询响应期间消息被重复处理。

### 令牌锁

一次只能有一个 Weixin 网关实例使用给定令牌。适配器在启动时获取作用域锁，在关闭时释放它。如果另一个网关已经在使用相同的令牌，启动失败并显示信息性错误消息。

## 所有环境变量

| 变量 | 必需 | 默认值 | 描述 |
|----------|----------|---------|-------------|
| `WEIXIN_ACCOUNT_ID` | ✅ | — | iLink 机器人账户 ID（从二维码登录获取） |
| `WEIXIN_TOKEN` | ✅ | — | iLink 机器人令牌（从二维码登录自动保存） |
| `WEIXIN_BASE_URL` | — | `https://ilinkai.weixin.qq.com` | iLink API 基础 URL |
| `WEIXIN_CDN_BASE_URL` | — | `https://novac2c.cdn.weixin.qq.com/c2c` | 媒体传输的 CDN 基础 URL |
| `WEIXIN_DM_POLICY` | — | `open` | 私信访问策略：`open`、`allowlist`、`disabled`、`pairing` |
| `WEIXIN_GROUP_POLICY` | — | `disabled` | 群组访问策略：`open`、`allowlist`、`disabled` |
| `WEIXIN_ALLOWED_USERS` | — | _(空)_ | 私信白名单的逗号分隔用户 ID |
| `WEIXIN_GROUP_ALLOWED_USERS` | — | _(空)_ | 群组白名单的逗号分隔**群组聊天 ID**（不是成员用户 ID）。变量名是旧版的——它期望群组 ID，而不是用户 ID。 |
| `WEIXIN_HOME_CHANNEL` | — | — | cron/通知输出的聊天 ID |
| `WEIXIN_HOME_CHANNEL_NAME` | — | `Home` | 主页频道的显示名称 |
| `WEIXIN_ALLOW_ALL_USERS` | — | — | 允许所有用户的网关级标志（设置向导使用） |

## 故障排除

| 问题 | 修复 |
|---------|-----|
| `Weixin startup failed: aiohttp and cryptography are required` | 安装两者：`pip install aiohttp cryptography` |
| `Weixin startup failed: WEIXIN_TOKEN is required` | 运行 `hermes gateway setup` 完成二维码登录，或手动设置 `WEIXIN_TOKEN` |
| `Weixin startup failed: WEIXIN_ACCOUNT_ID is required` | 在您的 `.env` 中设置 `WEIXIN_ACCOUNT_ID` 或运行 `hermes gateway setup` |
| `Another local Hermes gateway is already using this Weixin token` | 先停止其他网关实例——每个令牌只允许一个轮询器 |
| 会话过期（`errcode=-14`） | 您的登录会话已过期。重新运行 `hermes gateway setup` 扫描新的二维码 |
| 设置期间二维码过期 | 二维码自动刷新最多 3 次。如果持续过期，请检查您的网络连接 |
| 机器人在私信中不响应 | 检查 `WEIXIN_DM_POLICY`——如果设置为 `allowlist`，发送者必须在 `WEIXIN_ALLOWED_USERS` 中 |
| 机器人忽略群组消息 | 群组策略默认为 `disabled`。设置 `WEIXIN_GROUP_POLICY=open` 或 `allowlist`——但请注意，二维码登录的 iLink 机器人身份（`...@im.bot`）通常根本无法接收普通微信群消息。如果网关日志显示群组消息没有原始入站事件，限制在于 iLink 端，而不是 Hermes。 |
| 媒体下载/上传失败 | 确保安装了 `cryptography`。检查到 `novac2c.cdn.weixin.qq.com` 的网络访问 |
| `Blocked unsafe URL (SSRF protection)` | 出站媒体 URL 指向私有/内部地址。只允许公共 URL |
| 语音消息显示为文本 | 如果微信提供转录，适配器使用文本。这是预期行为 |
| 消息出现重复 | 适配器按消息 ID 去重。如果您看到重复，检查是否有多个网关实例在运行 |
| `iLink POST ... HTTP 4xx/5xx` | iLink 服务的 API 错误。检查您的令牌有效性和网络连接 |
| 终端二维码不渲染 | 使用 messaging extra 重新安装：`pip install hermes-agent[messaging]`。或者，打开二维码上方打印的 URL |
