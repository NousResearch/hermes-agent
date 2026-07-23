---
sidebar_position: 15
title: "微信（Weixin）"
description: "通过 iLink Bot API 将 Hermes Agent 连接到个人微信账号"
---

# 微信（Weixin / WeChat）

将 Hermes 连接到 [微信](https://weixin.qq.com/)（WeChat），腾讯的个人即时通讯平台。该适配器使用腾讯的 **iLink Bot API** 对接个人微信账号——与企业微信（WeCom）不同。消息通过长轮询（long-polling）方式传递，无需公网端点或 webhook。

:::info
本适配器适用于**个人微信账号**（微信）。如需对接企业微信，请参阅 [WeCom 适配器](./wecom.md)。
:::

:::warning iLink bot 身份——普通微信群可能无法使用
扫码登录后，Hermes 连接的是一个 **iLink bot 身份**（例如 `a5ace6fd482e@im.bot`），**而非**可完全脚本化的普通个人微信账号。具体影响如下：

- iLink bot 身份通常**无法像普通联系人一样被邀请进入普通微信群**。
- 对于大多数 bot 类型账号，iLink 通常**不会将普通微信群事件**（包括对扫码登录所用个人账号的 `@` 提及）推送到网关。
- `@` 提及用于扫码的个人微信账号，**不等同于** `@` 提及 iLink bot——两者是独立身份。
- 下方的 `WEIXIN_GROUP_POLICY` / `WEIXIN_GROUP_ALLOWED_USERS` 设置仅在 iLink 实际为你的账号类型返回群事件时才生效。若 iLink 不返回群事件，无论策略如何配置，群消息都不会到达 Hermes。

实际部署中，大多数情况下只有发送给 iLink bot 的私信（DM）能可靠工作。若配置完成后群消息仍无法送达，限制来自 iLink 侧，而非 Hermes。只要 `WEIXIN_GROUP_POLICY` 设置为 `disabled` 以外的值，网关在启动时会记录一条 `WARNING`。
:::

## 前置条件

- 一个个人微信账号
- Python 包：`aiohttp` 和 `cryptography`
- 使用 `messaging` 扩展安装 Hermes 时已内置终端二维码渲染功能

安装所需依赖：

```bash
pip install aiohttp cryptography
# 可选：用于终端二维码显示
cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"
```

## 配置步骤

### 1. 运行配置向导

连接微信账号最简便的方式是通过交互式配置向导：

```bash
hermes gateway setup
```

在提示中选择 **Weixin**。向导将执行以下步骤：

1. 向 iLink Bot API 请求二维码
2. 在终端中显示二维码（或提供 URL）
3. 等待你用微信手机端扫描二维码
4. 提示你在手机上确认登录
5. 自动将账号凭据保存至 `~/.hermes/weixin/accounts/`

确认后，你将看到如下消息：

```
微信连接成功，account_id=your-account-id
```

向导会保存 `account_id`、`token` 和 `base_url`，无需手动配置。

### 2. 配置环境变量

完成首次扫码登录后，在 `~/.hermes/.env` 中至少设置账号 ID：

```bash
WEIXIN_ACCOUNT_ID=your-account-id

# 可选：覆盖 token（通常由扫码登录自动保存）
# WEIXIN_TOKEN=your-bot-token

# 可选：限制访问权限
WEIXIN_DM_POLICY=open
WEIXIN_ALLOWED_USERS=user_id_1,user_id_2

# 可选：恢复旧版多行拆分行为
# WEIXIN_SPLIT_MULTILINE_MESSAGES=true

# 可选：cron/通知的默认频道
WEIXIN_HOME_CHANNEL=chat_id
WEIXIN_HOME_CHANNEL_NAME=Home
```

### 3. 启动网关

```bash
hermes gateway
```

适配器将恢复已保存的凭据，连接到 iLink API，并开始长轮询消息。

## 功能特性

- **长轮询传输** — 无需公网端点、webhook 或 WebSocket
- **扫码登录** — 通过 `hermes gateway setup` 扫码连接
- **私信（DM）消息** — 可配置访问策略；群消息功能取决于 iLink 是否实际为所连接身份推送群事件（iLink bot 账号通常不推送，详见上方警告）
- **媒体支持** — 图片、视频、文件和语音消息
- **AES-128-ECB 加密 CDN** — 所有媒体传输自动加解密
- **上下文 token 持久化** — 基于磁盘的回复连续性，重启后仍可保持
- **Markdown 格式化** — 保留 Markdown 格式（包括标题、表格和代码块），支持 Markdown 的微信客户端可原生渲染
- **智能消息分块** — 未超出长度限制时保持单条消息气泡；仅超长内容在逻辑边界处拆分
- **正在输入提示** — 代理处理消息时在微信客户端显示"正在输入…"状态
- **SSRF 防护** — 下载前验证外发媒体 URL
- **消息去重** — 5 分钟滑动窗口防止重复处理
- **自动重试与退避** — 从瞬时 API 错误中自动恢复

## 配置选项

在 `config.yaml` 的 `platforms.weixin.extra` 下设置：

| 键 | 默认值 | 说明 |
|-----|---------|-------------|
| `account_id` | — | iLink Bot 账号 ID（必填） |
| `token` | — | iLink Bot token（必填，由扫码登录自动保存） |
| `base_url` | `https://ilinkai.weixin.qq.com` | iLink API 基础 URL |
| `cdn_base_url` | `https://novac2c.cdn.weixin.qq.com/c2c` | 媒体传输 CDN 基础 URL |
| `dm_policy` | `open` | 私信访问策略：`open`、`allowlist`、`disabled`、`pairing` |
| `group_policy` | `disabled` | 群组访问策略：`open`、`allowlist`、`disabled` |
| `allow_from` | `[]` | 允许发送私信的用户 ID（当 dm_policy=allowlist 时生效） |
| `group_allow_from` | `[]` | 允许的群组 ID（当 group_policy=allowlist 时生效） |
| `split_multiline_messages` | `false` | 为 `true` 时，将多行回复拆分为多条消息（旧版行为）；为 `false` 时，多行回复保持为单条消息，除非超出长度限制。 |

## 访问策略

### 私信策略

控制哪些用户可以向 bot 发送私信：

| 值 | 行为 |
|-------|----------|
| `open` | 任何人均可向 bot 发送私信（默认） |
| `allowlist` | 仅 `allow_from` 中的用户 ID 可发送私信 |
| `disabled` | 忽略所有私信 |
| `pairing` | 配对模式（用于初始设置） |

```bash
WEIXIN_DM_POLICY=allowlist
WEIXIN_ALLOWED_USERS=user_id_1,user_id_2
```

### 群组策略

控制 bot 在哪些群组中响应消息，**前提是 iLink 为所连接身份推送了群事件**。对于扫码登录的 iLink bot 身份（例如 `...@im.bot`），群事件通常根本不会被推送，因此该策略可能不起作用——详见页面顶部的 iLink bot 限制警告。

| 值 | 行为 |
|-------|----------|
| `open` | bot 在所有群组中响应（如果事件被推送） |
| `allowlist` | bot 仅在 `group_allow_from` 中列出的群组 ID 中响应（如果事件被推送） |
| `disabled` | 忽略所有群消息（默认） |

```bash
WEIXIN_GROUP_POLICY=allowlist
# 注意：这是以逗号分隔的群聊 ID 列表，而非成员用户 ID，
# 尽管变量名中包含"USERS"。配置时请注意区分。
WEIXIN_GROUP_ALLOWED_USERS=group_id_1,group_id_2
```

:::note
微信的默认群组策略为 `disabled`（与企业微信默认为 `open` 不同）。这是有意为之——个人微信账号可能加入了很多群，且 iLink bot 身份通常根本无法接收普通微信群消息。若将 `WEIXIN_GROUP_POLICY` 设置为 `disabled` 以外的值，网关在启动时会记录一条 `WARNING`。
:::

## 媒体支持

### 入站（接收）

适配器接收用户发送的媒体附件，从微信 CDN 下载并解密，然后在本地缓存供代理处理：

| 类型 | 处理方式 |
|------|-----------------| 
| **图片** | 下载、AES 解密后缓存为 JPEG。 |
| **视频** | 下载、AES 解密后缓存为 MP4。 |
| **文件** | 下载、AES 解密后缓存，保留原始文件名。 |
| **语音** | 若有文字转录，则提取为文本；否则下载音频（SILK 格式）并缓存。 |

**引用消息：** 引用（回复）消息中的媒体也会被提取，以便代理了解用户回复的上下文。

### AES-128-ECB 加密 CDN

微信媒体文件通过加密 CDN 传输。适配器透明处理加解密：

- **入站：** 使用 `encrypted_query_param` URL 从 CDN 下载加密媒体，再使用消息载荷中提供的每文件密钥进行 AES-128-ECB 解密。
- **出站：** 使用随机 AES-128-ECB 密钥在本地加密文件，上传至 CDN，并在出站消息中包含加密引用。
- AES 密钥为 16 字节（128 位）。密钥可能以原始 base64 或十六进制编码形式到达——适配器两种格式均支持。
- 需要安装 `cryptography` Python 包。

无需任何配置——加解密自动完成。

### 出站（发送）

| 方法 | 发送内容 |
|--------|--------------|
| `send` | 带 Markdown 格式的文本消息 | 
| `send_image` / `send_image_file` | 原生图片消息（通过 CDN 上传） |
| `send_document` | 文件附件（通过 CDN 上传） |
| `send_video` | 视频消息（通过 CDN 上传） |

所有出站媒体均通过加密 CDN 上传流程处理：

1. 生成随机 AES-128 密钥
2. 使用 AES-128-ECB + PKCS#7 填充加密文件
3. 向 iLink API 请求上传 URL（`getuploadurl`）
4. 将密文上传至 CDN
5. 发送包含加密媒体引用的消息

## 上下文 Token 持久化

iLink Bot API 要求在每条出站消息中回传 `context_token`（针对特定对话方）。适配器维护一个基于磁盘的上下文 token 存储：

- Token 按账号+对话方保存至 `~/.hermes/weixin/accounts/<account_id>.context-tokens.json`
- 启动时恢复之前保存的 token
- 每条入站消息都会更新该发送方的已存储 token
- 出站消息自动包含最新的上下文 token

这确保了即使网关重启后，回复连续性也不会中断。

## Markdown 格式化

通过 iLink Bot API 连接的微信客户端可以直接渲染 Markdown，因此适配器保留 Markdown 而不对其进行改写：

- **标题** 保持为 Markdown 标题格式（`#`、`##` 等）
- **表格** 保持为 Markdown 表格
- **代码围栏** 保持为围栏代码块
- **多余空行** 在围栏代码块外折叠为双换行

## 消息分块

消息在不超出平台限制时以单条消息发送。仅超长内容才会被拆分发送：

- 最大消息长度：**4000 个字符**
- 未超出限制的消息保持完整，即使包含多个段落或换行
- 超长消息在逻辑边界处拆分（段落、空行、代码围栏）
- 代码围栏尽可能保持完整（除非围栏本身超出限制，否则不在块中间拆分）
- 超长的单个块回退到基础适配器的截断逻辑
- 发送多个分块时，块间延迟 0.3 秒，防止触发微信频率限制

## 正在输入提示

适配器在微信客户端中显示输入状态：

1. 消息到达时，适配器通过 `getconfig` API 获取 `typing_ticket`
2. 输入票据（typing ticket）按用户缓存 10 分钟
3. `send_typing` 发送开始输入信号；`stop_typing` 发送停止输入信号
4. 网关在代理处理消息期间自动触发输入提示

## 长轮询连接

适配器使用 HTTP 长轮询（而非 WebSocket）接收消息：

### 工作原理

1. **连接：** 验证凭据并启动轮询循环
2. **轮询：** 以 35 秒超时调用 `getupdates`；服务器保持请求直到消息到达或超时
3. **分发：** 入站消息通过 `asyncio.create_task` 并发分发
4. **同步缓冲区：** 持久化同步游标（`get_updates_buf`）保存至磁盘，确保重启后从正确位置恢复

### 重试行为

发生 API 错误时，适配器采用简单的重试策略：

| 条件 | 行为 |
|-----------|----------|
| 瞬时错误（第 1–2 次） | 2 秒后重试 |
| 持续错误（第 3 次及以上） | 退避 30 秒后重置计数器 |
| 会话过期（`errcode=-14`） | 暂停 10 分钟（可能需要重新登录） |
| 超时 | 立即重新轮询（正常长轮询行为） |

### 去重

入站消息使用消息 ID 在 5 分钟窗口内去重，防止网络抖动或轮询响应重叠时重复处理。

### Token 锁

同一时间只有一个微信网关实例可以使用给定的 token。适配器在启动时获取作用域锁，关闭时释放。若另一个网关已在使用相同 token，启动将失败并显示详细错误信息。

## 所有环境变量

| 变量 | 必填 | 默认值 | 说明 |
|----------|----------|---------|-------------|
| `WEIXIN_ACCOUNT_ID` | ✅ | — | iLink Bot 账号 ID（来自扫码登录） |
| `WEIXIN_TOKEN` | ✅ | — | iLink Bot token（由扫码登录自动保存） |
| `WEIXIN_BASE_URL` | — | `https://ilinkai.weixin.qq.com` | iLink API 基础 URL |
| `WEIXIN_CDN_BASE_URL` | — | `https://novac2c.cdn.weixin.qq.com/c2c` | 媒体传输 CDN 基础 URL |
| `WEIXIN_DM_POLICY` | — | `open` | 私信访问策略：`open`、`allowlist`、`disabled`、`pairing` |
| `WEIXIN_GROUP_POLICY` | — | `disabled` | 群组访问策略：`open`、`allowlist`、`disabled` |
| `WEIXIN_ALLOWED_USERS` | — | _（空）_ | 私信白名单的逗号分隔用户 ID |
| `WEIXIN_GROUP_ALLOWED_USERS` | — | _（空）_ | 群组白名单的逗号分隔**群聊 ID**（非成员用户 ID）。变量名为历史遗留，实际填写的是群 ID 而非用户 ID。 |
| `WEIXIN_HOME_CHANNEL` | — | — | cron/通知输出的聊天 ID |
| `WEIXIN_HOME_CHANNEL_NAME` | — | `Home` | 默认频道的显示名称 |
| `WEIXIN_ALLOW_ALL_USERS` | — | — | 网关级别的允许所有用户标志（由配置向导使用） |

## 故障排查

| 问题 | 解决方法 |
|---------|-----|
| `Weixin startup failed: aiohttp and cryptography are required` | 安装两者：`pip install aiohttp cryptography` |
| `Weixin startup failed: WEIXIN_TOKEN is required` | 运行 `hermes gateway setup` 完成扫码登录，或手动设置 `WEIXIN_TOKEN` |
| `Weixin startup failed: WEIXIN_ACCOUNT_ID is required` | 在 `.env` 中设置 `WEIXIN_ACCOUNT_ID`，或运行 `hermes gateway setup` |
| `Another local Hermes gateway is already using this Weixin token` | 先停止另一个网关实例——每个 token 只允许一个轮询器 |
| 会话过期（`errcode=-14`） | 登录会话已过期。重新运行 `hermes gateway setup` 扫描新二维码 |
| 配置过程中二维码过期 | 二维码最多自动刷新 3 次。若持续过期，请检查网络连接 |
| Bot 不响应私信 | 检查 `WEIXIN_DM_POLICY`——若设置为 `allowlist`，发送方必须在 `WEIXIN_ALLOWED_USERS` 中 |
| Bot 忽略群消息 | 群组策略默认为 `disabled`。设置 `WEIXIN_GROUP_POLICY=open` 或 `allowlist`——但请注意，扫码登录的 iLink bot 身份（`...@im.bot`）通常根本无法接收普通微信群消息。若网关日志中没有群消息的原始入站事件，限制来自 iLink 侧，而非 Hermes。 |
| 媒体下载/上传失败 | 确保已安装 `cryptography`。检查对 `novac2c.cdn.weixin.qq.com` 的网络访问 |
| `Blocked unsafe URL (SSRF protection)` | 出站媒体 URL 指向私有/内部地址，仅允许公网 URL |
| 语音消息显示为文本 | 若微信提供了转录文本，适配器会使用文本内容，这是预期行为 |
| 消息出现重复 | 适配器通过消息 ID 去重。若仍出现重复，检查是否有多个网关实例在运行 |
| `iLink POST ... HTTP 4xx/5xx` | iLink 服务返回 API 错误。检查 token 有效性和网络连通性 |
| 终端二维码无法渲染 | 使用 messaging 扩展重新安装：`cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"`。或者，打开二维码上方打印的 URL |