---
sidebar_position: 15
title: "微信（Weixin）"
description: "通过 iLink Bot API 将 Hermes Agent 连接到微信 ClawBot"
---

# 微信（Weixin / WeChat）

本页的 Python 依赖命令使用 [PM 准备的源码环境](https://hermes-agent.nousresearch.com/docs/reference/package-management/#developer-workflow)。
依赖变更后，请重新激活该 checkout 并重启 Hermes。

该适配器使用腾讯的 **iLink Bot API**，将 Hermes 连接到由个人微信绑定的 ClawBot。消息通过 HTTP 长轮询接收，无需为收消息配置公网回调地址。本文说明当前 Hermes 插件的实际行为；Hermes 的本地过滤、分块和重试策略不等同于微信服务端的统一协议限制。

下文以默认数据目录 `~/.hermes` 为例；设置了 `HERMES_HOME` 或使用独立配置档案时，应以当前实例的数据目录为准。

:::info
本适配器适用于**个人微信账号**（微信）。如需对接企业微信，请参阅 [WeCom 适配器](./wecom.md)。
:::

:::warning 接入范围：绑定用户与 ClawBot 的私聊
扫码连接得到的是 iLink Bot 身份，不会把扫码用户的普通微信账号、联系人列表和微信群交给 Hermes 管理。

当前扫码向导面向绑定用户与 ClawBot 的私聊，自动关闭群聊。插件中保留的群聊过滤配置不提供普通微信群接入能力，修改这些配置不能让微信后台产生或投递群消息。本文不将普通微信群列为已支持的接入场景。
:::

## 前置条件

- 一个个人微信账号
- Python 包：`aiohttp` 和 `cryptography`
- 使用 `messaging` 扩展安装 Hermes 时已内置终端二维码渲染功能

安装所需依赖：

```bash
python -c "import pm; pm.sync_venv(['messaging'], explicit=True)"
```

## 配置步骤

### 1. 运行配置向导

在终端直接运行微信登录命令：

```bash
hermes weixin login
```

也可以使用通用配置向导：

```bash
hermes gateway setup
```

使用通用向导时选择 **Weixin**。默认未启用两个测试开关时，微信向导执行以下步骤：

1. 向 iLink Bot API 请求二维码
2. 在终端中显示二维码（或提供 URL）
3. 等待你用微信手机端扫描二维码
4. 按微信后台返回的状态，必要时在终端输入手机显示的 PIN，并在手机上确认
5. 自动将账号凭据保存至 `~/.hermes/weixin/accounts/`
6. 自动允许绑定的微信账号发消息，并将通知发给该账号；关闭群聊，不再询问配对审批或群聊权限

配置完成后，你将看到如下消息：

```
微信配置完成，可以直接给机器人发消息。
```

向导保存后台返回的账号 ID、Bot token 和 API 地址，并自动完成消息接收与通知设置。允许使用和接收通知的用户来自绑定确认响应的 `ilink_user_id`；在他人绑定流程中，该用户与前一步的实名授权操作者可能不同。无需选择 `pairing`、`approve` 或群聊权限。若后台未返回绑定用户标识，向导会停止，不修改现有 `.env` 配置。

### 2. 查看自动保存的配置（可选）

扫码成功后无需手动设置以下内容。向导会自动保存账号，并使用这些访问设置：

```bash
WEIXIN_ACCOUNT_ID=your-account-id
WEIXIN_TOKEN=your-bot-token
WEIXIN_DM_POLICY=allowlist
WEIXIN_ALLOWED_USERS=bound-weixin-user-id
WEIXIN_ALLOW_ALL_USERS=false
WEIXIN_GROUP_POLICY=disabled
WEIXIN_GROUP_ALLOWED_USERS=
WEIXIN_HOME_CHANNEL=bound-weixin-user-id
WEIXIN_HOME_CHANNEL_NAME=Weixin
```

### 3. 启动网关

```bash
hermes gateway
```

适配器会读取已保存的凭据并开始长轮询。若网关已作为后台服务运行，完成新的绑定后执行 `hermes gateway restart`，让运行中的网关加载新账号。

## 功能特性

- **长轮询传输** — 无需公网端点、webhook 或 WebSocket
- **扫码登录** — 通过 `hermes weixin login` 或 `hermes gateway setup` 连接，支持按需处理实名授权和 PIN
- **私信消息** — 扫码向导自动允许绑定用户使用，群聊关闭
- **媒体支持** — 接收图片、视频、文件和语音；发送音频时默认作为文件附件
- **AES-128-ECB 加密 CDN** — 自动处理加密媒体的加解密
- **上下文 token 持久化** — 重启后恢复已保存的会话令牌；令牌是否有效仍由后台决定
- **Markdown 格式化** — 保留 Markdown 格式（包括标题、表格和代码块），支持 Markdown 的微信客户端可原生渲染
- **消息分块** — 超长内容按逻辑边界拆分；简短的多行聊天文本也可能拆成多条消息
- **正在输入提示** — 代理处理消息时在微信客户端显示"正在输入…"状态
- **SSRF 防护** — 下载前验证外发媒体 URL
- **消息去重** — 5 分钟滑动窗口防止重复处理
- **自动重试与退避** — 从瞬时 API 错误中自动恢复

## 配置选项

在 `config.yaml` 的 `platforms.weixin.extra` 下设置：

| 键 | 默认值 | 说明 |
|-----|---------|-------------|
| `account_id` | — | iLink Bot 账号 ID，扫码向导自动保存 |
| `token` | — | 兼容读取的 Bot token；向导将凭据保存到 `.env` 和账号文件，正常接入无需手填 |
| `base_url` | `https://ilinkai.weixin.qq.com` | iLink API 基础 URL |
| `cdn_base_url` | `https://novac2c.cdn.weixin.qq.com/c2c` | 媒体传输 CDN 基础 URL |
| `dm_policy` | `pairing`（扫码向导设为 `allowlist`） | 私信访问策略：`open`、`allowlist`、`disabled`、`pairing` |
| `group_policy` | `disabled` | Hermes 保留的群消息过滤策略，不表示支持普通微信群接入 |
| `allow_from` | 未配置时读取 `WEIXIN_ALLOWED_USERS` | `allowlist` 模式允许的用户；显式设置空列表会拒绝全部用户 |
| `group_allow_from` | 未配置时读取 `WEIXIN_GROUP_ALLOWED_USERS` | 保留的群 ID 过滤列表，正常扫码连接无需配置 |
| `split_multiline_messages` | `false` | 为 `true` 时按顶层行拆分；为 `false` 时通常保留完整消息，但简短聊天式多行文本仍可能拆分 |
| `text_batch_delay_seconds` | `0.3` | 普通入站文本的静默等待秒数，最大 `2.0`；接近分片阈值的文本使用下一项 |
| `text_batch_split_delay_seconds` | `1.0` | 接近分片阈值时的入站文本等待秒数，最大 `4.0`，且不低于上一项 |
| `send_chunk_delay_seconds` | `1.5` | 连续发送出站文本分块的间隔秒数，属于 Hermes 客户端策略 |

若不需要入站文本的额外合并等待，可将 `text_batch_delay_seconds` 和 `text_batch_split_delay_seconds` 都设为 `0`。

## 高级访问策略（正常扫码连接无需配置）

以下是 Hermes 的内部消息过滤规则，不代表微信提供配对审批或群聊接入能力。扫码向导不再展示这些选项，会自动允许绑定用户、关闭群聊。

### 私信策略

对微信后台已经投递给 Hermes 的私信进行过滤。允许列表不是邀请机制，也不能让未由后台投递的消息到达 Hermes。

| 值 | 行为 |
|-------|----------|
| `open` | 允许收到的所有私信，还需显式设置 `WEIXIN_ALLOW_ALL_USERS=true` 或网关级允许开关 |
| `allowlist` | 仅 `allow_from` 中的用户 ID 可发送私信 |
| `disabled` | 忽略所有私信 |
| `pairing` | 旧版 Hermes 配对审批规则，扫码向导不再使用 |

扫码向导会将 `WEIXIN_DM_POLICY` 设为 `allowlist`，将 `WEIXIN_ALLOWED_USERS` 设为本次绑定响应中的用户 ID。正常使用无需手工维护列表。

### 保留的群消息过滤项

`group_policy` 和 `group_allow_from` 是 Hermes 内部保留的过滤配置，不能视为微信协议的群聊能力开关。当前向导始终写入 `WEIXIN_GROUP_POLICY=disabled`，并清空群允许列表；不提供开启普通微信群的操作步骤。

## 媒体支持

### 入站（接收）

适配器接收用户发送的媒体附件，从微信 CDN 下载并解密，然后在本地缓存供代理处理：

| 类型 | 处理方式 |
|------|-----------------|
| **图片** | 下载并按消息提供的密钥解密，再交由 Hermes 图片缓存处理；缺少密钥时当前实现按明文处理。 |
| **视频** | 下载、按需解密并缓存，当前缓存使用 `.mp4` 文件名。 |
| **文件** | 下载、AES 解密后缓存，保留原始文件名。 |
| **语音** | 存在音频媒体时优先下载并缓存为 `.silk`，交由 Hermes 的语音处理流程转写；仅有 `voice_item.text`、没有音频媒体时才使用后台转写文本。 |

**引用消息：** 引用（回复）消息中的媒体也会被提取，以便代理了解用户回复的上下文。

### AES-128-ECB 加密 CDN

微信媒体文件通过加密 CDN 传输。适配器透明处理加解密：

- **入站：** 媒体引用中的字段名为 `encrypt_query_param`，构造 CDN 下载 URL 时的参数名为 `encrypted_query_param`。当前 Hermes 优先使用该参数构造 URL，缺少它时再使用经过微信 CDN 主机校验的 `full_url`。有密钥时执行 AES-128-ECB 解密。
- **出站：** 使用随机 AES-128-ECB 密钥在本地加密文件，上传至 CDN，并在出站消息中包含加密引用。
- AES 密钥为 16 字节。`media.aes_key` 接受两种编码：16 字节原始密钥的 Base64，或 32 位十六进制密钥字符串的 Base64；图片还兼容 `image_item.aeskey` 中的原始十六进制字符串。出站媒体使用十六进制字符串的 Base64。
- 需要安装 `cryptography` Python 包。

无需任何配置——加解密自动完成。

### 出站（发送）

| 方法 | 发送内容 |
|--------|--------------|
| `send` | 保留主要 Markdown 结构的文本消息 |
| `send_image` / `send_image_file` | 原生图片消息（通过 CDN 上传） |
| `send_document` | 根据文件类型选择图片、视频或文件附件（通过 CDN 上传） |
| `send_video` | 视频消息（通过 CDN 上传） |
| `send_voice` | 文件附件；当前不承诺发送原生语音气泡 |

所有出站媒体均通过加密 CDN 上传流程处理：

1. 生成随机 AES-128 密钥
2. 计算明文大小、MD5 和填充后的密文大小，请求 `getuploadurl`
3. 使用 AES-128-ECB + PKCS#7 填充加密文件
4. 将密文 POST 至 `upload_full_url`，或使用 `upload_param` 构造的上传 URL；成功响应需为 HTTP 200 且包含 `x-encrypted-param`
5. 发送包含加密媒体引用的消息；有说明文本时，当前实现将说明和媒体分别发送

## 上下文 Token 持久化

回复对应会话时，应回传入站消息中该对话方的 `context_token`。适配器维护一个基于磁盘的上下文 token 存储：

- Token 按账号+对话方保存至 `~/.hermes/weixin/accounts/<account_id>.context-tokens.json`
- 启动时恢复之前保存的 token
- 已接收处理的入站消息带有非空 `context_token` 时，更新该发送方的已存储 token
- 出站消息在有已存令牌时回传该令牌；没有令牌时，当前客户端仍会尝试发送

持久化只能恢复已有令牌，不能延长后台会话有效期，也不能保证主动通知一定送达。遇到会话失效响应且存在旧令牌时，当前实现会清除旧令牌并不带它重试一次；仍失败时应让用户先给机器人发消息，必要时重新绑定。

## Markdown 格式化

Hermes 保留主要 Markdown 结构；最终显示效果取决于微信客户端。格式化时还会整理空行，并对较长的普通文本行换行：

- **标题** 保持为 Markdown 标题格式（`#`、`##` 等）
- **表格** 保持为 Markdown 表格
- **代码围栏** 保持为围栏代码块
- **多余空行** 在围栏代码块外折叠为双换行

## 消息分块

当前 Hermes 的文本分块策略如下；这些是客户端实现参数，不是对微信服务端最大长度或频率上限的声明：

- 当前单块文本长度上限：**2000 个字符**（`MAX_MESSAGE_LENGTH`）
- 通常保留未超长的消息；符合简短聊天式文本规则的 2–6 行内容可能拆成多个气泡
- 超长消息在逻辑边界处拆分（段落、空行、代码围栏）
- 代码围栏尽可能保持完整（除非围栏本身超出限制，否则不在块中间拆分）
- 超长的单个块回退到基础适配器的截断逻辑
- 连续发送多个文本分块时，默认间隔 **1.5 秒**，可由 `send_chunk_delay_seconds` 调整；服务端仍可能返回限流

## 正在输入提示

适配器在取得有效票据后发送输入状态信号，最终显示由微信客户端决定：

1. 消息到达且无有效缓存票据时，适配器通过 `getconfig` API 获取 `typing_ticket`
2. 输入票据（typing ticket）按用户缓存 10 分钟
3. `send_typing` 发送开始输入信号；`stop_typing` 发送停止输入信号；缺少有效票据时先尝试重新获取，仍无票据则不发送
4. 网关在代理处理消息期间自动触发输入提示

## 长轮询连接

适配器使用 HTTP 长轮询（而非 WebSocket）接收消息：

### 工作原理

1. **连接：** 检查本地所需依赖和凭据是否齐全，取得本地 token 锁后启动轮询；账号凭据是否有效以实际 API 响应为准
2. **轮询：** 首次以 35 秒超时调用 `getupdates`；后续采用响应中有效的 `longpolling_timeout_ms` 建议值
3. **分发：** 入站消息通过 `asyncio.create_task` 并发分发
4. **同步游标：** 将返回的非空 `get_updates_buf` 保存至磁盘，重启后从已保存游标继续拉取

### 重试行为

以下是 Hermes 的 `getupdates` 长轮询循环策略，属于客户端实现；绑定轮询采用后文所述规则：

| 条件 | 行为 |
|-----------|----------|
| 瞬时错误（第 1–2 次） | 2 秒后重试 |
| 持续错误（第 3 次及以上） | 退避 30 秒后重置计数器 |
| 会话过期（`ret` 或 `errcode` 为 `-14`） | 暂停 10 分钟（可能需要重新登录） |
| 超时 | 立即重新轮询（正常长轮询行为） |

### 去重

入站消息使用消息 ID 在 5 分钟窗口内去重；当前实现还会按发送方和非空文本的指纹去重。因此，同一用户在该窗口内重复发送相同文本，也可能被忽略。

### Token 锁

Hermes 在本地取得该 Bot token 的作用域锁，避免多个本地网关同时轮询。发现本地锁冲突时会拒绝启动；这不是跨机器的分布式锁，也不是微信协议提供的互斥保证。

## 所有环境变量

| 变量 | 必填 | 默认值 | 说明 |
|----------|----------|---------|-------------|
| `WEIXIN_ACCOUNT_ID` | ✅ | — | iLink Bot 账号 ID（来自扫码登录） |
| `WEIXIN_TOKEN` | 需有 Bot token | — | 扫码向导自动保存；未显式提供 token 时，适配器也会尝试读取该账号的已存凭据 |
| `WEIXIN_BASE_URL` | — | `https://ilinkai.weixin.qq.com` | iLink API 基础 URL |
| `WEIXIN_CDN_BASE_URL` | — | `https://novac2c.cdn.weixin.qq.com/c2c` | 媒体传输 CDN 基础 URL |
| `WEIXIN_DM_POLICY` | — | `pairing`（扫码向导设为 `allowlist`） | 私信访问策略：`open`、`allowlist`、`disabled`、`pairing` |
| `WEIXIN_GROUP_POLICY` | — | `disabled` | Hermes 保留的群过滤策略；当前扫码向导关闭群聊 |
| `WEIXIN_ALLOWED_USERS` | — | _（空）_ | 私信白名单的逗号分隔用户 ID |
| `WEIXIN_GROUP_ALLOWED_USERS` | — | _（空）_ | 群组白名单的逗号分隔**群聊 ID**（非成员用户 ID）。变量名为历史遗留，实际填写的是群 ID 而非用户 ID。 |
| `WEIXIN_HOME_CHANNEL` | — | 未设置 | cron/通知目标；扫码向导自动设为绑定用户 |
| `WEIXIN_HOME_CHANNEL_NAME` | — | 向导设为 `Weixin` | 默认通知目标的显示名称 |
| `WEIXIN_ALLOW_ALL_USERS` | — | 未启用 | `open` 模式的显式授权标志；扫码向导设为 `false` |

## 故障排查

| 问题 | 解决方法 |
|---------|-----|
| `Weixin startup failed: aiohttp and cryptography are required` | 安装两者：`python -c "import pm; pm.sync_venv(['messaging'], explicit=True)"` |
| `Weixin startup failed: WEIXIN_TOKEN is required` | 运行 `hermes gateway setup` 完成扫码登录，或手动设置 `WEIXIN_TOKEN` |
| `Weixin startup failed: WEIXIN_ACCOUNT_ID is required` | 在 `.env` 中设置 `WEIXIN_ACCOUNT_ID`，或运行 `hermes gateway setup` |
| `Another local Hermes gateway is already using this Weixin token` | 先停止本机上持有该 token 锁的其他 Hermes 网关实例 |
| 会话过期（`ret` 或 `errcode` 为 `-14`） | 当前后台会话不可用；可重新运行 `hermes weixin login`，完成后重启后台网关 |
| 主动发送失败并报 `ret=-2 errmsg=prepare failed` 或 `unknown error` | 当前客户端将其作为会话未就绪处理；有旧 `context_token` 时会去掉令牌重试一次，不保证重试成功。仍失败时，让对方先给机器人发消息，必要时重新绑定。该情形不会进入普通 `-2` 限流冷却 |
| 配置过程中二维码过期或 PIN 错误次数过多 | 最多展示 3 张绑定码，即首次取码加最多 2 次刷新。次数用尽，或实名 ticket 距到期不超过 60 秒时停止，重新运行登录 |
| Bot 不响应私信 | 检查 `WEIXIN_DM_POLICY`——若设置为 `allowlist`，发送方必须在 `WEIXIN_ALLOWED_USERS` 中 |
| Bot 忽略普通微信群消息 | 当前扫码接入面向绑定用户与 ClawBot 私聊，不提供普通微信群接入；开启本地群过滤选项不会增加后台投递能力 |
| 媒体下载/上传失败 | 确保已安装 `cryptography`。检查对 `novac2c.cdn.weixin.qq.com` 的网络访问 |
| `Blocked unsafe URL (SSRF protection)` | 出站媒体 URL 指向私有/内部地址，仅允许公网 URL |
| 语音消息显示为文本 | 有音频时由 Hermes 处理和转写；仅收到文字时使用 `voice_item.text`。发送音频则默认为文件附件 |
| 消息出现重复 | 适配器通过消息 ID 去重。若仍出现重复，检查是否有多个网关实例在运行 |
| `iLink POST ... HTTP 4xx/5xx` | iLink 服务返回 API 错误。检查 token 有效性和网络连通性 |
| 终端二维码无法渲染 | 在已激活的源码环境中同步 `messaging` 扩展后重试；用于展示二维码的内容是 `qrcode_img_content`，不要使用仅供轮询的原始 `qrcode` 值 |

## 3.0 绑定：实名授权与 PIN 测试

当前实现向后台上报的版本字符串是 `3.0.0/hermes-1.0.0`：Bot 请求的 `base_info.channel_version` 与实名 CGI 的 `plugin_version` 使用相同字符串。前三段数字用于后台的版本兼容判断，后缀标识 Hermes 插件版本。这是客户端上报的版本声明，不表示本插件已实现上游全部功能。
**两个测试开关均默认关闭，且独立生效**。它们只影响新的扫码绑定，不影响已连接账号的收发消息。

```bash
# Default: no owner prompt and no proactive PIN request
hermes weixin login

# Request PIN verification
hermes weixin login --verify-pin

# Prompt for self/other binding; N starts real-name authorization
hermes weixin login --prompt-bind-owner

# Enable both independent test switches
hermes weixin login --prompt-bind-owner --verify-pin
```

`--prompt-bind-owner` 启用后，输入 `Y` 或直接回车表示本人绑定，跳过实名授权；输入 `N` 表示绑定他人微信。
绑定他人微信时，先由操作者用**已完成微信支付实名的本人微信**扫描实名授权二维码。该步骤是实名信息授权，不会代替开通微信支付实名；授权通过后，再由待绑定用户扫描 Bot 绑定二维码。

实名查询状态 `auth_status` 为 `0` 时继续等待，为 `1` 才请求绑定码，为 `2` 时停止；为 `3` 时最多重新获取一次实名 ticket，并展示新的授权码。再次无效或重取失败则停止。实名授权等待最多 480 秒，且不超过 ticket 的 `expire_time`；重取 ticket 不延长本次授权等待时限。

`--verify-pin` 只控制主动请求：为二维码申请的 POST 请求增加 `bot_spam_business_data` 字符串，内容为 `{"verify_pin":1}` 的 UTF-8 Base64 编码，即 `eyJ2ZXJpZnlfcGluIjoxfQ==`。关闭时省略该字段，不发送 `verify_pin: 0`。

**是否要求 PIN 由服务端返回的状态决定**。任何包含 `need_verifycode` 的状态（包括 `confirmed_but_need_verifycode`）都会提示输入，即使没有设置 `--verify-pin`。用户在终端输入手机显示的数字后，客户端在**下一次** `get_qrcode_status` GET 请求中通过 `verify_code` 提交；PIN 不发送到实名 CGI。

若状态仍要求 PIN，可重新输入；收到 `scaned` 时清除暂存数字；只有 `confirmed` 且凭据齐全时才保存登录账号，不能将 `confirmed_but_need_verifycode` 当作绑定完成。实名凭证和 PIN 不写入账号文件或 `.env`。当前本人/他人选择和 PIN 输入通过终端完成。

`expired` 或 `verify_code_blocked` 会刷新绑定码，最多展示 3 张（首次加最多 2 次刷新）。换码清除暂存 PIN，保留本次实名 ticket 和 PIN 请求开关，并将状态轮询地址重置为固定 iLink 地址；实名 ticket 距到期不超过 60 秒时不再刷新。收到 `binded_redirect` 表示已连接，不保存新的凭据，也不重写当前连接配置。

这两个开关的配置位置为 `platforms.weixin.extra`。以下示例保持默认关闭；测试时将需要的开关改为 `true`，再重新运行登录命令：

```yaml
platforms:
  weixin:
    extra:
      prompt_bind_owner: false
      verify_pin: false
```

配置值必须是 YAML 布尔值 `true` / `false`。命令行参数优先；`--no-prompt-bind-owner` 和 `--no-verify-pin` 可单次覆盖已开启的配置。
参数不会写回配置文件。登录成功后自动允许绑定的微信账号使用，将通知发送给该账号，并关闭群聊；不会再出现配对审批、用户白名单或群聊权限选择。重新绑定运行中的账号后执行 `hermes gateway restart`。

以下为通用的实名接口源站与代理选项，可作为命令行参数或同名的 `platforms.weixin.extra` 配置项使用。正常连接微信服务时无需设置：

| 参数 / 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `--acctweb-base-url` / `acctweb_base_url` | `https://weixin110.qq.com` | 仅覆盖实名服务的协议、主机和端口；接口路径固定为 `/security/acct/botrealnameverifycgi` |
| `--acctweb-proxy` / `acctweb_proxy` | 空 | 仅实名接口使用的 HTTP/HTTPS 代理地址 |

例如：

```bash
hermes weixin login --prompt-bind-owner --verify-pin --acctweb-base-url http://127.0.0.1:8080
```

实名测试地址和代理不会改变绑定二维码的 iLink 服务地址或已连接账号的消息服务地址。
