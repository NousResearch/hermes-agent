---
sidebar_position: 6
title: "WhatsApp Business（Cloud API）"
description: "通过 Meta 官方 Business Cloud API 将 Hermes Agent 设置为 WhatsApp 机器人"
---

# WhatsApp Business Cloud API 设置

Hermes 可以通过 Meta **官方** WhatsApp Business Cloud API 连接到 WhatsApp。这是生产级方案：无需 Node.js 桥接子进程、无二维码、无账号封禁风险。

代价是：

- 你需要一个 **Meta Business 账户**（不是个人 WhatsApp）。
- 机器人运行在专用的企业手机号码上，不是你的个人号码。
- Hermes 网关需要一个**公共 HTTPS URL**，以便 Meta 通过 webhook 投递入站消息。
- 用户最后一条消息超过 24 小时后的回复需要预先批准的**模板**（这是 Meta 的"客服窗口"规则，不是 Hermes 的限制）。

如果这些约束不适合你的使用场景，[Baileys 桥接集成](./whatsapp.md)是替代方案 —— 个人账户、无需公共 URL，但非官方且有封号风险。

:::tip 我该用哪个？
- **Cloud API（本指南）** —— 运行真正的企业机器人，追求稳定性，愿意配合 Meta 验证 + 模板流程
- **[Baileys 桥接](./whatsapp.md)** —— 个人项目、快速演示、单用户设置，愿意承担机器人手机号码被封的风险
:::

---

## 快速开始

```bash
hermes whatsapp-cloud
```

向导会引导你完成每个凭据，粘贴时逐一验证（捕获最常见的设置陷阱 —— 将电话号码粘贴到 Phone Number ID 字段），并打印需要在向导之外完成的后续步骤的确切说明（启动 cloudflared、配置 Meta 的 webhook 面板）。

本页其余部分是手动参考。

---

## 前置条件

1. **一个 Meta Business 账户。** 在 [business.facebook.com](https://business.facebook.com/) 创建。
2. **一个启用了 WhatsApp 的 Meta 应用。** 参见下方的"创建 Meta 应用"。
3. **一种将本地端口暴露到公共互联网的方式**，支持 HTTPS。推荐 Cloudflare Tunnel（`cloudflared`）—— 免费、无需端口转发、无需域名。ngrok、你自己的域名配合反向代理 + TLS，或网关直接绑定到公共 IP 的 VPS 也可以。
4. **可选但推荐**：`PATH` 上有 ffmpeg，这样出站语音消息会渲染为原生 WhatsApp 语音笔记气泡（绿色波形）而不是 MP3 音频附件。没有时 Hermes 会优雅降级。

---

## 创建 Meta 应用

1. 访问 [developers.facebook.com/apps](https://developers.facebook.com/apps) → **Create App**。
2. 选择用例：**"Connect with customers through WhatsApp"** → **Next**。
3. 选择或创建一个商业组合。审查发布要求。确认 → **Create app**。
4. 创建后你会进入 **Customize use case → Connect on WhatsApp → Quickstart**。点击 **Start using the API** → 现在在 **API Setup** 页面。
5. 确保已链接 WhatsApp Business Account（WABA）。如果你在步骤 3 创建了新组合，会自动创建一个。在 API Setup 页面验证。

你需要从 Dashboard 获取以下值 —— 向导按此顺序提示：

| 值 | Dashboard 位置 | 字段格式 | 说明 |
|---|---|---|---|
| **Phone Number ID** | App Dashboard → WhatsApp → API Setup → "From" 下拉菜单下方 | 数字，15-17 位 | **不是**电话号码本身。最常见的设置错误就是在这里粘贴实际电话号码。 |
| **Access Token** | App Dashboard → WhatsApp → API Setup → "Generate access token" | 以 `EAA` 开头，100+ 字符 | 临时令牌有效期 24 小时 —— 参见下方"永久令牌"了解生产环境。 |
| **App Secret** | App Dashboard → Settings → Basic → 点击 App secret 旁的 "Show" | 32 位小写十六进制 | 用于验证传入 webhook 签名。没有它，入站投递会以 503 拒绝。 |
| **App ID**（可选） | App Dashboard → Settings → Basic | 数字，15-16 位 | 消息功能不需要，用于分析。 |
| **WABA ID**（可选） | App Dashboard → WhatsApp → API Setup → 顶部附近 | 数字，15+ 位 | 消息功能不需要，用于分析。 |

---

## 永久令牌（生产环境）

临时访问令牌在 **24 小时**后过期，这意味着今天生成的令牌明天就停止工作。对于生产部署，使用 **System User 永久令牌**：

1. 访问 [business.facebook.com/latest/settings](https://business.facebook.com/latest/settings) → **System users**（左侧边栏）。
2. **Add** → 名称（例如 `hermes-bot`）→ 角色：**Admin**。
3. 选择新用户 → **Assign Assets**：
   - 选择你的应用 → 在 Full control 下切换 **Manage app**。
   - 选择你的 WhatsApp 账户 → 在 Full control 下切换 **Manage WhatsApp Business Accounts**。
   - 点击 **Assign assets**。
4. **Generate token**，包含以下权限：
   - `business_management`
   - `whatsapp_business_messaging`
   - `whatsapp_business_management`
5. 设置 **token expiration: Never**。
6. 复制令牌 → 更新 `~/.hermes/.env` 中的 `WHATSAPP_CLOUD_ACCESS_TOKEN` → 重启网关。

System User 令牌不会过期，除非你主动撤销。

---

## 将 Hermes 暴露到互联网

Cloud API 通过 HTTPS POST 向你的 webhook URL 投递入站消息 —— 这意味着 Hermes 网关必须能从 Meta 服务器访问。三种常见方式：

### Cloudflare Tunnel（推荐）

免费、无需端口转发、支持 Windows / macOS / Linux。与网关并行作为独立进程运行。

**安装：**

```bash
# Windows
winget install Cloudflare.cloudflared

# macOS
brew install cloudflared

# Linux
# Download the binary from https://github.com/cloudflare/cloudflared/releases
```

**运行快速隧道**（无需 Cloudflare 账户 —— 给你一个 `https://<random>.trycloudflare.com` URL）：

```bash
cloudflared tunnel --url http://localhost:8090
```

记下打印的 URL —— 那是你将提供给 Meta 的地址。

:::warning 快速隧道会轮换
免费快速隧道 URL 每次重启 `cloudflared` 都会变化。要获得稳定的 URL，请用 `cloudflared tunnel login` 登录并创建命名隧道。免费 Cloudflare 账户可获得无限命名隧道 —— 参阅 [Cloudflare 文档](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)了解命名隧道工作流。
:::

### ngrok

```bash
ngrok http 8090
```

免费层每次重启显示不同的 URL。付费层给你一个稳定子域名。

### 你自己的域名 + 反向代理

如果你已有带 TLS 证书的服务器（Caddy、nginx 等），将路由指向 `localhost:8090`。这是生产环境最稳定的选项，但需要现有基础设施。

---

## 在 Meta 侧配置 Webhook

隧道运行后：

1. 记下隧道打印的公共 URL —— 假设 `https://abc123.trycloudflare.com`。
2. 生成一个 **Verify Token** —— 向导用 `secrets.token_urlsafe(32)` 为你完成；如果手动配置，运行：
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   将其保存为 `~/.hermes/.env` 中的 `WHATSAPP_CLOUD_VERIFY_TOKEN`。
3. 启动 Hermes 网关：`hermes gateway`。
4. 在 Meta App Dashboard → **WhatsApp → Configuration**（或 **Use cases → Customize → Configuration**，取决于 UI 版本）→ 点击 Webhook 部分的 **Edit**。
5. 填写：
   - **Callback URL**：`https://abc123.trycloudflare.com/whatsapp/webhook`
   - **Verify Token**：步骤 2 的字符串（必须完全匹配）
6. 点击 **Verify and save**。Meta 向你的 URL 发送 GET 请求，网关回显 challenge，Meta 将 webhook 标记为已验证。
7. 在 **Webhook fields** 下，点击 **Manage** → 订阅 **messages** 字段。这告诉 Meta 将入站消息实际投递到你的 webhook。

**手动验证循环**（从第三个终端）：

```bash
TUNNEL="https://abc123.trycloudflare.com"
VERIFY="<your verify token>"

# Should print HTTP 200 with body "hello"
curl -i "$TUNNEL/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=$VERIFY&hub.challenge=hello"

# Health endpoint — should show verify_token_configured: true and app_secret_configured: true
curl "$TUNNEL/health"
```

---

## 收件人白名单（Meta 侧）

在开发模式下（你的应用通过 App Review 之前），Meta 限制你的机器人可以发消息的号码：

1. App Dashboard → WhatsApp → API Setup → **To** 下拉菜单。
2. 点击 **Manage phone number list**。
3. 添加你想发消息的电话号码（你的、你的团队的、友善的测试人员）。Meta 通过 SMS 或 WhatsApp 向每个号码发送 6 位验证码。

开发模式最多 5 个号码。通过 App Review 后移除此限制。

---

## 允许列表（Hermes 侧）

除了 Meta 的收件人白名单外，Hermes 还有自己的按平台允许列表，控制**Agent 处理哪些传入消息**。添加到 `~/.hermes/.env`：

```bash
# Comma-separated phone numbers, country code, no '+' / spaces / dashes
WHATSAPP_CLOUD_ALLOWED_USERS=15551234567,15557654321

# Or allow everyone (only safe in combination with Meta's recipient whitelist)
# WHATSAPP_CLOUD_ALLOW_ALL_USERS=true
```

向导在步骤 6 中设置此项。没有允许列表时，**所有传入消息都被拒绝** —— 这是有意为之，这样即使收件人白名单被放宽，机器人也不会被随机号码调用。

---

## 完善机器人的 WhatsApp 资料

WhatsApp 在聊天头部和联系人列表中显示机器人的**名称和头像**。这些无法通过 Cloud API 设置 —— 它们在 Meta Business Manager 中。

机器人运行后，前往 **[business.facebook.com/wa/manage/phone-numbers](https://business.facebook.com/wa/manage/phone-numbers/)**，点击你的电话号码，你会看到：

| 内容 | 位置 | 说明 |
|---|---|---|
| **显示名称** | 电话号码页面顶部 | 更改需通过 Meta 的名称审核流程（约 24-48 小时）。 |
| **头像** | 电话号码页面顶部 | 正方形图片，建议 ≥640×640px。立即更新。 |
| **简介/描述/网站/邮箱/营业时间/类别** | "Edit profile" 按钮 | 用户点击机器人名称时显示在信息面板中。纯装饰性。 |
| **认证徽章**（绿色对勾） | Business Manager → Security Center → Start Verification | 需要 Meta 独立的企业验证流程。 |

`hermes whatsapp-cloud` 向导在设置结束时打印这些链接。这些都不是机器人运行所必需的 —— 纯粹是让机器人对外观更完善的美化。

---

## 配置参考

所有设置存放在 `~/.hermes/.env` 中。必填值以**粗体**标注。

| 变量 | 默认值 | 说明 |
|---|---|---|
| **`WHATSAPP_CLOUD_PHONE_NUMBER_ID`** | — | API Setup 中的 15-17 位 ID。**不是**电话号码。 |
| **`WHATSAPP_CLOUD_ACCESS_TOKEN`** | — | Meta 访问令牌（以 `EAA` 开头）。临时 24 小时或 System User 永久。 |
| **`WHATSAPP_CLOUD_APP_SECRET`** | — | Settings → Basic 中的 32 位十六进制。没有它，入站以 503 拒绝。 |
| **`WHATSAPP_CLOUD_VERIFY_TOKEN`** | — | GET 握手的共享密钥。由向导自动生成。 |
| **`WHATSAPP_CLOUD_ALLOWED_USERS`** | — | 逗号分隔的允许发消息的 wa_ids。 |
| `WHATSAPP_CLOUD_ALLOW_ALL_USERS` | `false` | 设为 `true` 绕过允许列表。 |
| `WHATSAPP_CLOUD_APP_ID` | — | 可选，用于未来的分析集成。 |
| `WHATSAPP_CLOUD_WABA_ID` | — | 可选，用于未来的分析集成。 |
| `WHATSAPP_CLOUD_WEBHOOK_HOST` | `0.0.0.0` | Webhook 服务器绑定的接口。 |
| `WHATSAPP_CLOUD_WEBHOOK_PORT` | `8090` | Webhook 服务器绑定的端口。必须与隧道转发的端口匹配。 |
| `WHATSAPP_CLOUD_WEBHOOK_PATH` | `/whatsapp/webhook` | Meta POST 到的 URL 路径。 |
| `WHATSAPP_CLOUD_API_VERSION` | `v20.0` | Meta Graph API 版本。仅在 Meta 文档推荐新版本时覆盖。 |
| `WHATSAPP_CLOUD_HOME_CHANNEL` | — | 用作机器人主频道的 wa_id（用于定时任务等）。 |

你可以同时启用 **Baileys**（`whatsapp`）和 **Cloud**（`whatsapp_cloud`）适配器，指向不同的电话号码。

---

## 功能

### 入站

- **文本消息** —— 直接传递给 Agent。
- **图片** —— 自动下载并附加到 Agent 输入。具有原生视觉能力的模型（Claude、GPT-4o、Gemini 等）直接读取图片；非视觉模型接收自动生成的文本描述。
- **语音笔记** —— 自动下载为 `.ogg`，通过你配置的 STT Provider（本地 faster-whisper、OpenAI/Nous、Groq 等）转录，然后作为文本传递给 Agent。
- **文档** —— 自动下载。小型可读文本文件（`.txt`、`.md`、`.json`、`.py`、`.csv` 等）最大 100KB 内联到 Agent 输入中，使其无需工具调用即可读取。较大的文件在本地缓存供 Agent 的其他工具访问。
- **按钮点击** —— 当用户点击机器人之前发送的按钮（澄清选择、命令审批、斜杠命令确认）时，点击直接路由到正确的处理器。过时的点击回退为常规文本输入。
- **回复上下文** —— 当用户回复之前的机器人消息时，Agent 会看到原始消息作为上下文。

### 出站

- **文本** —— Markdown 自动转换为 WhatsApp 风格语法（`**bold**` → `*bold*`、`~~strike~~` → `~strike~`、标题 → 粗体、`[link](url)` → `link (url)`）。长消息按 4096 字符分块拆分。
- **图片** —— 支持 Agent 生成的图片和本地图片文件，作为原生照片附件投递。
- **语音消息** —— 文本转语音输出通过 ffmpeg 转换为原生 WhatsApp 语音笔记气泡（绿色波形）。未安装 ffmpeg 时回退为 MP3 音频附件。参见下方"语音消息"。
- **视频/文档** —— 都支持，作为原生附件发送。

### 交互式 UX

当 Agent 调用以下任何流程时，Hermes 使用 WhatsApp 原生交互式消息 —— 点击即答按钮而非"回复数字"提示：

- **`clarify` 工具** —— 多选问题渲染为快速回复按钮（1-3 个选项）或点击展开的列表面板（4+ 个选项）。选择"✏️ Other"让用户输入自由形式答案，Agent 会收到作为解析结果。
- **危险命令审批** —— 当 Agent 的终端/代码执行触发受限命令时，用户看到 `✅ Approve` / `❌ Deny` 按钮，而无需输入 `/approve` 或 `/deny`。
- **斜杠命令确认** —— 特权命令如 `/reload-mcp` 显示 `✅ Approve Once` / `🔒 Always` / `❌ Cancel` 按钮。

所有交互式提示在按钮渲染失败时（例如旧版 WhatsApp 客户端）优雅降级为纯文本。

### 已读回执和输入指示器

Hermes 立即确认入站消息：

- 你的消息在网关收到后立即显示**蓝色双勾**。
- 机器人名称在你的 WhatsApp 聊天中在 Agent 准备回复时显示 **"typing…"**。
- 机器人第一条响应消息到达时，输入指示器自动消失。

这让你清楚地知道机器人是已看到你的消息还是仍在处理回复。

### 语音消息

WhatsApp 区分"语音笔记"（绿色波形气泡）和普通音频文件附件。区别纯粹在于编解码器：语音笔记需要 `audio/ogg` 格式加 `opus` 编码。

Hermes TTS 生成 MP3。两种路径：

- **PATH 上有 ffmpeg**（推荐）—— 出站 TTS 被转换并作为正确的语音笔记到达。安装：
  - Windows：`winget install Gyan.FFmpeg`
  - macOS：`brew install ffmpeg`
  - Linux：包管理器
- **没有 ffmpeg** —— 出站 TTS 作为 MP3 音频附件到达。播放没问题，只是看起来不像语音笔记。网关日志中会发出一次性警告。

你可以通过健康端点检查网关是否找到 ffmpeg：

```bash
curl http://localhost:8090/health
# look for "ffmpeg_present": true
```

---

## 已知限制

### 24 小时对话窗口

Meta 只在用户最后一条入站消息后 24 小时内允许**自由格式消息**。在此窗口外，Meta API 唯一接受的是预先批准的**消息模板**。

**实际含义：**

- 响应式聊天（用户私信 → 机器人 24 小时内回复 → 用户回复 → ...）永远有效。覆盖 >95% 的正常机器人使用。
- **定时任务在间隔 > 24 小时后向 WhatsApp 投递**会因 Graph 错误码 `131047`（"Re-engagement message"）失败。
- **长时间运行的 `delegate_task` 异步结果**超过 24 小时会以相同方式失败。
- **Webhook 订阅者**将外部事件路由到 WhatsApp 时，如果用户最近没有私信过机器人，会失败。

Hermes 在系统提示中警告 Agent 关于此窗口，因此模型在安排延迟消息时知道要提及它。

消息模板支持（窗口外发送的解决方案）在 Hermes 中尚未实现。如果你需要它，请 [提交 issue](https://github.com/NousResearch/hermes-agent/issues) —— 已在计划中，但等待明确的需求信号。

### 群聊

Cloud API 有有限的群组支持（能力层级由 Meta 控制）。Hermes 的 `whatsapp_cloud` 适配器在 v1 中目前**仅处理私信**。如果需要群聊，请使用 Baileys 桥接。

### 出站速率限制

Meta 的默认吞吐量为**每个企业手机号码每秒 80 条消息**，可升级。Hermes 目前不在客户端执行此限制 —— 极高吞吐量的发送可能触达 Meta 的限制。

---

## 故障排除

### Meta Dashboard 中设置验证失败（"URL couldn't be validated"）

几乎总是以下之一：

- **隧道 URL 错误或过时** —— cloudflared 快速隧道会轮换。获取新 URL 并更新 `.env` 和 Meta Dashboard。
- **Verify Token 不匹配** —— `~/.hermes/.env` 中 `WHATSAPP_CLOUD_VERIFY_TOKEN` 的令牌必须与你在 Meta Dashboard 中输入的完全匹配。先运行上面的 curl 探测确认本地验证握手有效。
- **网关未运行** —— 检查 `hermes gateway` 是否启动。
- **App Secret 未设置** —— 没有它，Hermes 以 503 拒绝入站 POST。Meta 解释为"无法验证"。

### `graph error 100`: Object with ID '...' does not exist

你将电话号码（10-11 位）粘贴到了 `WHATSAPP_CLOUD_PHONE_NUMBER_ID`，而不是 Phone Number ID（Meta 的 15-17 位内部 ID）。请重新检查 API Setup 页面 —— Phone Number ID 显示在 "From" 下拉菜单*下方*。

向导现在用验证器捕获了此问题，但如果你手动配置，值得了解。

### `graph error 190`: Authentication Error

你的访问令牌无效。子码：

- `subcode 463` —— 令牌过期。临时令牌有效期 24 小时。重新生成，或切换到 System User 永久令牌（见上文）。
- `subcode 467` —— 令牌已失效（被撤销或密码已更改）。
- 其他 190 —— 令牌生成时缺少所需权限。确保已选择全部三个（`business_management`、`whatsapp_business_messaging`、`whatsapp_business_management`）。

### `graph error 131047`: Re-engagement message

24 小时对话窗口已过期（参见"已知限制"）。选择：

- 要求用户先私信机器人以重新打开窗口。
- 等待模板支持落地到 Hermes。

### 入站消息：`media metadata fetch failed (status=401)`

与出站相同的 401 根因（`graph error 190`）—— 访问令牌无效或过期。修复令牌。

### 机器人回复显示为原始 JSON / 工具调用泄漏

常见原因：为 `whatsapp_cloud` 配置的工具集缺少 Agent 想要调用的工具。检查 `hermes tools list` 并确认平台使用的是 `hermes-whatsapp`（默认 Cloud 适配器工具集，与 Baileys 相同）。

如果模型发出工具调用形式的文本而非结构化调用，通常意味着工具集实际上为空。参见 `hermes_cli/platforms.py` 了解平台 → 默认工具集映射。

### STT（语音笔记转录）返回空 / "could not transcribe"

默认 `stt.provider: local` 需要 `pip install faster-whisper`。如果你是 Nous 订阅者，可以通过 Meta 的托管音频网关路由 STT：

```bash
hermes config set stt.provider openai
hermes config set stt.use_gateway true
hermes gateway restart
```

这使用你的 Nous Portal 访问令牌，而不需要单独的 OpenAI 密钥。

---

## 安全说明

- **将 App Secret 视同密码** —— 任何持有它的人都可以伪造 Hermes 会接受为真实的 webhook 载荷。
- **Verify Token 是共享密钥** —— 泄露的风险较低（最坏情况有人可以将 Meta 的 webhook 重新订阅到他们的另一个 URL），但仍然要避免提交它。
- **Access Token 是你机器人的身份** —— System User 令牌等同于长期 API 密钥。如果部署被入侵，立即轮换。
- **Webhook 端点在设置了 `WHATSAPP_CLOUD_APP_SECRET` 时仅接受签名请求** —— 即使在开发环境中也保持设置。没有它，网关以 HTTP 503 拒绝入站投递。
- **`/health` 端点未认证** —— 安全暴露，因为它只报告配置存在性布尔值，不报告值本身。但如果你不想暴露它，在反向代理/隧道层限制访问。

---

## 与 Baileys 桥接的对比

| | Baileys（`hermes whatsapp`） | Cloud API（`hermes whatsapp-cloud`） |
|---|---|---|
| 账户类型 | 个人 | 企业 |
| 设置 | 扫描二维码 | Meta 应用 + WABA + 令牌 |
| 依赖 | Node.js + npm | 纯 Python（httpx + aiohttp） |
| 进程 | 托管 Node 子进程 | aiohttp webhook 服务器 |
| 需要公共 URL？ | 否 | 是 |
| 账号封禁风险 | 有（非官方 API） | 无（官方支持） |
| 入站 | 轮询 Node 桥接 | 从 Meta Webhook POST |
| 出站 | 本地桥接 → Baileys | HTTPS 到 graph.facebook.com |
| 群组 | 完整支持 | 仅私信（v1） |
| 24 小时窗口 | 无限制 | 硬性规则 —— 之后需要模板 |
| 语音笔记（出站） | 原生 | 原生（需 ffmpeg），否则 MP3 回退 |
| 已读回执 | 否 | 是（蓝色双勾） |
| 输入指示器 | 否 | 是（响应时自动消失） |
| 交互按钮 | 仅文本回退 | 原生（澄清、审批、斜杠确认） |
| 生产使用 | 有风险（Meta 可能封号） | 专为生产设计 |

大多数运行 Hermes 个人项目的用户偏好 Baileys。大多数运行面向客户机器人的用户偏好 Cloud API。

---

## 另请参阅

- [Meta 官方 WhatsApp Business Cloud API 文档](https://developers.facebook.com/documentation/business-messaging/whatsapp/) —— 底层平台、定价、App Review 和 Meta 侧速率限制的权威参考。
- [WhatsApp（Baileys 桥接）设置](whatsapp.md) —— 个人项目的替代集成。
- [消息平台概览](index.md) —— 所有消息集成一览。
