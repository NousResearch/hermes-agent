---
sidebar_position: 12
title: "Google Chat"
description: "使用 Cloud Pub/Sub 将 Hermes Agent 设置为 Google Chat 机器人"
---

# Google Chat 设置

将 Hermes Agent 作为机器人接入 Google Chat。该集成使用 Cloud Pub/Sub 拉取订阅接收入站事件，使用 Chat REST API 发送出站消息。与 Slack Socket Mode 或 Telegram 长轮询的使用体验相当：Hermes 进程无需公网 URL、隧道或 TLS 证书。它直接连接、认证并监听订阅——就像 Telegram 机器人通过 token 监听一样。

:::note Workspace 版本
Google Chat 是 Google Workspace 的一部分。你可以在个人 Workspace（通过 Google 注册的 `@yourdomain.com`）或拥有管理员权限可发布应用的企业 Workspace 中使用此集成。仅有 Gmail 账号的用户无法托管 Chat 应用。
:::

## 概览

| 组件 | 值 |
|-----------|-------|
| **依赖库** | `google-cloud-pubsub`、`google-api-python-client`、`google-auth` |
| **入站传输** | Cloud Pub/Sub 拉取订阅（无需公网端点） |
| **出站传输** | Chat REST API（`chat.googleapis.com`） |
| **认证** | 在订阅上具有 `roles/pubsub.subscriber` 的 Service Account JSON |
| **用户标识** | Chat 资源名称（`users/{id}`）+ 邮箱 |

---

## 第一步：创建或选择 GCP 项目

你需要一个 Google Cloud 项目来托管 Pub/Sub topic（主题）。如果还没有，请在 [console.cloud.google.com](https://console.cloud.google.com) 创建——个人账号有免费额度，足以覆盖机器人流量。

记下项目 ID（例如 `my-chat-bot-123`），后续每一步都会用到。

---

## 第二步：启用两个 API

在控制台中，进入 **APIs & Services → Library**，启用：

- **Google Chat API**
- **Cloud Pub/Sub API**

个人机器人产生的流量完全在免费额度内。

---

## 第三步：创建 Service Account

**IAM & Admin → Service Accounts → Create Service Account。**

- 名称：`hermes-chat-bot`
- 跳过"Grant this service account access to project"步骤。你只需要在特定订阅上配置 IAM，**不要**授予项目级别的 Pub/Sub 角色。

创建完成后，打开该 SA，进入 **Keys → Add Key → Create new key → JSON**，下载文件。将其保存到只有 Hermes 可读的位置（例如 `~/.hermes/google-chat-sa.json`，`chmod 600`）。

:::caution 不存在"Chat Bot Caller"角色
一个常见错误是搜索 Chat 专属 IAM 角色并在项目级别授予。该角色并不存在。Chat 机器人的权限来自被安装到某个 space（空间），而非 IAM。你的 SA 只需要在下一步创建的订阅上具有 Pub/Sub subscriber 权限。
:::

---

## 第四步：创建 Pub/Sub topic 和订阅

**Pub/Sub → Topics → Create topic。**

- Topic ID：`hermes-chat-events`
- 其余选项保持默认。

创建完成后，topic 详情页有 **Subscriptions** 标签页。在此创建一个订阅：

- Subscription ID：`hermes-chat-events-sub`
- 投递类型：**Pull**
- 消息保留：**7 天**（这样 Hermes 重启后积压消息不会丢失）
- 其余保持默认。

---

## 第五步：在 topic 上配置 IAM 绑定（关键）

在 **topic**（不是订阅）上添加一个 IAM 主体：

- 主体：`chat-api-push@system.gserviceaccount.com`
- 角色：`Pub/Sub Publisher`

若不配置此项，Google Chat 将无法向你的 topic 发布事件，机器人将永远收不到任何消息。

---

## 第六步：在订阅上配置 IAM 绑定

在 **订阅** 上，将你自己的 Service Account 添加为主体：

- 主体：`hermes-chat-bot@<your-project>.iam.gserviceaccount.com`
- 角色：`Pub/Sub Subscriber`

同时在同一订阅上授予 `Pub/Sub Viewer`——Hermes 在启动时会调用 `subscription.get()` 进行可达性检查。

---

## 第七步：配置 Chat 应用

进入 **APIs & Services → Google Chat API → Configuration**。

- **App name**：用户看到的名称（"Hermes"即可）。
- **Avatar URL**：任意公开 PNG 图片（Google 提供了一些默认选项）。
- **Description**：显示在应用目录中的简短说明。
- **Functionality**：启用 **Receive 1:1 messages** 和 **Join spaces and group conversations**。
- **Connection settings**：选择 **Cloud Pub/Sub**，输入 topic 名称 `projects/<your-project>/topics/hermes-chat-events`。
- **Visibility**：限制为你的 Workspace（或特定用户）——测试期间不要向所有人开放。

保存。

---

## 第八步：在测试 space 中安装机器人

在浏览器中打开 Google Chat。在 **+ New Chat** 菜单中搜索应用名称，向其发起私信。第一次发消息时，Google 会发送一个 `ADDED_TO_SPACE` 事件，Hermes 用它来缓存机器人自身的 `users/{id}`，以便过滤自发消息。

---

## 第九步：配置 Hermes

在 `~/.hermes/.env` 中添加 Google Chat 配置段：

```bash
# 必填
GOOGLE_CHAT_PROJECT_ID=my-chat-bot-123
GOOGLE_CHAT_SUBSCRIPTION_NAME=projects/my-chat-bot-123/subscriptions/hermes-chat-events-sub
GOOGLE_CHAT_SERVICE_ACCOUNT_JSON=/home/you/.hermes/google-chat-sa.json

# 授权 — 粘贴允许与机器人对话的用户邮箱
GOOGLE_CHAT_ALLOWED_USERS=you@yourdomain.com,coworker@yourdomain.com

# 可选
GOOGLE_CHAT_HOME_CHANNEL=spaces/AAAA...         # cron 任务的默认投递目标
GOOGLE_CHAT_MAX_MESSAGES=1                      # Pub/Sub FlowControl；1 表示每个会话串行执行命令
GOOGLE_CHAT_MAX_BYTES=16777216                  # 16 MiB — 在途消息字节上限
```

项目 ID 也可回退到 `GOOGLE_CLOUD_PROJECT`，SA 路径可回退到 `GOOGLE_APPLICATION_CREDENTIALS`——使用你偏好的约定即可。

安装 Google Chat 适配器所需的依赖（目前没有发布 Hermes extra，请直接安装）：

```bash
pip install google-cloud-pubsub google-api-python-client google-auth google-auth-oauthlib
```

启动 gateway（网关）：

```bash
hermes gateway
```

你应该会看到如下日志：

```
[GoogleChat] Connected; project=my-chat-bot-123, subscription=<redacted>,
             bot_user_id=users/XXXX, flow_control(msgs=1, bytes=16777216)
```

在测试私信中发送"hola"。机器人会先发送一条"Hermes is thinking…"占位消息，然后原地编辑该消息为真实回复——不会留下"消息已删除"的墓碑。

---

## 格式化与功能

Google Chat 支持有限的 Markdown 子集：

| 支持 | 不支持 |
|-----------|---------------|
| `*粗体*`、`_斜体_`、`~删除线~`、`` `代码` `` | 标题、列表 |
| 通过 URL 内联图片 | 交互式 Card v2 按钮（此 gateway 为 v1） |
| 原生文件附件（执行 `/setup-files` 后——见第十步） | 原生语音消息 / 圆形视频消息 |

Agent 的系统 prompt（提示词）包含 Google Chat 专属提示，使其了解这些限制，避免使用无法渲染的格式。

消息大小限制：每条消息 4000 个字符。较长的 agent 回复会自动拆分为多条消息。

Thread（线程）支持：当用户在 thread 中回复时，Hermes 会检测 `thread.name` 并在同一 thread 中发送回复，每个 thread 对应独立的 Hermes 会话。

---

## 第十步：原生附件投递（可选）

默认情况下，机器人可以发送文本、通过 URL 内联图片，以及音频/视频/文档的下载卡片。若要投递**原生** Chat 附件——即人工拖放文件时出现的文件 widget——每位用户需通过一次性 OAuth 流程授权机器人。

### 为何需要单独的流程

Google Chat 的 `media.upload` 端点会硬拒绝 service account 认证：

> This method doesn't support app authentication with a service account.
> Authenticate with a user account.

没有任何 IAM 角色或 scope 能解决这个问题。该端点只接受用户凭据。因此，机器人在上传文件时必须*以用户身份*操作——具体来说，是以请求文件的用户身份。

### 一次性宿主机设置

1. 在同一 GCP 项目中，进入 **APIs & Services → Credentials**。
2. **Create credentials → OAuth client ID → Desktop app**。
3. 下载 JSON 文件，移动到运行 Hermes 的宿主机上。
4. 在宿主机上，向 Hermes 注册该客户端：

```bash
python -m gateway.platforms.google_chat_user_oauth \
    --client-secret /path/to/client_secret.json
```

该命令会写入 `~/.hermes/google_chat_user_client_secret.json`。这是共享基础设施——它标识 OAuth *应用*，而非某个具体用户。无论后续有多少用户授权，每台宿主机只需一个文件。

### 每用户授权（在 Chat 中操作）

每位用户在与机器人的私信中执行一次流程：

1. 向机器人发送 `/setup-files`，机器人回复当前状态和下一步操作。
2. 发送 `/setup-files start`，机器人回复一个 OAuth URL。
3. 打开该 URL，点击 **Allow**，浏览器会尝试加载 `http://localhost:1/?...&code=...` 并失败。这是预期行为——auth code 在地址栏的 URL 中。
4. 复制失败的 URL（或仅复制 `code=...` 的值），粘贴回 Chat 中作为 `/setup-files <PASTED_URL>`。机器人将其换取 refresh token。

token 保存在 `~/.hermes/google_chat_user_tokens/<sanitized_email>.json`。该用户私信中后续的文件请求将使用*其*token，机器人以其身份上传，消息投递到其 space。

如需撤销：`/setup-files revoke` 仅删除该用户的 token，其他用户的 token 不受影响。

### Scope

该流程仅请求一个 scope：`chat.messages.create`。它同时覆盖 `media.upload` 和引用已上传 `attachmentDataRef` 的 `messages.create`。没有 Drive，没有更广泛的 Chat scope——这是有意为之的最小权限原则。

### 多用户行为

当请求者尚无每用户 token 时，机器人会回退到 `~/.hermes/google_chat_user_token.json` 中的旧版单用户 token（如果存在于多用户支持之前的安装中）。两者均不可用时，机器人会发送清晰的文字提示，告知请求者运行 `/setup-files`。

用户撤销只清除自己的槽位。某用户 token 产生的 401/403 只驱逐该用户的缓存，不影响其他用户。

---

## 故障排查

**发送"hola"后机器人没有任何响应。**

1. 在控制台检查 Pub/Sub 订阅是否有未投递消息。如果有，说明 Hermes 未通过认证——验证 `GOOGLE_CHAT_SERVICE_ACCOUNT_JSON`，并确认 SA 在订阅上具有 `Pub/Sub Subscriber` 角色。
2. 如果订阅中消息数为零，说明 Google Chat 没有发布消息。再次检查 **topic** 上的 IAM 绑定：`chat-api-push@system.gserviceaccount.com` 必须具有 `Pub/Sub Publisher` 角色。
3. 检查 `hermes gateway` 日志中是否有 `[GoogleChat] Connected`。如果看到 `[GoogleChat] Config validation failed`，错误信息会告诉你需要修复哪个环境变量。

**机器人有回复，但显示的是错误信息而非 agent 的答案。**

检查日志中是否有 `[GoogleChat] Pub/Sub stream died`——如果反复出现，可能是 SA 凭据已轮换或订阅已被删除。重试 10 次后，适配器会将自身标记为致命错误。

**每条出站消息都返回"403 Forbidden"。**

机器人已被从 space 中移除，或你在 Chat API 控制台中撤销了它。在 space 中重新安装（下一个 `ADDED_TO_SPACE` 事件会自动恢复消息发送功能）。

**出现过多"Rate limit hit"警告。**

Chat API 默认配额为每个 space 每分钟 60 条消息。如果 agent 产生的长流式回复超过该限制，适配器会以指数退避重试——但用户仍会感受到延迟。建议使用简洁回复，或在 GCP 控制台中提升配额。

**机器人持续发送"/setup-files"提示而非文件。**

请求者没有每用户 OAuth token，也没有旧版回退。在其私信中运行 `/setup-files` 并按照第十步操作。交换完成后，下次文件请求将原生上传，无需重启 gateway。

**`/setup-files start` 提示"No client credentials stored on the host."**

一次性宿主机设置未完成。在运行 Hermes 的宿主机终端中执行：

```bash
python -m gateway.platforms.google_chat_user_oauth \
    --client-secret /path/to/client_secret.json
```

然后再次发送 `/setup-files start`。

**`/setup-files <PASTED_URL>` 提示"Token exchange failed."**

auth code 是一次性的且有效期很短（通常几分钟）。发送 `/setup-files start` 获取新 URL 后重试。

---

## 安全说明

- **Service Account scope**：适配器请求 `chat.bot` 和 `pubsub` scope。IAM 应作为实际执行层——仅授予 SA 最小权限（订阅上的 `roles/pubsub.subscriber` + `roles/pubsub.viewer`），不要授予项目级或组织级 Pub/Sub 角色。
- **附件下载保护**：Hermes 只会将 SA bearer token 附加到主机名匹配 Google 自有域名短名单的 URL（`googleapis.com`、`drive.google.com`、`lh[3-6].googleusercontent.com` 等）。其他主机在发起 HTTP 请求前即被拒绝，以防范 SSRF 场景——即精心构造的事件将 bearer token 重定向到 GCE 元数据服务。
- **脱敏处理**：Service Account 邮箱、订阅路径和 topic 路径会被 `agent/redact.py` 从日志输出中剥离。调试信封转储（`GOOGLE_CHAT_DEBUG_RAW=1`）经过同一脱敏过滤器，以 DEBUG 级别记录。
- **合规性**：如果你计划将此机器人接入受监管的 Workspace（任何有数据驻留或 AI 治理政策的环境），请在首次安装前获得相应审批。
- **用户 OAuth scope**：每用户附件流程*仅*请求 `chat.messages.create`——覆盖 `media.upload` 及后续 `messages.create` 所需的最小权限。token 以明文 JSON 形式持久化在 `~/.hermes/google_chat_user_tokens/<sanitized_email>.json`（文件系统权限是保护手段——与 SA 密钥文件采用相同模型）。每个 token 归属于唯一一位用户；撤销操作仅限于该用户。