---
sidebar_position: 12
title: "Google Chat"
description: "使用 Cloud Pub/Sub 将 Hermes Agent 设置为 Google Chat 机器人"
---

# Google Chat 设置

将 Hermes Agent 连接到 Google Chat 作为机器人。该集成使用 Cloud Pub/Sub 拉取订阅来接收入站事件，并使用 Chat REST API 发送出站消息。与 Slack Socket Mode 或 Telegram 长轮询具有相同的人体工程学：你的 Hermes 进程不需要公网 URL、隧道或 TLS 证书。它连接、认证并监听订阅 —— 与 Telegram 机器人在 token 上监听的方式相同。

:::note 工作区版本
Google Chat 是 Google Workspace 的一部分。你可以将此集成用于个人 Workspace（通过 Google 注册的 `@yourdomain.com`）或你拥有发布应用管理员权限的工作 Workspace。仅有 Gmail 的账号无法托管 Chat 应用。
:::

## 概览

| 组件 | 值 |
|-----------|-------|
| **库** | `google-cloud-pubsub`、`google-api-python-client`、`google-auth` |
| **入站传输** | Cloud Pub/Sub 拉取订阅（无公网端点） |
| **出站传输** | Chat REST API（`chat.googleapis.com`） |
| **认证** | 服务账号 JSON，订阅上具有 `roles/pubsub.subscriber` |
| **用户识别** | Chat 资源名称（`users/{id}`）+ 邮箱 |

---

## 步骤 1：创建或选择一个 GCP 项目

你需要一个 Google Cloud 项目来托管 Pub/Sub 主题。如果你没有，请在 [console.cloud.google.com](https://console.cloud.google.com) 创建一个 —— 个人账户有免费层级，可以轻松覆盖机器人流量。

记下项目 ID（例如 `my-chat-bot-123`）。你将在每个后续步骤中使用它。

---

## 步骤 2：启用两个 API

在控制台中，转到 **APIs & Services → Library** 并启用：

- **Google Chat API**
- **Cloud Pub/Sub API**

两者对于个人机器人产生的流量都是免费的。

---

## 步骤 3：创建服务账号

**IAM & Admin → Service Accounts → Create Service Account。**

- 名称：`hermes-chat-bot`
- 跳过"授予此服务账号对项目的访问权限"步骤。你只需要特定订阅上的 IAM —— **不要**授予项目级 Pub/Sub 角色。

创建后，打开 SA，转到 **Keys → Add Key → Create new key → JSON** 并下载文件。将其保存在只有 Hermes 可以读取的地方（例如 `~/.hermes/google-chat-sa.json`，`chmod 600`）。

:::caution 没有"Chat Bot Caller"角色
一个常见错误是搜索特定于 Chat 的 IAM 角色并在项目级别授予它。该角色不存在。Chat 机器人的权限来自安装在空间中，而不是来自 IAM。你的 SA 只需要在下一步创建的主题的订阅上具有 Pub/Sub subscriber。
:::

---

## 步骤 4：创建 Pub/Sub 主题和订阅

**Pub/Sub → Topics → Create topic。**

- 主题 ID：`hermes-chat-events`
- 其他保持默认。

创建后，主题的详情页面有一个 **Subscriptions** 标签。创建一个：

- 订阅 ID：`hermes-chat-events-sub`
- 交付类型：**Pull**
- 消息保留：**7 天**（以便在 hermes 重启后保留积压）
- 其他保持默认。

---

## 步骤 5：主题上的 IAM 绑定（关键）

在**主题**上（不是订阅），添加一个 IAM 主体：

- 主体：`chat-api-push@system.gserviceaccount.com`
- 角色：`Pub/Sub Publisher`

没有这个，Google Chat 无法将事件发布到你的主题，你的机器人将永远不会收到任何东西。

---

## 步骤 6：订阅上的 IAM 绑定

在**订阅**上，将你自己的服务账号添加为主体：

- 主体：`hermes-chat-bot@<your-project>.iam.gserviceaccount.com`
- 角色：`Pub/Sub Subscriber`

同时在同一个订阅上授予 `Pub/Sub Viewer` —— Hermes 在启动时调用 `subscription.get()` 作为可达性检查。

---

## 步骤 7：配置 Chat 应用

转到 **APIs & Services → Google Chat API → Configuration**。

- **App name**：你想要用户看到的任何名称（"Hermes" 是合理的）。
- **Avatar URL**：任何公共 PNG（Google 有一些默认值）。
- **Description**：在应用目录中显示的简短句子。
- **Functionality**：启用 **Receive 1:1 messages** 和 **Join spaces and group conversations**。
- **Connection settings**：选择 **Cloud Pub/Sub**，输入主题名称 `projects/<your-project>/topics/hermes-chat-events`。
- **Visibility**：限制在你的 workspace（或特定用户）—— 在测试时不要向所有人发布。

保存。

---

## 步骤 8：在测试空间中安装机器人

在浏览器中打开 Google Chat。在 **+ New Chat** 菜单中搜索其名称，向你的应用发起 DM。第一次发送消息时，Google 会发送一个 `ADDED_TO_SPACE` 事件，Hermes 用它来缓存机器人在 `users/{id}` 中的自身 ID，用于自消息过滤。

---

## 步骤 9：配置 Hermes

将 Google Chat 部分添加到 `~/.hermes/.env`：

```bash
# 必需
GOOGLE_CHAT_PROJECT_ID=my-chat-bot-123
GOOGLE_CHAT_SUBSCRIPTION_NAME=projects/my-chat-bot-123/subscriptions/hermes-chat-events-sub
GOOGLE_CHAT_SERVICE_ACCOUNT_JSON=/home/you/.hermes/google-chat-sa.json

# 授权 — 粘贴允许与机器人交谈的人的邮箱
GOOGLE_CHAT_ALLOWED_USERS=you@yourdomain.com,coworker@yourdomain.com

# 可选
GOOGLE_CHAT_HOME_CHANNEL=spaces/AAAA...         # cron 作业的默认递送目标
GOOGLE_CHAT_MAX_MESSAGES=1                      # Pub/Sub FlowControl；1 序列化每个会话的命令
GOOGLE_CHAT_MAX_BYTES=16777216                  # 16 MiB — 飞行中消息字节上限
```

项目 ID 也可以回退到 `GOOGLE_CLOUD_PROJECT`，SA 路径可以回退到 `GOOGLE_APPLICATION_CREDENTIALS` —— 使用你喜欢的任何约定。

安装带有可选依赖的 Hermes：

```bash
pip install 'hermes-agent[google_chat]'
```

启动 gateway：

```bash
hermes gateway
```

你应该看到类似这样的日志行：

```
[GoogleChat] Connected; project=my-chat-bot-123, subscription=<redacted>,
             bot_user_id=users/XXXX, flow_control(msgs=1, bytes=16777216)
```

在测试 DM 中发送"hola"。机器人发布"Hermes is thinking…"标记，然后就地编辑同一条消息以显示真实响应 —— 没有"消息已删除"的墓碑。

---

## 格式和功能

Google Chat 只渲染有限的 markdown 子集：

| 支持 | 不支持 |
|-----------|---------------|
| `*bold*`、`_italic_`、`~strike~`、`` `code` `` | 标题、列表 |
| 通过 URL 内联图片 | 交互式卡片 v2 按钮（此 gateway 的 v1） |
| 原生文件附件（通过 `/setup-files` 后 —— 请参阅步骤 10） | 原生语音笔记/圆形视频笔记 |

agent 的系统提示包含特定于 Google Chat 的提示，因此它知道这些限制并避免使用无法渲染的格式。

消息大小限制：每条消息 4000 个字符。较长的 agent 响应会自动拆分到多条消息中。

主题支持：当用户在主题中回复时，Hermes 检测到 `thread.name` 并在同一主题中发布回复，因此每个主题都有独立的 Hermes 会话。

---

## 步骤 10：原生附件递送（可选）

开箱即用，机器人可以发布文本、通过 URL 内联图片，并为音频/视频/文档下载卡片。为了递送**原生** Chat 附件 —— 与人类拖放文件时获得的相同文件小部件 —— 每个用户通过每用户 OAuth 流程授权机器人一次。

### 为什么需要单独的流程

Google Chat 的 `media.upload` 端点硬性拒绝服务账号认证：

> 此方法不支持使用服务账号进行应用认证。
> 使用用户账号进行认证。

没有可以修复此问题的 IAM 角色或作用域。该端点只接受用户凭证。因此，机器人必须在上传文件时充当*用户* —— 具体来说，是请求文件的用户。

### 一次性主机设置

1. 在同一 GCP 项目中转到 **APIs & Services → Credentials**。
2. **创建凭证 → OAuth client ID → Desktop app**。
3. 下载 JSON。将其移动到运行 Hermes 的主机上。
4. 在主机上，向 Hermes 注册客户端：

```bash
python -m gateway.platforms.google_chat_user_oauth \
    --client-secret /path/to/client_secret.json
```

这会写入 `~/.hermes/google_chat_user_client_secret.json`。这是共享的基础设施 —— 它标识 OAuth *应用*，而不是任何个人用户。每个主机一个文件就足够了，无论以后有多少用户授权。

### 每用户授权（在聊天中）

每个用户在自己的 DM 中运行流程一次：

1. 他们向机器人发送 `/setup-files`。它回复状态和下一步。
2. 他们发送 `/setup-files start`。机器人回复一个 OAuth URL。
3. 他们打开 URL，点击 **Allow**，然后看着浏览器加载失败 `http://localhost:1/?...&code=...`。这种失败是预期的 —— 授权码在 URL 栏中。
4. 他们复制失败的 URL（或者只是 `code=...` 值）并将其作为 `/setup-files <PASTED_URL>` 粘贴回聊天中。机器人交换它以获取刷新 token。

token 落在 `~/.hermes/google_chat_user_tokens/<sanitized_email>.json`。该用户 DM 中的后续文件请求使用*他们的* token，因此机器人作为他们上传，消息落在他们的空间中。

以后要撤销：`/setup-files revoke` 只删除该用户的 token。其他用户的 token 不受影响。

### 作用域

该流程仅请求一个作用域：`chat.messages.create`。这涵盖了 `media.upload` 和引用上传的 `attachmentDataRef` 的 `messages.create`。没有 Drive，没有更广泛的 Chat 作用域 —— 这是有意为之的最小权限。

### 多用户行为

当请求者还没有每用户 token 时，机器人会回退到 `~/.hermes/google_chat_user_token.json`（如果来自预多用户安装）的旧版单用户 token。当两者都不可用时，机器人会发布一条明确的纯文本通知，告诉请求者运行 `/setup-files`。

用户撤销只清除他们自己的槽。一个用户 token 的 401/403 只会驱逐该用户的缓存。用户不会相互干扰。

---

## 故障排除

**发送"hola"后机器人保持沉默。**

1. 检查 Pub/Sub 订阅在控制台中有未送达的消息。
   如果有，Hermes 未通过认证 —— 验证 `GOOGLE_CHAT_SERVICE_ACCOUNT_JSON` 以及 SA 是否被列为订阅上的 `Pub/Sub Subscriber`。
2. 如果订阅有零条消息，Google Chat 没有发布。
   仔细检查**主题**上的 IAM 绑定：
   `chat-api-push@system.gserviceaccount.com` 必须具有 `Pub/Sub Publisher`。
3. 检查 `hermes gateway` 日志中的 `[GoogleChat] Connected`。如果你看到
   `[GoogleChat] Config validation failed`，错误消息会告诉你修复哪个环境变量。

**机器人回复但出现错误消息而不是 agent 的答案。**

检查日志中的 `[GoogleChat] Pub/Sub stream died` —— 如果这些重复，你的 SA 凭证可能已被轮换或订阅被删除。10 次尝试后，适配器将自己标记为致命。

**每条出站消息都出现"403 Forbidden"。**

机器人已从空间中移除，或者你在 Chat API 控制台中撤销了它。重新安装到空间中（下一个 `ADDED_TO_SPACE` 事件会自动重新启用消息传递）。

**太多"Rate limit hit"警告。**

Chat API 的默认配额允许每个空间每分钟 60 条消息。如果你的 agent 产生超过该限制的长流式响应，适配器会以指数退避重试 —— 但你仍会看到用户可见的延迟。考虑简洁的响应或在 GCP 控制台中提高配额。

**机器人不断发布"/setup-files"通知而不是文件。**

请求者没有每用户 OAuth token，也没有旧版回退。在他们的 DM 中运行 `/setup-files` 并按照步骤 10 进行。交换完成后，下一个文件请求会在不重启 gateway 的情况下原生上传。

**`/setup-files start` 说"No client credentials stored on the host."**

没有完成一次性主机设置。从运行 Hermes 的主机上的终端：

```bash
python -m gateway.platforms.google_chat_user_oauth \
    --client-secret /path/to/client_secret.json
```

然后再次发送 `/setup-files start`。

**`/setup-files <PASTED_URL>` 说"Token exchange failed."**

授权码是一次性的且生命周期很短（通常几分钟）。发送 `/setup-files start` 以获取新的 URL 并重试。

---

## 安全说明

- **服务账号范围**：适配器请求 `chat.bot` 和 `pubsub` 作用域。IAM 应该是实际的执行 —— 授予你的 SA 最小权限（订阅上的 `roles/pubsub.subscriber` + `roles/pubsub.viewer`），而不是项目级或组织级 Pub/Sub 角色。
- **附件下载保护**：Hermes 只会将 SA bearer token 附加到其主机与 Google 拥有的域的简短允许列表匹配（`googleapis.com`、`drive.google.com`、`lh[3-6].googleusercontent.com` 等）的 URL。任何其他主机在 HTTP 请求之前被拒绝，以防止精心设计的事件可能将 bearer token 重定向到 GCE 元数据服务的 SSRF 场景。
- **编辑**：服务账号邮箱、订阅路径和主题路径由 `agent/redact.py` 从日志输出中剥离。调试信封转储（`GOOGLE_CHAT_DEBUG_RAW=1`）通过相同的编辑过滤器路由，并在 DEBUG 级别记录。
- **合规性**：如果你计划将此机器人连接到受监管的工作区（任何具有数据驻留或 AI 治理策略的内容），请在首次安装前获得批准。
- **用户 OAuth 作用域**：每用户附件流程仅请求 `chat.messages.create` —— 涵盖 `media.upload` 以及后续 `messages.create` 的最小权限。Token 作为纯 JSON 持久化在 `~/.hermes/google_chat_user_tokens/<sanitized_email>.json`（文件系统权限是保护 —— 与 SA 密钥文件相同的模型）。每个 token 仅由一个用户拥有；撤销的作用域限定于该用户。
