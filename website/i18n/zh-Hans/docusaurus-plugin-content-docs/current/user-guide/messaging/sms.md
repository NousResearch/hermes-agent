---
sidebar_position: 8
sidebar_label: "SMS (Twilio)"
title: "SMS (Twilio)"
description: "通过 Twilio 将 Hermes Agent 设置为 SMS 聊天机器人"
---

# SMS 设置（Twilio）

Hermes 通过 [Twilio](https://www.twilio.com/) API 接入 SMS。用户向你的 Twilio 电话号码发送短信，即可获得 AI 回复——与 Telegram 或 Discord 的对话体验相同，但通过标准短信进行。

:::info 共享凭据
SMS gateway（网关）与可选的 [telephony skill](/reference/skills-catalog) 共享凭据。如果你已为语音通话或单次 SMS 配置了 Twilio，该 gateway 可直接使用相同的 `TWILIO_ACCOUNT_SID`、`TWILIO_AUTH_TOKEN` 和 `TWILIO_PHONE_NUMBER`。
:::

---

## 前提条件

- **Twilio 账户** — [在 twilio.com 注册](https://www.twilio.com/try-twilio)（提供免费试用）
- **具备 SMS 功能的 Twilio 电话号码**
- **可公开访问的服务器** — Twilio 在收到 SMS 时会向你的服务器发送 webhook
- **aiohttp** — `pip install 'hermes-agent[sms]'`

---

## 第一步：获取 Twilio 凭据

1. 前往 [Twilio 控制台](https://console.twilio.com/)
2. 从仪表板复制你的 **Account SID** 和 **Auth Token**
3. 前往 **Phone Numbers → Manage → Active Numbers**，记录 E.164 格式的电话号码（例如 `+15551234567`）

---

## 第二步：配置 Hermes

### 交互式设置（推荐）

```bash
hermes gateway setup
```

从平台列表中选择 **SMS (Twilio)**，向导将提示你输入凭据。

### 手动设置

在 `~/.hermes/.env` 中添加：

```bash
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+15551234567

# 安全：限制特定电话号码（推荐）
SMS_ALLOWED_USERS=+15559876543,+15551112222

# 可选：为 cron 任务投递设置主频道
SMS_HOME_CHANNEL=+15559876543
```

---

## 第三步：配置 Twilio Webhook

Twilio 需要知道将传入消息发送到哪里。在 [Twilio 控制台](https://console.twilio.com/) 中：

1. 前往 **Phone Numbers → Manage → Active Numbers**
2. 点击你的电话号码
3. 在 **Messaging → A MESSAGE COMES IN** 下，设置：
   - **Webhook**：`https://your-server:8080/webhooks/twilio`
   - **HTTP Method**：`POST`

:::tip 暴露你的 Webhook
如果你在本地运行 Hermes，请使用隧道工具暴露 webhook：

```bash
# 使用 cloudflared
cloudflared tunnel --url http://localhost:8080

# 使用 ngrok
ngrok http 8080
```

将生成的公网 URL 设置为你的 Twilio webhook。
:::

**将 `SMS_WEBHOOK_URL` 设置为你在 Twilio 中配置的相同 URL。** 这是 Twilio 签名验证所必需的——如果未设置，适配器将拒绝启动：

```bash
# 必须与 Twilio 控制台中的 webhook URL 一致
SMS_WEBHOOK_URL=https://your-server:8080/webhooks/twilio
```

webhook 端口默认为 `8080`，可通过以下方式覆盖：

```bash
SMS_WEBHOOK_PORT=3000
```

---

## 第四步：启动 Gateway

```bash
hermes gateway
```

你应该看到：

```
[sms] Twilio webhook server listening on 127.0.0.1:8080, from: +1555***4567
```

如果看到 `Refusing to start: SMS_WEBHOOK_URL is required`，请将 `SMS_WEBHOOK_URL` 设置为你在 Twilio 控制台中配置的公网 URL（参见第三步）。

向你的 Twilio 号码发送短信——Hermes 将通过 SMS 回复。

---

## 环境变量

| 变量 | 是否必填 | 说明 |
|----------|----------|-------------|
| `TWILIO_ACCOUNT_SID` | 是 | Twilio Account SID（以 `AC` 开头） |
| `TWILIO_AUTH_TOKEN` | 是 | Twilio Auth Token（同时用于 webhook 签名验证） |
| `TWILIO_PHONE_NUMBER` | 是 | 你的 Twilio 电话号码（E.164 格式） |
| `SMS_WEBHOOK_URL` | 是 | 用于 Twilio 签名验证的公网 URL——必须与 Twilio 控制台中的 webhook URL 一致 |
| `SMS_WEBHOOK_PORT` | 否 | Webhook 监听端口（默认：`8080`） |
| `SMS_WEBHOOK_HOST` | 否 | Webhook 绑定地址（默认：`0.0.0.0`） |
| `SMS_INSECURE_NO_SIGNATURE` | 否 | 设为 `true` 可禁用签名验证（仅限本地开发——**不适用于生产环境**） |
| `SMS_ALLOWED_USERS` | 否 | 允许聊天的 E.164 格式电话号码，逗号分隔 |
| `SMS_ALLOW_ALL_USERS` | 否 | 设为 `true` 允许所有人（不推荐） |
| `SMS_HOME_CHANNEL` | 否 | 用于 cron 任务／通知投递的电话号码 |
| `SMS_HOME_CHANNEL_NAME` | 否 | 主频道的显示名称（默认：`Home`） |

---

## SMS 特有行为

- **纯文本** — Markdown 会被自动剥离，因为 SMS 会将其渲染为字面字符
- **1600 字符限制** — 较长的回复会在自然边界处（换行符，其次是空格）拆分为多条消息
- **防回声** — 来自你自己 Twilio 号码的消息将被忽略，以防止循环
- **电话号码脱敏** — 日志中的电话号码会被脱敏处理以保护隐私

---

## 安全

### Webhook 签名验证

Hermes 通过验证 `X-Twilio-Signature` 头（HMAC-SHA1）来确认入站 webhook 确实来自 Twilio，防止攻击者注入伪造消息。

**`SMS_WEBHOOK_URL` 为必填项。** 将其设置为你在 Twilio 控制台中配置的公网 URL，否则适配器将拒绝启动。

如需在本地开发时不使用公网 URL，可禁用验证：

```bash
# 仅限本地开发——不适用于生产环境
SMS_INSECURE_NO_SIGNATURE=true
```

### 用户白名单

**Gateway 默认拒绝所有用户。** 请配置白名单：

```bash
# 推荐：限制特定电话号码
SMS_ALLOWED_USERS=+15559876543,+15551112222

# 或允许所有人（对于具有终端访问权限的机器人，不推荐）
SMS_ALLOW_ALL_USERS=true
```

:::warning
SMS 没有内置加密。除非你了解相关安全风险，否则不要通过 SMS 进行敏感操作。对于敏感场景，请优先使用 Signal 或 Telegram。
:::

---

## 故障排查

### 消息未到达

1. 检查 Twilio webhook URL 是否正确且可公开访问
2. 验证 `TWILIO_ACCOUNT_SID` 和 `TWILIO_AUTH_TOKEN` 是否正确
3. 在 Twilio 控制台 → **Monitor → Logs → Messaging** 中查看投递错误
4. 确保你的电话号码在 `SMS_ALLOWED_USERS` 中（或设置 `SMS_ALLOW_ALL_USERS=true`）

### 回复未发送

1. 检查 `TWILIO_PHONE_NUMBER` 是否正确设置（E.164 格式，带 `+`）
2. 验证你的 Twilio 账户是否有支持 SMS 的号码
3. 查看 Hermes gateway 日志中的 Twilio API 错误

### Webhook 端口冲突

如果 8080 端口已被占用，请更改端口：

```bash
SMS_WEBHOOK_PORT=3001
```

并在 Twilio 控制台中更新 webhook URL 以匹配新端口。