---
sidebar_position: 8
sidebar_label: "SMS (Twilio)"
title: "SMS (Twilio)"
description: "通过 Twilio 将 Hermes Agent 设置为 SMS 聊天机器人"
---

# SMS 设置（Twilio）

Hermes 通过 [Twilio](https://www.twilio.com/) API 连接到 SMS。人们给你的 Twilio 电话号码发短信，然后收到 AI 响应 —— 与 Telegram 或 Discord 相同的对话体验，但通过标准文本消息。

:::info 共享凭证
SMS gateway 与可选的[电话技能](/docs/reference/skills-catalog)共享凭证。如果你已经为语音通话或一次性 SMS 设置了 Twilio，gateway 使用相同的 `TWILIO_ACCOUNT_SID`、`TWILIO_AUTH_TOKEN` 和 `TWILIO_PHONE_NUMBER`。
:::

---

## 前提条件

- **Twilio 账号** — 在 [twilio.com](https://www.twilio.com/try-twilio) 注册（免费试用可用）
- **具有 SMS 功能的 Twilio 电话号码**
- **公开可访问的服务器** — SMS 到达时 Twilio 向你的服务器发送 webhook
- **aiohttp** — `pip install 'hermes-agent[sms]'`

---

## 步骤 1：获取你的 Twilio 凭证

1. 转到 [Twilio Console](https://console.twilio.com/)
2. 从仪表板复制你的 **Account SID** 和 **Auth Token**
3. 转到 **Phone Numbers → Manage → Active Numbers** — 记下你的电话号码，格式为 E.164（例如 `+15551234567`）

---

## 步骤 2：配置 Hermes

### 交互式设置（推荐）

```bash
hermes gateway setup
```

从平台列表中选择 **SMS (Twilio)**。向导将提示你输入凭证。

### 手动设置

添加到 `~/.hermes/.env`：

```bash
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+15551234567

# 安全：限制为特定电话号码（推荐）
SMS_ALLOWED_USERS=+15559876543,+15551112222

# 可选：为 cron 作业递送设置主页频道
SMS_HOME_CHANNEL=+15559876543
```

---

## 步骤 3：配置 Twilio Webhook

Twilio 需要知道 SMS 到达时将消息发送到哪里。在 [Twilio Console](https://console.twilio.com/) 中：

1. 转到 **Phone Numbers → Manage → Active Numbers**
2. 点击你的电话号码
3. 在 **Messaging → A MESSAGE COMES IN** 下，设置：
   - **Webhook**：`https://your-server:8080/webhooks/twilio`
   - **HTTP Method**：`POST`

:::tip 暴露你的 Webhook
如果你在本地运行 Hermes，使用隧道来暴露 webhook：

```bash
# 使用 cloudflared
cloudflared tunnel --url http://localhost:8080

# 使用 ngrok
ngrok http 8080
```

将生成的公网 URL 设置为你的 Twilio webhook。
:::

**将 `SMS_WEBHOOK_URL` 设置为你配置在 Twilio 中的相同 URL。** 这用于 Twilio 签名验证 —— 适配器将在没有它的情况下拒绝启动：

```bash
# 必须与你的 Twilio Console 中的 webhook URL 匹配
SMS_WEBHOOK_URL=https://your-server:8080/webhooks/twilio
```

Webhook 端口默认为 `8080`。使用以下命令覆盖：

```bash
SMS_WEBHOOK_PORT=3000
```

---

## 步骤 4：启动 Gateway

```bash
hermes gateway
```

你应该看到：

```
[sms] Twilio webhook server listening on 0.0.0.0:8080, from: +1555***4567
```

如果你看到 `Refusing to start: SMS_WEBHOOK_URL is required`，请设置 `SMS_WEBHOOK_URL` 到你在 Twilio Console 中配置的公网 URL（请参阅步骤 3）。

向你的 Twilio 电话号码发短信 — Hermes 将通过 SMS 回复。

---

## 环境变量

| 变量 | 必需 | 描述 |
|----------|----------|-------------|
| `TWILIO_ACCOUNT_SID` | 是 | Twilio Account SID（以 `AC` 开头） |
| `TWILIO_AUTH_TOKEN` | 是 | Twilio Auth Token（也用于 webhook 签名验证） |
| `TWILIO_PHONE_NUMBER` | 是 | 你的 Twilio 电话号码（E.164 格式） |
| `SMS_WEBHOOK_URL` | 是 | 用于 Twilio 签名验证的公网 URL — 必须与你的 Twilio Console 中的 webhook URL 匹配 |
| `SMS_WEBHOOK_PORT` | 否 | Webhook 监听器端口（默认：`8080`） |
| `SMS_WEBHOOK_HOST` | 否 | Webhook 绑定地址（默认：`0.0.0.0`） |
| `SMS_INSECURE_NO_SIGNATURE` | 否 | 设置为 `true` 以禁用签名验证（仅本地开发 — **不适用于生产**） |
| `SMS_ALLOWED_USERS` | 否 | 允许聊天的逗号分隔的 E.164 电话号码 |
| `SMS_ALLOW_ALL_USERS` | 否 | 设置为 `true` 以允许任何人（不推荐） |
| `SMS_HOME_CHANNEL` | 否 | 用于 cron 作业/通知递送的电话号码 |
| `SMS_HOME_CHANNEL_NAME` | 否 | 主页频道的显示名称（默认：`Home`） |

---

## SMS 特定行为

- **仅纯文本** — Markdown 自动去除，因为 SMS 将其渲染为字面字符
- **1600 字符限制** — 较长的响应在自然边界（新行，然后是空格）处拆分到多条消息
- **回声预防** — 忽略来自你自己的 Twilio 电话号码的消息以防止循环
- **电话号码编辑** — 电话号码在日志中出于隐私原因被编辑

---

## 安全

### Webhook 签名验证

Hermes 通过验证 `X-Twilio-Signature` 标头（HMAC-SHA1）来验证入站 webhook 确实来自 Twilio。这可以防止攻击者注入伪造的消息。

**`SMS_WEBHOOK_URL` 是必需的。** 将其设置为你在 Twilio Console 中配置的公网 URL。适配器将在没有它的情况下拒绝启动。

对于没有公网 URL 的本地开发，你可以禁用验证：

```bash
# 仅本地开发 — 不适用于生产
SMS_INSECURE_NO_SIGNATURE=true
```

### 用户白名单

**gateway 默认拒绝所有用户。** 配置白名单：

```bash
# 推荐：限制为特定电话号码
SMS_ALLOWED_USERS=+15559876543,+15551112222

# 或允许所有人（对于具有终端访问权限的机器人不推荐）
SMS_ALLOW_ALL_USERS=true
```

:::warning
SMS 没有内置加密。除非你了解安全含义，否则不要将 SMS 用于敏感操作。对于敏感用例，首选 Signal 或 Telegram。
:::

---

## 故障排除

### 消息未到达

1. 检查你的 Twilio webhook URL 正确且公开可访问
2. 验证 `TWILIO_ACCOUNT_SID` 和 `TWILIO_AUTH_TOKEN` 正确
3. 检查 Twilio Console → **Monitor → Logs → Messaging** 中的递送错误
4. 确保你的电话号码在 `SMS_ALLOWED_USERS` 中（或 `SMS_ALLOW_ALL_USERS=true`）

### 回复未发送

1. 检查 `TWILIO_PHONE_NUMBER` 设置正确（E.164 格式，带 `+`）
2. 验证你的 Twilio 账号有 SMS 功能电话号码
3. 检查 Hermes gateway 日志中的 Twilio API 错误

### Webhook 端口冲突

如果端口 8080 已被使用，更改它：

```bash
SMS_WEBHOOK_PORT=3001
```

更新 Twilio Console 中的 webhook URL 以匹配。
