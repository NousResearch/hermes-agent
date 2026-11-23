---
sidebar_position: 7
title: "电子邮件"
description: "通过 IMAP/SMTP 将 Hermes Agent 设置为电子邮件助手"
---

# 电子邮件设置

Hermes 可以使用标准 IMAP 和 SMTP 协议接收并回复电子邮件。向 Agent 的邮箱地址发送邮件，它会在同一线程中回复——无需特殊客户端或 bot API。支持 Gmail、Outlook、Yahoo、Fastmail，以及任何支持 IMAP/SMTP 的邮件服务商。

:::info 无外部依赖
Email 适配器使用 Python 内置的 `imaplib`、`smtplib` 和 `email` 模块，无需额外安装软件包或外部服务。
:::

---

## 前提条件

- **为 Hermes Agent 准备一个专用邮箱账户**（不要使用个人邮箱）
- **在该邮箱账户上启用 IMAP**
- **如果使用 Gmail 或其他开启了双重验证的服务商，需要准备应用专用密码**

### Gmail 设置

1. 在 Google 账户上启用双重验证（2FA）
2. 前往 [应用专用密码](https://myaccount.google.com/apppasswords)
3. 创建一个新的应用专用密码（选择"邮件"或"其他"）
4. 复制这个 16 位密码——使用它代替常规密码

### Outlook / Microsoft 365

1. 前往 [安全设置](https://account.microsoft.com/security)
2. 如尚未启用，请开启双重验证
3. 在"其他安全选项"下创建应用专用密码
4. IMAP 主机：`outlook.office365.com`，SMTP 主机：`smtp.office365.com`

### 其他服务商

大多数邮件服务商支持 IMAP/SMTP。请查阅服务商文档，了解：
- IMAP 主机和端口（通常为端口 993，使用 SSL）
- SMTP 主机和端口（通常为端口 587，使用 STARTTLS）
- 是否需要应用专用密码

---

## 第一步：配置 Hermes

最简便的方式：

```bash
hermes gateway setup
```

从平台菜单中选择 **Email**。向导会提示输入邮箱地址、密码、IMAP/SMTP 主机以及允许的发件人。

### 手动配置

在 `~/.hermes/.env` 中添加：

```bash
# 必填
EMAIL_ADDRESS=hermes@gmail.com
EMAIL_PASSWORD=abcd efgh ijkl mnop    # 应用专用密码（非常规密码）
EMAIL_IMAP_HOST=imap.gmail.com
EMAIL_SMTP_HOST=smtp.gmail.com

# 安全设置（推荐）
EMAIL_ALLOWED_USERS=your@email.com,colleague@work.com
# 必填：接收服务器在最上层 Authentication-Results 中写入的精确 authserv-id（Gmail 为 mx.google.com）。
# 该服务器须先移除声称此 id 的入站头部，再写入自己的结果。
EMAIL_AUTHSERV_ID=mx.google.com

# 可选
EMAIL_IMAP_PORT=993                    # 默认：993（IMAP SSL）
EMAIL_SMTP_PORT=587                    # 默认：587（SMTP STARTTLS）
EMAIL_POLL_INTERVAL=15                 # 收件箱检查间隔（秒），默认：15
EMAIL_HOME_ADDRESS=your@email.com      # cron 任务的默认投递目标
```

---

## 第二步：启动 Gateway

```bash
hermes gateway              # 在前台运行
hermes gateway install      # 安装为用户服务
sudo hermes gateway install --system   # 仅 Linux：开机自启的系统服务
```

启动时，适配器会：
1. 测试 IMAP 和 SMTP 连接
2. 将收件箱中所有现有邮件标记为"已读"（仅处理新邮件）
3. 开始轮询新邮件

---

## 工作原理

### 接收邮件

适配器按可配置的间隔（默认：15 秒）轮询 IMAP 收件箱中的未读邮件。对于每封新邮件：

- **主题行**作为上下文包含在内（例如 `[Subject: Deploy to production]`）
- **回复邮件**（主题以 `Re:` 开头）跳过主题前缀——线程上下文已经建立
- **附件**会缓存到本地：
  - 图片（JPEG、PNG、GIF、WebP）→ 可供视觉工具使用
  - 文档（PDF、ZIP 等）→ 可供文件访问工具使用
- **纯 HTML 邮件**会剥离标签以提取纯文本
- **自发邮件**会被过滤，防止回复循环
- **自动化/无回复发件人**会被静默忽略——`noreply@`、`mailer-daemon@`、`bounce@`、`no-reply@`，以及包含 `Auto-Submitted`、`Precedence: bulk` 或 `List-Unsubscribe` 头部的邮件

### 发送回复

回复通过 SMTP 发送，并正确维护邮件线程：

- **In-Reply-To** 和 **References** 头部用于维持线程
- **主题行**保留并添加 `Re:` 前缀（不会出现 `Re: Re:` 重复）
- **Message-ID** 使用 Agent 的域名生成
- 回复以纯文本（UTF-8）发送

### 文件附件

Agent 可以在回复中发送文件附件。在响应中包含 `MEDIA:/path/to/file`，该文件将作为附件添加到发出的邮件中。

### 跳过附件

如需忽略所有传入附件（用于防范恶意软件或节省带宽），在 `config.yaml` 中添加：

```yaml
platforms:
  email:
    skip_attachments: true
```

启用后，附件和内嵌部分会在解码前被跳过，邮件正文文本仍正常处理。

---

## 访问控制

电子邮件访问遵循与所有其他 Hermes 平台相同的模式：

1. **设置了 `EMAIL_ALLOWED_USERS`** → 仅处理来自这些地址的邮件
2. **未设置白名单** → 未知发件人会收到配对码
3. **`EMAIL_ALLOW_ALL_USERS=true`** → 接受任意发件人（请谨慎使用）

**发件人认证开启时（默认），必须设置 `authserv_id` pin。** 未设置 `platforms.email.authserv_id`（`EMAIL_AUTHSERV_ID`）时，Hermes 无法区分你的服务器写入的 `Authentication-Results` 头与发件人伪造的头（自建 MTA 若不写入自己的结果，发件人的头就会位于最上层），因此不信任任何头部：所有需要已认证 `From:` 的邮件都会被丢弃，gateway 日志会给出一次带修复方法的警告。获取方法：打开一封服务器投递的邮件原文，复制**最上层** `Authentication-Results` 头中第一个 `;` 之前的标识，例如 Gmail 为 `mx.google.com`。若该标识随邮件变化，说明它是逐跳的主机名而非 authserv-id：RFC 8601 规定 authserv-id 是管理域的稳定名称，请让 MTA 写入固定值（例如 OpenDKIM / OpenDMARC 的 `AuthservID`）并 pin 该值。id 匹配只是 pin，并不证明来源：还须确认 MTA 会先移除声称该 id 的入站 `Authentication-Results` 头，再添加自己的头部。若无法 pin 稳定的 id，可显式设置 `platforms.email.require_authenticated_sender: false`（或 `EMAIL_TRUST_FROM_HEADER=true`）接受风险，此时 `From:` 将按原样被信任。

:::warning
**请务必配置 `EMAIL_ALLOWED_USERS`。** 若不配置，任何知道 Agent 邮箱地址的人都可以发送命令。Agent 默认具有终端访问权限。
:::

---

## 故障排查

| 问题 | 解决方案 |
|---------|----------|
| 启动时出现 **"IMAP connection failed"** | 检查 `EMAIL_IMAP_HOST` 和 `EMAIL_IMAP_PORT`。确保账户已启用 IMAP。对于 Gmail，在设置 → 转发和 POP/IMAP 中启用。 |
| 启动时出现 **"SMTP connection failed"** | 检查 `EMAIL_SMTP_HOST` 和 `EMAIL_SMTP_PORT`。确认密码正确（Gmail 请使用应用专用密码）。 |
| **未收到邮件** | 检查 `EMAIL_ALLOWED_USERS` 是否包含发件人邮箱。若 gateway 日志出现 `authserv-id is not configured`，请将 `EMAIL_AUTHSERV_ID` 设为服务器写入的最上层 `Authentication-Results` 头中的精确 id。检查垃圾邮件文件夹——部分服务商会将自动回复标记为垃圾邮件。 |
| **"Authentication failed"** | 对于 Gmail，必须使用应用专用密码，而非常规密码。请先确保已启用双重验证。 |
| **重复回复** | 确保只有一个 gateway 实例在运行。检查 `hermes gateway status`。 |
| **响应缓慢** | 默认轮询间隔为 15 秒。设置 `EMAIL_POLL_INTERVAL=5` 可加快响应速度（但会增加 IMAP 连接次数）。 |
| **回复未归入线程** | 适配器使用 In-Reply-To 头部。部分邮件客户端（尤其是网页版）可能无法正确将自动回复归入线程。 |

---

## 安全

:::warning
**请使用专用邮箱账户。** 不要使用个人邮箱——Agent 会将密码存储在 `.env` 文件中，并通过 IMAP 拥有完整的收件箱访问权限。
:::

- 使用**应用专用密码**代替主密码（Gmail 开启双重验证后必须如此）
- 设置 `EMAIL_ALLOWED_USERS` 以限制可与 Agent 交互的用户
- 将 `EMAIL_AUTHSERV_ID` 设为接收 MTA 最上层 `Authentication-Results` 头的精确 id，并确认该 MTA 会移除伪造此 id 的入站头部
- 密码存储在 `~/.hermes/.env` 中——请保护此文件（`chmod 600`）
- IMAP 默认使用 SSL（端口 993），SMTP 默认使用 STARTTLS（端口 587）——连接已加密

---

## 环境变量参考

| 变量 | 是否必填 | 默认值 | 说明 |
|----------|----------|---------|-------------|
| `EMAIL_ADDRESS` | 是 | — | Agent 的邮箱地址 |
| `EMAIL_PASSWORD` | 是 | — | 邮箱密码或应用专用密码 |
| `EMAIL_IMAP_HOST` | 是 | — | IMAP 服务器主机（例如 `imap.gmail.com`） |
| `EMAIL_SMTP_HOST` | 是 | — | SMTP 服务器主机（例如 `smtp.gmail.com`） |
| `EMAIL_IMAP_PORT` | 否 | `993` | IMAP 服务器端口 |
| `EMAIL_SMTP_PORT` | 否 | `587` | SMTP 服务器端口 |
| `EMAIL_POLL_INTERVAL` | 否 | `15` | 收件箱检查间隔（秒） |
| `EMAIL_ALLOWED_USERS` | 否 | — | 允许的发件人地址，逗号分隔 |
| `EMAIL_AUTHSERV_ID` | 是（除非关闭发件人认证） | — | 接收服务器最上层 `Authentication-Results` 头中的精确 authserv-id；未设置时所有需要已认证 `From:` 的邮件都会被丢弃 |
| `EMAIL_HOME_ADDRESS` | 否 | — | cron 任务的默认投递目标 |
| `EMAIL_ALLOW_ALL_USERS` | 否 | `false` | 允许所有发件人（不推荐） |