---
sidebar_position: 7
title: "Email"
description: "通过 IMAP/SMTP 将 Hermes Agent 设置为电子邮件助手"
---

# 电子邮件设置

Hermes 可以使用标准 IMAP 和 SMTP 协议接收和回复电子邮件。向 agent 的地址发送电子邮件，它会在同一线程中回复 —— 无需特殊的客户端或机器人 API。适用于 Gmail、Outlook、Yahoo、Fastmail 或任何支持 IMAP/SMTP 的提供商。

:::info 无外部依赖
电子邮件适配器使用 Python 内置的 `imaplib`、`smtplib` 和 `email` 模块。不需要额外的包或外部服务。
:::

---

## 前提条件

- **专用的电子邮件账号** 用于你的 Hermes agent（不要使用你的个人电子邮件）
- 电子邮件账号上**启用 IMAP**
- 如果使用 Gmail 或其他支持 2FA 的提供商，需要**应用密码**

### Gmail 设置

1. 在你的 Google 账号上启用 2 因素认证
2. 前往 [应用密码](https://myaccount.google.com/apppasswords)
3. 创建一个新的应用密码（选择"邮件"或"其他"）
4. 复制 16 位密码 —— 你将使用它来代替你的常规密码

### Outlook / Microsoft 365

1. 前往 [安全设置](https://account.microsoft.com/security)
2. 如果尚未激活，请启用 2FA
3. 在"其他安全选项"下创建应用密码
4. IMAP 主机：`outlook.office365.com`，SMTP 主机：`smtp.office365.com`

### 其他提供商

大多数电子邮件提供商都支持 IMAP/SMTP。检查你的提供商的文档以了解：

- IMAP 主机和端口（通常是端口 993，使用 SSL）
- SMTP 主机和端口（通常是端口 587，使用 STARTTLS）
- 是否需要应用密码

---

## 步骤 1：配置 Hermes

最简单的方法：

```bash
hermes gateway setup
```

从平台菜单中选择 **Email**。向导会提示你输入电子邮件地址、密码、IMAP/SMTP 主机和允许的发送者。

### 手动配置

添加到 `~/.hermes/.env`：

```bash
# 必需
EMAIL_ADDRESS=hermes@gmail.com
EMAIL_PASSWORD=abcd efgh ijkl mnop    # 应用密码（不是你的常规密码）
EMAIL_IMAP_HOST=imap.gmail.com
EMAIL_SMTP_HOST=smtp.gmail.com

# 安全（推荐）
EMAIL_ALLOWED_USERS=your@email.com,colleague@work.com

# 可选
EMAIL_IMAP_PORT=993                    # 默认：993（IMAP SSL）
EMAIL_SMTP_PORT=587                    # 默认：587（SMTP STARTTLS）
EMAIL_POLL_INTERVAL=15                 # 收件箱检查间隔秒数（默认：15）
EMAIL_HOME_ADDRESS=your@email.com      # cron 作业的默认递送目标
```

---

## 步骤 2：启动 Gateway

```bash
hermes gateway              # 在前台运行
hermes gateway install      # 安装为用户服务
sudo hermes gateway install --system   # 仅 Linux：启动时系统服务
```

启动时，适配器：

1. 测试 IMAP 和 SMTP 连接
2. 将所有现有收件箱消息标记为"已读"（仅处理新电子邮件）
3. 开始轮询新消息

---

## 工作原理

### 接收消息

适配器以可配置的间隔（默认：15 秒）轮询 IMAP 收件箱中未读（UNSEEN）的消息。对于每封新电子邮件：

- **主题行** 作为上下文包含（例如 `[Subject: Deploy to production]`）
- **回复电子邮件**（主题以 `Re:` 开头）跳过主题前缀 —— 线程上下文已经建立
- **附件** 在本地缓存：
  - 图片（JPEG、PNG、GIF、WebP）→ 可供视觉工具使用
  - 文档（PDF、ZIP 等）→ 可供文件访问
- **仅 HTML 的电子邮件** 已去除标签以提取纯文本
- **自消息** 被过滤掉以防止回复循环
- **自动/noreply 发送者** 被静默忽略 —— `noreply@`、`mailer-daemon@`、`bounce@`、`no-reply@`，以及带有 `Auto-Submitted`、`Precedence: bulk` 或 `List-Unsubscribe` 标头的电子邮件

### 发送回复

回复通过 SMTP 发送，并带有正确的电子邮件线程：

- **In-Reply-To** 和 **References** 标头维护线程
- **主题行** 保留 `Re:` 前缀（不会双重 `Re: Re:`）
- **Message-ID** 使用 agent 的域生成
- 回复以纯文本（UTF-8）发送

### 文件附件

agent 可以在回复中发送文件附件。在响应中包含 `MEDIA:/path/to/file`，文件就会附加到出站电子邮件中。

### 跳过附件

要忽略所有入站附件（用于恶意软件防护或节省带宽），请添加到你的 `config.yaml`：

```yaml
platforms:
  email:
    skip_attachments: true
```

启用后，在负载解码之前会跳过附件和内联部分。电子邮件正文文本仍会正常处理。

---

## 访问控制

电子邮件访问遵循与所有其他 Hermes 平台相同的模式：

1. **设置了 `EMAIL_ALLOWED_USERS`** → 仅处理来自这些地址的电子邮件
2. **未设置许可名单** → 未知发送者会获得配对码
3. **`EMAIL_ALLOW_ALL_USERS=true`** → 接受任何发送者（谨慎使用）

:::warning
**始终配置 `EMAIL_ALLOWED_USERS`。** 没有它，任何知道 agent 电子邮件地址的人都可以发送命令。默认情况下，agent 具有终端访问权限。
:::

---

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| 启动时出现 **"IMAP connection failed"** | 验证 `EMAIL_IMAP_HOST` 和 `EMAIL_IMAP_PORT`。确保账号上已启用 IMAP。对于 Gmail，在 设置 → 转发和 POP/IMAP 中启用它。 |
| 启动时出现 **"SMTP connection failed"** | 验证 `EMAIL_SMTP_HOST` 和 `EMAIL_SMTP_PORT`。检查你的密码是否正确（对于 Gmail，使用应用密码）。 |
| **未收到消息** | 检查 `EMAIL_ALLOWED_USERS` 是否包含发送者的电子邮件。检查垃圾邮件文件夹 —— 某些提供商会标记自动回复。 |
| **"Authentication failed"** | 对于 Gmail，你必须使用应用密码，而不是你的常规密码。确保首先启用了 2FA。 |
| **重复回复** | 确保只有一个 gateway 实例正在运行。检查 `hermes gateway status`。 |
| **响应缓慢** | 默认轮询间隔为 15 秒。使用 `EMAIL_POLL_INTERVAL=5` 减少以获得更快的响应（但会有更多的 IMAP 连接）。 |
| **回复未形成线程** | 适配器使用 In-Reply-To 标头。某些电子邮件客户端（尤其是基于 Web 的）可能无法正确地将自动消息形成线程。 |

---

## 安全

:::warning
**使用专用的电子邮件账号。** 不要使用你的个人电子邮件 —— agent 在 `.env` 中存储密码，并通过 IMAP 具有完整的收件箱访问权限。
:::

- 使用**应用密码**而不是你的主密码（对于带有 2FA 的 Gmail 是必需的）
- 设置 `EMAIL_ALLOWED_USERS` 以限制谁可以与 agent 交互
- 密码存储在 `~/.hermes/.env` 中 —— 保护此文件（`chmod 600`）
- IMAP 默认使用 SSL（端口 993），SMTP 使用 STARTTLS（端口 587）—— 连接已加密

---

## 环境变量参考

| 变量 | 必需 | 默认 | 描述 |
|----------|----------|---------|-------------|
| `EMAIL_ADDRESS` | 是 | — | Agent 的电子邮件地址 |
| `EMAIL_PASSWORD` | 是 | — | 电子邮件密码或应用密码 |
| `EMAIL_IMAP_HOST` | 是 | — | IMAP 服务器主机（例如 `imap.gmail.com`） |
| `EMAIL_SMTP_HOST` | 是 | — | SMTP 服务器主机（例如 `smtp.gmail.com`） |
| `EMAIL_IMAP_PORT` | 否 | `993` | IMAP 服务器端口 |
| `EMAIL_SMTP_PORT` | 否 | `587` | SMTP 服务器端口 |
| `EMAIL_POLL_INTERVAL` | 否 | `15` | 收件箱检查间隔秒数 |
| `EMAIL_ALLOWED_USERS` | 否 | — | 逗号分隔的允许发送者地址列表 |
| `EMAIL_HOME_ADDRESS` | 否 | — | cron 作业的默认递送目标 |
| `EMAIL_ALLOW_ALL_USERS` | 否 | `false` | 允许所有发送者（不推荐） |
