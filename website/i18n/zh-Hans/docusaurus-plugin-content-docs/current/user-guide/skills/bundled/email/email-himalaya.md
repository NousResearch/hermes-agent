---
title: "Himalaya — Himalaya CLI: IMAP/SMTP email from terminal"
sidebar_label: "Himalaya"
description: "Himalaya CLI：从终端收发 IMAP/SMTP 邮件"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Himalaya

Himalaya CLI：从终端收发 IMAP/SMTP 邮件。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/email/himalaya` |
| 版本 | `1.1.0` |
| 作者 | community |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Email`, `IMAP`, `SMTP`, `CLI`, `Communication` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Himalaya 邮件 CLI

Himalaya 是一个 CLI 邮件客户端，支持通过 IMAP、SMTP、Notmuch 或 Sendmail 后端从终端管理邮件。

## 参考资料

- `references/configuration.md`（配置文件设置 + IMAP/SMTP 认证）
- `references/message-composition.md`（用于撰写邮件的 MML 语法）

## 前置条件

1. 已安装 Himalaya CLI（运行 `himalaya --version` 验证）
2. 配置文件位于 `~/.config/himalaya/config.toml`
3. 已配置 IMAP/SMTP 凭据（密码安全存储）

### 安装

```bash
# 预编译二进制（Linux/macOS — 推荐）
curl -sSL https://raw.githubusercontent.com/pimalaya/himalaya/master/install.sh | PREFIX=~/.local sh

# macOS 通过 Homebrew
brew install himalaya

# 或通过 cargo（任何支持 Rust 的平台）
cargo install himalaya --locked
```

## 配置设置

运行交互式向导以设置账户：

```bash
himalaya account configure
```

或手动创建 `~/.config/himalaya/config.toml`：

```toml
[accounts.personal]
email = "you@example.com"
display-name = "Your Name"
default = true

backend.type = "imap"
backend.host = "imap.example.com"
backend.port = 993
backend.encryption.type = "tls"
backend.login = "you@example.com"
backend.auth.type = "password"
backend.auth.cmd = "pass show email/imap"  # or use keyring

message.send.backend.type = "smtp"
message.send.backend.host = "smtp.example.com"
message.send.backend.port = 587
message.send.backend.encryption.type = "start-tls"
message.send.backend.login = "you@example.com"
message.send.backend.auth.type = "password"
message.send.backend.auth.cmd = "pass show email/smtp"

# Folder aliases (himalaya v1.2.0+ syntax). Required whenever the
# server's folder names don't match himalaya's canonical names
# (inbox/sent/drafts/trash). Gmail is the common case — see
# `references/configuration.md` for the `[Gmail]/Sent Mail` mapping.
folder.aliases.inbox = "INBOX"
folder.aliases.sent = "Sent"
folder.aliases.drafts = "Drafts"
folder.aliases.trash = "Trash"
```

> **关于别名语法的注意事项。** v1.2.0 之前的文档使用 `[accounts.NAME.folder.alias]` 子节（单数 `alias`）。v1.2.0 会静默忽略该形式——TOML 解析正常，但别名解析器从不读取它，因此每次查找都会回退到规范名称。在 Gmail 上，这意味着 SMTP 投递成功*之后*保存到已发送文件夹会失败，且 `himalaya message send` 以非零状态退出。任何在该退出码上重试的调用方（agent、脚本、用户）都会重新执行整个发送流程——包括 SMTP——从而向收件人产生重复邮件。请始终使用 `folder.aliases.X`（复数、点分键，直接位于 `[accounts.NAME]` 下）。

## Hermes 集成说明

- **读取、列出、搜索、移动、删除**均可直接通过终端工具完成
- **撰写/回复/转发**——推荐使用管道输入（`cat << EOF | himalaya template send`）以确保可靠性。交互式 `$EDITOR` 模式可配合 `pty=true` + 后台 + 进程工具使用，但需要了解编辑器及其命令
- 使用 `--output json` 获取结构化输出，便于程序化解析
- `himalaya account configure` 向导需要交互式输入——请使用 PTY 模式：`terminal(command="himalaya account configure", pty=true)`

## 常用操作

### 列出文件夹

```bash
himalaya folder list
```

### 列出邮件

列出 INBOX 中的邮件（默认）：

```bash
himalaya envelope list
```

列出指定文件夹中的邮件：

```bash
himalaya envelope list --folder "Sent"
```

分页列出：

```bash
himalaya envelope list --page 1 --page-size 20
```

### 搜索邮件

```bash
himalaya envelope list from john@example.com subject meeting
```

### 阅读邮件

按 ID 阅读邮件（显示纯文本）：

```bash
himalaya message read 42
```

导出原始 MIME：

```bash
himalaya message export 42 --full
```

### 回复邮件

在 Hermes 中非交互式回复，请读取原始邮件、撰写回复并通过管道发送：

```bash
# 获取回复模板，编辑后发送
himalaya template reply 42 | sed 's/^$/\nYour reply text here\n/' | himalaya template send
```

或手动构建回复：

```bash
cat << 'EOF' | himalaya template send
From: you@example.com
To: sender@example.com
Subject: Re: Original Subject
In-Reply-To: <original-message-id>

Your reply here.
EOF
```

全部回复（交互式——需要 $EDITOR，建议改用上述模板方式）：

```bash
himalaya message reply 42 --all
```

### 转发邮件

```bash
# 获取转发模板并通过管道修改后发送
himalaya template forward 42 | sed 's/^To:.*/To: newrecipient@example.com/' | himalaya template send
```

### 撰写新邮件

**非交互式（在 Hermes 中使用此方式）**——通过 stdin 管道传入邮件：

```bash
cat << 'EOF' | himalaya template send
From: you@example.com
To: recipient@example.com
Subject: Test Message

Hello from Himalaya!
EOF
```

或使用 headers 标志：

```bash
himalaya message write -H "To:recipient@example.com" -H "Subject:Test" "Message body here"
```

注意：不带管道输入的 `himalaya message write` 会打开 `$EDITOR`。配合 `pty=true` + 后台模式可以使用，但管道方式更简单可靠。

### 移动/复制邮件

移动到文件夹：

```bash
himalaya message move "Archive" 42
```

复制到文件夹：

```bash
himalaya message copy "Important" 42
```

### 删除邮件

```bash
himalaya message delete 42
```

### 管理标志

添加标志：

```bash
himalaya flag add 42 --flag seen
```

移除标志：

```bash
himalaya flag remove 42 --flag seen
```

## 多账户

列出账户：

```bash
himalaya account list
```

使用指定账户：

```bash
himalaya --account work envelope list
```

## 附件

保存邮件附件：

```bash
himalaya attachment download 42
```

保存到指定目录：

```bash
himalaya attachment download 42 --downloads-dir ~/Downloads
```

## 输出格式

大多数命令支持 `--output` 以获取结构化输出：

```bash
himalaya envelope list --output json
himalaya envelope list --output plain
```

## 调试

启用调试日志：

```bash
RUST_LOG=debug himalaya envelope list
```

完整追踪与回溯：

```bash
RUST_LOG=trace RUST_BACKTRACE=1 himalaya envelope list
```

## 提示

- 使用 `himalaya --help` 或 `himalaya <command> --help` 查看详细用法。
- 消息 ID 相对于当前文件夹；切换文件夹后请重新列出。
- 如需撰写带附件的富文本邮件，请使用 MML 语法（参见 `references/message-composition.md`）。
- 使用 `pass`、系统密钥环或输出密码的命令安全存储密码。