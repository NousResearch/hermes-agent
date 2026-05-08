---
title: "1Password — 设置并使用 1Password CLI (op)"
sidebar_label: "1Password"
description: "设置并使用 1Password CLI (op)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 1Password

设置并使用 1Password CLI (op)。在安装 CLI、启用桌面应用集成、登录以及读取/注入命令密钥时使用。

## 技能元数据

| | |
|---|---|
| 来源 | 可选技能 — 使用 `hermes skills install official/security/1password` 安装 |
| 路径 | `optional-skills/security/1password` |
| 版本 | `1.0.0` |
| 作者 | arceus77-7, 由 Hermes Agent 增强 |
| 许可证 | MIT |
| 标签 | `security`, `secrets`, `1password`, `op`, `cli` |

## 参考：完整的 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时智能体看到的指令。
:::

# 1Password CLI

当用户希望通过 1Password 管理密钥而不是明文环境变量或文件时使用此技能。

## 要求

- 1Password 账户
- 已安装 1Password CLI（`op`）
- 之一：桌面应用集成、服务账户令牌（`OP_SERVICE_ACCOUNT_TOKEN`）或 Connect 服务器
- `tmux` 可用，以便在 Hermes 终端调用期间保持稳定的认证会话（仅桌面应用流程）

## 何时使用

- 安装或配置 1Password CLI
- 使用 `op signin` 登录
- 读取密钥引用如 `op://Vault/Item/field`
- 使用 `op inject` 将密钥注入配置/模板
- 通过 `op run` 运行带密钥环境变量的命令

## 认证方法

### 服务账户（推荐用于 Hermes）

在 `~/.hermes/.env` 中设置 `OP_SERVICE_ACCOUNT_TOKEN`（技能首次加载时会提示）。
不需要桌面应用。支持 `op read`、`op inject`、`op run`。

```bash
export OP_SERVICE_ACCOUNT_TOKEN="your-token-here"
op whoami  # 验证 — 应显示 Type: SERVICE_ACCOUNT
```

### 桌面应用集成（交互式）

1. 在 1Password 桌面应用中启用：设置 → 开发者 → 与 1Password CLI 集成
2. 确保应用已解锁
3. 运行 `op signin` 并批准生物识别提示

### Connect 服务器（自托管）

```bash
export OP_CONNECT_HOST="http://localhost:8080"
export OP_CONNECT_TOKEN="your-connect-token"
```

## 设置

1. 安装 CLI：

```bash
# macOS
brew install 1password-cli

# Linux（官方包/安装文档）
# 参见 references/get-started.md 获取特定发行版的链接。

# Windows（winget）
winget install AgileBits.1Password.CLI
```

2. 验证：

```bash
op --version
```

3. 选择上面的认证方法并配置。

## Hermes 执行模式（桌面应用流程）

Hermes 终端命令默认是非交互式的，可能在调用之间丢失认证上下文。
为了在使用桌面应用集成时可靠地使用 `op`，在专用 tmux 会话中运行登录和密钥操作。

注意：使用 `OP_SERVICE_ACCOUNT_TOKEN` 时不需要此操作 — 令牌在终端调用之间自动保持。

```bash
SOCKET_DIR="${TMPDIR:-/tmp}/hermes-tmux-sockets"
mkdir -p "$SOCKET_DIR"
SOCKET="$SOCKET_DIR/hermes-op.sock"
SESSION="op-auth-$(date +%Y%m%d-%H%M%S)"

tmux -S "$SOCKET" new -d -s "$SESSION" -n shell

# 登录（在提示时在桌面应用中批准）
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "eval \"\$(op signin --account my.1password.com)\"" Enter

# 验证认证
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "op whoami" Enter

# 示例读取
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "op read 'op://Private/Npmjs/one-time password?attribute=otp'" Enter

# 如需要时捕获输出
tmux -S "$SOCKET" capture-pane -p -J -t "$SESSION":0.0 -S -200

# 清理
tmux -S "$SOCKET" kill-session -t "$SESSION"
```

## 常见操作

### 读取密钥

```bash
op read "op://app-prod/db/password"
```

### 获取 OTP

```bash
op read "op://app-prod/npm/one-time password?attribute=otp"
```

### 注入到模板

```bash
echo "db_password: {{ op://app-prod/db/password }}" | op inject
```

### 使用密钥环境变量运行命令

```bash
export DB_PASSWORD="op://app-prod/db/password"
op run -- sh -c '[ -n "$DB_PASSWORD" ] && echo "DB_PASSWORD is set" || echo "DB_PASSWORD missing"'
```

## 安全措施

- 除非用户明确请求，否则永远不要将原始密钥打印回用户。
- 优先使用 `op run` / `op inject` 而不是将密钥写入文件。
- 如果命令失败并显示"account is not signed in"，在同一 tmux 会话中重新运行 `op signin`。
- 如果桌面应用集成不可用（无头/CI），使用服务账户令牌流程。

## CI / 无头说明

对于非交互式使用，使用 `OP_SERVICE_ACCOUNT_TOKEN` 进行身份验证，避免交互式 `op signin`。
服务账户需要 CLI v2.18.0+。

## 参考

- `references/get-started.md`
- `references/cli-examples.md`
- https://developer.1password.com/docs/cli/
- https://developer.1password.com/docs/service-accounts/
