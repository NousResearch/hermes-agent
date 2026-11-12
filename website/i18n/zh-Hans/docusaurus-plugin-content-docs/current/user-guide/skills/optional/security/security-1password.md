---
title: "1Password — 设置并使用 1Password CLI (op)"
sidebar_label: "1Password"
description: "设置并使用 1Password CLI (op)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 1Password

设置并使用 1Password CLI (op)。适用于安装 CLI、启用桌面应用集成、登录，以及为命令读取/注入密钥的场景。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/security/1password` 安装 |
| 路径 | `optional-skills/security/1password` |
| 版本 | `1.0.0` |
| 作者 | arceus77-7，由 Hermes Agent 增强 |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `security`, `secrets`, `1password`, `op`, `cli` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# 1Password CLI

当用户希望通过 1Password 管理密钥，而非使用明文环境变量或文件时，使用此 skill。

## 前置要求

- 1Password 账户
- 已安装 1Password CLI（`op`）
- 以下之一：桌面应用集成、服务账户令牌（`OP_SERVICE_ACCOUNT_TOKEN`）或 Connect 服务器
- `tmux` 可用，用于在 Hermes 终端调用期间保持稳定的已认证会话（仅限桌面应用流程）

## 使用场景

- 安装或配置 1Password CLI
- 使用 `op signin` 登录
- 读取形如 `op://Vault/Item/field` 的密钥引用
- 使用 `op inject` 将密钥注入配置/模板
- 通过 `op run` 以密钥环境变量运行命令

## 认证方式

### 服务账户（推荐用于 Hermes）

在 `~/.hermes/.env` 中设置 `OP_SERVICE_ACCOUNT_TOKEN`（skill 首次加载时会提示输入）。
无需桌面应用。支持 `op read`、`op inject`、`op run`。

```bash
export OP_SERVICE_ACCOUNT_TOKEN="your-token-here"
op whoami  # verify — should show Type: SERVICE_ACCOUNT
```

### 桌面应用集成（交互式）

1. 在 1Password 桌面应用中启用：设置 → 开发者 → 与 1Password CLI 集成
2. 确保应用已解锁
3. 运行 `op signin` 并通过生物识别提示授权

### Connect 服务器（自托管）

```bash
export OP_CONNECT_HOST="http://localhost:8080"
export OP_CONNECT_TOKEN="your-connect-token"
```

## 设置步骤

1. 安装 CLI：

```bash
# macOS
brew install 1password-cli

# Linux (official package/install docs)
# See references/get-started.md for distro-specific links.

# Windows (winget)
winget install AgileBits.1Password.CLI
```

2. 验证：

```bash
op --version
```

3. 选择上述认证方式之一并进行配置。

## Hermes 执行模式（桌面应用流程）

Hermes 终端命令默认为非交互式，且在多次调用之间可能丢失认证上下文。
若要在桌面应用集成下可靠使用 `op`，请在专用 tmux 会话中执行登录和密钥操作。

注意：使用 `OP_SERVICE_ACCOUNT_TOKEN` 时**无需**此操作 — 令牌会在终端调用之间自动持久化。

```bash
SOCKET_DIR="${TMPDIR:-/tmp}/hermes-tmux-sockets"
mkdir -p "$SOCKET_DIR"
SOCKET="$SOCKET_DIR/hermes-op.sock"
SESSION="op-auth-$(date +%Y%m%d-%H%M%S)"

tmux -S "$SOCKET" new -d -s "$SESSION" -n shell

# Sign in (approve in desktop app when prompted)
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "eval \"\$(op signin --account my.1password.com)\"" Enter

# Verify auth
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "op whoami" Enter

# Example read
tmux -S "$SOCKET" send-keys -t "$SESSION":0.0 -- "op read 'op://Private/Npmjs/one-time password?attribute=otp'" Enter

# Capture output when needed
tmux -S "$SOCKET" capture-pane -p -J -t "$SESSION":0.0 -S -200

# Cleanup
tmux -S "$SOCKET" kill-session -t "$SESSION"
```

## 常用操作

### 读取密钥

```bash
op read "op://app-prod/db/password"
```

### 获取 OTP

```bash
op read "op://app-prod/npm/one-time password?attribute=otp"
```

### 注入模板

```bash
echo "db_password: {{ op://app-prod/db/password }}" | op inject
```

### 以密钥环境变量运行命令

```bash
export DB_PASSWORD="op://app-prod/db/password"
op run -- sh -c '[ -n "$DB_PASSWORD" ] && echo "DB_PASSWORD is set" || echo "DB_PASSWORD missing"'
```

## 使用限制

- 除非用户明确请求该值，否则不得将原始密钥打印给用户。
- 优先使用 `op run` / `op inject`，而非将密钥写入文件。
- 若命令报错"account is not signed in"，请在同一 tmux 会话中重新运行 `op signin`。
- 若桌面应用集成不可用（无头环境/CI），请使用服务账户令牌流程。

## CI / 无头环境说明

非交互式使用时，请通过 `OP_SERVICE_ACCOUNT_TOKEN` 进行认证，避免使用交互式 `op signin`。
服务账户需要 CLI v2.18.0+。

## 参考资料

- `references/get-started.md`
- `references/cli-examples.md`
- https://developer.1password.com/docs/cli/
- https://developer.1password.com/docs/service-accounts/