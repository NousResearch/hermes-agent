---
title: "Github Auth — GitHub auth setup: HTTPS tokens, SSH keys, gh CLI login"
sidebar_label: "Github Auth"
description: "GitHub auth 设置：HTTPS 令牌、SSH 密钥、gh CLI 登录"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Github Auth

GitHub auth 设置：HTTPS 令牌、SSH 密钥、gh CLI 登录。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/github/github-auth` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `GitHub`, `Authentication`, `Git`, `gh-cli`, `SSH`, `Setup` |
| 相关 skill | [`github-pr-workflow`](/user-guide/skills/bundled/github/github-github-pr-workflow), [`github-code-review`](/user-guide/skills/bundled/github/github-github-code-review), [`github-issues`](/user-guide/skills/bundled/github/github-github-issues), [`github-repo-management`](/user-guide/skills/bundled/github/github-github-repo-management) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# GitHub 认证设置

此 skill 用于配置认证，使 agent 能够操作 GitHub 仓库、PR、issue 和 CI。涵盖两条路径：

- **`git`（始终可用）** — 使用 HTTPS 个人访问令牌（personal access token）或 SSH 密钥
- **`gh` CLI（如已安装）** — 更丰富的 GitHub API 访问，认证流程更简单

## 检测流程

当用户要求你操作 GitHub 时，首先执行以下检查：

```bash
# Check what's available
git --version
gh --version 2>/dev/null || echo "gh not installed"

# Check if already authenticated
gh auth status 2>/dev/null || echo "gh not authenticated"
git config --global credential.helper 2>/dev/null || echo "no git credential helper"
```

**决策树：**
1. 若 `gh auth status` 显示已认证 → 直接使用 `gh` 处理所有操作
2. 若 `gh` 已安装但未认证 → 使用下方"gh auth"方法
3. 若 `gh` 未安装 → 使用下方"仅 git"方法（无需 sudo）

---

## 方法一：仅 Git 认证（无 gh，无 sudo）

适用于任何已安装 `git` 的机器，无需 root 权限。

### 选项 A：HTTPS 配合个人访问令牌（推荐）

最通用的方法——适用于所有环境，无需 SSH 配置。

**第一步：创建个人访问令牌**

告知用户访问：**https://github.com/settings/tokens**

- 点击"Generate new token (classic)"
- 填写名称，如"hermes-agent"
- 选择权限范围（scope）：
  - `repo`（完整仓库访问——读、写、推送、PR）
  - `workflow`（触发和管理 GitHub Actions）
  - `read:org`（如需操作组织仓库）
- 设置有效期（90 天是合理的默认值）
- 复制令牌——此后不会再次显示

**第二步：配置 git 存储令牌**

```bash
# Set up the credential helper to cache credentials
# "store" saves to ~/.git-credentials in plaintext (simple, persistent)
git config --global credential.helper store

# Now do a test operation that triggers auth — git will prompt for credentials
# Username: <their-github-username>
# Password: <paste the personal access token, NOT their GitHub password>
git ls-remote https://github.com/<their-username>/<any-repo>.git
```

首次输入凭据后，将被保存并在后续所有操作中复用。

**替代方案：cache helper（凭据在内存中过期）**

```bash
# Cache in memory for 8 hours (28800 seconds) instead of saving to disk
git config --global credential.helper 'cache --timeout=28800'
```

**替代方案：直接将令牌写入远程 URL（按仓库设置）**

```bash
# Embed token in the remote URL (avoids credential prompts entirely)
git remote set-url origin https://<username>:<token>@github.com/<owner>/<repo>.git
```

**第三步：配置 git 身份信息**

```bash
# Required for commits — set name and email
git config --global user.name "Their Name"
git config --global user.email "their-email@example.com"
```

**第四步：验证**

```bash
# Test push access (this should work without any prompts now)
git ls-remote https://github.com/<their-username>/<any-repo>.git

# Verify identity
git config --global user.name
git config --global user.email
```

### 选项 B：SSH 密钥认证

适合偏好 SSH 或已有密钥的用户。

**第一步：检查现有 SSH 密钥**

```bash
ls -la ~/.ssh/id_*.pub 2>/dev/null || echo "No SSH keys found"
```

**第二步：如需则生成密钥**

```bash
# Generate an ed25519 key (modern, secure, fast)
ssh-keygen -t ed25519 -C "their-email@example.com" -f ~/.ssh/id_ed25519 -N ""

# Display the public key for them to add to GitHub
cat ~/.ssh/id_ed25519.pub
```

告知用户在以下地址添加公钥：**https://github.com/settings/keys**
- 点击"New SSH key"
- 粘贴公钥内容
- 填写标题，如"hermes-agent-&lt;machine-name>"

**第三步：测试连接**

```bash
ssh -T git@github.com
# Expected: "Hi <username>! You've successfully authenticated..."
```

**第四步：配置 git 使用 SSH 访问 GitHub**

```bash
# Rewrite HTTPS GitHub URLs to SSH automatically
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

**第五步：配置 git 身份信息**

```bash
git config --global user.name "Their Name"
git config --global user.email "their-email@example.com"
```

---

## 方法二：gh CLI 认证

若已安装 `gh`，一步即可完成 API 访问和 git 凭据配置。

### 浏览器交互登录（桌面环境）

```bash
gh auth login
# Select: GitHub.com
# Select: HTTPS
# Authenticate via browser
```

### 基于令牌登录（无头环境 / SSH 服务器）

```bash
echo "<THEIR_TOKEN>" | gh auth login --with-token

# Set up git credentials through gh
gh auth setup-git
```

### 验证

```bash
gh auth status
```

---

## 不使用 gh 调用 GitHub API

当 `gh` 不可用时，仍可使用 `curl` 配合个人访问令牌访问完整的 GitHub API。其他 GitHub skill 的降级方案均采用此方式。

### 为 API 调用设置令牌

```bash
# Option 1: Export as env var (preferred — keeps it out of commands)
export GITHUB_TOKEN="<token>"

# Then use in curl calls:
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user
```

### 从 Git 凭据中提取令牌

若已通过 `credential.helper store` 配置 git 凭据，可提取令牌：

```bash
# Read from git credential store
uv run python3 "${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/git-credential-token.py"
```

### 辅助函数：检测认证方式

在任何 GitHub 工作流开始时使用此模式：

```bash
# Try gh first, fall back to git + curl
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  echo "AUTH_METHOD=gh"
elif [ -n "$GITHUB_TOKEN" ]; then
  echo "AUTH_METHOD=curl"
elif [ -f ~/.hermes/.env ] && grep -q "^GITHUB_TOKEN=" ~/.hermes/.env; then
  export GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" ~/.hermes/.env | head -1 | cut -d= -f2 | tr -d '\n\r')
  echo "AUTH_METHOD=curl"
elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
  export GITHUB_TOKEN=$(uv run python3 "${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/git-credential-token.py")
  echo "AUTH_METHOD=curl"
else
  echo "AUTH_METHOD=none"
  echo "Need to set up authentication first"
fi
```

---

## 故障排查

| 问题 | 解决方案 |
|---------|----------|
| `git push` 要求输入密码 | GitHub 已禁用密码认证。请使用个人访问令牌作为密码，或切换至 SSH |
| `remote: Permission to X denied` | 令牌可能缺少 `repo` scope——请重新生成并选择正确的 scope |
| `fatal: Authentication failed` | 缓存的凭据可能已过期——运行 `git credential reject` 后重新认证 |
| `ssh: connect to host github.com port 22: Connection refused` | 尝试通过 HTTPS 端口使用 SSH：在 `~/.ssh/config` 中为 `Host github.com` 添加 `Port 443` 和 `Hostname ssh.github.com` |
| 凭据不持久 | 检查 `git config --global credential.helper`——必须为 `store` 或 `cache` |
| 多个 GitHub 账号 | 在 `~/.ssh/config` 中为不同主机别名配置不同 SSH 密钥，或使用按仓库设置的凭据 URL |
| `gh: command not found` 且无 sudo | 使用上方方法一（仅 git）——无需安装任何软件 |