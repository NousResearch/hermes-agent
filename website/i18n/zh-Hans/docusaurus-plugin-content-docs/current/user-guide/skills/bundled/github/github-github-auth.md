---
title: "GitHub 认证 — GitHub 认证设置：HTTPS 令牌、SSH 密钥、gh CLI 登录"
sidebar_label: "GitHub 认证"
description: "GitHub 认证设置：HTTPS 令牌、SSH 密钥、gh CLI 登录"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# GitHub 认证

GitHub 认证设置：HTTPS 令牌、SSH 密钥、gh CLI 登录。

## 技能元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/github/github-auth` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `GitHub`、`认证`、`Git`、`gh-cli`、`SSH`、`设置` |
| 相关技能 | [`github-pr-workflow`](/docs/user-guide/skills/bundled/github/github-github-pr-workflow)、[`github-code-review`](/docs/user-guide/skills/bundled/github/github-github-code-review)、[`github-issues`](/docs/user-guide/skills/bundled/github/github-github-issues)、[`github-repo-management`](/docs/user-guide/skills/bundled/github/github-github-repo-management) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# GitHub 认证设置

此技能设置认证，使代理能够处理 GitHub 仓库、PR、Issue 和 CI。它涵盖两条路径：

- **`git`（始终可用）** — 使用 HTTPS 个人访问令牌或 SSH 密钥
- **`gh` CLI（如果已安装）** — 更丰富的 GitHub API 访问，认证流程更简单

## 检测流程

当用户要求你使用 GitHub 时，首先运行此检查：

```bash
# 检查可用的工具
git --version
gh --version 2>/dev/null || echo "gh not installed"

# 检查是否已认证
gh auth status 2>/dev/null || echo "gh not authenticated"
git config --global credential.helper 2>/dev/null || echo "no git credential helper"
```

**决策树：**
1. 如果 `gh auth status` 显示已认证 → 可以使用了，用 `gh` 处理所有操作
2. 如果 `gh` 已安装但未认证 → 使用下面的"gh auth"方法
3. 如果 `gh` 未安装 → 使用下面的"仅 git"方法（无需 sudo）

---

## 方法 1：仅 Git 认证（无 gh、无 sudo）

这适用于任何安装了 `git` 的机器。无需 root 权限。

### 选项 A：使用个人访问令牌的 HTTPS（推荐）

这是最通用的方法 — 在任何地方都能工作，不需要 SSH 配置。

**步骤 1：创建个人访问令牌**

告诉用户前往：**https://github.com/settings/tokens**

- 点击"Generate new token (classic)"
- 给它起个名字，如"hermes-agent"
- 选择权限范围：
  - `repo`（完全仓库访问权限 — 读取、写入、推送、PR）
  - `workflow`（触发和管理 GitHub Actions）
  - `read:org`（如果使用组织仓库）
- 设置过期时间（90 天是一个好的默认值）
- 复制令牌 — 它不会再显示

**步骤 2：配置 git 存储令牌**

```bash
# 设置凭证助手缓存凭证
# "store"以明文保存到 ~/.git-credentials（简单、持久）
git config --global credential.helper store

# 现在做一次测试操作来触发认证 — git 会提示输入凭证
# 用户名：<他们的 GitHub 用户名>
# 密码：<粘贴个人访问令牌，而不是 GitHub 密码>
git ls-remote https://github.com/<their-username>/<any-repo>.git
```

输入一次凭证后，它们会被保存并在所有未来操作中重复使用。

**替代方案：cache helper（凭证从内存过期）**

```bash
# 在内存中缓存 8 小时（28800 秒）而不是保存到磁盘
git config --global credential.helper 'cache --timeout=28800'
```

**替代方案：直接在远程 URL 中设置令牌（按仓库）**

```bash
# 在远程 URL 中嵌入令牌（完全避免凭证提示）
git remote set-url origin https://<username>:<token>@github.com/<owner>/<repo>.git
```

**步骤 3：配置 git 身份**

```bash
# 提交必需 — 设置名称和邮箱
git config --global user.name "Their Name"
git config --global user.email "their-email@example.com"
```

**步骤 4：验证**

```bash
# 测试推送权限（现在应该不需要任何提示就能工作）
git ls-remote https://github.com/<their-username>/<any-repo>.git

# 验证身份
git config --global user.name
git config --global user.email
```

### 选项 B：SSH 密钥认证

适合偏好 SSH 或已经设置好密钥的用户。

**步骤 1：检查现有 SSH 密钥**

```bash
ls -la ~/.ssh/id_*.pub 2>/dev/null || echo "No SSH keys found"
```

**步骤 2：需要时生成密钥**

```bash
# 生成 ed25519 密钥（现代、安全、快速）
ssh-keygen -t ed25519 -C "their-email@example.com" -f ~/.ssh/id_ed25519 -N ""

# 显示公钥供他们添加到 GitHub
cat ~/.ssh/id_ed25519.pub
```

告诉用户在以下位置添加公钥：**https://github.com/settings/keys**
- 点击"New SSH key"
- 粘贴公钥内容
- 起个标题，如"hermes-agent-&lt;machine-name&gt;"

**步骤 3：测试连接**

```bash
ssh -T git@github.com
# 预期："Hi <username>! You've successfully authenticated..."
```

**步骤 4：配置 git 对 GitHub 使用 SSH**

```bash
# 自动将 HTTPS GitHub URL 重写为 SSH
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

**步骤 5：配置 git 身份**

```bash
git config --global user.name "Their Name"
git config --global user.email "their-email@example.com"
```

---

## 方法 2：gh CLI 认证

如果 `gh` 已安装，它一步就能处理 API 访问和 git 凭证。

### 交互式浏览器登录（桌面）

```bash
gh auth login
# 选择：GitHub.com
# 选择：HTTPS
# 通过浏览器认证
```

### 基于令牌的登录（无头/SSH 服务器）

```bash
echo "<THEIR_TOKEN>" | gh auth login --with-token

# 通过 gh 设置 git 凭证
gh auth setup-git
```

### 验证

```bash
gh auth status
```

---

## 不使用 gh 调用 GitHub API

当 `gh` 不可用时，你仍然可以使用 `curl` 和个人访问令牌访问完整的 GitHub API。这是其他 GitHub 技能实现其回退的方式。

### 设置 API 调用的令牌

```bash
# 选项 1：导出为环境变量（推荐 — 避免在命令中暴露）
export GITHUB_TOKEN="<token>"

# 然后在 curl 调用中使用：
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user
```

### 从 Git 凭证中提取令牌

如果 git 凭证已配置（通过 credential.helper store），可以提取令牌：

```bash
# 从 git 凭证存储中读取
grep "github.com" ~/.git-credentials 2>/dev/null | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|'
```

### 辅助：检测认证方法

在任何 GitHub 工作流开始时使用此模式：

```bash
# 先尝试 gh，回退到 git + curl
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  echo "AUTH_METHOD=gh"
elif [ -n "$GITHUB_TOKEN" ]; then
  echo "AUTH_METHOD=curl"
elif [ -f ~/.hermes/.env ] && grep -q "^GITHUB_TOKEN=" ~/.hermes/.env; then
  export GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" ~/.hermes/.env | head -1 | cut -d= -f2 | tr -d '\n\r')
  echo "AUTH_METHOD=curl"
elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
  export GITHUB_TOKEN=$(grep "github.com" ~/.git-credentials | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|')
  echo "AUTH_METHOD=curl"
else
  echo "AUTH_METHOD=none"
  echo "Need to set up authentication first"
fi
```

---

## 故障排除

| 问题 | 解决方案 |
|---------|----------|
| `git push` 要求输入密码 | GitHub 已禁用密码认证。使用个人访问令牌作为密码，或切换到 SSH |
| `remote: Permission to X denied` | 令牌可能缺少 `repo` 权限 — 使用正确的权限范围重新生成 |
| `fatal: Authentication failed` | 缓存的凭证可能已过期 — 运行 `git credential reject` 然后重新认证 |
| `ssh: connect to host github.com port 22: Connection refused` | 尝试通过 HTTPS 端口的 SSH：在 `~/.ssh/config` 中添加 `Host github.com`，设置 `Port 443` 和 `Hostname ssh.github.com` |
| 凭证不持久 | 检查 `git config --global credential.helper` — 必须是 `store` 或 `cache` |
| 多个 GitHub 账户 | 在 `~/.ssh/config` 中使用不同密钥的主机别名，或按仓库使用凭证 URL |
| `gh: command not found` + 无 sudo | 使用上面的仅 git 方法 1 — 无需安装 |
