---
title: "Codex — 委托编码给 OpenAI Codex CLI（功能、PR）"
sidebar_label: "Codex"
description: "委托编码给 OpenAI Codex CLI（功能、PR）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Codex

委托编码给 OpenAI Codex CLI（功能、PR）。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/autonomous-ai-agents/codex` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `Coding-Agent`、`Codex`、`OpenAI`、`Code-Review`、`Refactoring` |
| 相关技能 | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code)、[`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时触发的完整技能定义。这是代理激活技能时看到的指令。
:::

# Codex CLI

通过 Hermes 终端将编码任务委托给 [Codex](https://github.com/openai/codex)。Codex 是 OpenAI 的自主编码代理 CLI。

## 使用场景

- 构建功能
- 重构
- PR 审查
- 批量问题修复

需要 codex CLI 和 git 仓库。

## 前置条件

- 安装 Codex：`npm install -g @openai/codex`
- 配置 OpenAI 认证：`OPENAI_API_KEY` 或来自 Codex CLI 登录流程的 Codex OAuth 凭据
- **必须在 git 仓库内运行** — Codex 拒绝在 git 目录外运行
- 在终端调用中使用 `pty=true` — Codex 是交互式终端应用

对于 Hermes 本身，`model.provider: openai-codex` 在 `hermes auth add openai-codex` 之后使用 `~/.hermes/auth.json` 中 Hermes 管理的 Codex OAuth。对于独立 Codex CLI，有效的 CLI OAuth 会话可能在 `~/.codex/auth.json` 下；不要将缺少 `OPENAI_API_KEY` 单独视为 Codex 认证缺失的证明。

## 一次性任务

```
terminal(command="codex exec 'Add dark mode toggle to settings'", workdir="~/project", pty=true)
```

对于临时工作（Codex 需要 git 仓库）：
```
terminal(command="cd $(mktemp -d) && git init && codex exec 'Build a snake game in Python'", pty=true)
```

## 后台模式（长任务）

```
# 以 PTY 在后台启动
terminal(command="codex exec --full-auto 'Refactor the auth module'", workdir="~/project", background=true, pty=true)
# 返回 session_id

# 监控进度
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# 如果 Codex 提问，发送输入
process(action="submit", session_id="<id>", data="yes")

# 需要时终止
process(action="kill", session_id="<id>")
```

## 关键标志

| 标志 | 效果 |
|------|--------|
| `exec "prompt"` | 一次性执行，完成后退出 |
| `--full-auto` | 沙箱但自动批准工作区内的文件更改 |
| `--yolo` | 无沙箱，无审批（最快，最危险） |

## PR 审查

克隆到临时目录以安全审查：

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && codex review --base origin/main", pty=true)
```

## 使用工作树并行修复问题

```
# 创建工作树
terminal(command="git worktree add -b fix/issue-78 /tmp/issue-78 main", workdir="~/project")
terminal(command="git worktree add -b fix/issue-99 /tmp/issue-99 main", workdir="~/project")

# 在每个中启动 Codex
terminal(command="codex --yolo exec 'Fix issue #78: <description>. Commit when done.'", workdir="/tmp/issue-78", background=true, pty=true)
terminal(command="codex --yolo exec 'Fix issue #99: <description>. Commit when done.'", workdir="/tmp/issue-99", background=true, pty=true)

# 监控
process(action="list")

# 完成后，推送并创建 PR
terminal(command="cd /tmp/issue-78 && git push -u origin fix/issue-78")
terminal(command="gh pr create --repo user/repo --head fix/issue-78 --title 'fix: ...' --body '...'")

# 清理
terminal(command="git worktree remove /tmp/issue-78", workdir="~/project")
```

## 批量 PR 审查

```
# 获取所有 PR refs
terminal(command="git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'", workdir="~/project")

# 并行审查多个 PR
terminal(command="codex exec 'Review PR #86. git diff origin/main...origin/pr/86'", workdir="~/project", background=true, pty=true)
terminal(command="codex exec 'Review PR #87. git diff origin/main...origin/pr/87'", workdir="~/project", background=true, pty=true)

# 发布结果
terminal(command="gh pr comment 86 --body '<review>'", workdir="~/project")
```

## 规则

1. **始终使用 `pty=true`** — Codex 是交互式终端应用，没有 PTY 会挂起
2. **需要 git 仓库** — Codex 不在 git 目录外运行。临时工作使用 `mktemp -d && git init`
3. **一次性任务使用 `exec`** — `codex exec "prompt"` 干净地运行并退出
4. **构建时使用 `--full-auto`** — 在沙箱内自动批准更改
5. **长任务用后台** — 使用 `background=true` 并用 `process` 工具监控
6. **不要干扰** — 用 `poll`/`log` 监控，对长时间运行的任务保持耐心
7. **并行没问题** — 同时运行多个 Codex 进程进行批量工作
