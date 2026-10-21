---
title: "Blackbox — 将编码任务委托给 Blackbox AI CLI 代理"
sidebar_label: "Blackbox"
description: "将编码任务委托给 Blackbox AI CLI 代理"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Blackbox

将编码任务委托给 Blackbox AI CLI 代理。这是一个内置评判机制的多模型代理，可将任务分发给多个 LLM 并选出最佳结果。需要安装 blackbox CLI 及 Blackbox AI API 密钥。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/autonomous-ai-agents/blackbox` 安装 |
| 路径 | `optional-skills/autonomous-ai-agents/blackbox` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent (Nous Research) |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Coding-Agent`, `Blackbox`, `Multi-Agent`, `Judge`, `Multi-Model` |
| 相关 skill | [`claude-code`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`codex`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`hermes-agent`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是代理在 skill 激活时所看到的指令内容。
:::

# Blackbox CLI

通过 Hermes 终端将编码任务委托给 [Blackbox AI](https://www.blackbox.ai/)。Blackbox 是一个多模型编码代理 CLI，可将任务分发给多个 LLM（Claude、Codex、Gemini、Blackbox Pro），并使用评判机制选出最佳实现。

该 CLI 为[开源项目](https://github.com/blackboxaicode/cli)（GPL-3.0，TypeScript，fork 自 Gemini CLI），支持交互式会话、非交互式单次执行、检查点（checkpointing）、MCP 以及视觉模型切换。

## 前置条件

- 已安装 Node.js 20+
- 已安装 Blackbox CLI：`npm install -g @blackboxai/cli`
- 或从源码安装：
  ```
  git clone https://github.com/blackboxaicode/cli.git
  cd cli && npm install && npm install -g .
  ```
- 从 [app.blackbox.ai/dashboard](https://app.blackbox.ai/dashboard) 获取 API 密钥
- 配置：运行 `blackbox configure` 并输入 API 密钥
- 在终端调用中使用 `pty=true` — Blackbox CLI 是交互式终端应用

## 单次任务

```
terminal(command="blackbox --prompt 'Add JWT authentication with refresh tokens to the Express API'", workdir="/path/to/project", pty=true)
```

快速临时工作：
```
terminal(command="cd $(mktemp -d) && git init && blackbox --prompt 'Build a REST API for todos with SQLite'", pty=true)
```

## 后台模式（长时任务）

对于需要数分钟的任务，使用后台模式以便监控进度：

```
# Start in background with PTY
terminal(command="blackbox --prompt 'Refactor the auth module to use OAuth 2.0'", workdir="~/project", background=true, pty=true)
# Returns session_id

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send input if Blackbox asks a question
process(action="submit", session_id="<id>", data="yes")

# Kill if needed
process(action="kill", session_id="<id>")
```

## 检查点与恢复

Blackbox CLI 内置检查点支持，可暂停并恢复任务：

```
# After a task completes, Blackbox shows a checkpoint tag
# Resume with a follow-up task:
terminal(command="blackbox --resume-checkpoint 'task-abc123-2026-03-06' --prompt 'Now add rate limiting to the endpoints'", workdir="~/project", pty=true)
```

## 会话命令

在交互式会话中，可使用以下命令：

| 命令 | 效果 |
|---------|--------|
| `/compress` | 压缩对话历史以节省 token |
| `/clear` | 清除历史并重新开始 |
| `/stats` | 查看当前 token 用量 |
| `Ctrl+C` | 取消当前操作 |

## PR 审查

克隆到临时目录以避免修改工作树：

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && blackbox --prompt 'Review this PR against main. Check for bugs, security issues, and code quality.'", pty=true)
```

## 并行工作

为独立任务启动多个 Blackbox 实例：

```
terminal(command="blackbox --prompt 'Fix the login bug'", workdir="/tmp/issue-1", background=true, pty=true)
terminal(command="blackbox --prompt 'Add unit tests for auth'", workdir="/tmp/issue-2", background=true, pty=true)

# Monitor all
process(action="list")
```

## 多模型模式

Blackbox 的独特功能是将同一任务分发给多个模型并对结果进行评判。通过 `blackbox configure` 配置要使用的模型 — 选择多个提供商以启用 Chairman/judge 工作流，CLI 将评估不同模型的输出并选出最佳结果。

## 关键参数

| 参数 | 效果 |
|------|--------|
| `--prompt "task"` | 非交互式单次执行 |
| `--resume-checkpoint "tag"` | 从已保存的检查点恢复 |
| `--yolo` | 自动批准所有操作和模型切换 |
| `blackbox session` | 启动交互式聊天会话 |
| `blackbox configure` | 更改设置、提供商、模型 |
| `blackbox info` | 显示系统信息 |

## 视觉支持

Blackbox 自动检测输入中的图像，并可切换至多模态分析。VLM 模式：
- `"once"` — 仅针对当前查询切换模型
- `"session"` — 在整个会话期间切换
- `"persist"` — 保持当前模型（不切换）

## Token 限制

通过 `.blackboxcli/settings.json` 控制 token 用量：
```json
{
  "sessionTokenLimit": 32000
}
```

## 规则

1. **始终使用 `pty=true`** — Blackbox CLI 是交互式终端应用，没有 PTY 将会挂起
2. **使用 `workdir`** — 确保代理专注于正确的目录
3. **长任务使用后台模式** — 使用 `background=true` 并通过 `process` 工具监控
4. **不要干预** — 使用 `poll`/`log` 监控，不要因为速度慢就终止会话
5. **报告结果** — 完成后检查变更内容并向用户汇总
6. **积分需要花钱** — Blackbox 使用积分制；多模型模式消耗积分更快
7. **检查前置条件** — 在尝试委托前确认 `blackbox` CLI 已安装