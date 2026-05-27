---
title: "Opencode — 将编码任务委托给 OpenCode CLI（功能开发、PR 审查）"
sidebar_label: "Opencode"
description: "将编码任务委托给 OpenCode CLI（功能开发、PR 审查）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Opencode

将编码任务委托给 OpenCode CLI（功能开发、PR 审查）。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/autonomous-ai-agents/opencode` |
| 版本 | `1.2.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Coding-Agent`, `OpenCode`, `Autonomous`, `Refactoring`, `Code-Review` |
| 相关 skill | [`claude-code`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`codex`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`hermes-agent`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# OpenCode CLI

使用 [OpenCode](https://opencode.ai) 作为由 Hermes 终端/进程工具编排的自主编码工作器。OpenCode 是一个支持多 provider、开源的 AI 编码 agent，具备 TUI（终端用户界面）和 CLI。

## 适用场景

- 用户明确要求使用 OpenCode
- 需要外部编码 agent 来实现/重构/审查代码
- 需要长时间运行的编码会话并定期检查进度
- 需要在隔离的工作目录/worktree 中并行执行任务

## 前置条件

- 已安装 OpenCode：`npm i -g opencode-ai@latest` 或 `brew install anomalyco/tap/opencode`
- 已配置认证：`opencode auth login` 或设置 provider 环境变量（OPENROUTER_API_KEY 等）
- 验证：`opencode auth list` 应显示至少一个 provider
- 代码任务推荐使用 Git 仓库
- 交互式 TUI 会话需要 `pty=true`

## 二进制文件解析（重要）

Shell 环境可能会解析到不同的 OpenCode 二进制文件。如果你的终端与 Hermes 的行为不一致，请检查：

```
terminal(command="which -a opencode")
terminal(command="opencode --version")
```

如有需要，可固定使用明确的二进制路径：

```
terminal(command="$HOME/.opencode/bin/opencode run '...'", workdir="~/project", pty=true)
```

## 单次任务

使用 `opencode run` 执行有边界的非交互式任务：

```
terminal(command="opencode run 'Add retry logic to API calls and update tests'", workdir="~/project")
```

使用 `-f` 附加上下文文件：

```
terminal(command="opencode run 'Review this config for security issues' -f config.yaml -f .env.example", workdir="~/project")
```

使用 `--thinking` 显示模型思考过程：

```
terminal(command="opencode run 'Debug why tests fail in CI' --thinking", workdir="~/project")
```

强制指定特定模型：

```
terminal(command="opencode run 'Refactor auth module' --model openrouter/anthropic/claude-sonnet-4", workdir="~/project")
```

## 交互式会话（后台运行）

对于需要多轮交互的迭代工作，在后台启动 TUI：

```
terminal(command="opencode", workdir="~/project", background=true, pty=true)
# 返回 session_id

# 发送 prompt（提示词）
process(action="submit", session_id="<id>", data="Implement OAuth refresh flow and add tests")

# 监控进度
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# 发送后续输入
process(action="submit", session_id="<id>", data="Now add error handling for token expiry")

# 干净退出 — Ctrl+C
process(action="write", session_id="<id>", data="\x03")
# 或直接终止进程
process(action="kill", session_id="<id>")
```

**重要：** 不要使用 `/exit`——它不是有效的 OpenCode 命令，会打开 agent 选择器对话框。请使用 Ctrl+C（`\x03`）或 `process(action="kill")` 退出。

### TUI 快捷键

| 按键 | 操作 |
|-----|--------|
| `Enter` | 提交消息（如有需要可按两次） |
| `Tab` | 在 agent 之间切换（build/plan） |
| `Ctrl+P` | 打开命令面板 |
| `Ctrl+X L` | 切换会话 |
| `Ctrl+X M` | 切换模型 |
| `Ctrl+X N` | 新建会话 |
| `Ctrl+X E` | 打开编辑器 |
| `Ctrl+C` | 退出 OpenCode |

### 恢复会话

退出后，OpenCode 会打印会话 ID。使用以下命令恢复：

```
terminal(command="opencode -c", workdir="~/project", background=true, pty=true)  # 继续上次会话
terminal(command="opencode -s ses_abc123", workdir="~/project", background=true, pty=true)  # 指定会话
```

## 常用标志

| 标志 | 用途 |
|------|-----|
| `run 'prompt'` | 单次执行后退出 |
| `--continue` / `-c` | 继续上次 OpenCode 会话 |
| `--session <id>` / `-s` | 继续指定会话 |
| `--agent <name>` | 选择 OpenCode agent（build 或 plan） |
| `--model provider/model` | 强制使用指定模型 |
| `--format json` | 机器可读的输出/事件 |
| `--file <path>` / `-f` | 向消息附加文件 |
| `--thinking` | 显示模型思考块 |
| `--variant <level>` | 推理强度（high、max、minimal） |
| `--title <name>` | 为会话命名 |
| `--attach <url>` | 连接到正在运行的 opencode 服务器 |

## 操作流程

1. 验证工具就绪状态：
   - `terminal(command="opencode --version")`
   - `terminal(command="opencode auth list")`
2. 对于有边界的任务，使用 `opencode run '...'`（无需 pty）。
3. 对于迭代任务，使用 `background=true, pty=true` 启动 `opencode`。
4. 使用 `process(action="poll"|"log")` 监控长时间运行的任务。
5. 如果 OpenCode 请求输入，通过 `process(action="submit", ...)` 响应。
6. 使用 `process(action="write", data="\x03")` 或 `process(action="kill")` 退出，切勿使用 `/exit`。
7. 向用户汇总文件变更、测试结果及后续步骤。

## PR 审查工作流

OpenCode 内置 PR 命令：

```
terminal(command="opencode pr 42", workdir="~/project", pty=true)
```

或在临时克隆中审查以实现隔离：

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && opencode run 'Review this PR vs main. Report bugs, security risks, test gaps, and style issues.' -f $(git diff origin/main --name-only | head -20 | tr '\n' ' ')", pty=true)
```

## 并行工作模式

使用独立的工作目录/worktree 避免冲突：

```
terminal(command="opencode run 'Fix issue #101 and commit'", workdir="/tmp/issue-101", background=true, pty=true)
terminal(command="opencode run 'Add parser regression tests and commit'", workdir="/tmp/issue-102", background=true, pty=true)
process(action="list")
```

## 会话与成本管理

列出历史会话：

```
terminal(command="opencode session list")
```

查看 token 用量和费用：

```
terminal(command="opencode stats")
terminal(command="opencode stats --days 7 --models anthropic/claude-sonnet-4")
```

## 注意事项

- 交互式 `opencode`（TUI）会话需要 `pty=true`。`opencode run` 命令**不需要** pty。
- `/exit` **不是**有效命令——它会打开 agent 选择器。请使用 Ctrl+C 退出 TUI。
- PATH 不匹配可能导致选择错误的 OpenCode 二进制文件/模型配置。
- 如果 OpenCode 看起来卡住了，在终止前先检查日志：
  - `process(action="log", session_id="<id>")`
- 避免多个并行 OpenCode 会话共享同一工作目录。
- 在 TUI 中可能需要按两次 Enter 才能提交（第一次确认文本，第二次发送）。

## 验证

冒烟测试：

```
terminal(command="opencode run 'Respond with exactly: OPENCODE_SMOKE_OK'")
```

成功标准：
- 输出包含 `OPENCODE_SMOKE_OK`
- 命令退出时无 provider/模型错误
- 对于代码任务：预期文件已变更且测试通过

## 规则

1. 单次自动化任务优先使用 `opencode run`——更简单且无需 pty。
2. 仅在需要迭代时使用交互式后台模式。
3. 始终将 OpenCode 会话限定在单个仓库/工作目录内。
4. 对于长时间任务，从 `process` 日志中提供进度更新。
5. 报告具体结果（文件变更、测试情况、剩余风险）。
6. 使用 Ctrl+C 或 kill 退出交互式会话，切勿使用 `/exit`。