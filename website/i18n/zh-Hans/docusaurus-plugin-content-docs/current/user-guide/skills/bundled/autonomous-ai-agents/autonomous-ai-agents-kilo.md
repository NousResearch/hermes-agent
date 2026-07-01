---
title: "Kilo — 将编码任务委托给 Kilo CLI（功能开发、PR 审查）"
sidebar_label: "Kilo"
description: "将编码任务委托给 Kilo CLI（功能开发、PR 审查）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kilo

将编码任务委托给 Kilo CLI（功能开发、PR 审查）。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/autonomous-ai-agents/kilo` |
| 版本 | `1.0.0` |
| 作者 | Kilo Team |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Coding-Agent`, `Kilo`, `Autonomous`, `Refactoring`, `Code-Review` |
| 相关 skill | [`claude-code`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`codex`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`hermes-agent`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), [`opencode`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Kilo CLI

使用 [Kilo](https://kilo.ai/cli) 作为由 Hermes 终端/进程工具编排的自主编码工作器。Kilo 是一个开源 AI 编码 agent（OpenCode 的分支），具备 CLI 与网页控制台，支持 500+ provider 模型，并内置 agent（Code、Plan、Ask、Debug、Review）。

## 适用场景

- 用户明确要求使用 Kilo
- 需要外部编码 agent 来实现/重构/审查代码
- 需要长时间运行的编码会话并定期检查进度
- 需要在隔离的工作目录/worktree 中并行执行任务
- 需要 CI/CD 流水线的全自主运行

## 前置条件

- 已安装 Kilo：`npm i -g @kilocode/cli`、`brew install Kilo-Org/tap/kilo`、`curl -fsSL https://kilo.ai/cli/install | bash`，或 AUR `paru -S kilo-bin`
- 已配置认证：`kilo auth login`（推荐使用 Kilo 账户——无需 API key 即可开始）或设置 provider 环境变量 / `kilo auth login <provider>`
- 验证：`kilo auth list` 应显示至少一个凭证
- 代码任务推荐使用 Git 仓库
- 交互式（`-i`）与后台会话需要 `pty=true`

## 二进制文件解析（重要）

Shell 环境可能会解析到不同的 Kilo 二进制文件。如果你的终端与 Hermes 的行为不一致，请检查：

```
terminal(command="which -a kilo")
terminal(command="kilo --version")
```

如有需要，可固定使用明确的二进制路径：

```
terminal(command="$HOME/.kilo/bin/kilo run '...'", workdir="~/project", pty=true)
```

## 单次任务

使用 `kilo run` 执行有边界的非交互式任务（无需 pty）：

```
terminal(command="kilo run 'Add retry logic to API calls and update tests'", workdir="~/project")
```

使用 `-f` 附加上下文文件：

```
terminal(command="kilo run 'Review this config for security issues' -f config.yaml -f .env.example", workdir="~/project")
```

使用 `--thinking` 显示模型思考过程：

```
terminal(command="kilo run 'Debug why tests fail in CI' --thinking", workdir="~/project")
```

以 `provider/model` 形式强制指定特定模型：

```
terminal(command="kilo run 'Refactor auth module' --model anthropic/claude-sonnet-4", workdir="~/project")
```

机器可读输出以便解析：

```
terminal(command="kilo run 'List all functions in src/' --format json", workdir="~/project")
```

## 自主模式（CI/CD）

Kilo 面向流水线的核心功能。`--auto` 禁用所有权限提示并自动批准每项操作，同时追踪 `task` 子会话的权限请求，使派生的子 agent 工作能够无人值守地进行：

```
terminal(command="kilo run --auto 'run tests and fix any failures'", workdir="~/project")
```

仅在可信环境中使用 `--auto`——它允许 agent 在无确认的情况下执行任何操作。对于仍想避免提示的临时单次任务，`--dangerously-skip-permissions` 会自动批准一次权限请求——但它仍然去除了审批关卡，因此仅可在隔离或可信的工作目录中使用，切勿用于用户的主要仓库。

## 内置 agent

Kilo 内置了可根据任务切换的专用 agent。传入 `--agent <name>`：

```
terminal(command="kilo run --agent review 'Review this PR vs main for bugs and security issues'", workdir="~/project")
terminal(command="kilo run --agent plan 'Design the schema for a multi-tenant billing service'", workdir="~/project")
```

| Agent | 用途 |
|------|-----|
| `code` | 默认。根据自然语言实现和编辑代码 |
| `plan` | 在编写代码前设计架构并撰写实现计划 |
| `ask` | 在不改动文件的情况下回答关于代码库的问题 |
| `debug` | 排查并追踪问题 |
| `review` | 从性能、安全、风格和测试覆盖率角度审查变更 |

## 交互式会话（后台运行）

对于需要多轮交互的迭代工作，运行 `kilo run -i`（直接交互的 split-footer 模式；需要 TTY，因此使用 `pty=true`）：

```
terminal(command="kilo run -i", workdir="~/project", background=true, pty=true)
# 返回 session_id

# 发送 prompt
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

对于长时间运行的有边界任务，运行 `kilo run '...'` 并使用 `background=true, pty=true`，以同样方式监控。

**注意：** 直接运行 `kilo`（无子命令）会在浏览器中打开本地 Kilo Console——它不是终端会话。终端编排请使用 `kilo run`（单次）或 `kilo run -i`（交互式）。

使用 Ctrl+C（`\x03`）或 `process(action="kill")` 退出交互式会话。不要依赖 `/exit`——请使用 kill 路径。

### 恢复会话

退出后，Kilo 会记录会话 ID。使用以下命令恢复：

```
terminal(command="kilo run -c", workdir="~/project", background=true, pty=true)  # 继续上次会话
terminal(command="kilo run -s ses_abc123", workdir="~/project", background=true, pty=true)  # 指定会话
```

在继续之前分叉（新 ID，保留历史）：

```
terminal(command="kilo run --fork -s ses_abc123 'Try a different approach'", workdir="~/project")
```

获取云端会话并在本地继续：

```
terminal(command="kilo run --cloud-fork -s ses_abc123 'Continue this cloud session locally'", workdir="~/project")
```

## 常用标志

| 标志 | 用途 |
|------|-----|
| `run 'prompt'` | 单次执行后退出 |
| `--auto` | 完全自主：自动批准所有权限（CI/CD） |
| `--continue` / `-c` | 继续上次 Kilo 会话 |
| `--session <id>` / `-s` | 继续指定会话 |
| `--fork` | 在继续前分叉会话（需要 `-c` 或 `-s`） |
| `--cloud-fork` | 获取云端会话并在本地继续（配合 `-s`） |
| `--agent <name>` | 选择 Kilo agent（code、plan、ask、debug、review） |
| `--model provider/model` / `-m` | 强制使用指定模型 |
| `--format json` | 机器可读的输出/事件 |
| `--file <path>` / `-f` | 向消息附加文件 |
| `--thinking` | 显示模型思考块 |
| `--variant <level>` | 推理强度（high、max、minimal） |
| `--title <name>` | 为会话命名 |
| `--interactive` / `-i` | 在直接交互 split-footer 模式下运行（需要 TTY） |
| `--share` | 分享会话 |
| `--dangerously-skip-permissions` | 自动批准一次权限请求 |

## 操作流程

1. 验证工具就绪状态：
   - `terminal(command="kilo --version")`
   - `terminal(command="kilo auth list")`
2. 对于有边界的任务，使用 `kilo run '...'`（无需 pty）。
3. 对于 CI/CD，使用 `kilo run --auto '...'`。
4. 对于迭代任务，使用 `kilo run -i` 配合 `background=true, pty=true`，或对长时间有边界任务使用 `kilo run '...'` 配合 `background=true, pty=true`。
5. 使用 `process(action="poll"|"log")` 监控长时间运行的任务。
6. 如果 Kilo 请求输入，通过 `process(action="submit", ...)` 响应。
7. 使用 `process(action="write", data="\x03")` 或 `process(action="kill")` 退出。
8. 向用户汇总文件变更、测试结果及后续步骤。

## PR 审查工作流

Kilo 内置 PR 命令，会获取并检出 PR 分支，导入 PR 正文中引用的会话，然后运行 Kilo：

```
terminal(command="kilo pr 42", workdir="~/project", pty=true)
```

或在临时克隆中审查以实现隔离：

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout <PR_NUMBER> && kilo run 'Review this PR vs main. Report bugs, security risks, test gaps, and style issues.' -f $(git diff origin/main --name-only | head -20 | tr '\n' ' ')")
```

## 并行工作模式

使用独立的工作目录/worktree 避免冲突：

```
terminal(command="kilo run --auto 'Fix issue #101 and commit'", workdir="/tmp/issue-101", background=true, pty=true)
terminal(command="kilo run --auto 'Add parser regression tests and commit'", workdir="/tmp/issue-102", background=true, pty=true)
process(action="list")
```

## 会话与成本管理

列出历史会话：

```
terminal(command="kilo session list")
```

查看 token 用量和费用：

```
terminal(command="kilo stats")
terminal(command="kilo stats --days 7 --models anthropic/claude-sonnet-4")
```

## 注意事项

- `kilo run`（默认，非交互式）会自动拒绝权限请求，除非设置了 `--auto` 或 `--dangerously-skip-permissions`。
- `kilo run` 单次任务**不需要** pty。`kilo run -i`（交互式）需要 TTY stdout——请使用 `pty=true`。
- 直接运行 `kilo` 会打开网页版 Kilo Console，而非终端会话。终端编排请使用 `kilo run` / `kilo run -i`。
- PATH 不匹配可能导致选择错误的 Kilo 二进制文件/模型配置。
- 如果 Kilo 看起来卡住了，在终止前先检查日志：
  - `process(action="log", session_id="<id>")`
- 避免多个并行 Kilo 会话共享同一工作目录。
- `--auto` 很危险——仅在可信环境中使用（它会批准每项操作）。
- `--dangerously-skip-permissions` 会为单次任务去除审批关卡——请像对待 `--auto` 一样，仅在隔离或可信的工作目录中使用，切勿用于用户的主要仓库。

## 验证

冒烟测试：

```
terminal(command="kilo run 'Respond with exactly: KILO_SMOKE_OK'")
```

成功标准：

- 输出包含 `KILO_SMOKE_OK`
- 命令退出时无 provider/模型错误
- 对于代码任务：预期文件已变更且测试通过

## 规则

1. 单次自动化任务优先使用 `kilo run`——更简单且无需 pty。
2. 仅在 CI/CD 流水线使用 `--auto`——它会自动批准一切。
3. 仅在需要迭代时使用交互式模式（`-i`）。
4. 始终将 Kilo 会话限定在单个仓库/工作目录内。
5. 对于长时间任务，从 `process` 日志中提供进度更新。
6. 报告具体结果（文件变更、测试情况、剩余风险）。
7. 使用 Ctrl+C 或 kill 退出交互式会话——使用 kill 路径，而非 `/exit`。
