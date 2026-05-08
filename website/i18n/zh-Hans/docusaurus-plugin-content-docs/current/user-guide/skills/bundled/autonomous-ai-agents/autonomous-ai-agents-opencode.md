---
title: "Opencode — 委托编码给 OpenCode CLI（功能、PR 审查）"
sidebar_label: "Opencode"
description: "委托编码给 OpenCode CLI（功能、PR 审查）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Opencode

委托编码给 OpenCode CLI（功能、PR 审查）。

## 技能元数据

| | |
|---|---|
| 来源 | 捆绑（默认安装） |
| 路径 | `skills/autonomous-ai-agents/opencode` |
| 版本 | `1.2.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 标签 | `Coding-Agent`、`OpenCode`、`Autonomous`、`Refactoring`、`Code-Review` |
| 相关技能 | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code)、[`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex)、[`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 加载此技能时触发的完整技能定义。这是代理激活技能时看到的指令。
:::

# OpenCode CLI

使用 [OpenCode](https://opencode.ai) 作为由 Hermes 终端/进程工具编排的自主编码工作器。OpenCode 是一个提供商无关的、开源的、带有 TUI 和 CLI 的 AI 编码代理。

## 使用场景

- 用户明确要求使用 OpenCode
- 你希望外部编码代理实现/重构/审查代码
- 你需要带有进度检查的长时间运行编码会话
- 你希望在隔离的工作目录/工作树中并行执行任务

## 前置条件

- 安装 OpenCode：`npm i -g opencode-ai@latest` 或 `brew install anomalyco/tap/opencode`
- 配置认证：`opencode auth login` 或设置提供商环境变量（OPENROUTER_API_KEY 等）
- 验证：`opencode auth list` 应显示至少一个提供商
- 代码任务需要 git 仓库（推荐）
- 交互式 TUI 会话使用 `pty=true`

## 二进制文件解析（重要）

Shell 环境可能解析不同的 OpenCode 二进制文件。如果你的终端和 Hermes 之间的行为不同，检查：

```
terminal(command="which -a opencode")
terminal(command="opencode --version")
```

如果需要，固定显式二进制路径：

```
terminal(command="$HOME/.opencode/bin/opencode run '...'", workdir="~/project", pty=true)
```

## 一次性任务

使用 `opencode run` 进行有界、非交互式任务：

```
terminal(command="opencode run 'Add retry logic to API calls and update tests'", workdir="~/project")
```

用 `-f` 附加上下文文件：

```
terminal(command="opencode run 'Review this config for security issues' -f config.yaml -f .env.example", workdir="~/project")
```

用 `--thinking` 显示模型思考：

```
terminal(command="opencode run 'Debug why tests fail in CI' --thinking", workdir="~/project")
```

强制指定模型：

```
terminal(command="opencode run 'Refactor auth module' --model openrouter/anthropic/claude-sonnet-4", workdir="~/project")
```

## 交互式会话（后台）

对于需要多次交流的迭代工作，在后台启动 TUI：

```
terminal(command="opencode", workdir="~/project", background=true, pty=true)
# 返回 session_id

# 发送提示
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

**重要：** 不要使用 `/exit` — 它不是有效的 OpenCode 命令，将打开代理选择器对话框。使用 Ctrl+C（`\x03`）或 `process(action="kill")` 退出。

### TUI 键盘绑定

| 键 | 操作 |
|-----|--------|
| `Enter` | 提交消息（如需要按两次） |
| `Tab` | 在代理之间切换（build/plan） |
| `Ctrl+P` | 打开命令面板 |
| `Ctrl+X L` | 切换会话 |
| `Ctrl+X M` | 切换模型 |
| `Ctrl+X N` | 新会话 |
| `Ctrl+X E` | 打开编辑器 |
| `Ctrl+C` | 退出 OpenCode |

### 恢复会话

退出后，OpenCode 打印会话 ID。用以下方式恢复：

```
terminal(command="opencode -c", workdir="~/project", background=true, pty=true)  # 继续上次会话
terminal(command="opencode -s ses_abc123", workdir="~/project", background=true, pty=true)  # 特定会话
```

## 常用标志

| 标志 | 用途 |
|------|-----|
| `run 'prompt'` | 一次性执行并退出 |
| `--continue` / `-c` | 继续上次 OpenCode 会话 |
| `--session <id>` / `-s` | 继续特定会话 |
| `--agent <name>` | 选择 OpenCode 代理（build 或 plan） |
| `--model provider/model` | 强制指定模型 |
| `--format json` | 机器可读输出/事件 |
| `--file <path>` / `-f` | 附加文件到消息 |
| `--thinking` | 显示模型思考块 |
| `--variant <level>` | 推理努力（high、max、minimal） |
| `--title <name>` | 命名会话 |
| `--attach <url>` | 连接到正在运行的 opencode 服务器 |

## 步骤

1. 验证工具准备情况：
   - `terminal(command="opencode --version")`
   - `terminal(command="opencode auth list")`
2. 对于有界任务，使用 `opencode run '...'`（无需 pty）。
3. 对于迭代任务，使用 `background=true, pty=true` 启动 `opencode`。
4. 使用 `process(action="poll"|"log")` 监控长任务。
5. 如果 OpenCode 要求输入，通过 `process(action="submit", ...)` 响应。
6. 使用 `process(action="write", data="\x03")` 或 `process(action="kill")` 退出。
7. 向用户总结文件更改、测试结果和后续步骤。

## PR 审查工作流

OpenCode 有内置 PR 命令：

```
terminal(command="opencode pr 42", workdir="~/project", pty=true)
```

或在临时克隆中审查以实现隔离：

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && opencode run 'Review this PR vs main. Report bugs, security risks, test gaps, and style issues.' -f $(git diff origin/main --name-only | head -20 | tr '\n' ' ')", pty=true)
```

## 并行工作模式

使用独立工作目录/工作树避免冲突：

```
terminal(command="opencode run 'Fix issue #101 and commit'", workdir="/tmp/issue-101", background=true, pty=true)
terminal(command="opencode run 'Add parser regression tests and commit'", workdir="/tmp/issue-102", background=true, pty=true)
process(action="list")
```

## 会话和成本管理

列出过去的会话：

```
terminal(command="opencode session list")
```

检查令牌使用量和成本：

```
terminal(command="opencode stats")
terminal(command="opencode stats --days 7 --models anthropic/claude-sonnet-4")
```

## 陷阱

- 交互式 `opencode`（TUI）会话需要 `pty=true`。`opencode run` 命令**不需要** pty。
- `/exit` **不是**有效命令 — 它会打开代理选择器。使用 Ctrl+C 退出 TUI。
- PATH 不匹配可能选择错误的 OpenCode 二进制/模型配置。
- 如果 OpenCode 看起来卡住，在终止前检查日志：
  - `process(action="log", session_id="<id>")`
- 避免多个并行 OpenCode 会话共享一个工作目录。
- 在 TUI 中提交时可能需要按两次 Enter（一次完成文本，一次发送）。

## 验证

冒烟测试：

```
terminal(command="opencode run 'Respond with exactly: OPENCODE_SMOKE_OK'")
```

成功标准：
- 输出包含 `OPENCODE_SMOKE_OK`
- 命令退出时没有提供商/模型错误
- 对于代码任务：预期文件已更改且测试通过

## 规则

1. 一次性自动化首选 `opencode run` — 更简单，不需要 pty。
2. 仅在需要迭代时使用交互式后台模式。
3. 始终将 OpenCode 会话限定在单个仓库/工作目录。
4. 对于长任务，提供来自 `process` 日志的进度更新。
5. 报告具体结果（文件更改、测试、剩余风险）。
6. 使用 Ctrl+C 或 kill 退出交互式会话，不要用 `/exit`。
