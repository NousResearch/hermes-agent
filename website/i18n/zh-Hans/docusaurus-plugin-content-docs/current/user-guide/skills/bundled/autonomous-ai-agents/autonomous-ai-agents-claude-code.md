---
title: "Claude Code — 将编码任务委托给 Claude Code CLI（功能、PR）"
sidebar_label: "Claude Code"
description: "将编码任务委托给 Claude Code CLI（功能、PR）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Claude Code

将编码任务委托给 Claude Code CLI（功能、PR）。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/autonomous-ai-agents/claude-code` |
| 版本 | `2.2.0` |
| 作者 | Hermes Agent + Teknium |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `Coding-Agent`, `Claude`, `Anthropic`, `Code-Review`, `Refactoring`, `PTY`, `Automation` |
| 相关 skill | [`codex`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`hermes-agent`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), [`opencode`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode), [`kilo`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-kilo) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# Claude Code — Hermes 编排指南

通过 Hermes 终端将编码任务委托给 [Claude Code](https://code.claude.com/docs/en/cli-reference)（Anthropic 的自主编码 agent CLI）。Claude Code v2.x 可以自主读取文件、编写代码、运行 shell 命令、派生子 agent 并管理 git 工作流。

## 前置条件

- **安装：** `npm install -g @anthropic-ai/claude-code`
- **认证：** 运行一次 `claude` 以登录（Pro/Max 使用浏览器 OAuth，或设置 `ANTHROPIC_API_KEY`）
- **控制台认证：** `claude auth login --console` 用于 API key 计费
- **SSO 认证：** `claude auth login --sso` 用于企业版
- **检查状态：** `claude auth status`（JSON）或 `claude auth status --text`（人类可读）
- **健康检查：** `claude doctor` — 检查自动更新器和安装健康状态
- **版本检查：** `claude --version`（需要 v2.x+）
- **更新：** `claude update` 或 `claude upgrade`

## 两种编排模式

Hermes 以两种根本不同的方式与 Claude Code 交互。请根据任务选择合适的模式。

### 模式一：Print 模式（`-p`）— 非交互式（大多数任务的首选）

Print 模式运行一次性任务，返回结果后退出。无需 PTY（伪终端），无交互式提示。这是最简洁的集成方式。

```
terminal(command="claude -p 'Add error handling to all API calls in src/' --allowedTools 'Read,Edit' --max-turns 10", workdir="/path/to/project", timeout=120)
```

**何时使用 print 模式：**
- 一次性编码任务（修复 bug、添加功能、重构）
- CI/CD 自动化和脚本
- 使用 `--json-schema` 进行结构化数据提取
- 管道输入处理（`cat file | claude -p "analyze this"`）
- 任何不需要多轮对话的任务

**Print 模式跳过所有交互式对话框** — 无工作区信任提示，无权限确认。这使其非常适合自动化场景。

### 模式二：通过 tmux 的交互式 PTY — 多轮会话

交互模式提供完整的对话式 REPL（交互式解释器），可以发送后续 prompt、使用斜杠命令，并实时观察 Claude 的工作过程。**需要 tmux 编排。**

```
# 启动 tmux 会话
terminal(command="tmux new-session -d -s claude-work -x 140 -y 40")

# 在其中启动 Claude Code
terminal(command="tmux send-keys -t claude-work 'cd /path/to/project && claude' Enter")

# 等待启动，然后发送任务
# （等待约 3-5 秒显示欢迎界面）
terminal(command="sleep 5 && tmux send-keys -t claude-work 'Refactor the auth module to use JWT tokens' Enter")

# 通过捕获面板监控进度
terminal(command="sleep 15 && tmux capture-pane -t claude-work -p -S -50")

# 发送后续任务
terminal(command="tmux send-keys -t claude-work 'Now add unit tests for the new JWT code' Enter")

# 完成后退出
terminal(command="tmux send-keys -t claude-work '/exit' Enter")
```

**何时使用交互模式：**
- 多轮迭代工作（重构 → 审查 → 修复 → 测试循环）
- 需要人工介入决策的任务
- 探索性编码会话
- 需要使用 Claude 斜杠命令时（`/compact`、`/review`、`/model`）

## PTY 对话框处理（交互模式的关键）

Claude Code 在首次启动时最多会显示两个确认对话框。**必须**通过 tmux send-keys 处理这些对话框。

### 对话框一：工作区信任（首次访问某目录时）
```
❯ 1. Yes, I trust this folder    ← 默认（直接按 Enter）
  2. No, exit
```
**处理方式：** `tmux send-keys -t <session> Enter` — 默认选项正确。

### 对话框二：绕过权限警告（仅在使用 --dangerously-skip-permissions 时）
```
❯ 1. No, exit                    ← 默认（错误选项！）
  2. Yes, I accept
```
**处理方式：** 必须先向下导航，再按 Enter：
```
tmux send-keys -t <session> Down && sleep 0.3 && tmux send-keys -t <session> Enter
```

### 健壮的对话框处理模式
```
# 使用权限绕过启动
terminal(command="tmux send-keys -t claude-work 'claude --dangerously-skip-permissions \"your task\"' Enter")

# 处理信任对话框（按 Enter 选择默认的"Yes"）
terminal(command="sleep 4 && tmux send-keys -t claude-work Enter")

# 处理权限对话框（按 Down 再按 Enter 选择"Yes, I accept"）
terminal(command="sleep 3 && tmux send-keys -t claude-work Down && sleep 0.3 && tmux send-keys -t claude-work Enter")

# 等待 Claude 工作
terminal(command="sleep 15 && tmux capture-pane -t claude-work -p -S -60")
```

**注意：** 某个目录首次接受信任后，信任对话框不会再次出现。只有权限对话框会在每次使用 `--dangerously-skip-permissions` 时重复出现。

## CLI 子命令

| 子命令 | 用途 |
|------------|---------|
| `claude` | 启动交互式 REPL |
| `claude "query"` | 以初始 prompt 启动 REPL |
| `claude -p "query"` | Print 模式（非交互式，完成后退出） |
| `cat file \| claude -p "query"` | 通过管道传入内容作为 stdin 上下文 |
| `claude -c` | 继续此目录中最近的对话 |
| `claude -r "id"` | 通过 ID 或名称恢复特定会话 |
| `claude auth login` | 登录（添加 `--console` 用于 API 计费，`--sso` 用于企业版） |
| `claude auth status` | 检查登录状态（返回 JSON；`--text` 为人类可读格式） |
| `claude mcp add <name> -- <cmd>` | 添加 MCP 服务器 |
| `claude mcp list` | 列出已配置的 MCP 服务器 |
| `claude mcp remove <name>` | 移除 MCP 服务器 |
| `claude agents` | 列出已配置的 agent |
| `claude doctor` | 对安装和自动更新器运行健康检查 |
| `claude update` / `claude upgrade` | 将 Claude Code 更新到最新版本 |
| `claude remote-control` | 启动服务器以从 claude.ai 或移动应用控制 Claude |
| `claude install [target]` | 安装原生构建（stable、latest 或特定版本） |
| `claude setup-token` | 设置长期认证 token（需要订阅） |
| `claude plugin` / `claude plugins` | 管理 Claude Code 插件 |
| `claude auto-mode` | 检查自动模式分类器配置 |

## Print 模式深度解析

### 结构化 JSON 输出
```
terminal(command="claude -p 'Analyze auth.py for security issues' --output-format json --max-turns 5", workdir="/project", timeout=120)
```

返回包含以下字段的 JSON 对象：
```json
{
  "type": "result",
  "subtype": "success",
  "result": "The analysis text...",
  "session_id": "75e2167f-...",
  "num_turns": 3,
  "total_cost_usd": 0.0787,
  "duration_ms": 10276,
  "stop_reason": "end_turn",
  "terminal_reason": "completed",
  "usage": { "input_tokens": 5, "output_tokens": 603, ... },
  "modelUsage": { "claude-sonnet-4-6": { "costUSD": 0.078, "contextWindow": 200000 } }
}
```

**关键字段：** `session_id` 用于恢复会话，`num_turns` 表示 agentic 循环次数，`total_cost_usd` 用于费用追踪，`subtype` 用于成功/错误检测（`success`、`error_max_turns`、`error_budget`）。

### 流式 JSON 输出
如需实时 token 流式传输，使用 `stream-json` 配合 `--verbose`：
```
terminal(command="claude -p 'Write a summary' --output-format stream-json --verbose --include-partial-messages", timeout=60)
```

返回换行符分隔的 JSON 事件。使用 jq 过滤实时文本：
```
claude -p "Explain X" --output-format stream-json --verbose --include-partial-messages | \
  jq -rj 'select(.type == "stream_event" and .event.delta.type? == "text_delta") | .event.delta.text'
```

流事件包含 `system/api_retry`，带有 `attempt`、`max_retries` 和 `error` 字段（例如 `rate_limit`、`billing_error`）。

### 双向流式传输
如需实时输入和输出流式传输：
```
claude -p "task" --input-format stream-json --output-format stream-json --replay-user-messages
```
`--replay-user-messages` 在 stdout 上重新发出用户消息以供确认。

### 管道输入
```
# 通过管道传入文件进行分析
terminal(command="cat src/auth.py | claude -p 'Review this code for bugs' --max-turns 1", timeout=60)

# 通过管道传入多个文件
terminal(command="cat src/*.py | claude -p 'Find all TODO comments' --max-turns 1", timeout=60)

# 通过管道传入命令输出
terminal(command="git diff HEAD~3 | claude -p 'Summarize these changes' --max-turns 1", timeout=60)
```

### 使用 JSON Schema 进行结构化提取
```
terminal(command="claude -p 'List all functions in src/' --output-format json --json-schema '{\"type\":\"object\",\"properties\":{\"functions\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"functions\"]}' --max-turns 5", workdir="/project", timeout=90)
```

从 JSON 结果中解析 `structured_output`。Claude 在返回前会根据 schema 验证输出。

### 会话续接
```
# 开始一个任务
terminal(command="claude -p 'Start refactoring the database layer' --output-format json --max-turns 10 > /tmp/session.json", workdir="/project", timeout=180)

# 使用会话 ID 恢复
terminal(command="claude -p 'Continue and add connection pooling' --resume $(cat /tmp/session.json | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"session_id\"])') --max-turns 5", workdir="/project", timeout=120)

# 或恢复同一目录中最近的会话
terminal(command="claude -p 'What did you do last time?' --continue --max-turns 1", workdir="/project", timeout=30)

# 派生会话（新 ID，保留历史）
terminal(command="claude -p 'Try a different approach' --resume <id> --fork-session --max-turns 10", workdir="/project", timeout=120)
```

### CI/脚本的精简模式
```
terminal(command="claude --bare -p 'Run all tests and report failures' --allowedTools 'Read,Bash' --max-turns 10", workdir="/project", timeout=180)
```

`--bare` 跳过 hook、插件、MCP 发现和 CLAUDE.md 加载。启动最快。需要 `ANTHROPIC_API_KEY`（跳过 OAuth）。

在精简模式下选择性加载上下文：
| 要加载的内容 | 标志 |
|---------|------|
| 系统 prompt 追加内容 | `--append-system-prompt "text"` 或 `--append-system-prompt-file path` |
| 设置 | `--settings <file-or-json>` |
| MCP 服务器 | `--mcp-config <file-or-json>` |
| 自定义 agent | `--agents '<json>'` |

### 过载时的备用模型
```
terminal(command="claude -p 'task' --fallback-model haiku --max-turns 5", timeout=90)
```
当默认模型过载时自动切换到指定模型（仅限 print 模式）。

## 完整 CLI 标志参考

### 会话与环境
| 标志 | 效果 |
|------|--------|
| `-p, --print` | 非交互式一次性模式（完成后退出） |
| `-c, --continue` | 恢复当前目录中最近的对话 |
| `-r, --resume <id>` | 通过 ID 或名称恢复特定会话（无 ID 时显示交互式选择器） |
| `--fork-session` | 恢复时创建新会话 ID 而非复用原始 ID |
| `--session-id <uuid>` | 为对话使用特定 UUID |
| `--no-session-persistence` | 不将会话保存到磁盘（仅限 print 模式） |
| `--add-dir <paths...>` | 授予 Claude 访问额外工作目录的权限 |
| `-w, --worktree [name]` | 在 `.claude/worktrees/<name>` 处的隔离 git worktree 中运行 |
| `--tmux` | 为 worktree 创建 tmux 会话（需要 `--worktree`） |
| `--ide` | 启动时自动连接到有效的 IDE |
| `--chrome` / `--no-chrome` | 启用/禁用 Chrome 浏览器集成以进行 Web 测试 |
| `--from-pr [number]` | 恢复与特定 GitHub PR 关联的会话 |
| `--file <specs...>` | 启动时下载的文件资源（格式：`file_id:relative_path`） |

### 模型与性能
| 标志 | 效果 |
|------|--------|
| `--model <alias>` | 模型选择：`sonnet`、`opus`、`haiku` 或完整名称如 `claude-sonnet-4-6` |
| `--effort <level>` | 推理深度：`low`、`medium`、`high`、`max`、`auto` |
| `--max-turns <n>` | 限制 agentic 循环次数（仅限 print 模式；防止失控） |
| `--max-budget-usd <n>` | 以美元为单位限制 API 花费（仅限 print 模式） |
| `--fallback-model <model>` | 默认模型过载时自动切换（仅限 print 模式） |
| `--betas <betas...>` | 在 API 请求中包含的 beta 头（仅限 API key 用户） |

### 权限与安全
| 标志 | 效果 |
|------|--------|
| `--dangerously-skip-permissions` | 自动批准所有工具使用（文件写入、bash、网络等） |
| `--allow-dangerously-skip-permissions` | 将绕过作为*选项*启用，但不默认启用 |
| `--permission-mode <mode>` | `default`、`acceptEdits`、`plan`、`auto`、`dontAsk`、`bypassPermissions` |
| `--allowedTools <tools...>` | 白名单特定工具（逗号或空格分隔） |
| `--disallowedTools <tools...>` | 黑名单特定工具 |
| `--tools <tools...>` | 覆盖内置工具集（`""` = 无，`"default"` = 全部，或工具名称） |

### 输出与输入格式
| 标志 | 效果 |
|------|--------|
| `--output-format <fmt>` | `text`（默认）、`json`（单个结果对象）、`stream-json`（换行符分隔） |
| `--input-format <fmt>` | `text`（默认）或 `stream-json`（实时流式输入） |
| `--json-schema <schema>` | 强制输出符合 schema 的结构化 JSON |
| `--verbose` | 完整的逐轮输出 |
| `--include-partial-messages` | 在消息块到达时包含部分消息（stream-json + print） |
| `--replay-user-messages` | 在 stdout 上重新发出用户消息（stream-json 双向） |

### 系统 Prompt 与上下文
| 标志 | 效果 |
|------|--------|
| `--append-system-prompt <text>` | **追加**到默认系统 prompt（保留内置能力） |
| `--append-system-prompt-file <path>` | **追加**文件内容到默认系统 prompt |
| `--system-prompt <text>` | **替换**整个系统 prompt（通常建议使用 --append） |
| `--system-prompt-file <path>` | 用文件内容**替换**系统 prompt |
| `--bare` | 跳过 hook、插件、MCP 发现、CLAUDE.md、OAuth（启动最快） |
| `--agents '<json>'` | 以 JSON 形式动态定义自定义子 agent |
| `--mcp-config <path>` | 从 JSON 文件加载 MCP 服务器（可重复使用） |
| `--strict-mcp-config` | 仅使用 `--mcp-config` 中的 MCP 服务器，忽略所有其他 MCP 配置 |
| `--settings <file-or-json>` | 从 JSON 文件或内联 JSON 加载额外设置 |
| `--setting-sources <sources>` | 逗号分隔的加载来源：`user`、`project`、`local` |
| `--plugin-dir <paths...>` | 仅在本次会话中从目录加载插件 |
| `--disable-slash-commands` | 禁用所有 skill/斜杠命令 |

### 调试
| 标志 | 效果 |
|------|--------|
| `-d, --debug [filter]` | 启用调试日志，可选类别过滤器（例如 `"api,hooks"`、`"!1p,!file"`） |
| `--debug-file <path>` | 将调试日志写入文件（隐式启用调试模式） |

### Agent 团队
| 标志 | 效果 |
|------|--------|
| `--teammate-mode <mode>` | agent 团队的显示方式：`auto`、`in-process` 或 `tmux` |
| `--brief` | 启用 `SendUserMessage` 工具用于 agent 间通信 |

### --allowedTools / --disallowedTools 的工具名称语法
```
Read                    # 所有文件读取
Edit                    # 文件编辑（现有文件）
Write                   # 文件创建（新文件）
Bash                    # 所有 shell 命令
Bash(git *)             # 仅 git 命令
Bash(git commit *)      # 仅 git commit 命令
Bash(npm run lint:*)    # 使用通配符的模式匹配
WebSearch               # Web 搜索能力
WebFetch                # Web 页面抓取
mcp__<server>__<tool>   # 特定 MCP 工具
```

## 设置与配置

### 设置优先级（从高到低）
1. **CLI 标志** — 覆盖所有设置
2. **本地项目：** `.claude/settings.local.json`（个人，已 gitignore）
3. **项目：** `.claude/settings.json`（共享，git 跟踪）
4. **用户：** `~/.claude/settings.json`（全局）

### 设置中的权限
```json
{
  "permissions": {
    "allow": ["Bash(npm run lint:*)", "WebSearch", "Read"],
    "ask": ["Write(*.ts)", "Bash(git push*)"],
    "deny": ["Read(.env)", "Bash(rm -rf *)"]
  }
}
```

### 记忆文件（CLAUDE.md）层级
1. **全局：** `~/.claude/CLAUDE.md` — 适用于所有项目
2. **项目：** `./CLAUDE.md` — 项目特定上下文（git 跟踪）
3. **本地：** `.claude/CLAUDE.local.md` — 个人项目覆盖（已 gitignore）

在交互模式中使用 `#` 前缀快速添加到记忆：`# Always use 2-space indentation`。

## 交互会话：斜杠命令

### 会话与上下文
| 命令 | 用途 |
|---------|---------|
| `/help` | 显示所有命令（包括自定义和 MCP 命令） |
| `/compact [focus]` | 压缩上下文以节省 token；CLAUDE.md 在压缩后保留。例如 `/compact focus on auth logic` |
| `/clear` | 清除对话历史，重新开始 |
| `/context` | 以彩色网格可视化上下文使用情况并提供优化建议 |
| `/cost` | 查看 token 使用情况，包含按模型和缓存命中的细分 |
| `/resume` | 切换到或恢复不同的会话 |
| `/rewind` | 回退到对话或代码中的上一个检查点 |
| `/btw <question>` | 提问附带问题而不增加上下文成本 |
| `/status` | 显示版本、连接状态和会话信息 |
| `/todos` | 列出对话中跟踪的待办事项 |
| `/exit` 或 `Ctrl+D` | 结束会话 |

### 开发与审查
| 命令 | 用途 |
|---------|---------|
| `/review` | 请求对当前更改进行代码审查 |
| `/security-review` | 对当前更改执行安全分析 |
| `/plan [description]` | 进入 Plan 模式并自动启动任务规划 |
| `/loop [interval]` | 在会话中安排定期任务 |
| `/batch` | 自动创建 worktree 用于大型并行更改（5-30 个 worktree） |

### 配置与工具
| 命令 | 用途 |
|---------|---------|
| `/model [model]` | 在会话中途切换模型（使用方向键调整 effort） |
| `/effort [level]` | 设置推理 effort：`low`、`medium`、`high`、`max` 或 `auto` |
| `/init` | 创建 CLAUDE.md 文件用于项目记忆 |
| `/memory` | 打开 CLAUDE.md 进行编辑 |
| `/config` | 打开交互式设置配置 |
| `/permissions` | 查看/更新工具权限 |
| `/agents` | 管理专用子 agent |
| `/mcp` | 管理 MCP 服务器的交互式 UI |
| `/add-dir` | 添加额外工作目录（适用于 monorepo） |
| `/usage` | 显示计划限制和速率限制状态 |
| `/voice` | 启用按键说话语音模式（20 种语言；按住 Space 录音，松开发送） |
| `/release-notes` | 版本发布说明的交互式选择器 |

### 自定义斜杠命令
创建 `.claude/commands/<name>.md`（项目共享）或 `~/.claude/commands/<name>.md`（个人）：

```markdown
# .claude/commands/deploy.md
Run the deploy pipeline:
1. Run all tests
2. Build the Docker image
3. Push to registry
4. Update the $ARGUMENTS environment (default: staging)
```

用法：`/deploy production` — `$ARGUMENTS` 将被用户输入替换。

### Skills（自然语言调用）
与斜杠命令（手动调用）不同，`.claude/skills/` 中的 skill 是 markdown 指南，当任务匹配时 Claude 会通过自然语言自动调用：

```markdown
# .claude/skills/database-migration.md
When asked to create or modify database migrations:
1. Use Alembic for migration generation
2. Always create a rollback function
3. Test migrations against a local database copy
```

## 交互会话：键盘快捷键

### 通用控制
| 按键 | 操作 |
|-----|--------|
| `Ctrl+C` | 取消当前输入或生成 |
| `Ctrl+D` | 退出会话 |
| `Ctrl+R` | 反向搜索命令历史 |
| `Ctrl+B` | 将运行中的任务移至后台 |
| `Ctrl+V` | 将图片粘贴到对话中 |
| `Ctrl+O` | 转录模式 — 查看 Claude 的思考过程 |
| `Ctrl+G` 或 `Ctrl+X Ctrl+E` | 在外部编辑器中打开 prompt |
| `Esc Esc` | 回退对话或代码状态/总结 |

### 模式切换
| 按键 | 操作 |
|-----|--------|
| `Shift+Tab` | 循环切换权限模式（普通 → 自动接受 → 计划） |
| `Alt+P` | 切换模型 |
| `Alt+T` | 切换思考模式 |
| `Alt+O` | 切换快速模式 |

### 多行输入
| 按键 | 操作 |
|-----|--------|
| `\` + `Enter` | 快速换行 |
| `Shift+Enter` | 换行（备选） |
| `Ctrl+J` | 换行（备选） |

### 输入前缀
| 前缀 | 操作 |
|--------|--------|
| `!` | 直接执行 bash，绕过 AI（例如 `!npm test`）。单独使用 `!` 可切换 shell 模式。 |
| `@` | 通过自动补全引用文件/目录（例如 `@./src/api/`） |
| `#` | 快速添加到 CLAUDE.md 记忆（例如 `# Use 2-space indentation`） |
| `/` | 斜杠命令 |

### 专业技巧："ultrathink"
在 prompt 中使用关键词 "ultrathink" 可在该轮次获得最大推理 effort。无论当前 `/effort` 设置如何，这都会触发最深层的思考模式。

## PR 审查模式

### 快速审查（Print 模式）
```
terminal(command="cd /path/to/repo && git diff main...feature-branch | claude -p 'Review this diff for bugs, security issues, and style problems. Be thorough.' --max-turns 1", timeout=60)
```

### 深度审查（交互式 + Worktree）
```
terminal(command="tmux new-session -d -s review -x 140 -y 40")
terminal(command="tmux send-keys -t review 'cd /path/to/repo && claude -w pr-review' Enter")
terminal(command="sleep 5 && tmux send-keys -t review Enter")  # 信任对话框
terminal(command="sleep 2 && tmux send-keys -t review 'Review all changes vs main. Check for bugs, security issues, race conditions, and missing tests.' Enter")
terminal(command="sleep 30 && tmux capture-pane -t review -p -S -60")
```

### 通过 PR 编号审查
```
terminal(command="claude -p 'Review this PR thoroughly' --from-pr 42 --max-turns 10", workdir="/path/to/repo", timeout=120)
```

### Claude Worktree 配合 tmux
```
terminal(command="claude -w feature-x --tmux", workdir="/path/to/repo")
```
在 `.claude/worktrees/feature-x` 创建隔离的 git worktree，并为其创建 tmux 会话。有 iTerm2 时使用原生面板；添加 `--tmux=classic` 使用传统 tmux。

## 并行 Claude 实例

同时运行多个独立的 Claude 任务：

```
# 任务一：修复后端
terminal(command="tmux new-session -d -s task1 -x 140 -y 40 && tmux send-keys -t task1 'cd ~/project && claude -p \"Fix the auth bug in src/auth.py\" --allowedTools \"Read,Edit\" --max-turns 10' Enter")

# 任务二：编写测试
terminal(command="tmux new-session -d -s task2 -x 140 -y 40 && tmux send-keys -t task2 'cd ~/project && claude -p \"Write integration tests for the API endpoints\" --allowedTools \"Read,Write,Bash\" --max-turns 15' Enter")

# 任务三：更新文档
terminal(command="tmux new-session -d -s task3 -x 140 -y 40 && tmux send-keys -t task3 'cd ~/project && claude -p \"Update README.md with the new API endpoints\" --allowedTools \"Read,Edit\" --max-turns 5' Enter")

# 监控所有任务
terminal(command="sleep 30 && for s in task1 task2 task3; do echo '=== '$s' ==='; tmux capture-pane -t $s -p -S -5 2>/dev/null; done")
```

## CLAUDE.md — 项目上下文文件

Claude Code 自动从项目根目录加载 `CLAUDE.md`。使用它来持久化项目上下文：

```markdown
# Project: My API

## Architecture
- FastAPI backend with SQLAlchemy ORM
- PostgreSQL database, Redis cache
- pytest for testing with 90% coverage target

## Key Commands
- `make test` — run full test suite
- `make lint` — ruff + mypy
- `make dev` — start dev server on :8000

## Code Standards
- Type hints on all public functions
- Docstrings in Google style
- 2-space indentation for YAML, 4-space for Python
- No wildcard imports
```

**要具体。** 不要写"写好代码"，而应写"JS 使用 2 空格缩进"或"测试文件以 `.test.ts` 后缀命名"。具体的指令可以减少纠错循环。

### 规则目录（模块化 CLAUDE.md）
对于规则较多的项目，使用规则目录代替单一庞大的 CLAUDE.md：
- **项目规则：** `.claude/rules/*.md` — 团队共享，git 跟踪
- **用户规则：** `~/.claude/rules/*.md` — 个人，全局

规则目录中的每个 `.md` 文件都作为额外上下文加载。这比将所有内容塞进单个 CLAUDE.md 更整洁。

### 自动记忆
Claude 自动将学到的项目上下文存储在 `~/.claude/projects/<project>/memory/` 中。
- **限制：** 每个项目 25KB 或 200 行
- 这与 CLAUDE.md 分开 — 这是 Claude 自己关于项目的笔记，跨会话积累

## 自定义子 Agent

在 `.claude/agents/`（项目）、`~/.claude/agents/`（个人）中定义专用 agent，或通过 `--agents` CLI 标志（会话）定义：

### Agent 位置优先级
1. `.claude/agents/` — 项目级，团队共享
2. `--agents` CLI 标志 — 会话特定，动态
3. `~/.claude/agents/` — 用户级，个人

### 创建 Agent
```markdown
# .claude/agents/security-reviewer.md
---
name: security-reviewer
description: Security-focused code review
model: opus
tools: [Read, Bash]
---
You are a senior security engineer. Review code for:
- Injection vulnerabilities (SQL, XSS, command injection)
- Authentication/authorization flaws
- Secrets in code
- Unsafe deserialization
```

调用方式：`@security-reviewer review the auth module`

### 通过 CLI 动态定义 Agent
```
terminal(command="claude --agents '{\"reviewer\": {\"description\": \"Reviews code\", \"prompt\": \"You are a code reviewer focused on performance\"}}' -p 'Use @reviewer to check auth.py'", timeout=120)
```

Claude 可以编排多个 agent："Use @db-expert to optimize queries, then @security to audit the changes."

## Hook — 事件触发自动化

在 `.claude/settings.json`（项目）或 `~/.claude/settings.json`（全局）中配置：

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Write(*.py)",
      "hooks": [{"type": "command", "command": "ruff check --fix $CLAUDE_FILE_PATHS"}]
    }],
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{"type": "command", "command": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -q 'rm -rf'; then echo 'Blocked!' && exit 2; fi"}]
    }],
    "Stop": [{
      "hooks": [{"type": "command", "command": "echo 'Claude finished a response' >> /tmp/claude-activity.log"}]
    }]
  }
}
```

### 全部 8 种 Hook 类型
| Hook | 触发时机 | 常见用途 |
|------|--------------|------------|
| `UserPromptSubmit` | Claude 处理用户 prompt 之前 | 输入验证、日志记录 |
| `PreToolUse` | 工具执行之前 | 安全门控、阻止危险命令（exit 2 = 阻止） |
| `PostToolUse` | 工具完成之后 | 自动格式化代码、运行 linter |
| `Notification` | 权限请求或等待输入时 | 桌面通知、告警 |
| `Stop` | Claude 完成响应时 | 完成日志记录、状态更新 |
| `SubagentStop` | 子 agent 完成时 | Agent 编排 |
| `PreCompact` | 上下文记忆被清除之前 | 备份会话转录 |
| `SessionStart` | 会话开始时 | 加载开发上下文（例如 `git status`） |

### Hook 环境变量
| 变量 | 内容 |
|----------|---------|
| `CLAUDE_PROJECT_DIR` | 当前项目路径 |
| `CLAUDE_FILE_PATHS` | 正在修改的文件 |
| `CLAUDE_TOOL_INPUT` | 工具参数（JSON 格式） |

### 安全 Hook 示例
```json
{
  "PreToolUse": [{
    "matcher": "Bash",
    "hooks": [{"type": "command", "command": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -qE 'rm -rf|git push.*--force|:(){ :|:& };:'; then echo 'Dangerous command blocked!' && exit 2; fi"}]
  }]
}
```

## MCP 集成

为数据库、API 和服务添加外部工具服务器：

```
# GitHub 集成
terminal(command="claude mcp add -s user github -- npx @modelcontextprotocol/server-github", timeout=30)

# PostgreSQL 查询
terminal(command="claude mcp add -s local postgres -- npx @anthropic-ai/server-postgres --connection-string postgresql://localhost/mydb", timeout=30)

# Puppeteer 用于 Web 测试
terminal(command="claude mcp add puppeteer -- npx @anthropic-ai/server-puppeteer", timeout=30)
```

### MCP 作用域
| 标志 | 作用域 | 存储位置 |
|------|-------|---------|
| `-s user` | 全局（所有项目） | `~/.claude.json` |
| `-s local` | 此项目（个人） | `.claude/settings.local.json`（已 gitignore） |
| `-s project` | 此项目（团队共享） | `.claude/settings.json`（git 跟踪） |

### Print/CI 模式中的 MCP
```
terminal(command="claude --bare -p 'Query database' --mcp-config mcp-servers.json --strict-mcp-config", timeout=60)
```
`--strict-mcp-config` 忽略除 `--mcp-config` 以外的所有 MCP 服务器。

在对话中引用 MCP 资源：`@github:issue://123`

### MCP 限制与调优
- **工具描述：** 每个服务器的工具描述和服务器指令上限为 2KB
- **结果大小：** 默认有上限；使用 `maxResultSizeChars` 注解允许最多 **500K** 字符的大型输出
- **输出 token：** `export MAX_MCP_OUTPUT_TOKENS=50000` — 限制 MCP 服务器的输出以防止上下文泛滥
- **传输方式：** `stdio`（本地进程）、`http`（远程）、`sse`（服务器发送事件）

## 监控交互会话

### 读取 TUI 状态
```
# 定期捕获以检查 Claude 是否仍在工作或等待输入
terminal(command="tmux capture-pane -t dev -p -S -10")
```

注意以下指示符：
- 底部的 `❯` = 等待您的输入（Claude 已完成或正在提问）
- `●` 行 = Claude 正在主动使用工具（读取、写入、运行命令）
- `⏵⏵ bypass permissions on` = 状态栏显示权限模式
- `◐ medium · /effort` = 状态栏中的当前 effort 级别
- `ctrl+o to expand` = 工具输出被截断（可在交互模式中展开）

### 上下文窗口健康状态
在交互模式中使用 `/context` 查看上下文使用情况的彩色网格。关键阈值：
- **&lt; 70%** — 正常运行，完整精度
- **70-85%** — 精度开始下降，考虑使用 `/compact`
- **> 85%** — 幻觉风险显著上升，使用 `/compact` 或 `/clear`

## 环境变量

| 变量 | 效果 |
|----------|--------|
| `ANTHROPIC_API_KEY` | 用于认证的 API key（OAuth 的替代方案） |
| `CLAUDE_CODE_EFFORT_LEVEL` | 默认 effort：`low`、`medium`、`high`、`max` 或 `auto` |
| `MAX_THINKING_TOKENS` | 限制思考 token 数量（设为 `0` 完全禁用思考） |
| `MAX_MCP_OUTPUT_TOKENS` | 限制 MCP 服务器的输出（默认值不固定；例如设为 `50000`） |
| `CLAUDE_CODE_NO_FLICKER=1` | 启用备用屏幕渲染以消除终端闪烁 |
| `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB` | 从子进程中清除凭据以提高安全性 |

## 成本与性能建议

1. **在 print 模式中使用 `--max-turns`** 以防止失控循环。大多数任务从 5-10 开始。
2. **使用 `--max-budget-usd`** 设置成本上限。注意：系统 prompt 缓存创建的最低成本约为 $0.05。
3. **简单任务使用 `--effort low`**（更快、更便宜）。复杂推理使用 `high` 或 `max`。
4. **CI/脚本使用 `--bare`** 以跳过插件/hook 发现开销。
5. **使用 `--allowedTools`** 限制为任务实际需要的工具（例如仅审查时使用 `Read`）。
6. **在交互会话中使用 `/compact`** 当上下文变大时。
7. **使用管道输入** 而非让 Claude 读取文件，当您只需要分析已知内容时。
8. **简单任务使用 `--model haiku`**（更便宜），复杂多步骤工作使用 `--model opus`。
9. **在 print 模式中使用 `--fallback-model haiku`** 以优雅处理模型过载。
10. **为不同任务开启新会话** — 会话持续 5 小时；新鲜上下文更高效。
11. **在 CI 中使用 `--no-session-persistence`** 以避免在磁盘上积累已保存的会话。

## 陷阱与注意事项

1. **交互模式需要 tmux** — Claude Code 是完整的 TUI 应用。在 Hermes 终端中单独使用 `pty=true` 可以工作，但 tmux 提供了 `capture-pane` 用于监控和 `send-keys` 用于输入，这对编排至关重要。
2. **`--dangerously-skip-permissions` 对话框默认为"No, exit"** — 必须按 Down 再按 Enter 才能接受。Print 模式（`-p`）完全跳过此步骤。
3. **`--max-budget-usd` 最低约为 $0.05** — 仅系统 prompt 缓存创建就需要这么多。设置更低会立即报错。
4. **`--max-turns` 仅限 print 模式** — 在交互会话中被忽略。
5. **Claude 可能使用 `python` 而非 `python3`** — 在没有 `python` 符号链接的系统上，Claude 的 bash 命令首次会失败，但它会自我纠正。
6. **会话恢复需要相同目录** — `--continue` 查找当前工作目录中最近的会话。
7. **`--json-schema` 需要足够的 `--max-turns`** — Claude 必须先读取文件才能生成结构化输出，这需要多轮次。
8. **信任对话框每个目录只出现一次** — 仅首次出现，之后缓存。
9. **后台 tmux 会话会持续存在** — 完成后始终使用 `tmux kill-session -t <name>` 清理。
10. **斜杠命令（如 `/commit`）仅在交互模式下有效** — 在 `-p` 模式中，用自然语言描述任务。
11. **`--bare` 跳过 OAuth** — 需要 `ANTHROPIC_API_KEY` 环境变量或设置中的 `apiKeyHelper`。
12. **上下文退化是真实存在的** — 上下文窗口使用率超过 70% 时，AI 输出质量会明显下降。使用 `/context` 监控并主动使用 `/compact`。

## Hermes Agent 规则

1. **单一任务优先使用 print 模式（`-p`）** — 更简洁，无需处理对话框，输出结构化
2. **多轮交互工作使用 tmux** — 编排 TUI 的唯一可靠方式
3. **始终设置 `workdir`** — 让 Claude 专注于正确的项目目录
4. **在 print 模式中设置 `--max-turns`** — 防止无限循环和失控成本
5. **监控 tmux 会话** — 使用 `tmux capture-pane -t <session> -p -S -50` 检查进度
6. **注意 `❯` 提示符** — 表示 Claude 正在等待输入（已完成或正在提问）
7. **清理 tmux 会话** — 完成后关闭它们以避免资源泄漏
8. **向用户报告结果** — 完成后总结 Claude 做了什么以及发生了什么变化
9. **不要终止慢速会话** — Claude 可能正在进行多步骤工作；检查进度而非直接终止
10. **使用 `--allowedTools`** — 将能力限制为任务实际需要的工具