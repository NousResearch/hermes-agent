---
title: "Antigravity CLI — 操作 Antigravity CLI (agy)：插件、认证、沙箱"
sidebar_label: "Antigravity CLI"
description: "操作 Antigravity CLI (agy)：插件、认证、沙箱"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Antigravity CLI

操作 Antigravity CLI (agy)：插件、认证、沙箱。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/autonomous-ai-agents/antigravity-cli` 安装 |
| 路径 | `optional-skills/autonomous-ai-agents/antigravity-cli` |
| 版本 | `0.1.0` |
| 作者 | Tony Simons (asimons81)、Hermes Agent |
| 许可证 | MIT |
| 平台 | linux、macos、windows |
| 标签 | `Coding-Agent`、`Antigravity`、`CLI`、`Auth`、`Plugins`、`Sandbox` |
| 相关技能 | [`grok`](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-grok)、[`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex)、[`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code)、[`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# Antigravity CLI (`agy`)

Antigravity CLI 的操作指南，调用方式为 `agy`。通过 Hermes `terminal` 工具运行所有 `agy` 命令；使用 `read_file` 检查其配置和日志。此技能是参考+操作流程——它不包装网络 API，因此不需要从 Hermes 本身进行认证。

## 何时使用

- 安装、更新或冒烟测试 `agy` 二进制文件
- 驱动非交互式 `agy --print` / `agy -p` 一次性任务
- 调试 Antigravity 认证、沙箱、权限或插件状态
- 读取 Antigravity 设置、快捷键、对话或日志

## 心智模型

Antigravity 有两层——区分它们，否则指引会出错：

1. **Shell 包装命令** — `agy help`、`agy install`、`agy plugin`、`agy update`、`agy changelog`。通过 `terminal` 工具运行。
2. **交互式会话内斜杠命令** — `/config`、`/permissions`、`/skills`、`/agents` 等。这些仅存在于运行中的 `agy` TUI 会话内，不在 shell 包装器上。

`agy help` 显示 shell 包装器界面，而非会话内斜杠命令。

## 前提条件

- PATH 上有 `agy` 二进制文件。通过 `terminal` 工具验证：`command -v agy && agy --version`。
- 此技能不需要任何环境变量或 API 密钥——Antigravity 通过 OS 密钥环/浏览器登录管理自己的认证（见下方认证部分）。

## 如何运行

通过 `terminal` 工具调用每个 `agy` 命令。示例：

```
terminal(command="agy --version")
terminal(command="agy help")
terminal(command="agy plugin list")
terminal(command="agy --print 'Summarize the repo in 3 bullets'", workdir="/path/to/project")
```

对于交互式多轮 TUI 会话，使用 `pty=true`（加 tmux 进行捕获/监控）启动 `agy`，与 `codex` / `claude-code` 技能使用的模式相同。对于一次性冒烟测试和脚本提示，优先使用 `agy --print`（非交互式）。

要检查 Antigravity 自身的文件，使用 `read_file` 读取下方核心路径中的路径——不要通过终端 `cat` 它们。

## 核心路径

- 二进制/入口点：`agy`
- 应用数据目录：`~/.gemini/antigravity-cli/`
- 设置文件：`~/.gemini/antigravity-cli/settings.json`
- 快捷键文件：`~/.gemini/antigravity-cli/keybindings.json`
- 日志：`~/.gemini/antigravity-cli/log/cli-*.log`
- 对话：`~/.gemini/antigravity-cli/conversations/`
- Brain 制品：`~/.gemini/antigravity-cli/brain/`
- 历史：`~/.gemini/antigravity-cli/history.jsonl`
- 插件暂存：`~/.gemini/antigravity-cli/plugins/<plugin_name>/`

## 快速参考

### 包装器命令
- `agy changelog`
- `agy help`
- `agy install`
- `agy plugin` / `agy plugins`
- `agy update`

### 有用标志
- `--add-dir`
- `--continue` / `-c`
- `--conversation`
- `--dangerously-skip-permissions`
- `--print` / `-p`
- `--print-timeout`
- `--prompt`
- `--prompt-interactive` / `-i`
- `--sandbox`
- `--log-file`
- `--version`

### 插件子命令（`agy plugin --help`）
- `list`、`import [source]`、`install <target>`、`uninstall <name>`、`enable <name>`、`disable <name>`、`validate [path]`、`link <mp> <target>`、`help`

### 安装标志（`agy install --help`）
- `--dir`、`--skip-aliases`、`--skip-path`

### 会话内斜杠命令
- **对话控制：** `/resume`（`/switch`）、`/rewind`（`/undo`）、`/rename <name>`、`/clear`、`/fork`、`/reset`、`/new`
- **设置和工具：** `/config`、`/settings`、`/permissions`、`/model`、`/keybindings`、`/statusline`、`/tasks`、`/skills`、`/mcp`、`/open <path>`、`/usage`、`/logout`、`/agents`
- **提示辅助：** `@` 路径自动完成、`esc esc` 清除提示（不在流式传输时）、`!` 直接运行终端命令、`?` 打开帮助

## 设置和权限

### 常用设置键（`settings.json`）
- `allowNonWorkspaceAccess`
- `colorScheme`
- `permissions.allow`
- `trustedWorkspaces`

### 权限模式
`request-review`、`always-proceed`、`strict`、`proceed-in-sandbox`。

### 沙箱行为
- `enableTerminalSandbox` 是 `settings.json` 中的布尔值；默认 `false`。
- 启动时覆盖（`--sandbox`、`--dangerously-skip-permissions`）可以为当前会话覆盖持久设置。

## 认证行为

- CLI 首先尝试 OS 安全密钥环。
- 没有已保存会话时，回退到基于浏览器的 Google 登录。
- 本地打开默认浏览器；通过 SSH 打印授权 URL 并期望认证码粘贴回来。
- `/logout` 移除已保存的凭据。

## 插件

- 插件暂存在 `~/.gemini/antigravity-cli/plugins/<plugin_name>/` 下。
- 它们可以包含技能、代理、规则、MCP 服务器和钩子。
- `agy plugin list` 返回无导入插件是有效的空状态。

## 陷阱

- `agy help` 显示包装器命令，而非交互式斜杠命令。
- `agy --version` 是安全的非交互式版本检查；`agy version` 是交互式的，在没有真实 TTY 时可能失败。
- 查找故障的首选位置：`~/.gemini/antigravity-cli/log/cli-*.log`（使用 `read_file` 读取）。
- 不要混淆持久 JSON 设置与启动时覆盖。
- `~/.gemini/antigravity-cli/bin/agentapi` 是 `agy agentapi` 的薄包装器。
- 在 WSL 上，token 存储基于文件，因此认证问题通常是本地文件/会话状态问题，而非仅浏览器问题。
- 工作空间身份可能依赖启动目录和 `.antigravitycli` 项目标记。

## 验证

确认安装是真实且可用的，全部通过 `terminal` 工具（使用 `read_file` 读取文件）：

1. `terminal(command="command -v agy")`
2. `terminal(command="agy --version")`
3. `terminal(command="agy help")`
4. `terminal(command="agy plugin list")`
5. 读取 `~/.gemini/antigravity-cli/settings.json`
6. 读取最新的 `~/.gemini/antigravity-cli/log/cli-*.log`
7. 如需，读取 `~/.gemini/antigravity-cli/keybindings.json`

## 支持文件

- `references/cli-docs.md` — 入门、使用和功能文档的精简笔记。
