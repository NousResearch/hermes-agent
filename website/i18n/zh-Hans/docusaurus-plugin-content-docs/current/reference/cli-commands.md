---
sidebar_position: 1
title: "CLI 命令参考"
description: "Hermes 终端命令与命令族的权威参考"
---

# CLI 命令参考

本页覆盖你在 shell 中运行的**终端命令**。

对于会话内的斜杠命令，请参阅 [Slash Commands Reference](./slash-commands.md)。

## 全局入口

```bash
hermes [global-options] <command> [subcommand/options]
```

### 全局选项（摘要）

| 选项 | 含义 |
|---|---|
| `--version`, `-V` | 显示版本并退出 |
| `--profile <name>`, `-p <name>` | 指定本次调用使用的配置文件 |
| `--resume <session>`, `-r <session>` | 按 ID 或标题恢复会话 |
| `--continue [name]`, `-c [name]` | 恢复最近的会话或匹配标题的最近会话 |
| `--worktree`, `-w` | 为并行代理工作流在独立 git worktree 中启动 |
| `--yolo` | 跳过危险命令的审批提示 |
| `--pass-session-id` | 将会话 ID 包含到系统提示中 |
| `--ignore-user-config` | 忽略 `~/.hermes/config.yaml`，使用内置默认值 |
| `--ignore-rules` | 跳过自动注入 `AGENTS.md`、`SOUL.md`、`.cursorrules`、memory 与预加载技能 |
| `--tui` | 启动 TUI（替代经典 CLI），等同于 `HERMES_TUI=1` |
| `--dev` | 与 `--tui` 一起使用：直接运行 TypeScript 源（TUI 开发者用） |

## 顶级命令（摘要）

常用命令示例：`hermes chat`, `hermes model`, `hermes gateway`, `hermes setup`, `hermes auth`, `hermes status`, `hermes cron`, `hermes kanban`, `hermes skills`, `hermes memory`, `hermes config`, `hermes logs`, `hermes backup`, `hermes profile` 等。

（原文包含完整表格，此处保留摘要以便快速查阅；完整条目在英文源文档）

## `hermes chat`

```bash
hermes chat [options]
```

常用选项：

| 选项 | 含义 |
|---|---|
| `-q`, `--query "..."` | 一次性非交互查询 |
| `-m`, `--model <model>` | 覆盖本次运行使用的模型 |
| `-t`, `--toolsets <csv>` | 启用逗号分隔的工具集列表 |
| `--provider <provider>` | 强制使用指定提供者 |
| `-s`, `--skills <name>` | 预加载一个或多个技能 |
| `-v`, `--verbose` | 详细输出 |
| `-Q`, `--quiet` | 程序化模式：抑制横幅/指示/工具预览 |
| `--image <path>` | 附加本地图片到单次查询 |
| `--worktree` | 为本次运行创建隔离 git worktree |
| `--checkpoints` | 在破坏性文件修改前启用文件系统检查点 |
| `--yolo` | 跳过审批提示 |
| `--ignore-user-config` | 忽略用户配置，使用内置默认（CI/调试用） |

示例：

```bash
hermes chat -q "Summarize the latest PRs"
hermes chat --provider openrouter --model anthropic/claude-sonnet-4.6
hermes chat --toolsets web,terminal,skills
```

### `hermes -z <prompt>` — 脚本化一次性调用

用于脚本或 CI：纯输入-输出，只有代理最终回复作为 stdout，无横幅或其他杂项。示例：

```bash
hermes -z "What's the capital of France?"
# → Paris.
```

## `hermes model`

交互式的提供者与模型选择器，用于添加新提供者、运行 OAuth、输入 API 密钥等（需在终端外运行）。

```bash
hermes model
```

注意：会话内的 `/model` 只能在已配置的提供者/模型间切换，不能添加新提供者或运行 OAuth。

## 其他命令

文档中对 `hermes gateway`、`hermes setup`、`hermes slack`、`hermes auth`、`hermes status`、`hermes cron`、`hermes kanban` 等命令有详尽子命令与示例。

（若需，我可以继续翻译该页剩余详细子命令表格与示例。）

---
---
sidebar_position: 1
title: "CLI 命令参考"
description: "Hermes 终端命令和命令族的权威参考"
---

# CLI 命令参考

本页介绍你从 shell 里运行的 **终端命令**。

关于聊天内的斜杠命令，请参见 [斜杠命令参考](./slash-commands.md)。

## 全局入口

```bash
hermes [global-options] <command> [subcommand/options]
```

### 全局选项

| 选项 | 说明 |
|--------|-------------|
| `--version`, `-V` | 显示版本并退出。 |
| `--profile <name>`, `-p <name>` | 选择本次调用要使用的 Hermes profile。会覆盖 `hermes profile use` 设置的粘性默认值。 |
| `--resume <session>`, `-r <session>` | 按 ID 或标题恢复之前的会话。 |
| `--continue [name]`, `-c [name]` | 恢复最近一次会话，或者最近一个匹配标题的会话。 |
| `--worktree`, `-w` | 在隔离的 git worktree 中启动，适合并行智能体工作流。 |
| `--yolo` | 跳过危险命令审批提示。 |
| `--pass-session-id` | 把 session ID 加到智能体的系统提示词中。 |
| `--ignore-user-config` | 忽略 `~/.hermes/config.yaml`，回退到内置默认值。`.env` 中的凭据仍会加载。 |
| `--ignore-rules` | 跳过 `AGENTS.md`、`SOUL.md`、`.cursorrules`、记忆和预加载技能的自动注入。 |
| `--tui` | 启动 [TUI](../user-guide/tui.md)，而不是经典 CLI。等同于 `HERMES_TUI=1`。 |
| `--dev` | 与 `--tui` 配合时：直接通过 `tsx` 运行 TypeScript 源码，而不是预构建 bundle（供 TUI 贡献者使用）。 |

## 顶层命令

| 命令 | 用途 |
|---------|---------|
| `hermes chat` | 与智能体进行交互式或一次性聊天。 |
| `hermes model` | 交互式选择默认提供商和模型。 |
| `hermes fallback` | 管理主模型出错时使用的回退提供商。 |
| `hermes gateway` | 运行或管理消息网关服务。 |
| `hermes setup` | 为全部或部分配置提供交互式设置向导。 |
| `hermes whatsapp` | 配置并配对 WhatsApp bridge。 |
| `hermes slack` | Slack 辅助工具（目前：生成包含所有命令的原生 slash app manifest）。 |
| `hermes auth` | 管理凭据 - 添加、列出、删除、重置、设置策略。处理 Codex/Nous/Anthropic 的 OAuth 流程。 |
| `hermes login` / `logout` | **已弃用** - 请改用 `hermes auth`。 |
| `hermes status` | 显示智能体、认证和平台状态。 |
| `hermes cron` | 查看并触发 cron 调度器。 |
| `hermes kanban` | 多 profile 协作看板（任务、链接、调度器）。 |
| `hermes webhook` | 管理用于事件驱动激活的动态 webhook 订阅。 |
| `hermes hooks` | 查看、批准或移除在 `config.yaml` 中声明的 shell-script hooks。 |
| `hermes doctor` | 诊断配置和依赖问题。 |
| `hermes dump` | 生成适合复制粘贴的支持 / 调试摘要。 |
| `hermes debug` | 调试工具 - 上传日志和系统信息以便支持排查。 |
| `hermes backup` | 将 Hermes home 目录备份为 zip 文件。 |
| `hermes checkpoints` | 检查 / 清理 / 清空 `~/.hermes/checkpoints/`（`/rollback` 使用的影子存储）。不带参数运行可查看状态概览。 |
| `hermes import` | 从 zip 文件恢复 Hermes 备份。 |
| `hermes logs` | 查看、尾随和过滤 agent / gateway / error 日志文件。 |
| `hermes config` | 显示、编辑、迁移和查询配置文件。 |
| `hermes pairing` | 审批或撤销消息配对码。 |
| `hermes skills` | 浏览、安装、发布、审计和配置技能。 |
| `hermes curator` | 后台技能维护 - 状态、运行、暂停、固定。参见 [Curator](../user-guide/features/curator.md)。 |
| `hermes memory` | 配置外部记忆提供商。插件专属子命令（例如 `hermes honcho`）会在对应提供商启用时自动注册。 |
| `hermes acp` | 将 Hermes 作为 ACP 服务器运行，用于编辑器集成。 |
| `hermes mcp` | 管理 MCP 服务器配置，并将 Hermes 作为 MCP 服务器运行。 |
| `hermes plugins` | 管理 Hermes Agent 插件（安装、启用、禁用、移除）。 |
| `hermes tools` | 按平台配置启用的工具。 |
| `hermes computer-use` | 安装或检查 cua-driver 后端（macOS Computer Use）。 |
| `hermes sessions` | 浏览、导出、清理、重命名和删除会话。 |
| `hermes insights` | 显示 token / 成本 / 活动分析。 |
| `hermes claw` | OpenClaw 迁移辅助工具。 |
| `hermes dashboard` | 启动用于管理配置、API key 和会话的 web dashboard。 |
| `hermes profile` | 管理 profiles - 多个彼此隔离的 Hermes 实例。 |
| `hermes completion` | 打印 shell 补全脚本（bash/zsh/fish）。 |
| `hermes version` | 显示版本信息。 |
| `hermes update` | 拉取最新代码并重新安装依赖。`--check` 只打印提交差异，不拉取；`--backup` 会在拉取前对 `HERMES_HOME` 做快照。 |
| `hermes uninstall` | 从系统中移除 Hermes。 |

### `hermes insights` {#hermes-insights}

`hermes insights` 用于查看会话 token 使用、成本估算和活动分解，适合在性能和费用排查时快速定位高消耗环节。

## `hermes chat`

```bash
hermes chat [options]
```

常用选项：

| 选项 | 说明 |
|--------|-------------|
| `-q`, `--query "..."` | 一次性、非交互式提示。 |
| `-m`, `--model <model>` | 覆盖本次运行所用的模型。 |
| `-t`, `--toolsets <csv>` | 启用一个逗号分隔的工具集列表。 |
| `--provider <provider>` | 强制使用某个提供商：`auto`、`openrouter`、`nous`、`openai-codex`、`copilot-acp`、`copilot`、`anthropic`、`gemini`、`google-gemini-cli`、`huggingface`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`、`kilocode`、`xiaomi`、`arcee`、`gmi`、`alibaba`、`alibaba-coding-plan`（别名 `alibaba_coding`）、`deepseek`、`nvidia`、`ollama-cloud`、`xai`（别名 `grok`）、`qwen-oauth`、`bedrock`、`opencode-zen`、`opencode-go`、`ai-gateway`、`azure-foundry`、`lmstudio`、`stepfun`、`tencent-tokenhub`（别名 `tencent`、`tokenhub`）。 |
| `-s`, `--skills <name>` | 为本次会话预加载一个或多个技能（可重复或用逗号分隔）。 |
| `-v`, `--verbose` | 详细输出。 |
| `-Q`, `--quiet` | 程序化模式：隐藏横幅、spinner 和工具预览。 |
| `--image <path>` | 为一次性查询附加本地图片。 |
| `--resume <session>` / `--continue [name]` | 直接从 `chat` 恢复一个会话。 |
| `--worktree` | 为本次运行创建一个隔离的 git worktree。 |
| `--checkpoints` | 在破坏性文件更改前启用文件系统 checkpoints。 |
| `--yolo` | 跳过审批提示。 |
| `--pass-session-id` | 将 session ID 传入系统提示词。 |
| `--ignore-user-config` | 忽略 `~/.hermes/config.yaml`，使用内置默认值。`.env` 中的凭据仍会加载。适合隔离 CI、可复现 bug 报告和第三方集成。 |
| `--ignore-rules` | 跳过 `AGENTS.md`、`SOUL.md`、`.cursorrules`、持久化记忆和预加载技能的自动注入。与 `--ignore-user-config` 结合可得到完全隔离的运行。 |
| `--source <tag>` | 会话来源标签，用于过滤（默认：`cli`）。第三方集成若不应出现在用户会话列表中，可使用 `tool`。 |
| `--max-turns <N>` | 每次对话轮次的最大工具调用迭代数（默认 90，或 `agent.max_turns` 中的配置）。 |

示例：

```bash
hermes
hermes chat -q "总结最新的 PR"
hermes chat --provider openrouter --model anthropic/claude-sonnet-4.6
hermes chat --toolsets web,terminal,skills
hermes chat --quiet -q "只返回 JSON"
hermes chat --worktree -q "审查这个仓库并打开一个 PR"
hermes chat --ignore-user-config --ignore-rules -q "在不使用我个人配置的情况下复现"
```

### `hermes -z <prompt>` - 脚本化一次性调用

对于程序化调用者（shell 脚本、CI、cron、管道输入提示词的父进程），`hermes -z` 是最纯粹的一次性入口：**单个提示输入，最终回复纯文本输出，stdout 和 stderr 上没有多余内容。**没有横幅，没有 spinner，没有工具预览，也没有 `Session:` 行 - 只有智能体最终回复的纯文本。

```bash
hermes -z "法国的首都是哪里？"
# → Paris.

# 父脚本可以干净地捕获响应：
answer=$(hermes -z "总结这个内容" < /path/to/file.txt)
```

每次运行时的覆盖项（不修改 `~/.hermes/config.yaml`）：

| 标志 | 等效环境变量 | 用途 |
|---|---|---|
| `-m` / `--model <model>` | `HERMES_INFERENCE_MODEL` | 覆盖本次运行模型 |
| `--provider <provider>` | `HERMES_INFERENCE_PROVIDER` | 覆盖本次运行提供商 |

```bash
hermes -z "…" --provider openrouter --model openai/gpt-5.5
# 或者：
HERMES_INFERENCE_MODEL=anthropic/claude-sonnet-4.6 hermes -z "…"
```

同一个智能体、同一套工具、同一套技能 - 只是去掉了所有交互 / 外观层。如果你还想把工具输出也放进 transcript，改用 `hermes chat -q`；`-z` 明确就是“我只要最终答案”。

## `hermes model`

交互式提供商 + 模型选择器。**这是用来添加新提供商、设置 API key 和运行 OAuth 流程的命令。**请在终端里运行它 - 不要在正在进行的 Hermes 聊天会话里运行。

```bash
hermes model
```

适合在这些场景使用：
- **添加新提供商**（OpenRouter、Anthropic、Copilot、DeepSeek、自定义等）
- 登录基于 OAuth 的提供商（Anthropic、Copilot、Codex、Nous Portal）
- 输入或更新 API key
- 在提供商专属模型列表中选择
- 配置自托管 / 自定义端点
- 将新默认值保存到 config 中

:::warning hermes model 与 /model 的区别
**`hermes model`**（在终端里、会话外运行）是**完整的提供商设置向导**。它可以添加新提供商、运行 OAuth 流程、提示输入 API key，并配置端点。

**`/model`**（在正在进行的 Hermes 聊天会话里输入）只能**在已设置好的提供商和模型之间切换**。它不能添加新提供商、运行 OAuth，也不会提示输入 API key。

**如果你需要添加新提供商：**先退出当前 Hermes 会话（`Ctrl+C` 或 `/quit`），然后在终端提示符下运行 `hermes model`。
:::

### `/model` 斜杠命令（会话中切换）

无需离开会话即可在已配置模型之间切换：

```
/model                              # 显示当前模型和可用选项
/model claude-sonnet-4              # 切换模型（自动检测提供商）
/model zai:glm-5                    # 切换提供商和模型
/model custom:qwen-2.5              # 使用自定义端点上的模型
/model custom                       # 从自定义端点自动检测模型
/model custom:local:qwen-2.5        # 使用一个命名的自定义提供商
/model openrouter:anthropic/claude-sonnet-4  # 切回云端
```

默认情况下，`/model` 的改动**只对当前会话生效**。加上 `--global` 可以把变化持久化到 `config.yaml`：

```
/model claude-sonnet-4 --global     # 切换并保存为新的默认值
```

:::info 如果我只看到 OpenRouter 模型怎么办？
如果你只配置了 OpenRouter，那么 `/model` 只会显示 OpenRouter 模型。要添加另一个提供商（Anthropic、DeepSeek、Copilot 等），请先退出会话，再在终端里运行 `hermes model`。
:::

提供商和 base URL 的变更会自动持久化到 `config.yaml`。当你从自定义端点切走时，旧的 base URL 会被清除，避免泄漏到其他提供商。

## `hermes gateway`

```bash
hermes gateway <subcommand>
```

子命令：

| 子命令 | 说明 |
|------------|-------------|
| `run` | 在前台运行网关。推荐用于 WSL、Docker 和 Termux。 |
| `start` | 启动已安装的 systemd / launchd 后台服务。 |
| `stop` | 停止服务（或前台进程）。 |
| `restart` | 重启服务。 |
| `status` | 显示服务状态。 |
| `install` | 作为 systemd（Linux）或 launchd（macOS）后台服务安装。 |
| `uninstall` | 移除已安装的服务。 |
| `setup` | 交互式消息平台设置。 |

选项：

| 选项 | 说明 |
|--------|-------------|
| `--all` | 用于 `start` / `restart` / `stop`：作用于**所有 profile 的**网关，而不只当前活动的 `HERMES_HOME`。适合你并排运行多个 profile，并在 `hermes update` 后一起重启它们。 |

:::tip WSL 用户
使用 `hermes gateway run`，不要用 `hermes gateway start` - WSL 的 systemd 支持不稳定。为了持续运行，可以把它包在 tmux 里：`tmux new -s hermes 'hermes gateway run'`。详情见 [WSL FAQ](/reference/faq#wsl-gateway-keeps-disconnecting-or-hermes-gateway-start-fails)。
:::

## `hermes setup`

```bash
hermes setup [model|tts|terminal|gateway|tools|agent] [--non-interactive] [--reset] [--quick] [--reconfigure]
```

**首次运行：** 启动首次配置向导。

**老用户（已配置过）：** 直接进入完整重配置向导 - 每个提示都会把当前值作为默认值，按 Enter 保留，或输入新值。不再是菜单式流程。

如果只想进入某个部分，而不是完整向导：

| 部分 | 说明 |
|---------|-------------|
| `model` | 提供商和模型设置。 |
| `terminal` | 终端后端和沙箱设置。 |
| `gateway` | 消息平台设置。 |
| `tools` | 按平台启用 / 禁用工具。 |
| `agent` | 智能体行为设置。 |

选项：

| 选项 | 说明 |
|--------|-------------|
| `--quick` | 针对返回用户的运行：只提示缺失或未设置的项目，跳过已配置项。 |
| `--non-interactive` | 使用默认值 / 环境变量，不进行交互提示。 |
| `--reset` | 在设置前把配置恢复为默认值。 |
| `--reconfigure` | 向后兼容别名 - 在现有安装上直接运行 `hermes setup` 现在默认就相当于这个行为。 |

## `hermes whatsapp`

```bash
hermes whatsapp
```