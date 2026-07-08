---
sidebar_position: 1
title: "CLI 命令参考"
description: "Hermes 终端命令及命令族的权威参考"
---

# CLI 命令参考

本页介绍从 shell 运行的**终端命令**。

关于聊天内斜杠命令，请参阅 [斜杠命令参考](./slash-commands.md)。

## 全局入口

```bash
hermes [global-options] <command> [subcommand/options]
```

### 全局选项

| 选项 | 说明 |
|--------|-------------|
| `--version`, `-V` | 显示版本并退出。 |
| `--profile <name>`, `-p <name>` | 选择本次调用使用的 Hermes profile（配置文件）。覆盖 `hermes profile use` 设置的粘性默认值。 |
| `--resume <session>`, `-r <session>` | 通过 ID 或标题恢复之前的会话。 |
| `--continue [name]`, `-c [name]` | 恢复最近的会话，或恢复最近一个匹配标题的会话。 |
| `--worktree`, `-w` | 在隔离的 git worktree 中启动，用于并行 agent 工作流。 |
| `--yolo` | 跳过危险命令的审批提示。 |
| `--pass-session-id` | 在 agent 的 system prompt（系统提示词）中包含会话 ID。 |
| `--ignore-user-config` | 忽略 `~/.hermes/config.yaml`，回退到内置默认值。`.env` 中的凭据仍会加载。 |
| `--ignore-rules` | 跳过 `AGENTS.md`、`SOUL.md`、`.cursorrules`、memory（记忆）和预加载 skill 的自动注入。 |
| `--tui` | 启动 [TUI](../user-guide/tui.md) 而非经典 CLI。等同于 `HERMES_TUI=1`。 |
| `--dev` | 与 `--tui` 配合使用：通过 `tsx` 直接运行 TypeScript 源码而非预构建包（供 TUI 贡献者使用）。 |

## 顶级命令

| 命令 | 用途 |
|---------|---------|
| `hermes chat` | 与 agent 进行交互式或单次聊天。 |
| `hermes model` | 交互式选择默认 provider 和模型。 |
| `hermes fallback` | 管理主模型出错时依次尝试的 fallback provider。 |
| `hermes gateway` | 运行或管理消息 gateway 服务。 |
| `hermes proxy` | 本地 OpenAI 兼容代理，附加 OAuth provider 凭据。参见 [订阅代理](../user-guide/features/subscription-proxy.md)。 |
| `hermes lsp` | 管理 Language Server Protocol 集成（为 write_file/patch 提供语义诊断）。 |
| `hermes setup` | 全部或部分配置的交互式设置向导。 |
| `hermes whatsapp` | 配置并配对 WhatsApp 桥接。 |
| `hermes slack` | Slack 辅助工具（当前功能：生成将每条命令注册为原生斜杠命令的 app manifest）。 |
| `hermes auth` | 管理凭据——添加、列出、删除、重置、设置策略。处理 Codex/Nous/Anthropic 的 OAuth 流程。 |
| `hermes login` / `logout` | **已弃用** — 请改用 `hermes auth`。 |
| `hermes status` | 显示 agent、auth 和平台状态。 |
| `hermes cron` | 检查并触发 cron 调度器。 |
| `hermes kanban` | 多 profile 协作看板（任务、链接、调度器）。 |
| `hermes webhook` | 管理用于事件驱动激活的动态 webhook 订阅。 |
| `hermes hooks` | 检查、审批或删除 `config.yaml` 中声明的 shell 脚本 hook。 |
| `hermes doctor` | 诊断配置和依赖问题。 |
| `hermes security audit` | 对 venv、plugin 依赖和固定 MCP 服务器进行按需供应链审计（OSV.dev）。 |
| `hermes dump` | 可直接复制粘贴的设置摘要，用于支持/调试。 |
| `hermes debug` | 调试工具——上传日志和系统信息以获取支持。 |
| `hermes backup` | 将 Hermes 主目录备份为 zip 文件。 |
| `hermes checkpoints` | 检查/修剪/清除 `~/.hermes/checkpoints/`（`/rollback` 使用的影子存储）。不带参数运行可查看状态概览。 |
| `hermes import` | 从 zip 文件恢复 Hermes 备份。 |
| `hermes logs` | 查看、跟踪和过滤 agent/gateway/错误日志文件。 |
| `hermes config` | 显示、编辑、迁移和查询配置文件。 |
| `hermes pairing` | 审批或撤销消息配对码。 |
| `hermes skills` | 浏览、安装、发布、审计和配置 skill。 |
| `hermes bundles` | 将多个 skill 归组到单个 `/<name>` 斜杠命令下。参见 [Skill Bundles](../user-guide/features/skills.md#skill-bundles)。 |
| `hermes curator` | 后台 skill 维护——状态、运行、暂停、固定。参见 [Curator](../user-guide/features/curator.md)。 |
| `hermes memory` | 配置外部 memory provider。当对应 provider 激活时，特定于 plugin 的子命令（如 `hermes honcho`）会自动注册。 |
| `hermes acp` | 将 Hermes 作为 ACP 服务器运行，用于编辑器集成。 |
| `hermes mcp` | 管理 MCP 服务器配置，并将 Hermes 作为 MCP 服务器运行。 |
| `hermes plugins` | 管理 Hermes Agent plugin（安装、启用、禁用、删除）。 |
| `hermes portal` | Nous Portal 状态、订阅链接和 Tool Gateway 路由。参见 [Tool Gateway](../user-guide/features/tool-gateway.md)。 |
| `hermes tools` | 按平台配置已启用的工具。 |
| `hermes computer-use` | 安装或检查 cua-driver 后端（macOS Computer Use）。 |
| `hermes sessions` | 浏览、导出、修剪、重命名和删除会话。 |
| `hermes insights` | 显示 token/费用/活动分析。 |
| `hermes claw` | OpenClaw 迁移辅助工具。 |
| `hermes dashboard` | 启动用于管理配置、API 密钥和会话的 Web 控制台。 |
| `hermes profile` | 管理 profile——多个隔离的 Hermes 实例。 |
| `hermes completion` | 打印 shell 补全脚本（bash/zsh/fish）。 |
| `hermes version` | 显示版本信息。 |
| `hermes update` | 拉取最新代码并重新安装依赖。`--check` 预览而不安装；`--backup` 在拉取前对 `HERMES_HOME` 进行快照。 |
| `hermes uninstall` | 从系统中删除 Hermes。 |

## `hermes chat`

```bash
hermes chat [options]
```

常用选项：

| 选项 | 说明 |
|--------|-------------|
| `-q`, `--query "..."` | 单次非交互式 prompt。 |
| `-m`, `--model <model>` | 覆盖本次运行的模型。 |
| `-t`, `--toolsets <csv>` | 启用逗号分隔的 toolset 集合。 |
| `--provider <provider>` | 强制指定 provider：`auto`、`openrouter`、`nous`、`openai-codex`、`copilot-acp`、`copilot`、`anthropic`、`gemini`、`huggingface`、`novita`（别名 `novita-ai`、`novitaai`）、`openai-api`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`、`kilocode`、`xiaomi`、`arcee`、`gmi`、`alibaba`、`alibaba-coding-plan`（别名 `alibaba_coding`）、`deepseek`、`nvidia`、`ollama-cloud`、`xai`（别名 `grok`）、`xai-oauth`（别名 `grok-oauth`）、`qwen-oauth`、`bedrock`、`opencode-zen`、`opencode-go`、`azure-foundry`、`lmstudio`、`stepfun`、`tencent-tokenhub`（别名 `tencent`、`tokenhub`）。 |
| `-s`, `--skills <name>` | 为会话预加载一个或多个 skill（可重复或逗号分隔）。 |
| `-v`, `--verbose` | 详细输出。 |
| `-Q`, `--quiet` | 程序化模式：抑制横幅/spinner/工具预览。 |
| `--image <path>` | 为单次查询附加本地图片。 |
| `--resume <session>` / `--continue [name]` | 直接从 `chat` 恢复会话。 |
| `--worktree` | 为本次运行创建隔离的 git worktree。 |
| `--checkpoints` | 在破坏性文件变更前启用文件系统 checkpoint。 |
| `--yolo` | 跳过审批提示。 |
| `--pass-session-id` | 将会话 ID 传入 system prompt。 |
| `--ignore-user-config` | 忽略 `~/.hermes/config.yaml`，使用内置默认值。`.env` 中的凭据仍会加载。适用于隔离的 CI 运行、可复现的 bug 报告和第三方集成。 |
| `--ignore-rules` | 跳过 `AGENTS.md`、`SOUL.md`、`.cursorrules`、持久 memory 和预加载 skill 的自动注入。与 `--ignore-user-config` 组合可实现完全隔离的运行。 |
| `--source <tag>` | 用于过滤的会话来源标签（默认：`cli`）。对于不应出现在用户会话列表中的第三方集成，使用 `tool`。 |
| `--max-turns <N>` | 每个对话轮次的最大工具调用迭代次数（默认：90，或 config 中的 `agent.max_turns`）。 |

示例：

```bash
hermes
hermes chat -q "Summarize the latest PRs"
hermes chat --provider openrouter --model anthropic/claude-sonnet-4.6
hermes chat --toolsets web,terminal,skills
hermes chat --quiet -q "Return only JSON"
hermes chat --worktree -q "Review this repo and open a PR"
hermes chat --ignore-user-config --ignore-rules -q "Repro without my personal setup"
```

### `hermes -z <prompt>` — 脚本化单次调用

对于程序化调用方（shell 脚本、CI、cron、通过管道传入 prompt 的父进程），`hermes -z` 是最纯粹的单次入口：**单个 prompt 输入，最终响应文本输出，stdout 和 stderr 上不输出任何其他内容。** 无横幅、无 spinner、无工具预览、无 `Session:` 行——只有 agent 的最终回复纯文本。

```bash
hermes -z "What's the capital of France?"
# → Paris.

# 父脚本可以干净地捕获响应：
answer=$(hermes -z "summarize this" < /path/to/file.txt)
```

单次运行覆盖（不修改 `~/.hermes/config.yaml`）：

| 标志 | 等效环境变量 | 用途 |
|---|---|---|
| `-m` / `--model <model>` | `HERMES_INFERENCE_MODEL` | 覆盖本次运行的模型 |
| `--provider <provider>` | _(无)_ | 覆盖本次运行的 provider |

```bash
hermes -z "…" --provider openrouter --model openai/gpt-5.5
# 或：
HERMES_INFERENCE_MODEL=anthropic/claude-sonnet-4.6 hermes -z "…"
```

相同的 agent、相同的工具、相同的 skill——只是剥离了所有交互式/装饰性层。如果你还需要在记录中包含工具输出，请改用 `hermes chat -q`；`-z` 专门用于"我只需要最终答案"的场景。

## `hermes model`

交互式 provider + 模型选择器。**这是添加新 provider、设置 API 密钥和运行 OAuth 流程的命令。** 从终端运行——不要在活跃的 Hermes 聊天会话内部运行。

```bash
hermes model
```

在以下情况使用此命令：
- **添加新 provider**（OpenRouter、Anthropic、Copilot、DeepSeek、自定义等）
- 登录基于 OAuth 的 provider（Anthropic、Copilot、Codex、Nous Portal）
- 输入或更新 API 密钥
- 从 provider 特定的模型列表中选择
- 配置自定义/自托管端点
- 将新默认值保存到 config

:::warning hermes model 与 /model——了解区别
**`hermes model`**（从终端运行，在任何 Hermes 会话外部）是**完整的 provider 设置向导**。它可以添加新 provider、运行 OAuth 流程、提示输入 API 密钥并配置端点。

**`/model`**（在活跃的 Hermes 聊天会话中输入）只能**在已设置好的 provider 和模型之间切换**。它无法添加新 provider、运行 OAuth 或提示输入 API 密钥。

**如果需要添加新 provider：** 先退出 Hermes 会话（`Ctrl+C` 或 `/quit`），然后从终端提示符运行 `hermes model`。
:::

### `/model` 斜杠命令（会话中途）

无需离开会话即可在已配置的模型之间切换：

```
/model                              # 显示当前模型和可用选项
/model claude-sonnet-4              # 切换模型（自动检测 provider）
/model zai:glm-5                    # 切换 provider 和模型
/model custom:qwen-2.5              # 在自定义端点上使用模型
/model custom                       # 从自定义端点自动检测模型
/model custom:local:qwen-2.5        # 使用命名的自定义 provider
/model openrouter:anthropic/claude-sonnet-4  # 切换回云端
```

默认情况下，`/model` 的更改**仅对当前会话生效**。添加 `--global` 可将更改持久化到 `config.yaml`：

```
/model claude-sonnet-4 --global     # 切换并保存为新默认值
```

:::info 如果我只看到 OpenRouter 模型怎么办？
如果你只配置了 OpenRouter，`/model` 将只显示 OpenRouter 模型。要添加其他 provider（Anthropic、DeepSeek、Copilot 等），请退出会话并从终端运行 `hermes model`。
:::

Provider 和 base URL 的更改会自动持久化到 `config.yaml`。从自定义端点切换走时，过时的 base URL 会被清除，以防止其泄漏到其他 provider。

## `hermes gateway`

```bash
hermes gateway <subcommand>
```

子命令：

| 子命令 | 说明 |
|------------|-------------|
| `run` | 在前台运行 gateway。推荐用于 WSL、Docker 和 Termux。 |
| `start` | 启动已安装的 systemd/launchd 后台服务。 |
| `stop` | 停止服务（或前台进程）。 |
| `restart` | 重启服务。 |
| `status` | 显示服务状态。 |
| `list` | 列出**所有 profile** 及每个 profile 的 gateway 当前是否运行（有 PID 时显示）。当你并行运行多个 profile 并需要单一概览时很方便。 |
| `install` | 安装为 systemd（Linux）或 launchd（macOS）后台服务。 |
| `uninstall` | 删除已安装的服务。 |
| `setup` | 交互式消息平台设置。 |

选项：

| 选项 | 说明 |
|--------|-------------|
| `--all` | 在 `start` / `restart` / `stop` 时：对**每个 profile** 的 gateway 执行操作，而不仅限于活跃的 `HERMES_HOME`。当你并行运行多个 profile 并希望在 `hermes update` 后全部重启时很有用。 |
| `--no-supervise` | 在 `run` 时：在 s6-overlay Docker 镜像内部，跳过 s6 自动监管，退回到 pre-s6 前台语义——gateway 作为容器主进程运行，无自动重启。在 s6 镜像之外为空操作。等同于设置 `HERMES_GATEWAY_NO_SUPERVISE=1`。 |

:::tip WSL 用户
使用 `hermes gateway run` 而非 `hermes gateway start`——WSL 的 systemd 支持不稳定。用 tmux 包裹以保持持久运行：`tmux new -s hermes 'hermes gateway run'`。详见 [WSL FAQ](/reference/faq#wsl-gateway-keeps-disconnecting-or-hermes-gateway-start-fails)。
:::

## `hermes lsp`

```bash
hermes lsp <subcommand>
```

管理 Language Server Protocol 集成。LSP 在后台运行真实的语言服务器（pyright、gopls、rust-analyzer 等），并将其诊断信息输入 `write_file` 和 `patch` 使用的写后检查。受 git 工作区检测限制——仅当 cwd 或编辑的文件位于 git worktree 内时，LSP 才会运行。

子命令：

| 子命令 | 说明 |
|------------|-------------|
| `status` | 显示服务状态、已配置的服务器、安装状态。 |
| `list` | 打印支持的服务器注册表。传入 `--installed-only` 可跳过缺失的服务器。 |
| `install <id>` | 主动安装某个服务器的二进制文件。 |
| `install-all` | 安装所有具有已知自动安装方案的服务器。 |
| `restart` | 关闭正在运行的客户端，以便下次编辑时重新启动。 |
| `which <id>` | 打印某个服务器的已解析二进制路径。 |

完整指南、支持的语言和配置项，请参阅 [LSP — 语义诊断](/user-guide/features/lsp)。

## `hermes setup`

```bash
hermes setup [model|tts|terminal|gateway|tools|agent] [--non-interactive] [--reset] [--quick] [--reconfigure] [--portal]
```

**首次运行：** 启动首次使用向导。

**已配置用户：** 直接进入完整重新配置向导——每个提示都以当前值作为默认值，按 Enter 保留或输入新值。无菜单。

跳转到某个部分而非完整向导：

| 部分 | 说明 |
|---------|-------------|
| `model` | Provider 和模型设置。 |
| `terminal` | 终端后端和沙箱设置。 |
| `gateway` | 消息平台设置。 |
| `tools` | 按平台启用/禁用工具。 |
| `agent` | Agent 行为设置。 |

选项：

| 选项 | 说明 |
|--------|-------------|
| `--quick` | 在已配置用户运行时：仅提示缺失或未设置的项目，跳过已配置的项目。 |
| `--non-interactive` | 使用默认值/环境变量，不显示提示。 |
| `--reset` | 在设置前将配置重置为默认值。 |
| `--reconfigure` | 向后兼容别名——在已有安装上裸运行 `hermes setup` 现在默认执行此操作。 |
| `--portal` | 一键 Nous Portal 设置：通过 OAuth 登录，将 Nous 设为推理 provider，并选择加入 [Tool Gateway](../user-guide/features/tool-gateway.md)。跳过向导其余部分。 |

## `hermes portal`

```bash
hermes portal [status|open|tools]
```

检查 Nous Portal 认证、Tool Gateway 路由，并访问订阅页面。不带子命令时运行 `status`。

| 子命令 | 说明 |
|------------|-------------|
| `status`（默认） | Portal 认证状态 + 每个工具的 Tool Gateway 路由摘要。不带子命令时也会显示。 |
| `open` | 在默认浏览器中打开 `portal.nousresearch.com/manage-subscription`。 |
| `tools` | 列出每个 Tool Gateway 合作伙伴（Firecrawl、FAL、OpenAI TTS、Browser Use、Modal）及哪些通过 Nous 路由。 |

关于 gateway 本身的配置，请参阅 [Tool Gateway](../user-guide/features/tool-gateway.md)。关于一键设置路径，请参阅上方的 `hermes setup --portal`。

## `hermes whatsapp`

```bash
hermes whatsapp
```

运行 WhatsApp 配对/设置流程，包括模式选择和二维码配对。

## `hermes slack`

```bash
hermes slack manifest              # 将 manifest 打印到 stdout
hermes slack manifest --write      # 写入 ~/.hermes/slack-manifest.json
hermes slack manifest --slashes-only  # 仅输出 features.slash_commands 数组
```

生成一个 Slack app manifest，将 `COMMAND_REGISTRY` 中的每条 gateway 命令（`/btw`、`/stop`、`/model` 等）注册为一等公民 Slack 斜杠命令——与 Discord 和 Telegram 保持一致。将输出粘贴到你的 Slack app 配置中：[https://api.slack.com/apps](https://api.slack.com/apps) → 你的 app → **Features → App Manifest → Edit**，然后点击 **Save**。如果 scope 或斜杠命令有变化，Slack 会提示重新安装。

| 标志 | 默认值 | 用途 |
|------|---------|---------|
| `--write [PATH]` | stdout | 写入文件而非 stdout。裸 `--write` 写入 `$HERMES_HOME/slack-manifest.json`。 |
| `--name NAME` | `Hermes` | Slack 中的机器人显示名称。 |
| `--description DESC` | 默认简介 | Slack app 目录中显示的机器人描述。 |
| `--slashes-only` | 关闭 | 仅输出 `features.slash_commands`，用于合并到手动维护的 manifest 中。 |

`hermes update` 后重新运行 `hermes slack manifest --write` 以获取新增命令。


## `hermes login` / `hermes logout` *（已弃用）*

:::caution
`hermes login` 已被移除。请使用 `hermes auth` 管理 OAuth 凭据，使用 `hermes model` 选择 provider，或使用 `hermes setup` 进行完整的交互式设置。
:::

## `hermes auth`

管理同一 provider 的密钥轮换凭据池。完整文档请参阅 [凭据池](/user-guide/features/credential-pools)。

```bash
hermes auth                                              # 交互式向导
hermes auth list                                         # 显示所有池
hermes auth list openrouter                              # 显示特定 provider
hermes auth add openrouter --api-key sk-or-v1-xxx        # 添加 API 密钥
hermes auth add anthropic --type oauth                   # 添加 OAuth 凭据
hermes auth remove openrouter 2                          # 按索引删除
hermes auth reset openrouter                             # 清除冷却时间
hermes auth status anthropic                             # 显示某 provider 的认证状态
hermes auth logout anthropic                             # 登出并清除已存储的认证状态
hermes auth spotify                                      # 通过 PKCE 将 Hermes 与 Spotify 认证
```

子命令：`add`、`list`、`remove`、`reset`、`status`、`logout`、`spotify`。不带子命令调用时，启动交互式管理向导。

## `hermes status`

```bash
hermes status [--all] [--deep]
```

| 选项 | 说明 |
|--------|-------------|
| `--all` | 以可分享的脱敏格式显示所有详情。 |
| `--deep` | 运行可能耗时更长的深度检查。 |

## `hermes cron`

```bash
hermes cron <list|create|edit|pause|resume|run|remove|status|tick>
```

| 子命令 | 说明 |
|------------|-------------|
| `list` | 显示已调度的任务。 |
| `create` / `add` | 从 prompt 创建调度任务，可通过重复 `--skill` 附加一个或多个 skill。 |
| `edit` | 更新任务的调度、prompt、名称、投递方式、重复次数或附加的 skill。支持 `--clear-skills`、`--add-skill` 和 `--remove-skill`。 |
| `pause` | 暂停任务而不删除。 |
| `resume` | 恢复已暂停的任务并计算下次未来运行时间。 |
| `run` | 在下次调度器 tick 时触发任务。 |
| `remove` | 删除调度任务。 |
| `status` | 检查 cron 调度器是否正在运行。 |
| `tick` | 运行到期任务一次后退出。 |

## `hermes kanban`

```bash
hermes kanban [--board <slug>] <action> [options]
```

多 profile、多项目协作看板。每个安装可托管多个看板（每个项目、仓库或领域一个）；每个看板是独立的队列，拥有自己的 SQLite 数据库和调度器作用域。新安装从名为 `default` 的单个看板开始，其数据库为 `~/.hermes/kanban.db`（向后兼容）；其他看板位于 `~/.hermes/kanban/boards/<slug>/kanban.db`。嵌入在 gateway 中的调度器每次 tick 扫描所有看板。

**全局标志（适用于以下所有操作）：**

| 标志 | 用途 |
|------|---------|
| `--board <slug>` | 操作特定看板。默认为当前看板（通过 `hermes kanban boards switch`、`HERMES_KANBAN_BOARD` 环境变量或 `default` 设置）。 |

**这是人工/脚本操作界面。** 调度器生成的 agent worker 通过专用的 `kanban_*` [toolset](/user-guide/features/kanban#how-workers-interact-with-the-board)（`kanban_show`、`kanban_complete`、`kanban_block`、`kanban_create`、`kanban_link`、`kanban_comment`、`kanban_heartbeat`；编排器 profile 还可使用 `kanban_list` 和 `kanban_unblock`）驱动看板，而非调用 `hermes kanban`。Worker 的环境中固定了 `HERMES_KANBAN_BOARD`，因此物理上无法看到其他看板。

| 操作 | 用途 |
|--------|---------|
| `init` | 如果缺少则创建 `kanban.db`。幂等操作。 |
| `boards list` / `boards ls` | 列出所有看板及任务数量。支持 `--json`、`--all`（包含已归档）。 |
| `boards create <slug>` | 创建新看板。标志：`--name`、`--description`、`--icon`、`--color`、`--switch`（设为活跃）。Slug 为 kebab-case，自动转小写。 |
| `boards switch <slug>` / `boards use` | 将 `<slug>` 持久化为活跃看板（写入 `~/.hermes/kanban/current`）。 |
| `boards show` / `boards current` | 打印当前活跃看板的名称、数据库路径和任务数量。 |
| `boards rename <slug> "<name>"` | 更改看板的显示名称。Slug 不可变。 |
| `boards rm <slug>` | 归档（默认）或硬删除看板。`--delete` 跳过归档步骤。已归档看板移至 `boards/_archived/<slug>-<ts>/`。`default` 看板拒绝此操作。 |
| `create "<title>"` | 在活跃看板上创建新任务。标志：`--body`、`--assignee`、`--parent`（可重复）、`--workspace scratch\|worktree\|dir:<path>`、`--tenant`、`--priority`、`--triage`、`--idempotency-key`、`--max-runtime`、`--max-retries`、`--skill`（可重复）。 |
| `list` / `ls` | 列出活跃看板上的任务。可用 `--mine`、`--assignee`、`--status`、`--tenant`、`--archived`、`--json` 过滤。 |
| `show <id>` | 显示任务及其评论和事件。`--json` 用于机器输出。 |
| `assign <id> <profile>` | 分配或重新分配。使用 `none` 取消分配。任务运行时拒绝此操作。 |
| `link <parent> <child>` | 添加依赖关系。检测循环依赖。两个任务必须在同一看板上。 |
| `unlink <parent> <child>` | 删除依赖关系。 |
| `claim <id>` | 原子性地认领就绪任务。打印已解析的工作区路径。 |
| `comment <id> "<text>"` | 追加评论。下一个认领该任务的 worker 会在其 `kanban_show()` 响应中读取到它。 |
| `complete <id>` | 将任务标记为完成。标志：`--result`、`--summary`、`--metadata`。 |
| `block <id> "<reason>"` | 将任务标记为等待人工输入。同时将原因追加为评论。 |
| `schedule <id> "<reason>"` | 将时间延迟/后续工作停放到 `scheduled` 状态，使其不显示为人工阻塞项。 |
| `unblock <id>` | 将已阻塞或已调度的任务返回就绪状态（如果依赖仍未完成则返回 `todo`）。 |
| `archive <id>` | 从默认列表中隐藏。`gc` 将删除 scratch 工作区。 |
| `tail <id>` | 跟踪任务的事件流。 |
| `dispatch` | 对活跃看板执行一次调度器扫描。标志：`--dry-run`、`--max N`、`--failure-limit N`、`--json`。 |
| `context <id>` | 打印 worker 将看到的完整上下文（标题 + 正文 + 父任务结果 + 评论）。 |
| `specify <id>` / `specify --all` | 通过辅助 LLM 将 triage 列中的任务细化为具体规格（标题 + 包含目标、方案、验收标准的正文），然后将其提升到 `todo`。标志：`--tenant`（将 `--all` 限定到一个 tenant）、`--author`、`--json`。在 `config.yaml` 的 `auxiliary.triage_specifier` 下配置模型。 |
| `decompose <id>` / `decompose --all` | 将 triage 列中的任务按描述拆分为子任务图，路由到专业 profile（编排器驱动路径）。当 LLM 判断任务不适合拆分时，回退到 specify 风格的单任务提升。与 `specify` 相同的标志。在 `config.yaml` 的 `auxiliary.kanban_decomposer` 下配置模型。当 `kanban.auto_decompose: true`（默认）时，每次调度器 tick 也会自动运行。参见 [自动与手动编排](/user-guide/features/kanban#auto-vs-manual-orchestration)。 |
| `gc` | 删除已归档任务的 scratch 工作区。 |

示例：

```bash
# 创建第二个看板并在不切换的情况下向其添加任务。
hermes kanban boards create atm10-server --name "ATM10 Server" --icon 🎮
hermes kanban --board atm10-server create "Restart server" --assignee ops

# 切换活跃看板以供后续调用使用。
hermes kanban boards switch atm10-server
hermes kanban list                  # 显示 atm10-server 的任务

# 归档看板（可恢复）或硬删除。
hermes kanban boards rm atm10-server
hermes kanban boards rm atm10-server --delete
```

看板解析顺序（优先级从高到低）：`--board <slug>` 标志 → `HERMES_KANBAN_BOARD` 环境变量 → `~/.hermes/kanban/current` 文件 → `default`。

所有操作也可作为 gateway 中的斜杠命令使用（`/kanban …`），参数界面相同——包括 `boards` 子命令和 `--board` 标志。

完整设计——与 Cline Kanban / Paperclip / NanoClaw / Gemini Enterprise 的对比、八种协作模式、四个用户故事、并发正确性证明——请参阅仓库中的 `docs/hermes-kanban-v1-spec.pdf` 或 [Kanban 用户指南](/user-guide/features/kanban)。

## `hermes webhook`

```bash
hermes webhook <subscribe|list|remove|test>
```

管理用于事件驱动 agent 激活的动态 webhook 订阅。需要在 config 中启用 webhook 平台——如未配置，将打印设置说明。

| 子命令 | 说明 |
|------------|-------------|
| `subscribe` / `add` | 创建 webhook 路由。返回要在你的服务上配置的 URL 和 HMAC 密钥。 |
| `list` / `ls` | 显示所有 agent 创建的订阅。 |
| `remove` / `rm` | 删除动态订阅。不影响 config.yaml 中的静态路由。 |
| `test` | 发送测试 POST 以验证订阅是否正常工作。 |

### `hermes webhook subscribe`

```bash
hermes webhook subscribe <name> [options]
```

| 选项 | 说明 |
|--------|-------------|
| `--prompt` | 带有 `{dot.notation}` payload 引用的 prompt 模板。 |
| `--events` | 要接受的逗号分隔事件类型（如 `issues,pull_request`）。为空则接受所有。 |
| `--description` | 人类可读的描述。 |
| `--skills` | 为 agent 运行加载的逗号分隔 skill 名称。 |
| `--deliver` | 投递目标：`log`（默认）、`telegram`、`discord`、`slack`、`github_comment`。 |
| `--deliver-chat-id` | 跨平台投递的目标聊天/频道 ID。 |
| `--secret` | 自定义 HMAC 密钥。省略时自动生成。 |
| `--deliver-only` | 跳过 agent——将渲染后的 `--prompt` 作为字面消息投递。零 LLM 成本，亚秒级投递。要求 `--deliver` 为真实目标（非 `log`）。 |

订阅持久化到 `~/.hermes/webhook_subscriptions.json`，webhook 适配器无需重启 gateway 即可热重载。

## `hermes doctor`

```bash
hermes doctor [--fix]
```

| 选项 | 说明 |
|--------|-------------|
| `--fix` | 尽可能尝试自动修复。 |

## `hermes dump`

```bash
hermes dump [--show-keys]
```

输出整个 Hermes 设置的紧凑纯文本摘要。专为复制粘贴到 Discord、GitHub issue 或 Telegram 寻求支持而设计——无 ANSI 颜色、无特殊格式，只有数据。

| 选项 | 说明 |
|--------|-------------|
| `--show-keys` | 显示脱敏的 API 密钥前缀（首尾各 4 个字符），而非仅显示 `set`/`not set`。 |

### 包含内容

| 部分 | 详情 |
|---------|---------|
| **Header** | Hermes 版本、发布日期、git commit hash |
| **Environment** | 操作系统、Python 版本、OpenAI SDK 版本 |
| **Identity** | 活跃 profile 名称、HERMES_HOME 路径 |
| **Model** | 已配置的默认模型和 provider |
| **Terminal** | 后端类型（local、docker、ssh 等） |
| **API keys** | 所有 22 个 provider/工具 API 密钥的存在性检查 |
| **Features** | 已启用的 toolset、MCP 服务器数量、memory provider |
| **Services** | Gateway 状态、已配置的消息平台 |
| **Workload** | Cron 任务数量、已安装 skill 数量 |
| **Config overrides** | 与默认值不同的所有 config 值 |

### 示例输出

```
--- hermes dump ---
version:          0.8.0 (2026.4.8) [af4abd2f]
os:               Linux 6.14.0-37-generic x86_64
python:           3.11.14
openai_sdk:       2.24.0
profile:          default
hermes_home:      ~/.hermes
model:            anthropic/claude-opus-4.6
provider:         openrouter
terminal:         local

api_keys:
  openrouter           set
  openai               not set
  anthropic            set
  nous                 not set
  firecrawl            set
  ...

features:
  toolsets:           all
  mcp_servers:        0
  memory_provider:    built-in
  gateway:            running (systemd)
  platforms:          telegram, discord
  cron_jobs:          3 active / 5 total
  skills:             42

config_overrides:
  agent.max_turns: 250
  compression.threshold: 0.85
  display.streaming: True
--- end dump ---
```

### 使用场景

- 在 GitHub 上报告 bug——将 dump 粘贴到 issue 中
- 在 Discord 中寻求帮助——在代码块中分享
- 与他人对比设置
- 出现问题时快速进行健全性检查

:::tip
`hermes dump` 专为分享而设计。交互式诊断请使用 `hermes doctor`。可视化概览请使用 `hermes status`。
:::

## `hermes debug`

```bash
hermes debug share [options]
```

将调试报告（系统信息 + 近期日志）上传到粘贴服务并获取可分享的 URL。适用于快速支持请求——包含帮助者诊断问题所需的一切信息。

| 选项 | 说明 |
|--------|-------------|
| `--lines <N>` | 每个日志文件包含的日志行数（默认：200）。 |
| `--expire <days>` | 粘贴过期天数（默认：7）。 |
| `--local` | 在本地打印报告而非上传。 |

报告包含系统信息（操作系统、Python 版本、Hermes 版本）、近期 agent 和 gateway 日志（每文件 512 KB 限制）以及脱敏的 API 密钥状态。密钥始终脱敏——不会上传任何密钥。

依次尝试的粘贴服务：paste.rs、dpaste.com。

### 示例

```bash
hermes debug share              # 上传调试报告，打印 URL
hermes debug share --lines 500  # 包含更多日志行
hermes debug share --expire 30  # 粘贴保留 30 天
hermes debug share --local      # 在终端打印报告（不上传）
```

## `hermes backup`

```bash
hermes backup [options]
```

创建 Hermes 配置、skill、会话和数据的 zip 归档。备份不包含 hermes-agent 代码库本身。

| 选项 | 说明 |
|--------|-------------|
| `-o`, `--output <path>` | zip 文件的输出路径（默认：`~/hermes-backup-<timestamp>.zip`）。 |
| `-q`, `--quick` | 快速快照：仅包含关键状态文件（config.yaml、state.db、.env、auth、cron 任务）。比完整备份快得多。 |
| `-l`, `--label <name>` | 快照标签（仅与 `--quick` 配合使用）。 |

备份使用 SQLite 的 `backup()` API 进行安全复制，因此即使 Hermes 正在运行也能正确工作（WAL 模式安全）。

**zip 中排除的内容：**

- `*.db-wal`、`*.db-shm`、`*.db-journal` — SQLite 的 WAL/共享内存/日志附属文件。`*.db` 文件已通过 `sqlite3.backup()` 获得一致快照；将活跃附属文件一并打包会导致恢复时看到半提交状态。
- `checkpoints/` — 每会话轨迹缓存。以 hash 为键，每次会话重新生成；无论如何都无法干净地移植到其他安装。
- `hermes-agent` 代码本身（这是用户数据备份，不是仓库快照）。

### 示例

```bash
hermes backup                           # 完整备份到 ~/hermes-backup-*.zip
hermes backup -o /tmp/hermes.zip        # 完整备份到指定路径
hermes backup --quick                   # 仅状态快速快照
hermes backup --quick --label "pre-upgrade"  # 带标签的快速快照
```

## `hermes checkpoints`

```bash
hermes checkpoints [COMMAND]
```

检查和管理 `~/.hermes/checkpoints/` 处的影子 git 存储——会话内 `/rollback` 命令的存储层。可随时安全运行；不需要 agent 正在运行。

| 子命令 | 说明 |
|------------|-------------|
| `status`（默认） | 显示总大小、项目数量和每个项目的详情。裸 `hermes checkpoints` 等同于此。 |
| `list` | `status` 的别名。 |
| `prune` | 强制执行清理——删除孤立和过期项目，GC 存储，强制执行大小上限。忽略 24 小时幂等性标记。 |
| `clear` | 删除整个 checkpoint 基础存储。不可逆；除非使用 `-f` 否则要求确认。 |
| `clear-legacy` | 仅删除 v1→v2 迁移产生的 `legacy-<timestamp>/` 归档。 |

### 选项

| 选项 | 子命令 | 说明 |
|--------|------------|-------------|
| `--limit N` | `status`、`list` | 最多列出的项目数（默认 20）。 |
| `--retention-days N` | `prune` | 删除 `last_touch` 早于 N 天的项目（默认 7）。 |
| `--max-size-mb N` | `prune` | 在孤立/过期清理后，删除每个项目最旧的 commit，直到总存储大小 ≤ N MB（默认 500）。 |
| `--keep-orphans` | `prune` | 跳过删除工作目录不再存在的项目。 |
| `-f`, `--force` | `clear`、`clear-legacy` | 跳过确认提示。 |

### 示例

```bash
hermes checkpoints                                  # 状态概览
hermes checkpoints prune --retention-days 3         # 激进清理
hermes checkpoints prune --max-size-mb 200          # 一次性收紧大小上限
hermes checkpoints clear-legacy -f                  # 删除 v1 归档目录
hermes checkpoints clear -f                         # 清除所有内容
```

完整架构和会话内命令，请参阅 [Checkpoints 与 `/rollback`](../user-guide/checkpoints-and-rollback.md)。

## `hermes import`

```bash
hermes import <zipfile> [options]
```

将之前创建的 Hermes 备份恢复到 Hermes 主目录。归档中的所有文件会覆盖 Hermes 主目录中的现有文件；`--force` 仅跳过当目标已有 Hermes 安装时触发的确认提示。

| 选项 | 说明 |
|--------|-------------|
| `-f`, `--force` | 跳过已有安装的确认提示。 |

:::warning
导入前请停止 gateway，以避免与正在运行的进程冲突。
:::

### 示例
```bash
hermes import ~/hermes-backup-20260423.zip           # 覆盖现有配置前提示确认
hermes import ~/hermes-backup-20260423.zip --force   # 不提示直接覆盖
```

## `hermes logs`

```bash
hermes logs [log_name] [options]
```

查看、跟踪和过滤 Hermes 日志文件。所有日志存储在 `~/.hermes/logs/`（非默认 profile 存储在 `<profile>/logs/`）。

### 日志文件

| 名称 | 文件 | 记录内容 |
|------|------|-----------------|
| `agent`（默认） | `agent.log` | 所有 agent 活动——API 调用、工具调度、会话生命周期（INFO 及以上） |
| `errors` | `errors.log` | 仅警告和错误——agent.log 的过滤子集 |
| `gateway` | `gateway.log` | 消息 gateway 活动——平台连接、消息调度、webhook 事件 |

### 选项

| 选项 | 说明 |
|--------|-------------|
| `log_name` | 要查看的日志：`agent`（默认）、`errors`、`gateway`，或 `list` 以显示可用文件及大小。 |
| `-n`, `--lines <N>` | 显示的行数（默认：50）。 |
| `-f`, `--follow` | 实时跟踪日志，类似 `tail -f`。按 Ctrl+C 停止。 |
| `--level <LEVEL>` | 显示的最低日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`。 |
| `--session <ID>` | 过滤包含会话 ID 子字符串的行。 |
| `--since <TIME>` | 显示相对时间之前的行：`30m`、`1h`、`2d` 等。支持 `s`（秒）、`m`（分钟）、`h`（小时）、`d`（天）。 |
| `--component <NAME>` | 按组件过滤：`gateway`、`agent`、`tools`、`cli`、`cron`。 |

### 示例

```bash
# 查看 agent.log 的最后 50 行（默认）
hermes logs

# 实时跟踪 agent.log
hermes logs -f

# 查看 gateway.log 的最后 100 行
hermes logs gateway -n 100

# 仅显示最近一小时的警告和错误
hermes logs --level WARNING --since 1h

# 按特定会话过滤
hermes logs --session abc123

# 从 30 分钟前开始跟踪 errors.log
hermes logs errors --since 30m -f

# 列出所有日志文件及其大小
hermes logs list
```

### 过滤

过滤器可以组合使用。当多个过滤器同时激活时，日志行必须通过**所有**过滤器才会显示：

```bash
# 最近 2 小时内包含会话 "tg-12345" 的 WARNING+ 行
hermes logs --level WARNING --since 2h --session tg-12345
```

当 `--since` 激活时，没有可解析时间戳的行会被包含（它们可能是多行日志条目的续行）。当 `--level` 激活时，没有可检测级别的行会被包含。

### 日志轮转

Hermes 使用 Python 的 `RotatingFileHandler`。旧日志会自动轮转——查找 `agent.log.1`、`agent.log.2` 等。`hermes logs list` 子命令显示所有日志文件，包括已轮转的。

## `hermes config`

```bash
hermes config <subcommand>
```

子命令：

| 子命令 | 说明 |
|------------|-------------|
| `show` | 显示当前 config 值。 |
| `edit` | 在编辑器中打开 `config.yaml`。 |
| `set <key> <value>` | 设置 config 值。 |
| `path` | 打印 config 文件路径。 |
| `env-path` | 打印 `.env` 文件路径。 |
| `check` | 检查缺失或过期的 config。 |
| `migrate` | 交互式添加新引入的选项。 |

## `hermes pairing`

```bash
hermes pairing <list|approve|revoke|clear-pending>
```

| 子命令 | 说明 |
|------------|-------------|
| `list` | 显示待处理和已审批的用户。 |
| `approve <platform> <code>` | 审批配对码。 |
| `revoke <platform> <user-id>` | 撤销用户的访问权限。 |
| `clear-pending` | 清除待处理的配对码。 |

## `hermes skills`

```bash
hermes skills <subcommand>
```

子命令：

| 子命令 | 说明 |
|------------|-------------|
| `browse` | 分页浏览 skill 注册表。 |
| `search` | 搜索 skill 注册表。 |
| `install` | 安装 skill。 |
| `inspect` | 预览 skill 而不安装。 |
| `list` | 列出已安装的 skill。 |
| `check` | 检查已安装的 hub skill 是否有上游更新。 |
| `update` | 在有上游变更时重新安装 hub skill。 |
| `audit` | 重新扫描已安装的 hub skill。 |
| `uninstall` | 删除通过 hub 安装的 skill。 |
| `reset` | 通过清除 manifest 条目，取消将捆绑 skill 标记为 `user_modified` 的状态。使用 `--restore` 时，还会将用户副本替换为捆绑版本。 |
| `publish` | 将 skill 发布到注册表。 |
| `snapshot` | 导出/导入 skill 配置。 |
| `tap` | 管理自定义 skill 来源。 |
| `config` | 按平台交互式启用/禁用 skill 配置。 |

常用示例：

```bash
hermes skills browse
hermes skills browse --source official
hermes skills search react --source skills-sh
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect official/security/1password
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install official/migration/openclaw-migration
hermes skills install skills-sh/anthropics/skills/pdf --force
hermes skills install https://sharethis.chat/SKILL.md                     # 直接 URL（单文件 SKILL.md）
hermes skills install https://example.com/SKILL.md --name my-skill        # frontmatter 无名称时覆盖名称
hermes skills check
hermes skills update
hermes skills config
hermes skills reset google-workspace
hermes skills reset google-workspace --restore --yes
```

注意：
- `--force` 可以覆盖第三方/社区 skill 的非危险性策略阻止。
- `--force` 不覆盖 `dangerous` 扫描结论。
- `--source skills-sh` 搜索公共 `skills.sh` 目录。
- `--source well-known` 允许你将 Hermes 指向暴露 `/.well-known/skills/index.json` 的站点。
- `--source browse-sh` 搜索 [browse.sh](https://browse.sh) 包含 200+ 站点特定浏览器自动化 skill 的目录。标识符形如 `browse-sh/airbnb.com/search-listings-ddgioa`。
- 传入 `http(s)://…/*.md` URL 可直接安装单文件 SKILL.md。当 frontmatter 没有 `name:` 且 URL slug 不是有效标识符时，交互式终端会提示输入名称；非交互式界面（TUI 内的 `/skills install`、gateway 平台）需要改用 `--name <x>`。

## `hermes bundles`

```bash
hermes bundles <subcommand>
```

Skill bundle 将多个 skill 归组到一个 `/<bundle-name>` 斜杠命令下。调用 bundle 会将每个引用的 skill 加载到单个合并的用户消息中。存储位置：`~/.hermes/skill-bundles/<slug>.yaml`。YAML schema 和行为请参阅 [Skill Bundles](../user-guide/features/skills.md#skill-bundles)。

子命令：

| 子命令 | 说明 |
|------------|-------------|
| `list` | 列出已安装的 bundle（不带子命令时的默认行为） |
| `show <name>` | 显示某个 bundle 的名称、描述、skill 和文件路径 |
| `create <name>` | 创建新 bundle。传入 `--skill <id>`（可重复）或省略以进行交互式输入。支持 `--description`、`--instruction`、`--force`。 |
| `delete <name>` | 删除 bundle 文件 |
| `reload` | 重新扫描 `~/.hermes/skill-bundles/` 并报告新增/删除的 bundle |

示例：

```bash
hermes bundles create backend-dev \
  --skill github-code-review \
  --skill test-driven-development \
  --skill github-pr-workflow \
  -d "Backend feature work"

hermes bundles list
hermes bundles show backend-dev
hermes bundles delete backend-dev
```

在聊天会话中，`/bundles` 列出已安装的 bundle，`/<bundle-name>` 加载某个 bundle。

## `hermes curator`

```bash
hermes curator <subcommand>
```

Curator 是一个辅助模型后台任务，定期审查 agent 创建的 skill，修剪过期的，合并重叠的，并归档过时的。捆绑和通过 hub 安装的 skill 不会被触及。归档可恢复；不会发生自动删除。

| 子命令 | 说明 |
|------------|-------------|
| `status` | 显示 curator 状态和 skill 统计 |
| `run` | 立即触发 curator 审查（阻塞直到 LLM 处理完成） |
| `run --background` | 在后台线程中启动 LLM 处理并立即返回 |
| `run --dry-run` | 仅预览——生成审查报告但不进行任何修改 |
| `backup` | 手动对 `~/.hermes/skills/` 进行 tar.gz 快照（curator 在每次真实运行前也会自动快照） |
| `rollback` | 从快照恢复 `~/.hermes/skills/`（默认使用最新快照） |
| `rollback --list` | 列出可用快照 |
| `rollback --id <ts>` | 按 id 恢复特定快照 |
| `rollback -y` | 跳过确认提示 |
| `pause` | 暂停 curator 直到恢复 |
| `resume` | 恢复已暂停的 curator |
| `pin <skill>` | 固定 skill，使 curator 永不自动转换其状态 |
| `unpin <skill>` | 取消固定 skill |
| `restore <skill>` | 恢复已归档的 skill |
| `archive <skill>` | 手动归档 skill |
| `prune` | 手动修剪 curator 通常会清理的 skill |
| `list-archived` | 列出已归档的 skill（可通过 `restore` 恢复） |

在全新安装时，第一次计划运行会延迟一个完整的 `interval_hours`（默认 7 天）——gateway 不会在 `hermes update` 后的第一次 tick 时立即执行 curator。使用 `hermes curator run --dry-run` 在此之前预览。

行为和配置请参阅 [Curator](../user-guide/features/curator.md)。

## `hermes fallback`

```bash
hermes fallback <subcommand>
```

管理 fallback provider 链。当主模型因速率限制、过载或连接错误而失败时，按顺序尝试 fallback provider。

| 子命令 | 说明 |
|------------|-------------|
| `list`（别名：`ls`） | 显示当前 fallback 链（不带子命令时的默认行为） |
| `add` | 选择 provider + 模型（与 `hermes model` 相同的选择器）并追加到链末尾 |
| `remove`（别名：`rm`） | 选择要从链中删除的条目 |
| `clear` | 删除所有 fallback 条目 |

参见 [Fallback Providers](../user-guide/features/fallback-providers.md)。

## `hermes hooks`

```bash
hermes hooks <subcommand>
```

检查 `~/.hermes/config.yaml` 中声明的 shell 脚本 hook，针对合成 payload 测试它们，并管理 `~/.hermes/shell-hooks-allowlist.json` 处的首次使用同意许可名单。

| 子命令 | 说明 |
|------------|-------------|
| `list`（别名：`ls`） | 列出已配置的 hook 及其匹配器、超时和同意状态 |
| `test <event>` | 针对合成 payload 触发匹配 `<event>` 的所有 hook |
| `revoke`（别名：`remove`、`rm`） | 删除某个命令的许可名单条目（下次重启后生效） |
| `doctor` | 检查每个已配置的 hook：可执行位、许可名单、mtime 漂移、JSON 有效性和合成运行计时 |

事件签名和 payload 格式请参阅 [Hooks](../user-guide/features/hooks.md)。

## `hermes memory`

```bash
hermes memory <subcommand>
```

设置和管理外部 memory provider plugin。可用 provider：honcho、openviking、mem0、hindsight、holographic、retaindb、byterover、supermemory。同一时间只能有一个外部 provider 处于活跃状态。内置 memory（MEMORY.md/USER.md）始终处于活跃状态。

子命令：

| 子命令 | 说明 |
|------------|-------------|
| `setup` | 交互式 provider 选择和配置。 |
| `status` | 显示当前 memory provider 配置。 |
| `off` | 禁用外部 provider（仅使用内置）。 |

:::info Provider 特定子命令
当外部 memory provider 处于活跃状态时，它可能会注册自己的顶级 `hermes <provider>` 命令用于 provider 特定管理（例如 Honcho 激活时的 `hermes honcho`）。未激活的 provider 不暴露其子命令。运行 `hermes --help` 查看当前已连接的命令。
:::

## `hermes acp`

```bash
hermes acp
```

将 Hermes 作为 ACP（Agent Client Protocol）stdio 服务器启动，用于编辑器集成。

相关入口：

```bash
hermes-acp
python -m acp_adapter
```

首先安装支持：

```bash
cd ~/.hermes/hermes-agent && uv pip install -e '.[acp]'
```

参见 [ACP 编辑器集成](../user-guide/features/acp.md) 和 [ACP 内部原理](../developer-guide/acp-internals.md)。

## `hermes mcp`

```bash
hermes mcp <subcommand>
```

管理 MCP（Model Context Protocol）服务器配置，并将 Hermes 作为 MCP 服务器运行。

| 子命令 | 说明 |
|------------|-------------|
| `serve [-v\|--verbose]` | 将 Hermes 作为 MCP 服务器运行——向其他 agent 暴露对话。 |
| `add <name> [--url URL] [--command CMD] [--args ...] [--auth oauth\|header]` | 添加 MCP 服务器并自动发现工具。 |
| `remove <name>`（别名：`rm`） | 从 config 中删除 MCP 服务器。 |
| `list`（别名：`ls`） | 列出已配置的 MCP 服务器。 |
| `test <name>` | 测试与 MCP 服务器的连接。 |
| `configure <name>`（别名：`config`） | 切换服务器的工具选择。 |
| `login <name>` | 强制重新认证基于 OAuth 的 MCP 服务器。 |

参见 [MCP 配置参考](./mcp-config-reference.md)、[在 Hermes 中使用 MCP](../guides/use-mcp-with-hermes.md) 和 [MCP 服务器模式](../user-guide/features/mcp.md#running-hermes-as-an-mcp-server)。

## `hermes plugins`

```bash
hermes plugins [subcommand]
```

统一的 plugin 管理——通用 plugin、memory provider 和 context engine 集于一处。不带子命令运行 `hermes plugins` 会打开包含两个部分的复合交互界面：

- **General Plugins** — 多选复选框，用于启用/禁用已安装的 plugin
- **Provider Plugins** — 单选配置，用于 Memory Provider 和 Context Engine。在某个类别上按 ENTER 打开单选选择器。

| 子命令 | 说明 |
|------------|-------------|
| *（无）* | 复合交互界面——通用 plugin 切换 + provider plugin 配置。 |
| `install <identifier> [--force]` | 从 Git URL 或 `owner/repo` 安装 plugin。 |
| `update <name>` | 拉取已安装 plugin 的最新变更。 |
| `remove <name>`（别名：`rm`、`uninstall`） | 删除已安装的 plugin。 |
| `enable <name>` | 启用已禁用的 plugin。 |
| `disable <name>` | 禁用 plugin 而不删除。 |
| `list`（别名：`ls`） | 列出已安装的 plugin 及启用/禁用状态。 |

Provider plugin 选择保存到 `config.yaml`：
- `memory.provider` — 活跃 memory provider（为空 = 仅内置）
- `context.engine` — 活跃 context engine（`"compressor"` = 内置默认值）

通用 plugin 禁用列表存储在 `config.yaml` 的 `plugins.disabled` 下。

参见 [Plugins](../user-guide/features/plugins.md) 和 [构建 Hermes Plugin](../developer-guide/plugins/index.md)。

## `hermes tools`

```bash
hermes tools [--summary]
```

| 选项 | 说明 |
|--------|-------------|
| `--summary` | 打印当前已启用工具摘要并退出。 |

不带 `--summary` 时，启动交互式按平台工具配置界面。

## `hermes computer-use`

```bash
hermes computer-use <subcommand>
```

子命令：

| 子命令 | 说明 |
|------------|-------------|
| `install` | 运行上游 cua-driver 安装程序（仅 macOS）。 |
| `install --upgrade` | 即使 cua-driver 已在 PATH 中也重新运行安装程序。上游脚本始终拉取最新版本，因此这会执行原地升级。 |
| `status` | 打印 `cua-driver` 是否在 `$PATH` 中以及已安装的版本。 |

`hermes computer-use install` 是安装 `computer_use` toolset 使用的 [cua-driver](https://github.com/trycua/cua) 二进制文件的稳定入口。它运行与首次启用 Computer Use 时 `hermes tools` 调用的相同上游安装程序，因此如果 toolset 切换未触发安装（例如在已配置用户的设置中），可以安全地用于重新运行安装。

`hermes update` 在更新结束时，如果 cua-driver 在 PATH 中，会自动重新运行上游安装程序，因此大多数用户不需要手动调用 `--upgrade`。当上游发布了你现在就想要的修复，而不想等待下次 Hermes 更新时，使用此选项。

## `hermes sessions`

```bash
hermes sessions <subcommand>
```

子命令：

| 子命令 | 说明 |
|------------|-------------|
| `list` | 列出最近的会话。 |
| `browse` | 带搜索和恢复功能的交互式会话选择器。 |
| `export <output> [--session-id ID]` | 将会话导出为 JSONL。 |
| `delete <session-id>` | 删除单个会话。 |
| `prune` | 删除旧会话。 |
| `stats` | 显示会话存储统计信息。 |
| `rename <session-id> <title>` | 设置或更改会话标题。 |

## `hermes insights`

```bash
hermes insights [--days N] [--source platform]
```

| 选项 | 说明 |
|--------|-------------|
| `--days <n>` | 分析最近 `n` 天（默认：30）。 |
| `--source <platform>` | 按来源过滤，如 `cli`、`telegram` 或 `discord`。 |

## `hermes claw`

```bash
hermes claw migrate [options]
```

将 OpenClaw 设置迁移到 Hermes。从 `~/.openclaw`（或自定义路径）读取并写入 `~/.hermes`。自动检测旧版目录名（`~/.clawdbot`、`~/.moltbot`）和配置文件名（`clawdbot.json`、`moltbot.json`）。

| 选项 | 说明 |
|--------|-------------|
| `--dry-run` | 预览将迁移的内容而不写入任何内容。 |
| `--preset <name>` | 迁移预设：`full`（所有兼容设置）或 `user-data`（排除基础设施配置）。两种预设都不导入密钥——需要显式传入 `--migrate-secrets`。 |
| `--overwrite` | 在冲突时覆盖现有 Hermes 文件（默认：当计划有冲突时拒绝应用）。 |
| `--migrate-secrets` | 在迁移中包含 API 密钥。即使在 `--preset full` 下也需要显式指定。 |
| `--no-backup` | 跳过迁移前对 `~/.hermes/` 的 zip 快照（默认情况下，在应用前会将单个还原点归档写入 `~/.hermes/backups/pre-migration-*.zip`；可用 `hermes import` 恢复）。 |
| `--source <path>` | 自定义 OpenClaw 目录（默认：`~/.openclaw`）。 |
| `--workspace-target <path>` | 工作区说明（AGENTS.md）的目标目录。 |
| `--skill-conflict <mode>` | 处理 skill 名称冲突：`skip`（默认）、`overwrite` 或 `rename`。 |
| `--yes` | 跳过确认提示。 |

### 迁移内容

迁移涵盖 30+ 个类别，包括 persona、memory、skill、模型 provider、消息平台、agent 行为、会话策略、MCP 服务器、TTS 等。条目要么**直接导入**到 Hermes 等效项，要么**归档**以供手动审查。

**直接导入：** SOUL.md、MEMORY.md、USER.md、AGENTS.md、skill（4 个源目录）、默认模型、自定义 provider、MCP 服务器、消息平台 token 和许可名单（Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost）、agent 默认值（推理努力程度、压缩、人工延迟、时区、沙箱）、会话重置策略、审批规则、TTS 配置、浏览器设置、工具设置、执行超时、命令许可名单、gateway 配置以及来自 3 个来源的 API 密钥。

**归档以供手动审查：** Cron 任务、plugin、hook/webhook、memory 后端（QMD）、skill 注册表配置、UI/身份、日志、多 agent 设置、频道绑定、IDENTITY.md、TOOLS.md、HEARTBEAT.md、BOOTSTRAP.md。

**API 密钥解析**按优先级顺序检查三个来源：config 值 → `~/.openclaw/.env` → `auth-profiles.json`。所有 token 字段处理纯字符串、环境变量模板（`${VAR}`）和 SecretRef 对象。

完整的 config 键映射、SecretRef 处理详情和迁移后检查清单，请参阅**[完整迁移指南](../guides/migrate-from-openclaw.md)**。

### 示例

```bash
# 预览将迁移的内容
hermes claw migrate --dry-run

# 完整迁移（所有兼容设置，不含密钥）
hermes claw migrate --preset full

# 包含 API 密钥的完整迁移
hermes claw migrate --preset full --migrate-secrets

# 仅迁移用户数据（不含密钥），覆盖冲突
hermes claw migrate --preset user-data --overwrite

# 从自定义 OpenClaw 路径迁移
hermes claw migrate --source /home/user/old-openclaw
```

## `hermes dashboard`

```bash
hermes dashboard [options]
```

启动 Web 控制台——基于浏览器的界面，用于管理配置、API 密钥和监控会话。需要 `cd ~/.hermes/hermes-agent && uv pip install -e ".[web]"`（FastAPI + Uvicorn）。内嵌浏览器 Chat 标签页始终可用，但额外需要 `pty` extra（`cd ~/.hermes/hermes-agent && uv pip install -e ".[web,pty]"`）以及 POSIX PTY 环境（如 Linux、macOS 或 WSL2）。完整文档请参阅 [Web 控制台](/user-guide/features/web-dashboard)。

| 选项 | 默认值 | 说明 |
|--------|---------|-------------|
| `--port` | `9119` | Web 服务器运行端口 |
| `--host` | `127.0.0.1` | 绑定地址 |
| `--no-open` | — | 不自动打开浏览器 |
| `--insecure` | 关闭 | 允许绑定到非 localhost 主机。会在网络上暴露控制台凭据；仅在受信任的网络控制下使用。 |
| `--stop` | — | 停止正在运行的 `hermes dashboard` 进程并退出。 |
| `--status` | — | 列出正在运行的 `hermes dashboard` 进程并退出。 |

```bash
# 默认——在浏览器中打开 http://127.0.0.1:9119
hermes dashboard

# 自定义端口，不打开浏览器
hermes dashboard --port 8080 --no-open
```

## `hermes profile`

```bash
hermes profile <subcommand>
```

管理 profile——多个隔离的 Hermes 实例，每个实例拥有自己的 config、会话、skill 和主目录。

| 子命令 | 说明 |
|------------|-------------|
| `list` | 列出所有 profile。 |
| `use <name>` | 设置粘性默认 profile。 |
| `create <name> [--clone] [--clone-all] [--clone-from <source>] [--no-alias]` | 创建新 profile。`--clone` 从活跃 profile 复制 config、`.env`、`SOUL.md` 和 skills。`--clone-all` 复制所有状态。`--clone-from` 指定源 profile，除非与 `--clone-all` 配合使用，否则会隐含 config 克隆。 |
| `delete <name> [-y]` | 删除 profile。 |
| `show <name>` | 显示 profile 详情（主目录、config 等）。 |
| `alias <name> [--remove] [--name NAME]` | 管理快速访问 profile 的包装脚本。 |
| `rename <old> <new>` | 重命名 profile。 |
| `export <name> [-o FILE]` | 将 profile 导出为 `.tar.gz` 归档（本地备份）。 |
| `import <archive> [--name NAME]` | 从 `.tar.gz` 归档导入 profile（本地恢复）。 |
| `install <source> [--name N] [--alias] [--force] [-y]` | 从 git URL 或本地目录安装 profile 发行版。 |
| `update <name> [--force-config] [-y]` | 重新拉取发行版；保留用户数据（memory、会话、auth）。 |
| `info <name>` | 显示 profile 的发行版 manifest（版本、依赖、来源）。 |

示例：

```bash
hermes profile list
hermes profile create work --clone
hermes profile use work
hermes profile alias work --name h-work
hermes profile export work -o work-backup.tar.gz
hermes profile import work-backup.tar.gz --name restored
hermes profile install github.com/user/my-distro --alias
hermes profile update work
hermes -p work chat -q "Hello from work profile"
```

## `hermes completion`

```bash
hermes completion [bash|zsh|fish]
```

将 shell 补全脚本打印到 stdout。在 shell profile 中 source 输出内容，即可对 Hermes 命令、子命令和 profile 名称进行 Tab 补全。

示例：

```bash
# Bash
hermes completion bash >> ~/.bashrc

# Zsh
hermes completion zsh >> ~/.zshrc

# Fish
hermes completion fish > ~/.config/fish/completions/hermes.fish
```

## `hermes update`

```bash
hermes update [--check] [--backup] [--restart-gateway]
```

拉取最新的 `hermes-agent` 代码并在受管理的 venv 中重新安装依赖，然后重新运行安装后 hook（MCP 服务器、skill 同步、补全安装）。可在运行中的安装上安全执行。使用 `--check` 查看你的检出是否落后于 `origin/main`，而不安装。

| 选项 | 说明 |
|--------|-------------|
| `--check` | 并排打印当前 commit 和最新 `origin/main` commit，同步时退出码为 0，落后时为 1。不拉取、不安装、不重启任何内容。 |
| `--backup` | 在拉取前创建 `HERMES_HOME` 的带标签预更新快照（config、auth、会话、skill、配对数据）。默认**关闭**——之前的始终备份行为在大型主目录上每次更新会增加数分钟。通过 `config.yaml` 中的 `update.backup: true` 永久开启。 |
| `--restart-gateway` | 成功更新后重启正在运行的 gateway 服务。如果安装了多个 profile，隐含 `--all` 语义。 |

附加行为：

- **配对数据快照。** 即使 `--backup` 关闭，`hermes update` 也会在 `git pull` 前对 `~/.hermes/pairing/` 和 Feishu 评论规则进行轻量快照。如果拉取覆盖了你正在编辑的文件，可以用 `hermes backup restore --state pre-update` 回滚。
- **旧版 `hermes.service` 警告。** 如果 Hermes 检测到预重命名的 `hermes.service` systemd 单元（而非当前的 `hermes-gateway.service`），会打印一次性迁移提示，帮助你避免循环重启问题。
- **退出码。** 成功时为 `0`，拉取/安装/安装后错误时为 `1`，阻止 `git pull` 的意外工作树变更时为 `2`。

## 维护命令

| 命令 | 说明 |
|---------|-------------|
| `hermes version` | 打印版本信息。 |
| `hermes update` | 拉取最新变更并重新安装依赖。 |
| `hermes uninstall [--full] [--yes]` | 删除 Hermes，可选择删除所有 config/数据。 |

## 另请参阅

- [斜杠命令参考](./slash-commands.md)
- [CLI 界面](../user-guide/cli.md)
- [会话](../user-guide/sessions.md)
- [Skill 系统](../user-guide/features/skills.md)
- [皮肤与主题](../user-guide/features/skins.md)