---
sidebar_position: 1
title: "CLI 命令参考"
description: "Hermes 终端命令和命令家族的权威参考"
---

# CLI 命令参考

本页面涵盖从 Shell 运行的**终端命令**。

有关聊天内斜杠命令，请参见 [斜杠命令参考](./slash-commands.md)。

## 全局入口点

```bash
hermes [全局选项] <命令> [子命令/选项]
```

### 全局选项

| 选项 | 描述 |
|--------|-------------|
| `--version`, `-V` | 显示版本并退出。 |
| `--profile <名称>`, `-p <名称>` | 选择此次调用使用的 Hermes profile。会覆盖 `hermes profile use` 设置的默认 profile。 |
| `--resume <会话>`, `-r <会话>` | 通过 ID 或标题恢复之前的会话。 |
| `--continue [名称]`, `-c [名称]` | 恢复最近的会话，或匹配标题的最近会话。 |
| `--worktree`, `-w` | 为并行代理工作流启动一个隔离的 git worktree。 |
| `--yolo` | 跳过危险命令确认提示。 |
| `--pass-session-id` | 在代理的系统提示中包含会话 ID。 |
| `--ignore-user-config` | 忽略 `~/.hermes/config.yaml` 并回退到内置默认值。`.env` 中的凭证仍会被加载。 |
| `--ignore-rules` | 跳过 `AGENTS.md`、`SOUL.md`、`.cursorrules`、记忆和预加载技能的自动注入。 |
| `--tui` | 启动 [TUI](../user-guide/tui.md) 而不是经典 CLI。等同于 `HERMES_TUI=1`。 |
| `--dev` | 配合 `--tui`：通过 `tsx` 直接运行 TypeScript 源码而不是预构建的包（供 TUI 贡献者使用）。 |

## 顶级命令

| 命令 | 用途 |
|---------|---------|
| `hermes chat` | 与代理进行交互式或一次性聊天。 |
| `hermes model` | 交互式选择默认 provider 和模型。 |
| `hermes fallback` | 管理主模型出错时尝试的备用 provider。 |
| `hermes gateway` | 运行或管理消息网关服务。 |
| `hermes setup` | 所有或部分配置的交互式设置向导。 |
| `hermes whatsapp` | 配置和配对 WhatsApp 桥接。 |
| `hermes slack` | Slack 辅助工具（目前：生成包含所有命令作为原生斜杠的 app manifest）。 |
| `hermes auth` | 管理凭证 — 添加、列出、移除、重置、设置策略。处理 Codex/Nous/Anthropic 的 OAuth 流程。 |
| `hermes login` / `logout` | **已弃用** — 请改用 `hermes auth`。 |
| `hermes status` | 显示代理、认证和平台状态。 |
| `hermes cron` | 检查和触发 cron 调度器。 |
| `hermes kanban` | 多 profile 协作看板（任务、链接、调度器）。 |
| `hermes webhook` | 管理用于事件驱动激活的动态 webhook 订阅。 |
| `hermes hooks` | 检查、批准或移除 `config.yaml` 中声明的 shell 脚本钩子。 |
| `hermes doctor` | 诊断配置和依赖问题。 |
| `hermes dump` | 用于支持/调试的可复制设置摘要。 |
| `hermes debug` | 调试工具 — 上传日志和系统信息以获取支持。 |
| `hermes backup` | 将 Hermes 主目录备份为 zip 文件。 |
| `hermes checkpoints` | 检查/清理 `~/.hermes/checkpoints/`（/`rollback` 使用的隐藏存储）。不带参数运行以查看状态概览。 |
| `hermes import` | 从 zip 文件恢复 Hermes 备份。 |
| `hermes logs` | 查看、跟踪和过滤代理/网关/错误日志文件。 |
| `hermes config` | 显示、编辑、迁移和查询配置文件。 |
| `hermes pairing` | 批准或撤销消息配对码。 |
| `hermes skills` | 浏览、安装、发布、审计和配置技能。 |
| `hermes curator` | 后台技能维护 — 状态、运行、暂停、固定。请参见 [Curator](../user-guide/features/curator.md)。 |
| `hermes memory` | 配置外部记忆 provider。特定插件的子命令（如 `hermes honcho`）在其 provider 激活时自动注册。 |
| `hermes acp` | 将 Hermes 作为 ACP 服务器运行以实现编辑器集成。 |
| `hermes mcp` | 管理 MCP 服务器配置和将 Hermes 作为 MCP 服务器运行。 |
| `hermes plugins` | 管理 Hermes Agent 插件（安装、启用、禁用、移除）。 |
| `hermes tools` | 配置每个平台启用的工具。 |
| `hermes sessions` | 浏览、导出、清理、重命名和删除会话。 |
| `hermes insights` | 显示 token/费用/活动分析。 |
| `hermes fallback` | 备用 provider 链的交互式管理器。 |
| `hermes claw` | OpenClaw 迁移辅助工具。 |
| `hermes dashboard` | 启动 Web 仪表板以管理配置、API 密钥和会话。 |
| `hermes profile` | 管理 profile — 多个隔离的 Hermes 实例。 |
| `hermes completion` | 打印 shell 补全脚本（bash/zsh/fish）。 |
| `hermes version` | 显示版本信息。 |
| `hermes update` | 拉取最新代码并重新安装依赖。`--check` 仅打印提交差异而不拉取；`--backup` 在拉取前拍摄 `HERMES_HOME` 快照。 |
| `hermes uninstall` | 从系统中移除 Hermes。 |

## `hermes chat`

```bash
hermes chat [选项]
```

常用选项：

| 选项 | 描述 |
|--------|-------------|
| `-q`, `--query "..."` | 一次性、非交互式提示。 |
| `-m`, `--model <模型>` | 覆盖此次运行的模型。 |
| `-t`, `--toolsets <csv>` | 启用逗号分隔的工具集。 |
| `--provider <provider>` | 强制使用某个 provider：`auto`、`openrouter`、`nous`、`openai-codex`、`copilot-acp`、`copilot`、`anthropic`、`gemini`、`google-gemini-cli`、`huggingface`、`zai`、`kimi-coding`、`kimi-coding-cn`、`minimax`、`minimax-cn`、`minimax-oauth`、`kilocode`、`xiaomi`、`arcee`、`gmi`、`alibaba`、`alibaba-coding-plan`（别名 `alibaba_coding`）、`deepseek`、`nvidia`、`ollama-cloud`、`xai`（别名 `grok`）、`qwen-oauth`、`bedrock`、`opencode-zen`、`opencode-go`、`ai-gateway`、`azure-foundry`、`tencent-tokenhub`（别名 `tencent`、`tokenhub`）。 |
| `-s`, `--skills <名称>` | 为会话预加载一个或多个技能（可重复或逗号分隔）。 |
| `-v`, `--verbose` | 详细输出。 |
| `-Q`, `--quiet` | 程序化模式：抑制横幅/旋转器/工具预览。 |
| `--image <路径>` | 将本地图片附加到单个查询。 |
| `--resume <会话>` / `--continue [名称]` | 直接从 `chat` 恢复会话。 |
| `--worktree` | 为此次运行创建隔离的 git worktree。 |
| `--checkpoints` | 在破坏性文件更改之前启用文件系统检查点。 |
| `--yolo` | 跳过确认提示。 |
| `--pass-session-id` | 将会话 ID 传递到系统提示。 |
| `--ignore-user-config` | 忽略 `~/.hermes/config.yaml` 并使用内置默认值。`.env` 中的凭证仍会被加载。适用于隔离的 CI 运行、可重现的 bug 报告和第三方集成。 |
| `--ignore-rules` | 跳过 `AGENTS.md`、`SOUL.md`、`.cursorrules`、持久记忆和预加载技能的自动注入。配合 `--ignore-user-config` 可实现完全隔离的运行。 |
| `--source <标签>` | 会话源标签用于过滤（默认：`cli`）。对第三方集成使用 `tool`，这样不会出现在用户会话列表中。 |
| `--max-turns <N>` | 每轮对话的最大工具调用迭代次数（默认：90，或配置中的 `agent.max_turns`）。 |

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

### `hermes -z <提示>` — 脚本化一次性

对于程序化调用者（shell 脚本、CI、cron、父进程通过管道输入提示），`hermes -z` 是最纯粹的一次性入口：**单提示输入，最终响应文本输出，stdout 或 stderr 上没有任何其他内容。** 无横幅、无旋转器、无工具预览、无 `Session:` 行 — 只有代理的最终回复作为纯文本。

```bash
hermes -z "What's the capital of France?"
# → Paris.

# 父脚本可以干净地捕获响应：
answer=$(hermes -z "summarize this" < /path/to/file.txt)
```

每次运行的覆盖（不修改 `~/.hermes/config.yaml`）：

| 标志 | 等效环境变量 | 用途 |
|---|---|---|
| `-m` / `--model <模型>` | `HERMES_INFERENCE_MODEL` | 覆盖此次运行的模型 |
| `--provider <provider>` | `HERMES_INFERENCE_PROVIDER` | 覆盖此次运行的 provider |

```bash
hermes -z "…" --provider openrouter --model openai/gpt-5.5
# 或：
HERMES_INFERENCE_MODEL=anthropic/claude-sonnet-4.6 hermes -z "…"
```

相同的代理、相同的工具、相同的技能 — 只是剥离了所有交互式/装饰层。如果你也需要工具输出到记录中，请改用 `hermes chat -q`；`-z` 明确用于"我只需要最终答案"。

## `hermes model`

交互式 provider + 模型选择器。**这是添加新 provider、设置 API 密钥和运行 OAuth 流程的命令。** 从终端运行 — 不是在活跃的 Hermes 聊天会话内运行。

```bash
hermes model
```

使用此命令的场景：
- **添加新 provider**（OpenRouter、Anthropic、Copilot、DeepSeek、自定义等）
- 登录基于 OAuth 的 provider（Anthropic、Copilot、Codex、Nous Portal）
- 输入或更新 API 密钥
- 从 provider 特定的模型列表中选择
- 配置自定义/自托管端点
- 将新的默认值保存到配置

:::warning hermes model vs /model — 了解区别
**`hermes model`**（从终端运行，在任何 Hermes 会话之外）是**完整的 provider 设置向导**。它可以添加新 provider、运行 OAuth 流程、提示输入 API 密钥和配置端点。

**`/model`**（在活跃的 Hermes 聊天会话中输入）只能**在已设置的 provider 和模型之间切换**。它不能添加新 provider、运行 OAuth 或提示输入 API 密钥。

**如果需要添加新 provider：** 先退出 Hermes 会话（`Ctrl+C` 或 `/quit`），然后从终端提示符运行 `hermes model`。
:::

### `/model` 斜杠命令（会话中）

在不离开会话的情况下切换已配置的模型：

```
/model                              # 显示当前模型和可用选项
/model claude-sonnet-4              # 切换模型（自动检测 provider）
/model zai:glm-5                    # 切换 provider 和模型
/model custom:qwen-2.5              # 使用自定义端点上的模型
/model custom                       # 从自定义端点自动检测模型
/model custom:local:qwen-2.5        # 使用命名自定义 provider
/model openrouter:anthropic/claude-sonnet-4  # 切换回云端
```

默认情况下，`/model` 的更改**仅应用于当前会话**。添加 `--global` 以将更改持久化到 `config.yaml`：

```
/model claude-sonnet-4 --global     # 切换并保存为新默认值
```

:::info 如果只看到 OpenRouter 模型怎么办？
如果你只配置了 OpenRouter，`/model` 将只显示 OpenRouter 模型。要添加其他 provider（Anthropic、DeepSeek、Copilot 等），请退出会话并从终端运行 `hermes model`。
:::

Provider 和 base URL 更改会自动持久化到 `config.yaml`。当从自定义端点切换离开时，旧的 base URL 会被清除以防止泄露到其他 provider。

## `hermes gateway`

```bash
hermes gateway <子命令>
```

子命令：

| 子命令 | 描述 |
|------------|-------------|
| `run` | 在前台运行网关。推荐用于 WSL、Docker 和 Termux。 |
| `start` | 启动已安装的 systemd/launchd 后台服务。 |
| `stop` | 停止服务（或前台进程）。 |
| `restart` | 重启服务。 |
| `status` | 显示服务状态。 |
| `install` | 安装为 systemd（Linux）或 launchd（macOS）后台服务。 |
| `uninstall` | 移除已安装的服务。 |
| `setup` | 交互式消息平台设置。 |

选项：

| 选项 | 描述 |
|--------|-------------|
| `--all` | 在 `start` / `restart` / `stop` 时：对**每个 profile 的**网关进行操作，而不仅仅是活动的 `HERMES_HOME`。如果你并行运行多个 profile 并想在 `hermes update` 后全部重启，这很有用。 |

:::tip WSL 用户
使用 `hermes gateway run` 而不是 `hermes gateway start` — WSL 的 systemd 支持不可靠。用 tmux 包装以保持持久性：`tmux new -s hermes 'hermes gateway run'`。详情请参见 [WSL FAQ](/docs/reference/faq#wsl-gateway-keeps-disconnecting-or-hermes-gateway-start-fails)。
:::

## `hermes setup`

```bash
hermes setup [model|tts|terminal|gateway|tools|agent] [--non-interactive] [--reset] [--quick] [--reconfigure]
```

**首次运行：** 启动首次使用向导。

**回访用户（已配置）：** 直接进入完整重配置向导 — 每个提示显示其当前值作为默认值，按 Enter 保持或输入新值。无菜单。

跳转到某个部分而不是完整向导：

| 部分 | 描述 |
|---------|-------------|
| `model` | Provider 和模型设置。 |
| `terminal` | 终端后端和沙箱设置。 |
| `gateway` | 消息平台设置。 |
| `tools` | 按平台启用/禁用工具。 |
| `agent` | 代理行为设置。 |

选项：

| 选项 | 描述 |
|--------|-------------|
| `--quick` | 回访用户运行时：仅提示缺失或未设置的项目。跳过已配置的项目。 |
| `--non-interactive` | 使用默认值/环境值而不提示。 |
| `--reset` | 在设置前重置配置为默认值。 |
| `--reconfigure` | 向后兼容别名 — 已安装上的裸 `hermes setup` 现在默认执行此操作。 |

## `hermes whatsapp`

```bash
hermes whatsapp
```

运行 WhatsApp 配对/设置流程，包括模式选择和二维码配对。

## `hermes slack`

```bash
hermes slack manifest              # 打印 manifest 到 stdout
hermes slack manifest --write      # 写入到 ~/.hermes/slack-manifest.json
hermes slack manifest --slashes-only  # 仅输出 features.slash_commands 数组
```

生成一个 Slack app manifest，将 `COMMAND_REGISTRY`（`/btw`、`/stop`、`/model`、…）中的每个网关命令注册为一等的 Slack 斜杠命令 — 与 Discord 和 Telegram 对等。将输出粘贴到 Slack app 配置中：
[https://api.slack.com/apps](https://api.slack.com/apps) → 你的 app →
**Features → App Manifest → Edit**，然后**保存**。如果 scope 或斜杠命令更改，Slack 会提示重新安装。

| 标志 | 默认值 | 用途 |
|------|---------|---------|
| `--write [路径]` | stdout | 写入文件而不是 stdout。裸 `--write` 写入 `$HERMES_HOME/slack-manifest.json`。 |
| `--name 名称` | `Hermes` | Slack 中的 Bot 显示名称。 |
| `--description 描述` | 默认简介 | Slack app 目录中显示的 Bot 描述。 |
| `--slashes-only` | 关闭 | 仅输出 `features.slash_commands` 以合并到手动维护的 manifest 中。 |

在 `hermes update` 后再次运行 `hermes slack manifest --write` 以获取新命令。

## `hermes login` / `hermes logout` *（已弃用）*

:::caution
`hermes login` 已被移除。请使用 `hermes auth` 管理 OAuth 凭证，使用 `hermes model` 选择 provider，或使用 `hermes setup` 进行完整的交互式设置。
:::

## `hermes auth`

管理同一 provider 密钥轮换的凭证池。请参见 [凭证池](/docs/user-guide/features/credential-pools) 获取完整文档。

```bash
hermes auth                                              # 交互式向导
hermes auth list                                         # 显示所有池
hermes auth list openrouter                              # 显示特定 provider
hermes auth add openrouter --api-key sk-or-v1-xxx        # 添加 API 密钥
hermes auth add anthropic --type oauth                   # 添加 OAuth 凭证
hermes auth remove openrouter 2                          # 按索引移除
hermes auth reset openrouter                             # 清除冷却时间
```

子命令：`add`、`list`、`remove`、`reset`。不带子命令调用时，启动交互式管理向导。

## `hermes status`

```bash
hermes status [--all] [--deep]
```

| 选项 | 描述 |
|--------|-------------|
| `--all` | 以可共享的编辑格式显示所有详细信息。 |
| `--deep` | 运行可能需要更长时间的深度检查。 |

## `hermes cron`

```bash
hermes cron <list|create|edit|pause|resume|run|remove|status|tick>
```

| 子命令 | 描述 |
|------------|-------------|
| `list` | 显示计划的作业。 |
| `create` / `add` | 从提示创建计划作业，可选择通过重复 `--skill` 附加一个或多个技能。 |
| `edit` | 更新作业的计划、提示、名称、交付、重复次数或附加的技能。支持 `--clear-skills`、`--add-skill` 和 `--remove-skill`。 |
| `pause` | 暂停作业而不删除。 |
| `resume` | 恢复暂停的作业并计算其下一次运行时间。 |
| `run` | 在下一个调度器刻度上触发作业。 |
| `remove` | 删除计划作业。 |
| `status` | 检查 cron 调度器是否正在运行。 |
| `tick` | 运行到期的作业一次并退出。 |

## `hermes kanban`

```bash
hermes kanban [--board <slug>] <操作> [选项]
```

多 profile、多项目协作看板。每个安装可以托管多个看板（每个项目、仓库或域一个）；每个看板是一个独立队列，有自己的 SQLite 数据库和调度器作用域。新安装以一个名为 `default` 的看板开始，其数据库为 `~/.hermes/kanban.db`（向后兼容）；其他看板位于 `~/.hermes/kanban/boards/<slug>/kanban.db`。嵌入网关的调度器在每个刻度上扫描每个看板。

**全局标志（适用于下面的每个操作）：**

| 标志 | 用途 |
|------|---------|
| `--board <slug>` | 在特定看板上进行操作。默认为当前看板（通过 `hermes kanban boards switch`、`HERMES_KANBAN_BOARD` 环境变量或 `default` 设置）。 |

**这是人类/脚本表面。** 调度器生成的代理工作线程通过专用的 `kanban_*` [工具集](/docs/user-guide/features/kanban#how-workers-interact-with-the-board)（`kanban_show`、`kanban_complete`、`kanban_block`、`kanban_create`、`kanban_link`、`kanban_comment`、`kanban_heartbeat`）驱动看板，而不是调用 `hermes kanban`。工作线程在环境中固定了 `HERMES_KANBAN_BOARD`，因此物理上无法看到其他看板。

| 操作 | 用途 |
|--------|---------|
| `init` | 如果缺失则创建 `kanban.db`。幂等。 |
| `boards list` / `boards ls` | 列出所有看板及任务计数。`--json`、`--all`（包括已归档）。 |
| `boards create <slug>` | 创建新看板。标志：`--name`、`--description`、`--icon`、`--color`、`--switch`（设为活动）。Slug 是 kebab-case，自动转小写。 |
| `boards switch <slug>` / `boards use` | 将 `<slug>` 持久化为活动看板（写入 `~/.hermes/kanban/current`）。 |
| `boards show` / `boards current` | 打印当前活动看板的名称、数据库路径和任务计数。 |
| `boards rename <slug> "<名称>"` | 更改看板的显示名称。Slug 不可变。 |
| `boards rm <slug>` | 归档（默认）或硬删除看板。`--delete` 跳过归档步骤。已归档的看板移到 `boards/_archived/<slug>-<ts>/`。拒绝 `default`。 |
| `create "<标题>"` | 在活动看板上创建新任务。标志：`--body`、`--assignee`、`--parent`（可重复）、`--workspace scratch|worktree|dir:<路径>`、`--tenant`、`--priority`、`--triage`、`--idempotency-key`、`--max-runtime`、`--skill`（可重复）。 |
| `list` / `ls` | 列出活动看板上的任务。使用 `--mine`、`--assignee`、`--status`、`--tenant`、`--archived`、`--json` 过滤。 |
| `show <id>` | 显示带有评论和事件的任務。`--json` 用于机器输出。 |
| `assign <id> <profile>` | 分配或重新分配。使用 `none` 取消分配。任务运行时拒绝。 |
| `link <父> <子>` | 添加依赖。循环检测。两个任务必须在同一看板上。 |
| `unlink <父> <子>` | 移除依赖。 |
| `claim <id>` | 原子性地声明就绪任务。打印解析的工作区路径。 |
| `comment <id> "<文本>"` | 追加评论。下一个声明该任务的工作线程在响应 `kanban_show()` 时读取它。 |
| `complete <id>` | 标记任务完成。标志：`--result`、`--summary`、`--metadata`。 |
| `block <id> "<原因>"` | 标记任务被阻止。也将原因追加为评论。 |
| `unblock <id>` | 将被阻止的任务恢复为就绪。 |
| `archive <id>` | 从默认列表中隐藏。`gc` 将移除临时工作区。 |
| `tail <id>` | 跟踪任务的事件流。 |
| `dispatch` | 在活动看板上执行一次调度器传递。标志：`--dry-run`、`--max N`、`--json`。 |
| `context <id>` | 打印工作线程将看到的完整上下文（标题 + 正文 + 父结果 + 评论）。 |
| `specify <id>` / `specify --all` | 通过辅助 LLM 将待分类列中的任务具体化为具体规格（包含目标、方法、验收标准的标题 + 正文），然后提升到 `todo`。标志：`--tenant`（将 `--all` 限定为一个租户）、`--author`、`--json`。在 `config.yaml` 的 `auxiliary.triage_specifier` 下配置模型。 |
| `gc` | 移除已归档任务的临时工作区。 |

示例：

```bash
# 创建第二个看板并在上面放置任务而不切换离开。
hermes kanban boards create atm10-server --name "ATM10 Server" --icon 🎮
hermes kanban --board atm10-server create "Restart server" --assignee ops

# 切换活动看板以进行后续调用。
hermes kanban boards switch atm10-server
hermes kanban list                  # 显示 atm10-server 任务

# 归档看板（可恢复）或硬删除。
hermes kanban boards rm atm10-server
hermes kanban boards rm atm10-server --delete
```

看板解析顺序（优先级从高到低）：`--board <slug>` 标志 → `HERMES_KANBAN_BOARD` 环境变量 → `~/.hermes/kanban/current` 文件 → `default`。

所有操作也可以在网关中作为斜杠命令使用（`/kanban …`），具有相同的参数表面 — 包括 `boards` 子命令和 `--board` 标志。

有关完整设计 — 与 Cline Kanban / Paperclip / NanoClaw / Gemini Enterprise 的比较、八种协作模式、四个用户故事、并发正确性证明 — 请参见仓库中的 `docs/hermes-kanban-v1-spec.pdf` 或 [Kanban 用户指南](/docs/user-guide/features/kanban)。

## `hermes webhook`

```bash
hermes webhook <subscribe|list|remove|test>
```

管理用于事件驱动代理激活的动态 webhook 订阅。需要 webhook 平台在配置中启用 — 如果未配置，打印设置说明。

| 子命令 | 描述 |
|------------|-------------|
| `subscribe` / `add` | 创建 webhook 路由。返回要配置在你的服务上的 URL 和 HMAC 密钥。 |
| `list` / `ls` | 显示所有代理创建的订阅。 |
| `remove` / `rm` | 删除动态订阅。config.yaml 中的静态路由不受影响。 |
| `test` | 发送测试 POST 以验证订阅是否正常工作。 |

### `hermes webhook subscribe`

```bash
hermes webhook subscribe <名称> [选项]
```

| 选项 | 描述 |
|--------|-------------|
| `--prompt` | 带有 `{dot.notation}` 有效载荷引用的提示模板。 |
| `--events` | 接受的逗号分隔事件类型（如 `issues,pull_request`）。空 = 全部。 |
| `--description` | 人类可读的描述。 |
| `--skills` | 逗号分隔的技能名称，用于代理运行。 |
| `--deliver` | 交付目标：`log`（默认）、`telegram`、`discord`、`slack`、`github_comment`。 |
| `--deliver-chat-id` | 跨平台交付的目标聊天/频道 ID。 |
| `--secret` | 自定义 HMAC 密钥。如果省略则自动生成。 |
| `--deliver-only` | 跳过代理 — 将渲染后的 `--prompt` 作为原文消息交付。零 LLM 成本，亚秒级交付。需要 `--deliver` 是真实目标（不是 `log`）。 |

订阅持久化到 `~/.hermes/webhook_subscriptions.json`，由 webhook 适配器热重载，无需重启网关。

## `hermes doctor`

```bash
hermes doctor [--fix]
```

| 选项 | 描述 |
|--------|-------------|
| `--fix` | 尽可能尝试自动修复。 |

## `hermes dump`

```bash
hermes dump [--show-keys]
```

输出你的整个 Hermes 设置的紧凑纯文本摘要。设计为在请求支持时复制粘贴到 Discord、GitHub issues 或 Telegram — 无 ANSI 颜色、无特殊格式，只有数据。

| 选项 | 描述 |
|--------|-------------|
| `--show-keys` | 显示编辑后的 API 密钥前缀（首尾各 4 个字符）而不是仅显示 `set`/`not set`。 |

### 包含内容

| 部分 | 详情 |
|---------|---------|
| **Header** | Hermes 版本、发布日期、git 提交哈希 |
| **Environment** | OS、Python 版本、OpenAI SDK 版本 |
| **Identity** | 活动 profile 名称、HERMES_HOME 路径 |
| **Model** | 配置的默认模型和 provider |
| **Terminal** | 后端类型（local、docker、ssh 等） |
| **API keys** | 所有 22 个 provider/工具 API 密钥的存在检查 |
| **Features** | 启用的工具集、MCP 服务器计数、记忆 provider |
| **Services** | 网关状态、配置的消息平台 |
| **Workload** | Cron 作业计数、已安装技能计数 |
| **Config overrides** | 与默认值不同的任何配置值 |

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

- 在 GitHub 上报告 bug — 将 dump 粘贴到 issue
- 在 Discord 上寻求帮助 — 在代码块中分享
- 将你的设置与他人比较
- 当某些东西不工作时进行快速健全性检查

:::tip
`hermes dump` 专门为分享而设计。对于交互式诊断，请使用 `hermes doctor`。对于可视化概览，请使用 `hermes status`。
:::

## `hermes debug`

```bash
hermes debug share [选项]
```

上传调试报告（系统信息 + 近期日志）到粘贴服务并获取可共享的 URL。对快速支持请求很有用 — 包含帮助者诊断问题所需的一切。

| 选项 | 描述 |
|--------|-------------|
| `--lines <N>` | 每个日志文件包含的日志行数（默认：200）。 |
| `--expire <天数>` | 粘贴过期天数（默认：7）。 |
| `--local` | 本地打印报告而不是上传。 |

报告包括系统信息（OS、Python 版本、Hermes 版本）、近期代理和网关日志（每个文件 512 KB 限制）以及编辑后的 API 密钥状态。密钥始终被编辑 — 不上传任何秘密。

尝试的粘贴服务顺序：paste.rs、dpaste.com。

### 示例

```bash
hermes debug share              # 上传调试报告，打印 URL
hermes debug share --lines 500  # 包含更多日志行
hermes debug share --expire 30  # 保留 30 天
hermes debug share --local      # 打印报告到终端（不上传）
```

## `hermes backup`

```bash
hermes backup [选项]
```

创建 Hermes 配置、技能、会话和数据的 zip 存档。备份不包括 hermes-agent 代码库本身。

| 选项 | 描述 |
|--------|-------------|
| `-o`, `--output <路径>` | zip 文件的输出路径（默认：`~/hermes-backup-<时间戳>.zip`）。 |
| `-q`, `--quick` | 快速快照：仅关键状态文件（config.yaml、state.db、.env、auth、cron 作业）。比完整备份快得多。 |
| `-l`, `--label <名称>` | 快照的标签（仅与 `--quick` 一起使用）。 |

备份使用 SQLite 的 `backup()` API 进行安全复制，因此在 Hermes 运行时也能正常工作（WAL 模式安全）。

**不包括在 zip 中的内容：**

- `*.db-wal`、`*.db-shm`、`*.db-journal` — SQLite 的 WAL/共享内存/日志附属文件。`*.db` 文件已经通过 `sqlite3.backup()` 获得了一致的快照；附带活的附属文件会在恢复时看到半提交状态。
- `checkpoints/` — 每个会话的轨迹缓存。按哈希键控，每会话重新生成；无论如何都不会干净地移植到另一个安装。
- `hermes-agent` 代码本身（这是用户数据备份，不是仓库快照）。

### 示例

```bash
hermes backup                           # 完整备份到 ~/hermes-backup-*.zip
hermes backup -o /tmp/hermes.zip        # 完整备份到特定路径
hermes backup --quick                   # 快速仅状态快照
hermes backup --quick --label "pre-upgrade"  # 带标签的快速快照
```

## `hermes checkpoints`

```bash
hermes checkpoints [命令]
```

检查和管理 `~/.hermes/checkpoints/` 处的隐藏 git 存储 — 会话内 `/rollback` 命令背后的存储层。随时可以安全运行；不需要代理正在运行。

| 子命令 | 描述 |
|------------|-------------|
| `status`（默认） | 显示总大小、项目计数和每个项目的细分。裸 `hermes checkpoints` 等效于此。 |
| `list` | `status` 的别名。 |
| `prune` | 强制清理扫描 — 删除孤立和过时项目、GC 存储、执行大小限制。忽略 24h 幂等性标记。 |
| `clear` | 删除整个检查点基础。不可逆；除非使用 `-f` 否则会请求确认。 |
| `clear-legacy` | 仅删除 v1→v2 迁移产生的 `legacy-<时间戳>/` 归档。 |

### 选项

| 选项 | 子命令 | 描述 |
|--------|------------|-------------|
| `--limit N` | `status`、`list` | 最多列出 N 个项目（默认 20）。 |
| `--retention-days N` | `prune` | 删除 `last_touch` 超过 N 天的项目（默认 7）。 |
| `--max-size-mb N` | `prune` | 在孤立/过时扫描之后，删除每个项目中最旧的提交，直到总存储大小 ≤ N MB（默认 500）。 |
| `--keep-orphans` | `prune` | 跳过删除工作目录不再存在的项目。 |
| `-f`, `--force` | `clear`、`clear-legacy` | 跳过确认提示。 |

### 示例

```bash
hermes checkpoints                                  # 状态概览
hermes checkpoints prune --retention-days 3         # 积极清理
hermes checkpoints prune --max-size-mb 200          # 一次性收紧大小限制
hermes checkpoints clear-legacy -f                  # 删除 v1 归档目录
hermes checkpoints clear -f                         # 清除所有内容
```

请参见 [检查点和 `/rollback`](../user-guide/checkpoints-and-rollback.md) 获取完整架构和会话内命令。

## `hermes import`

```bash
hermes import <zip文件> [选项]
```

将之前创建的 Hermes 备份恢复到你的 Hermes 主目录。存档中的所有文件覆盖 Hermes 主目录中的现有文件；`--force` 仅在目标已有 Hermes 安装时跳过触发的确认提示。

| 选项 | 描述 |
|--------|-------------|
| `-f`, `--force` | 跳过已有安装确认提示。 |

:::warning
导入前停止网关以避免与运行中的进程冲突。
:::

### 示例
```bash
hermes import ~/hermes-backup-20260423.zip           # 在覆盖现有配置前提示
hermes import ~/hermes-backup-20260423.zip --force   # 不提示直接覆盖
```

## `hermes logs`

```bash
hermes logs [日志名称] [选项]
```

查看、跟踪和过滤 Hermes 日志文件。所有日志存储在 `~/.hermes/logs/`（或非默认 profile 的 `<profile>/logs/`）。

### 日志文件

| 名称 | 文件 | 捕获内容 |
|------|------|-----------------|
| `agent`（默认） | `agent.log` | 所有代理活动 — API 调用、工具调度、会话生命周期（INFO 及以上） |
| `errors` | `errors.log` | 仅警告和错误 — agent.log 的过滤子集 |
| `gateway` | `gateway.log` | 消息网关活动 — 平台连接、消息调度、webhook 事件 |

### 选项

| 选项 | 描述 |
|--------|-------------|
| `日志名称` | 查看哪个日志：`agent`（默认）、`errors`、`gateway`，或 `list` 显示可用文件及大小。 |
| `-n`, `--lines <N>` | 显示的行数（默认：50）。 |
| `-f`, `--follow` | 实时跟踪日志，如 `tail -f`。按 Ctrl+C 停止。 |
| `--level <级别>` | 显示的最低日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`。 |
| `--session <ID>` | 过滤包含会话 ID 子串的行。 |
| `--since <时间>` | 从多久之前开始显示行：`30m`、`1h`、`2d` 等。支持 `s`（秒）、`m`（分）、`h`（时）、`d`（天）。 |
| `--component <名称>` | 按组件过滤：`gateway`、`agent`、`tools`、`cli`、`cron`。 |

### 示例

```bash
# 查看 agent.log 的最后 50 行（默认）
hermes logs

# 实时跟踪 agent.log
hermes logs -f

# 查看 gateway.log 的最后 100 行
hermes logs gateway -n 100

# 仅显示过去一小时的警告和错误
hermes logs --level WARNING --since 1h

# 按特定会话过滤
hermes logs --session abc123

# 从 30 分钟前开始跟踪 errors.log
hermes logs errors --since 30m -f

# 列出所有日志文件及大小
hermes logs list
```

### 过滤

过滤器可以组合。当多个过滤器处于活动状态时，日志行必须通过**所有**过滤器才会显示：

```bash
# 过去 2 小时内包含会话 "tg-12345" 的 WARNING+ 行
hermes logs --level WARNING --since 2h --session tg-12345
```

当 `--since` 处于活动状态时，包含不可解析时间戳的行会被包含（它们可能是多行日志条目的续行）。当 `--level` 处于活动状态时，包含不可检测级别的行会被包含。

### 日志轮转

Hermes 使用 Python 的 `RotatingFileHandler`。旧日志会自动轮转 — 查找 `agent.log.1`、`agent.log.2` 等。`hermes logs list` 子命令显示所有日志文件，包括轮转的文件。

## `hermes config`

```bash
hermes config <子命令>
```

子命令：

| 子命令 | 描述 |
|------------|-------------|
| `show` | 显示当前配置值。 |
| `edit` | 在编辑器中打开 `config.yaml`。 |
| `set <键> <值>` | 设置配置值。 |
| `path` | 打印配置文件路径。 |
| `env-path` | 打印 `.env` 文件路径。 |
| `check` | 检查缺失或过时的配置。 |
| `migrate` | 交互式添加新引入的选项。 |

## `hermes pairing`

```bash
hermes pairing <list|approve|revoke|clear-pending>
```

| 子命令 | 描述 |
|------------|-------------|
| `list` | 显示待批准和已批准的用户。 |
| `approve <平台> <代码>` | 批准配对码。 |
| `revoke <平台> <用户ID>` | 撤销用户的访问权限。 |
| `clear-pending` | 清除待处理的配对码。 |

## `hermes skills`

```bash
hermes skills <子命令>
```

子命令：

| 子命令 | 描述 |
|------------|-------------|
| `browse` | 技能注册中心的分页浏览器。 |
| `search` | 搜索技能注册中心。 |
| `install` | 安装技能。 |
| `inspect` | 预览技能而不安装。 |
| `list` | 列出已安装的技能。 |
| `check` | 检查已安装的中心技能是否有上游更新。 |
| `update` | 当有上游更改时重新安装中心技能。 |
| `audit` | 重新扫描已安装的中心技能。 |
| `uninstall` | 移除中心安装的技能。 |
| `reset` | 通过清除其 manifest 条目来取消标记为 `user_modified` 的捆绑技能。使用 `--restore` 也会用捆绑版本替换用户副本。 |
| `publish` | 将技能发布到注册中心。 |
| `snapshot` | 导出/导入技能配置。 |
| `tap` | 管理自定义技能源。 |
| `config` | 交互式按平台启用/禁用技能配置。 |

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
hermes skills install https://example.com/SKILL.md --name my-skill        # 当 frontmatter 没有 name 时覆盖名称
hermes skills check
hermes skills update
hermes skills config
hermes skills reset google-workspace
hermes skills reset google-workspace --restore --yes
```

注意：
- `--force` 可以覆盖第三方/社区技能的非危险策略阻止。
- `--force` 不会覆盖 `dangerous` 扫描裁决。
- `--source skills-sh` 搜索公共 `skills.sh` 目录。
- `--source well-known` 让你指向暴露 `/.well-known/skills/index.json` 的站点。
- 传递 `http(s)://…/*.md` URL 直接安装单文件 SKILL.md。当 frontmatter 没有 `name:` 且 URL slug 不是有效标识符时，交互式终端会提示输入名称；非交互式表面（TUI 内的 `/skills install`、网关平台）需要 `--name <x>` 代替。

## `hermes curator`

```bash
hermes curator <子命令>
```

curator 是一个辅助模型后台任务，定期审查代理创建的技能、清理过时技能、整合重叠技能和归档过时技能。捆绑和中心安装的技能不会被触碰。归档可恢复；永远不会自动删除。

| 子命令 | 描述 |
|------------|-------------|
| `status` | 显示 curator 状态和技能统计 |
| `run` | 立即触发 curator 审查（阻塞直到 LLM 传递完成） |
| `run --background` | 在后台线程启动 LLM 传递并立即返回 |
| `run --dry-run` | 仅预览 — 生成审查报告而无任何更改 |
| `backup` | 手动拍摄 `~/.hermes/skills/` 的 tar.gz 快照（curator 也会在每次实际运行前自动快照） |
| `rollback` | 从快照恢复 `~/.hermes/skills/`（默认为最新的） |
| `rollback --list` | 列出可用的快照 |
| `rollback --id <ts>` | 按 ID 恢复特定快照 |
| `rollback -y` | 跳过确认提示 |
| `pause` | 暂停 curator 直到恢复 |
| `resume` | 恢复暂停的 curator |
| `pin <技能>` | 固定技能以使 curator 永远不会自动转换它 |
| `unpin <技能>` | 取消固定技能 |
| `restore <技能>` | 恢复已归档的技能 |

在新安装上，第一次计划的传递会延迟一个完整的 `interval_hours`（默认 7 天）— 网关不会在 `hermes update` 后的第一个刻度上立即进行整理。使用 `hermes curator run --dry-run` 在发生之前预览。

请参见 [Curator](../user-guide/features/curator.md) 了解行为和配置。

## `hermes fallback`

```bash
hermes fallback <子命令>
```

管理备用 provider 链。当主模型因速率限制、过载或连接错误失败时，按顺序尝试备用 provider。

| 子命令 | 描述 |
|------------|-------------|
| `list`（别名：`ls`） | 显示当前备用链（无子命令时的默认值） |
| `add` | 选择 provider + 模型（与 `hermes model` 相同的选择器）并追加到链 |
| `remove`（别名：`rm`） | 选择要从链中删除的条目 |
| `clear` | 移除所有备用条目 |

请参见 [备用 Provider](../user-guide/features/fallback-providers.md)。

## `hermes hooks`

```bash
hermes hooks <子命令>
```

检查在 `~/.hermes/config.yaml` 中声明的 shell 脚本钩子，用合成有效载荷测试它们，并管理 `~/.hermes/shell-hooks-allowlist.json` 的首次使用同意列表。

| 子命令 | 描述 |
|------------|-------------|
| `list`（别名：`ls`） | 列出配置的钩子及其匹配器、超时和同意状态 |
| `test <事件>` | 对合成有效载荷触发每个匹配的 `<事件>` 钩子 |
| `revoke`（别名：`remove`、`rm`） | 移除命令的同意列表条目（下次重启时生效） |
| `doctor` | 检查每个配置的钩子：exec 位、同意列表、mtime 漂移、JSON 有效性和合成运行计时 |

请参见 [Hooks](../user-guide/features/hooks.md) 了解事件签名和有效载荷形状。

## `hermes memory`

```bash
hermes memory <子命令>
```

设置和管理外部记忆 provider 插件。可用 provider：honcho、openviking、mem0、hindsight、holographic、retaindb、byterover、supermemory。一次只能激活一个外部 provider。内置记忆（MEMORY.md/USER.md）始终处于活动状态。

子命令：

| 子命令 | 描述 |
|------------|-------------|
| `setup` | 交互式 provider 选择和配置。 |
| `status` | 显示当前记忆 provider 配置。 |
| `off` | 禁用外部 provider（仅内置）。 |

:::info Provider 特定的子命令
当外部记忆 provider 处于活动状态时，它可能注册自己的顶级 `hermes <provider>` 命令用于 provider 特定的管理（如活动时为 `hermes honcho`）。非活动 provider 不暴露其子命令。运行 `hermes --help` 查看当前已连接的内容。
:::

## `hermes acp`

```bash
hermes acp
```

将 Hermes 作为 ACP（Agent Client Protocol）stdio 服务器启动，用于编辑器集成。

相关入口点：

```bash
hermes-acp
python -m acp_adapter
```

先安装支持：

```bash
pip install -e '.[acp]'
```

请参见 [ACP 编辑器集成](../user-guide/features/acp.md) 和 [ACP 内部](../developer-guide/acp-internals.md)。

## `hermes mcp`

```bash
hermes mcp <子命令>
```

管理 MCP（Model Context Protocol）服务器配置和将 Hermes 作为 MCP 服务器运行。

| 子命令 | 描述 |
|------------|-------------|
| `serve [-v|--verbose]` | 将 Hermes 作为 MCP 服务器运行 — 向其他代理暴露对话。 |
| `add <名称> [--url URL] [--command CMD] [--args ...] [--auth oauth|header]` | 添加带有自动工具发现的 MCP 服务器。 |
| `remove <名称>`（别名：`rm`） | 从配置中移除 MCP 服务器。 |
| `list`（别名：`ls`） | 列出配置的 MCP 服务器。 |
| `test <名称>` | 测试与 MCP 服务器的连接。 |
| `configure <名称>`（别名：`config`） | 切换服务器的工具选择。 |

请参见 [MCP 配置参考](./mcp-config-reference.md)、[将 MCP 与 Hermes 结合使用](../guides/use-mcp-with-hermes.md) 和 [MCP 服务器模式](../user-guide/features/mcp.md#running-hermes-as-an-mcp-server)。

## `hermes plugins`

```bash
hermes plugins [子命令]
```

统一插件管理 — 通用插件、记忆 provider 和上下文引擎集中在一处。运行 `hermes plugins` 不带子命令会打开复合交互式屏幕，包含两部分：

- **通用插件** — 多选复选框用于启用/禁用已安装的插件
- **Provider 插件** — 记忆 Provider 和上下文引擎的单选配置。按 Enter 进入类别打开单选选择器。

| 子命令 | 描述 |
|------------|-------------|
| *（无）* | 复合交互式 UI — 通用插件切换 + provider 插件配置。 |
| `install <标识符> [--force]` | 从 Git URL 或 `owner/repo` 安装插件。 |
| `update <名称>` | 为已安装的插件拉取最新更改。 |
| `remove <名称>`（别名：`rm`、`uninstall`） | 移除已安装的插件。 |
| `enable <名称>` | 启用已禁用的插件。 |
| `disable <名称>` | 禁用插件而不移除。 |
| `list`（别名：`ls`） | 列出已安装的插件及其启用/禁用状态。 |

Provider 插件选择保存到 `config.yaml`：
- `memory.provider` — 活动记忆 provider（空 = 仅内置）
- `context.engine` — 活动上下文引擎（`"compressor"` = 内置默认）

通用插件禁用列表存储在 `config.yaml` 的 `plugins.disabled` 下。

请参见 [Plugins](../user-guide/features/plugins.md) 和 [构建 Hermes 插件](../guides/build-a-hermes-plugin.md)。

## `hermes tools`

```bash
hermes tools [--summary]
```

| 选项 | 描述 |
|--------|-------------|
| `--summary` | 打印当前启用工具摘要并退出。 |

不使用 `--summary` 时，启动交互式按平台工具配置 UI。

## `hermes sessions`

```bash
hermes sessions <子命令>
```

子命令：

| 子命令 | 描述 |
|------------|-------------|
| `list` | 列出最近的会话。 |
| `browse` | 带搜索和恢复的交互式会话选择器。 |
| `export <输出> [--session-id ID]` | 将会话导出为 JSONL。 |
| `delete <会话ID>` | 删除一个会话。 |
| `prune` | 删除旧会话。 |
| `stats` | 显示会话存储统计。 |
| `rename <会话ID> <标题>` | 设置或更改会话标题。 |

## `hermes insights`

```bash
hermes insights [--days N] [--source 平台]
```

| 选项 | 描述 |
|--------|-------------|
| `--days <n>` | 分析过去 `n` 天（默认：30）。 |
| `--source <平台>` | 按来源过滤，如 `cli`、`telegram` 或 `discord`。 |

## `hermes claw`

```bash
hermes claw migrate [选项]
```

将你的 OpenClaw 设置迁移到 Hermes。从 `~/.openclaw`（或自定义路径）读取并写入 `~/.hermes`。自动检测旧目录名（`~/.clawdbot`、`~/.moltbot`）和配置文件名（`clawdbot.json`、`moltbot.json`）。

| 选项 | 描述 |
|--------|-------------|
| `--dry-run` | 预览将要迁移的内容而不写入任何内容。 |
| `--preset <名称>` | 迁移预设：`full`（所有兼容设置）或 `user-data`（排除基础设施配置）。两个预设都不会导入秘密 — 显式传递 `--migrate-secrets`。 |
| `--overwrite` | 冲突时覆盖现有 Hermes 文件（默认：计划有冲突时拒绝应用）。 |
| `--migrate-secrets` | 在迁移中包含 API 密钥。即使在 `--preset full` 下也需要。 |
| `--no-backup` | 跳过迁移前的 `~/.hermes/` zip 快照（默认情况下，在应用前将单个恢复点存档写入 `~/.hermes/backups/pre-migration-*.zip`；可用 `hermes import` 恢复）。 |
| `--source <路径>` | 自定义 OpenClaw 目录（默认：`~/.openclaw`）。 |
| `--workspace-target <路径>` | 工作区指令（AGENTS.md）的目标目录。 |
| `--skill-conflict <模式>` | 处理技能名称冲突：`skip`（默认）、`overwrite` 或 `rename`。 |
| `--yes` | 跳过确认提示。 |

### 迁移内容

迁移涵盖 30+ 类别，涵盖 persona、记忆、技能、模型 provider、消息平台、代理行为、会话策略、MCP 服务器、TTS、浏览器等。項目要么**直接导入**到 Hermes 等效项，要么**归档**供手动审查。

**直接导入：** SOUL.md、MEMORY.md、USER.md、AGENTS.md、技能（4 个源目录）、默认模型、自定义 provider、MCP 服务器、消息平台令牌和允许列表（Telegram、Discord、Slack、WhatsApp、Signal、Matrix、Mattermost）、代理默认值（推理工作、压缩、人工延迟、时区、沙箱）、会话重置策略、批准规则、TTS 配置、浏览器设置、工具设置、exec 超时、命令允许列表、网关配置，以及来自 3 个来源的 API 密钥。

**归档供手动审查：** Cron 作业、插件、钩子/webhook、记忆后端（QMD）、技能注册配置、UI/身份、日志记录、多代理设置、频道绑定、IDENTITY.md、TOOLS.md、HEARTBEAT.md、BOOTSTRAP.md。

**API 密钥解析**按优先级顺序检查三个来源：配置值 → `~/.openclaw/.env` → `auth-profiles.json`。所有令牌字段处理纯字符串、环境模板（`${VAR}`）和 SecretRef 对象。

有关完整的配置键映射、SecretRef 处理细节和迁移后检查清单，请参见 **[完整迁移指南](../guides/migrate-from-openclaw.md)**。

### 示例

```bash
# 预览将要迁移的内容
hermes claw migrate --dry-run

# 完整迁移（所有兼容设置，无秘密）
hermes claw migrate --preset full

# 包括 API 密钥的完整迁移
hermes claw migrate --preset full --migrate-secrets

# 仅迁移用户数据（无秘密），覆盖冲突
hermes claw migrate --preset user-data --overwrite

# 从自定义 OpenClaw 路径迁移
hermes claw migrate --source /home/user/old-openclaw
```

## `hermes dashboard`

```bash
hermes dashboard [选项]
```

启动 Web 仪表板 — 用于管理配置、API 密钥和监控会话的基于浏览器的 UI。需要 `pip install hermes-agent[web]`（FastAPI + Uvicorn）。请参见 [Web 仪表板](/docs/user-guide/features/web-dashboard) 获取完整文档。

| 选项 | 默认值 | 描述 |
|--------|---------|-------------|
| `--port` | `9119` | Web 服务器运行的端口 |
| `--host` | `127.0.0.1` | 绑定地址 |
| `--no-open` | — | 不自动打开浏览器 |

```bash
# 默认 — 打开浏览器到 http://127.0.0.1:9119
hermes dashboard

# 自定义端口，无浏览器
hermes dashboard --port 8080 --no-open
```

## `hermes profile`

```bash
hermes profile <子命令>
```

管理 profile — 多个隔离的 Hermes 实例，每个都有自己的配置、会话、技能和主目录。

| 子命令 | 描述 |
|------------|-------------|
| `list` | 列出所有 profile。 |
| `use <名称>` | 设置固定的默认 profile。 |
| `create <名称> [--clone] [--clone-all] [--clone-from <源>] [--no-alias]` | 创建新 profile。`--clone` 从活动 profile 复制 config、`.env` 和 `SOUL.md`。`--clone-all` 复制所有状态。`--clone-from` 指定源 profile。 |
| `delete <名称> [-y]` | 删除 profile。 |
| `show <名称>` | 显示 profile 详情（主目录、配置等）。 |
| `alias <名称> [--remove] [--name 名称]` | 管理快速访问 profile 的包装脚本。 |
| `rename <旧> <新>` | 重命名 profile。 |
| `export <名称> [-o 文件]` | 将 profile 导出为 `.tar.gz` 存档。 |
| `import <存档> [--name 名称]` | 从 `.tar.gz` 存档导入 profile。 |

示例：

```bash
hermes profile list
hermes profile create work --clone
hermes profile use work
hermes profile alias work --name h-work
hermes profile export work -o work-backup.tar.gz
hermes profile import work-backup.tar.gz --name restored
hermes -p work chat -q "Hello from work profile"
```

## `hermes completion`

```bash
hermes completion [bash|zsh|fish]
```

打印 shell 补全脚本到 stdout。在 shell 配置文件中 source 输出以获得 Hermes 命令、子命令和 profile 名称的 tab 补全。

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

拉取最新的 `hermes-agent` 代码并重新安装 venv 中的依赖，然后重新运行安装后钩子（MCP 服务器、技能同步、补全安装）。在活动安装上运行安全。

| 选项 | 描述 |
|--------|-------------|
| `--check` | 并排打印当前提交和最新的 `origin/main` 提交，如果同步则退出 0，如果落后则退出 1。不拉取、安装或重启任何内容。 |
| `--backup` | 在拉取前创建带标签的 `HERMES_HOME` 更新前快照（配置、auth、会话、技能、配对数据）。默认**关闭** — 之前的始终备份行为在大主目录上为每次更新增加了数分钟。通过 `config.yaml` 中的 `update.backup: true` 永久开启。 |
| `--restart-gateway` | 成功更新后，重启运行中的网关服务。如果安装了多个 profile，则暗示 `--all` 语义。 |

附加行为：

- **配对数据快照。** 即使 `--backup` 关闭，`hermes update` 也会在 `git pull` 之前对 `~/.hermes/pairing/` 和飞书评论规则进行轻量级快照。如果拉取重写了你正在编辑的文件，可以用 `hermes backup restore --state pre-update` 回滚。
- **旧 `hermes.service` 警告。** 如果 Hermes 检测到预重命名的 `hermes.service` systemd 单元（而不是当前的 `hermes-gateway.service`），它会打印一次性迁移提示，这样你可以避免 flap-loop 问题。
- **退出码。** 成功时为 `0`，拉取/安装/安装后错误时为 `1`，阻止 `git pull` 的意外工作树更改时为 `2`。

## `hermes fallback`

```bash
hermes fallback           # 交互式管理器
```

管理备用 provider 链（当你的主 provider 遇到速率限制或返回致命错误时使用），无需手动编辑 `config.yaml`。重用 `hermes model` 的 provider 选择器 — 相同的 provider 列表、相同的凭证提示、相同的验证。

典型会话：

1. 按 `a` 添加备用 → 选择 provider（基于 OAuth 的 provider 打开浏览器；API 密钥 provider 提示输入密钥），然后选择具体模型。
2. 使用 `↑`/`↓` 重新排序备用（列表中第一个最先尝试）。
3. 按 `d` 移除一个。

所有更改持久化到 `config.yaml` 顶级的 `fallback_providers:` 列表。与 [凭证池](/docs/user-guide/features/credential-pools) 交互：池在 provider *内* 轮换密钥，备用切换到*不同*的 provider。

请参见 [备用 Provider](/docs/user-guide/features/fallback-providers) 了解行为详情以及与 `fallback_model`（旧版单一备用键）的交互。

## 维护命令

| 命令 | 描述 |
|---------|-------------|
| `hermes version` | 打印版本信息。 |
| `hermes update` | 拉取最新更改并重新安装依赖。 |
| `hermes uninstall [--full] [--yes]` | 移除 Hermes，可选择删除所有配置/数据。 |

## 另请参见

- [斜杠命令参考](./slash-commands.md)
- [CLI 界面](../user-guide/cli.md)
- [会话](../user-guide/sessions.md)
- [技能系统](../user-guide/features/skills.md)
- [皮肤和主题](../user-guide/features/skins.md)
