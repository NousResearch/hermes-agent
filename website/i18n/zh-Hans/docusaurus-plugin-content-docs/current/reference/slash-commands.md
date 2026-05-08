---
sidebar_position: 2
title: "斜杠命令参考"
description: "交互式 CLI 和消息平台的斜杠命令完整参考"
---

# 斜杠命令参考

Hermes 有两个斜杠命令界面，都由 `hermes_cli/commands.py` 中的中央 `COMMAND_REGISTRY` 驱动：

- **交互式 CLI 斜杠命令** — 由 `cli.py` 分发，命令注册表提供自动补全
- **消息斜杠命令** — 由 `gateway/run.py` 分发，命令注册表生成帮助文本和平台菜单

已安装的技能也会作为动态斜杠命令暴露在这两个界面上。这包括内置技能如 `/plan`，它会打开计划模式并在活动工作区/后端工作目录下的 `.hermes/plans/` 中保存 Markdown 计划。

## 交互式 CLI 斜杠命令

在 CLI 中输入 `/` 可打开自动补全菜单。内置命令不区分大小写。

### 会话

| 命令 | 描述 |
|---------|-------------|
| `/new`（别名：`/reset`） | 开始新会话（新会话 ID + 历史记录） |
| `/clear` | 清屏并开始新会话 |
| `/history` | 显示对话历史 |
| `/save` | 保存当前对话 |
| `/retry` | 重试上一条消息（重新发送给 agent） |
| `/undo` | 移除上一条用户/助手交换内容 |
| `/title` | 为当前会话设置标题（用法：`/title 我的会话名称`） |
| `/compress [聚焦主题]` | 手动压缩对话上下文（刷新记忆 + 总结）。可选的聚焦主题缩小总结保留的范围。 |
| `/rollback` | 列出或恢复文件系统检查点（用法：`/rollback [数字]`） |
| `/snapshot [create\|restore <id>\|prune]`（别名：`/snap`） | 创建或恢复 Hermes 配置/状态快照。`create [label]` 保存快照，`restore <id>` 恢复到该快照，`prune [N]` 删除旧快照，或不带参数列出所有快照。 |
| `/stop` | 终止所有运行中的后台进程 |
| `/queue <提示词>`（别名：`/q`） | 将提示词加入队列等待下一轮（不会中断当前 agent 响应）。 |
| `/steer <提示词>` | 注入一个运行中提示，在**下一次工具调用之后**到达 agent ——不打断，不产生新的用户轮次。文本在当前工具完成后追加到最后一个工具结果的内容中，给 agent 新的上下文而不破坏当前工具调用循环。用此命令在任务中途引导方向（如 agent 运行测试时输入"聚焦 auth 模块"）。 |
| `/goal <文本>` | 设置一个 Hermes 跨轮次努力达成的持久目标 —— 这是我们对 Ralph 循环的实现。每一轮后一个辅助评判模型决定目标是否完成；如果未完成，Hermes 自动继续。子命令：`/goal status`、`/goal pause`、`/goal resume`、`/goal clear`。默认预算为 20 轮（`goals.max_turns`）；任何真实用户消息会中断继续循环，状态在 `/resume` 后存活。详见[持久目标](/docs/user-guide/features/goals)。 |
| `/resume [名称]` | 恢复之前命名的会话 |
| `/redraw` | 强制完整 UI 重绘（从 tmux 调整大小后的终端漂移、鼠标选择残影等恢复） |
| `/status` | 显示会话信息 |
| `/agents`（别名：`/tasks`） | 显示当前会话中的活跃 agent 和运行中的任务。 |
| `/background <提示词>`（别名：`/bg`、`/btw`） | 在独立的后台会话中运行提示词。agent 独立处理您的提示词 —— 您当前的会话可自由进行其他工作。结果在任务完成时作为面板显示。详见 [CLI 后台会话](/docs/user-guide/cli#background-sessions)。 |
| `/branch [名称]`（别名：`/fork`） | 分支当前会话（探索不同路径） |

### 配置

| 命令 | 描述 |
|---------|-------------|
| `/config` | 显示当前配置 |
| `/model [模型名称]` | 显示或更改当前模型。支持：`/model claude-sonnet-4`、`/model provider:model`（切换提供商）、`/model custom:model`（自定义端点）、`/model custom:name:model`（命名自定义提供商）、`/model custom`（自动检测端点），以及用户定义的别名（`/model fav`、`/model grok` —— 参见[自定义模型别名](#custom-model-aliases)）。使用 `--global` 将更改持久化到 config.yaml。**注意：** `/model` 只能切换已配置的提供商。要添加新提供商，请退出会话并从终端运行 `hermes model`。 |
| `/personality` | 设置预定义人格 |
| `/verbose` | 循环切换工具进度显示：关闭 → 新 → 全部 → 详细。可通过配置为[消息平台启用](#notes)。 |
| `/fast [normal\|fast\|status]` | 切换快速模式 —— OpenAI Priority Processing / Anthropic Fast Mode。选项：`normal`、`fast`、`status`。 |
| `/reasoning` | 管理推理投入和显示（用法：`/reasoning [level\|show\|hide]`） |
| `/skin` | 显示或更改显示外观/主题 |
| `/statusbar`（别名：`/sb`） | 切换上下文/模型状态栏的显示开关 |
| `/voice [on\|off\|tts\|status]` | 切换 CLI 语音模式和语音播放。录音使用 `voice.record_key`（默认：`Ctrl+B`）。 |
| `/yolo` | 切换 YOLO 模式 —— 跳过所有危险命令确认提示。 |
| `/footer [on\|off\|status]` | 切换网关运行时元数据页脚在最终回复中的显示（显示模型、工具计数、计时）。 |
| `/busy [queue\|steer\|interrupt\|status]` | CLI 专用：控制在 Hermes 工作时按 Enter 键的行为 —— 将新消息加入队列、中途转向、或立即中断。 |
| `/indicator [kaomoji\|emoji\|unicode\|ascii]` | CLI 专用：选择 TUI 忙碌指示器样式。 |

### 工具和技能

| 命令 | 描述 |
|---------|-------------|
| `/tools [list\|disable\|enable] [名称...]` | 管理工具：列出可用工具，或在当前会话中禁用/启用特定工具。禁用工具会将其从 agent 的工具集中移除并触发会话重置。 |
| `/toolsets` | 列出可用工具集 |
| `/browser [connect\|disconnect\|status]` | 管理本地 Chrome CDP 连接。`connect` 将浏览器工具附加到运行中的 Chrome 实例（默认：`ws://localhost:9222`）。`disconnect` 分离。`status` 显示当前连接。如果未检测到调试器会自动启动 Chrome。 |
| `/skills` | 从在线注册表搜索、安装、检查或管理技能 |
| `/cron` | 管理计划任务（列出、添加/创建、编辑、暂停、恢复、运行、删除） |
| `/curator` | 后台技能维护 —— `status`、`run`、`pin`、`archive`。详见 [Curator](/docs/user-guide/features/curator)。 |
| `/kanban <操作>` | 在不离开聊天的情况下驱动多配置文件、多项目协作看板。完整的 `hermes kanban` 界面可用：`/kanban list`、`/kanban show t_abc`、`/kanban create "标题" --assignee X`、`/kanban comment t_abc "文本"`、`/kanban unblock t_abc`、`/kanban dispatch` 等。多看板支持：`/kanban boards list`、`/kanban boards create <slug>`、`/kanban boards switch <slug>`、`/kanban --board <slug> <操作>`。详见 [Kanban 斜杠命令](/docs/user-guide/features/kanban#kanban-slash-command)。 |
| `/reload-mcp`（别名：`/reload_mcp`） | 从 config.yaml 重新加载 MCP 服务器 |
| `/reload` | 将 `.env` 变量重新加载到运行中的会话（无需重启即可获取新 API 密钥） |
| `/plugins` | 列出已安装的插件及其状态 |

### 信息

| 命令 | 描述 |
|---------|-------------|
| `/help` | 显示此帮助消息 |
| `/usage` | 显示 token 使用量、成本明细、会话时长，以及 —— 当活跃提供商可用时 —— **账户限制**部分，包含从提供商 API 实时获取的剩余配额/积分/计划使用量。 |
| `/insights` | 显示使用洞察和分析（过去 30 天） |
| `/platforms`（别名：`/gateway`） | 显示网关/消息平台状态 |
| `/paste` | 附加剪贴板图像 |
| `/copy [数字]` | 将上一条助手回复复制到剪贴板（或带数字复制倒数第 N 条）。仅 CLI。 |
| `/image <路径>` | 为下一条提示词附加本地图像文件。 |
| `/debug` | 上传调试报告（系统信息 + 日志）并获取可分享链接。消息平台也可用。 |
| `/profile` | 显示活跃配置文件名和主目录 |
| `/gquota` | 显示 Google Gemini Code Assist 配额使用情况及进度条（仅在 `google-gemini-cli` 提供商活跃时可用）。 |

### 退出

| 命令 | 描述 |
|---------|-------------|
| `/quit` | 退出 CLI（也可：`/exit`）。 |

### 动态 CLI 斜杠命令

| 命令 | 描述 |
|---------|-------------|
| `/<技能名>` | 将任何已安装的技能作为按需命令加载。示例：`/gif-search`、`/github-pr-workflow`、`/excalidraw`。 |
| `/skills ...` | 从注册表和官方可选技能目录搜索、浏览、检查、安装、审计、发布和配置技能。 |

### 快速命令

用户定义的快速命令将短斜杠命令映射到 shell 命令或其他斜杠命令。在 `~/.hermes/config.yaml` 中配置：

```yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  deploy:
    type: exec
    command: scripts/deploy.sh
  inbox:
    type: alias
    target: /gmail unread
```

然后在 CLI 或消息平台中输入 `/status`、`/deploy` 或 `/inbox`。快速命令在分发时解析，可能不会出现在每个内置自动补全/帮助表中。

不支持仅字符串提示词快捷方式作为快速命令。将较长的可复用提示词放入技能，或使用 `type: alias` 指向现有斜杠命令。

### 自定义模型别名

为您常用的模型定义自己的短名称，然后可以在 CLI 或任何消息平台中通过 `/model <别名>` 调用。别名在两者中功能相同，无论是会话级别（默认）还是 `--global` 切换。

支持两种配置格式：

**完整格式** —— 固定确切模型、提供商，可选基础 URL。放在 `~/.hermes/config.yaml` 中：

```yaml
model_aliases:
  fav:
    model: claude-sonnet-4.6
    provider: anthropic
  grok:
    model: grok-4
    provider: x-ai
  ollama-qwen:
    model: qwen3-coder:30b
    provider: custom
    base_url: http://localhost:11434/v1
```

**简短格式** —— `provider/model` 在一个字符串中。从 shell 设置，无需编辑 YAML：

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

然后在聊天中：

```
/model fav            # 仅会话级别
/model grok --global  # 同时将 current-model 更改持久化到 config.yaml
```

用户别名优先于内置短名称，因此将别名命名为 `sonnet`、`kimi`、`opus` 等会覆盖内置名称。别名名称不区分大小写。

### 别名解析

命令支持前缀匹配：输入 `/h` 解析为 `/help`，`/mod` 解析为 `/model`。当前缀模糊（匹配多个命令）时，按注册表顺序第一个匹配优先。全命令名称和注册别名始终优先于前缀匹配。

## 消息斜杠命令

消息网关在 Telegram、Discord、Slack、WhatsApp、Signal、Email、Home Assistant 和 Teams 聊天中支持以下内置命令：

| 命令 | 描述 |
|---------|-------------|
| `/new` | 开始新对话。 |
| `/reset` | 重置对话历史。 |
| `/status` | 显示会话信息。 |
| `/stop` | 终止所有运行中的后台进程并中断运行中的 agent。 |
| `/model [provider:model]` | 显示或更改模型。支持提供商切换（`/model zai:glm-5`）、自定义端点（`/model custom:model`）、命名自定义提供商（`/model custom:local:qwen`）、自动检测（`/model custom`）以及用户定义别名（`/model fav`、`/model grok` —— 参见[自定义模型别名](#custom-model-aliases)）。使用 `--global` 将更改持久化到 config.yaml。**注意：** `/model` 只能切换已配置的提供商。要添加新提供商或设置 API 密钥，请在聊天会话外从终端使用 `hermes model`。 |
| `/personality [名称]` | 为会话设置人格覆盖。 |
| `/fast [normal\|fast\|status]` | 切换快速模式 —— OpenAI Priority Processing / Anthropic Fast Mode。 |
| `/retry` | 重试上一条消息。 |
| `/undo` | 移除上一条交换内容。 |
| `/sethome`（别名：`/set-home`） | 将当前聊天标记为投递的平台主页频道。 |
| `/compress [聚焦主题]` | 手动压缩对话上下文。可选的聚焦主题缩小总结保留的范围。 |
| `/topic [off\|help\|session-id]` | **仅 Telegram 私信。** 管理用户管理的多会话话题模式。`/topic` 启用或显示状态；`/topic off` 禁用并清除绑定；`/topic help` 显示用法；在话题内输入 `/topic <session-id>` 可恢复之前的会话。详见[多会话私信模式](/docs/user-guide/messaging/telegram#multi-session-dm-mode-topic)。 |
| `/title [名称]` | 设置或显示会话标题。 |
| `/resume [名称]` | 恢复之前命名的会话。 |
| `/usage` | 显示 token 使用量、预估成本明细（输入/输出）、上下文窗口状态、会话时长，以及 —— 当活跃提供商可用时 —— **账户限制**部分，包含从提供商 API 实时获取的剩余配额/积分。 |
| `/insights [天数]` | 显示使用分析。 |
| `/reasoning [level\|show\|hide]` | 更改推理投入或切换推理显示。 |
| `/voice [on\|off\|tts\|join\|channel\|leave\|status]` | 控制聊天中的语音回复。`join`/`channel`/`leave` 管理 Discord 语音频道模式。 |
| `/rollback [数字]` | 列出或恢复文件系统检查点。 |
| `/background <提示词>` | 在独立后台会话中运行提示词。结果在任务完成时传回同一聊天。详见[消息后台会话](/docs/user-guide/messaging/#background-sessions)。 |
| `/queue <提示词>`（别名：`/q`） | 将提示词加入下一轮队列而不中断当前轮次。 |
| `/steer <提示词>` | 在下一次工具调用后注入消息而不中断 —— 模型在下一轮迭代中获取它，而不是作为新轮次。 |
| `/goal <文本>` | 设置一个 Hermes 跨轮次努力达成的持久目标 —— 这是我们对 Ralph 循环的实现。评判模型在每轮后检查；如果未完成，Hermes 自动继续直到完成、您暂停/清除它、或达到轮次预算（默认 20）。子命令：`/goal status`、`/goal pause`、`/goal resume`、`/goal clear`。安全地用于状态/暂停/清除中途运行；设置新目标需要先 `/stop`。详见[持久目标](/docs/user-guide/features/goals)。 |
| `/footer [on\|off\|status]` | 切换最终回复中的运行时元数据页脚（显示模型、工具计数、计时）。 |
| `/curator [status\|run\|pin\|archive]` | 后台技能维护控制。 |
| `/kanban <操作>` | 从聊天驱动多配置文件、多项目协作看板 —— 与 CLI 相同的参数界面。绕过运行中 agent 保护，因此 `/kanban unblock t_abc`、`/kanban comment t_abc "..."`、`/kanban list --mine`、`/kanban boards switch <slug>` 等可在轮次中途工作。`/kanban create ...` 自动订阅发起聊天到新任务的终端事件。详见 [Kanban 斜杠命令](/docs/user-guide/features/kanban#kanban-slash-command)。 |
| `/reload-mcp`（别名：`/reload_mcp`） | 从配置重新加载 MCP 服务器。 |
| `/yolo` | 切换 YOLO 模式 —— 跳过所有危险命令确认提示。 |
| `/commands [页码]` | 浏览所有命令和技能（分页）。 |
| `/approve [session\|always]` | 批准并执行待处理的危险命令。`session` 仅批准本次会话；`always` 添加到永久白名单。 |
| `/deny` | 拒绝待处理的危险命令。 |
| `/update` | 将 Hermes Agent 更新到最新版本。 |
| `/restart` | 在排空前台活动运行后优雅地重启网关。网关重新上线后，会向请求者的聊天/线程发送确认。 |
| `/debug` | 上传调试报告（系统信息 + 日志）并获取可分享链接。 |
| `/help` | 显示消息帮助。 |
| `/<技能名>` | 按名称调用任何已安装的技能。 |

## 备注

- `/skin`、`/snapshot`、`/gquota`、`/reload`、`/tools`、`/toolsets`、`/browser`、`/config`、`/cron`、`/skills`、`/platforms`、`/paste`、`/image`、`/statusbar`、`/plugins`、`/busy`、`/indicator`、`/redraw`、`/clear`、`/history`、`/save`、`/copy` 和 `/quit` 是**仅 CLI**命令。
- `/verbose` **默认仅 CLI**，但可通过在 `config.yaml` 中设置 `display.tool_progress_command: true` 为消息平台启用。启用后会循环切换 `display.tool_progress` 模式并保存到配置。
- `/sethome`、`/update`、`/restart`、`/approve`、`/deny`、`/topic` 和 `/commands` 是**仅消息**命令。
- `/status`、`/background`、`/queue`、`/steer`、`/voice`、`/reload-mcp`、`/rollback`、`/debug`、`/fast`、`/footer`、`/curator`、`/kanban` 和 `/yolo` 在 **CLI 和消息网关**中均可使用。
- `/voice join`、`/voice channel` 和 `/voice leave` 仅在 Discord 上有意义。
