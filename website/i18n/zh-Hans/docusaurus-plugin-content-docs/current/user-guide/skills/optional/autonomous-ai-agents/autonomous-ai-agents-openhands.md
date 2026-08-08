---
title: "OpenHands — 将编码任务委托给 OpenHands CLI（模型无关、LiteLLM）"
sidebar_label: "OpenHands"
description: "将编码任务委托给 OpenHands CLI（模型无关、LiteLLM）"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Openhands

将编码任务委托给 OpenHands CLI（模型无关、LiteLLM）。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/autonomous-ai-agents/openhands` 安装 |
| 路径 | `optional-skills/autonomous-ai-agents/openhands` |
| 版本 | `0.1.0` |
| 作者 | Tim Koepsel (xzessmedia)、Hermes Agent |
| 许可证 | MIT |
| 平台 | linux、macos |
| 标签 | `Coding-Agent`、`OpenHands`、`Model-Agnostic`、`LiteLLM` |
| 相关技能 | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code)、[`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex)、[`opencode`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode)、[`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在此技能触发时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# OpenHands CLI

通过 `terminal` 工具将编码任务委托给 [OpenHands CLI](https://github.com/All-Hands-AI/OpenHands)。OpenHands 是模型无关的：任何 LiteLLM 支持的提供商（OpenAI、Anthropic、OpenRouter、DeepSeek、Ollama、vLLM 等）。

此技能是批处理/一次性委托的无头模式包装器。不从 Hermes 使用交互式文本 UI。

## 何时使用

- 用户希望将编码任务专门委托给 OpenHands。
- 用户想要在非 Anthropic / 非 OpenAI 提供商上运行的编码代理（DeepSeek、Qwen、Ollama、vLLM、Nous 等）——兄弟技能 `claude-code` 和 `codex` 绑定到单一供应商。
- 工作空间内的多步骤文件编辑 + shell 命令。

对于 Claude 原生，优先使用 `claude-code`。对于 OpenAI 原生，优先使用 `codex`。对于 Hermes 原生子代理，使用 `delegate_task`。

## 前提条件

1. 安装上游（需要 Python 3.12+ 和 `uv`）：

   ```
   terminal(command="uv tool install openhands --python 3.12")
   ```

   验证：`openhands --version`（撰写时当前为 `OpenHands CLI 1.16.0` / `SDK v1.21.0`）。

2. 选择模型并为 `--override-with-envs` 设置环境变量：

   ```
   export LLM_MODEL=openrouter/openai/gpt-4o-mini       # 或任何 LiteLLM slug
   export LLM_API_KEY=$OPENROUTER_API_KEY
   export LLM_BASE_URL=https://openrouter.ai/api/v1     # 原生 OpenAI 可省略
   ```

   `LLM_MODEL` 使用 LiteLLM 的完整 slug。当提供商是 OpenRouter 时 slug 是双重前缀：`openrouter/<vendor>/<model>`（例如 `openrouter/anthropic/claude-sonnet-4.5`）。原生 Anthropic：`anthropic/claude-sonnet-4-5`。原生 OpenAI：`openai/gpt-4o-mini`。

3. 抑制启动横幅，使 JSON 输出前面没有 ASCII 艺术：

   ```
   export OPENHANDS_SUPPRESS_BANNER=1
   ```

## 如何运行

始终通过 `terminal` 工具调用。始终传递 `--headless --json --override-with-envs --exit-without-confirmation` 用于自动化。

### 一次性任务

```
terminal(
  command="OPENHANDS_SUPPRESS_BANNER=1 LLM_MODEL=openrouter/openai/gpt-4o-mini LLM_API_KEY=$OPENROUTER_API_KEY LLM_BASE_URL=https://openrouter.ai/api/v1 openhands --headless --json --override-with-envs --exit-without-confirmation -t 'Add error handling to all API calls in src/'",
  workdir="/path/to/project",
  timeout=600
)
```

### 长任务后台运行

```
terminal(command="<same as above>", workdir="/path/to/project", background=true, notify_on_complete=true)
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")
```

### 恢复之前的对话

OpenHands 在每次运行结束时打印 `Conversation ID: <32-hex>` 和 `Hint: openhands --resume <dashed-uuid>` 行。使用破折号形式恢复：

```
terminal(
  command="OPENHANDS_SUPPRESS_BANNER=1 LLM_MODEL=... openhands --headless --json --override-with-envs --exit-without-confirmation --resume <dashed-uuid> -t 'Now fix the bug you found'",
  workdir="/path/to/project"
)
```

## 标志列表

已对照 `openhands --help`（CLI 1.16.0）验证。不在此表中的不是标志——通过环境变量或设置文件传递。

| 标志 | 效果 |
|------|--------|
| `--headless` | 无 UI，需要 `-t` 或 `-f`。自动批准所有操作（此模式下没有 `--llm-approve`）。 |
| `--json` | JSONL 事件流（需要 `--headless`）。 |
| `-t TEXT` | 任务提示。 |
| `-f PATH` | 从文件读取任务。 |
| `--resume [ID]` | 恢复对话。无 ID → 列出最近的。 |
| `--last` | 恢复最近的（配合 `--resume`）。 |
| `--override-with-envs` | 应用 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` 环境变量。没有此选项，OpenHands 使用 `~/.openhands/settings.json` 并忽略环境变量。 |
| `--exit-without-confirmation` | 不显示"确定要退出吗"对话框。 |
| `--always-approve` / `--yolo` | 自动批准每个操作（`--headless` 中的默认）。 |
| `--llm-approve` | 基于 LLM 的安全门控（仅交互式——在 headless 中不工作）。 |
| `--version` / `-v` | 打印版本并退出。 |

**没有 `--model`、`--max-iterations`、`--workspace`、`--sandbox`、`--sandbox-type` 标志。** 模型是 `LLM_MODEL`。工作空间是你传递给 `terminal` 工具的 `workdir`。沙箱/运行时是 `RUNTIME` 和 `SANDBOX_VOLUMES` 环境变量。

## JSON 事件模式

使用 `--json --headless` 时，OpenHands 输出 JSONL——每行一个 JSON 对象，加上少量非 JSON 状态行（`Initializing agent...`、`Agent is working`、`Agent finished`、最终摘要框、`Goodbye!`、`Conversation ID:`、`Hint:`）。过滤以 `{` 开头的行。

顶层 `kind` 字段区分事件：

- `MessageEvent` — 用户/代理文本回合。`source` 是 `user` 或 `agent`。
- `ActionEvent` — 代理选择了工具。读取 `tool_name`（`file_editor`、`terminal`、`finish`）和 `action.kind`（`FileEditorAction`、`TerminalAction`、`FinishAction`）。
- `ObservationEvent` — 工具结果。`observation.is_error` 是成功标志。`source` 是 `environment`。
- `ActionEvent` 内的 `FinishAction` 在 `action.message` 中携带代理的最终消息。

cli 首先打印 LiteLLM/Authlib 的所有 stderr — 见陷阱。仅解析 stdout，逐行忽略不以 `{` 开头的行。

## 陷阱

- **LiteLLM 在每次调用时的警告。** CLI 向 stderr 打印 `bedrock-runtime` 和 `sagemaker-runtime` 警告，因为未安装 `botocore`。加上 Authlib 弃用警告。这些是噪音，不是失败。将 stderr 管道到 `/dev/null` 或在显示给用户前过滤。
- **横幅垃圾。** 没有 `OPENHANDS_SUPPRESS_BANNER=1`，每次运行都以多行 `+--+` ASCII 框开始，宣传 SDK。始终导出它。
- **`--override-with-envs` 在自动化中是必需的。** 没有它，OpenHands 忽略 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` 并回退到 `~/.openhands/settings.json`。在全新安装中此文件不存在，CLI 会挂起等待首次运行设置。
- **模型 slug 是 LiteLLM 的，不是提供商的。** `openrouter/openai/gpt-4o-mini` 有效；在指向 OpenRouter 时 `openai/gpt-4o-mini` 无效。`anthropic/claude-sonnet-4-5`（连字符）是原生 Anthropic；`openrouter/anthropic/claude-sonnet-4.5`（点）是通过 OpenRouter。搞错 → 神秘的 LiteLLM 400。
- **`pip install openhands-ai` 是错误的包。** 那是旧版 V0 SDK。新 CLI 是 `uv tool install openhands --python 3.12`。没有维护的 conda 包。
- **恢复 ID 格式很挑剔。** CLI 以 `Conversation ID: f46573d9cfdb45e492ca189bde40019b`（无破折号）结束，然后是 `Hint: openhands --resume f46573d9-cfdb-45e4-92ca-189bde40019b`（有破折号）。使用破折号形式。
- **Headless 忽略 `--llm-approve`。** 如果你传递它，会得到 argparse 错误。Headless 模式硬编码 always-approve。
- **上游不支持 Windows。** OpenHands 文档要求 Windows 上使用 WSL。此技能相应地限制为 `[linux, macos]`。
- **`~/.openhands/conversations/<id>/` 会累积。** 每次运行持久化一个轨迹。如果运行批处理则清理它。
- **重量级安装（约 200 个包）。** 使用 `uv tool install`（隔离的 venv）以避免与活动项目的依赖冲突。

## 验证

```
terminal(
  command="OPENHANDS_SUPPRESS_BANNER=1 LLM_MODEL=openrouter/openai/gpt-4o-mini LLM_API_KEY=$OPENROUTER_API_KEY LLM_BASE_URL=https://openrouter.ai/api/v1 openhands --headless --json --override-with-envs --exit-without-confirmation -t 'Print the string OPENHANDS_OK to stdout via the terminal tool.'",
  workdir="/tmp",
  timeout=120
)
```

如果 JSONL 流以 `FinishAction` 结束且其 `action.message` 提到 `OPENHANDS_OK`，则安装工作正常。

## 相关

- [OpenHands GitHub](https://github.com/All-Hands-AI/OpenHands)
- [OpenHands CLI 命令参考](https://docs.openhands.dev/openhands/usage/cli/command-reference)
- 兄弟技能：`claude-code`（仅 Anthropic）、`codex`（仅 OpenAI）、`opencode`（通过 OpenCode 的多提供商）、`hermes-agent`（通过 `delegate_task` 的 Hermes 子代理）。
