---
sidebar_position: 6
title: "Event Hooks"
description: "在关键生命周期节点运行自定义代码——记录活动、发送告警、推送到 webhook"
---

# Event Hooks

Hermes 有四套 hook 系统，可在关键生命周期节点运行自定义代码：

| 系统 | 注册方式 | 运行环境 | 使用场景 |
|------|---------|---------|---------|
| **[Gateway hooks](#gateway-event-hooks)** | `~/.hermes/hooks/` 下的 `HOOK.yaml` + `handler.py` | 仅 Gateway | 日志、告警、webhook |
| **[Plugin hooks](#plugin-hooks)** | [插件](/user-guide/features/plugins)中的 `ctx.register_hook()` | CLI + Gateway | 工具拦截、指标采集、护栏 |
| **[Shell hooks](#shell-hooks)** | `~/.hermes/config.yaml` 中 `hooks:` 块指向的 shell 脚本 | CLI + Gateway | 用于阻断、自动格式化、上下文注入的即插即用脚本 |
| **[Outbound webhooks](#outbound-webhooks)** | `~/.hermes/config.yaml` 中的 `hooks.outbound:` 列表 | CLI + Gateway | 将签名后的生命周期事件推送到外部 HTTP endpoint |

Hook 回调错误会被隔离并记录，不会导致 agent 崩溃。但 hook 并非全是被动观察者：指令/控制类 hook 可改变流程，transform 可替换内容，shell `pre_tool_call` 还能阻断或在失败时关闭执行。

## Gateway Event Hooks

Gateway hooks 在 gateway 运行期间（Telegram、Discord、Slack、WhatsApp、Teams）自动触发，不会阻塞主 agent 管道。

### 创建 Hook

每个 hook 是 `~/.hermes/hooks/` 下的一个目录，包含两个文件：

```text
~/.hermes/hooks/
└── my-hook/
    ├── HOOK.yaml      # 声明要监听的事件
    └── handler.py     # Python 处理函数
```

#### HOOK.yaml

```yaml
name: my-hook
description: Log all agent activity to a file
events:
  - agent:start
  - agent:end
  - agent:step
```

`events` 列表决定哪些事件会触发你的处理器。可以订阅任意事件组合，包括 `command:*` 这样的通配符。

#### handler.py

```python
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path.home() / ".hermes" / "hooks" / "my-hook" / "activity.log"

async def handle(event_type: str, context: dict):
    """Called for each subscribed event. Must be named 'handle'."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        **context,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

**处理器规则：**
- 必须命名为 `handle`
- 接收 `event_type`（字符串）和 `context`（字典）
- 可以是 `async def` 或普通 `def`——两者均可
- 错误会被捕获并记录，不会导致 agent 崩溃

### 可用事件

| 事件 | 触发时机 | Context 键 |
|------|---------|-----------|
| `gateway:startup` | Gateway 进程启动 | `platforms`（活跃平台名称列表） |
| `session:start` | 新消息会话创建 | `platform`、`user_id`、`session_id`、`session_key` |
| `session:end` | 会话结束（重置前） | `platform`、`user_id`、`session_key` |
| `session:reset` | 用户执行 `/new` 或 `/reset` | `platform`、`user_id`、`session_key` |
| `agent:start` | Agent 开始处理消息 | `platform`、`user_id`、`session_id`、`message` |
| `agent:step` | 工具调用循环的每次迭代 | `platform`、`user_id`、`session_id`、`iteration`、`tool_names` |
| `agent:end` | Agent 完成处理 | `platform`、`user_id`、`session_id`、`message`、`response` |
| `command:*` | 任意斜杠命令执行 | `platform`、`user_id`、`command`、`args` |

#### 通配符匹配

注册了 `command:*` 的处理器会在任何 `command:` 事件（`command:model`、`command:reset` 等）触发时执行。通过单个订阅即可监控所有斜杠命令。

### 示例

#### Telegram 长任务告警

当 agent 执行超过 10 步时向自己发送消息：

```yaml
# ~/.hermes/hooks/long-task-alert/HOOK.yaml
name: long-task-alert
description: Alert when agent is taking many steps
events:
  - agent:step
```

```python
# ~/.hermes/hooks/long-task-alert/handler.py
import os
import httpx

THRESHOLD = 10
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_HOME_CHANNEL")

async def handle(event_type: str, context: dict):
    iteration = context.get("iteration", 0)
    if iteration == THRESHOLD and BOT_TOKEN and CHAT_ID:
        tools = ", ".join(context.get("tool_names", []))
        text = f"⚠️ Agent has been running for {iteration} steps. Last tools: {tools}"
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": text},
            )
```

#### 命令使用日志记录器

追踪哪些斜杠命令被使用：

```yaml
# ~/.hermes/hooks/command-logger/HOOK.yaml
name: command-logger
description: Log slash command usage
events:
  - command:*
```

```python
# ~/.hermes/hooks/command-logger/handler.py
import json
from datetime import datetime
from pathlib import Path

LOG = Path.home() / ".hermes" / "logs" / "command_usage.jsonl"

def handle(event_type: str, context: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now().isoformat(),
        "command": context.get("command"),
        "args": context.get("args"),
        "platform": context.get("platform"),
        "user": context.get("user_id"),
    }
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

#### 会话开始 Webhook

新会话时 POST 到外部服务：

```yaml
# ~/.hermes/hooks/session-webhook/HOOK.yaml
name: session-webhook
description: Notify external service on new sessions
events:
  - session:start
  - session:reset
```

```python
# ~/.hermes/hooks/session-webhook/handler.py
import httpx

WEBHOOK_URL = "https://your-service.example.com/hermes-events"

async def handle(event_type: str, context: dict):
    async with httpx.AsyncClient() as client:
        await client.post(WEBHOOK_URL, json={
            "event": event_type,
            **context,
        }, timeout=5)
```

### 教程：BOOT.md——每次 Gateway 启动时运行启动检查清单

这是社区中流行的一种模式：在 `~/.hermes/BOOT.md` 放置一个 Markdown 检查清单，让 agent 在每次 gateway 启动时执行一次。适用于"每次启动时检查隔夜 cron 失败情况，若有失败则在 Discord 上通知我"，或"汇总过去 24 小时的 deploy.log 并发布到 Slack #ops"等场景。

本教程展示如何以用户自定义 hook 的方式自行构建。Hermes 不内置 BOOT.md hook——你可以精确配置自己想要的行为。

#### 我们要构建什么

1. 在 `~/.hermes/BOOT.md` 放置一个包含自然语言启动指令的文件。
2. 一个监听 `gateway:startup` 的 gateway hook，它会生成一个一次性 agent，使用 gateway 已解析的模型和凭据，执行 BOOT.md 中的指令。
3. 一个 `[SILENT]` 约定，让 agent 在没有内容需要汇报时选择不发送消息。

#### 第一步：编写检查清单

创建 `~/.hermes/BOOT.md`。像给人类助手下达指令一样编写：

```markdown
# Startup Checklist

1. Run `hermes cron list` and check if any scheduled jobs failed overnight.
2. If any failed, send a summary to Discord #ops using the `send_message` tool.
3. Check if `/opt/app/deploy.log` has any ERROR lines from the last 24 hours. If yes, summarize them and include in the same Discord message.
4. If nothing went wrong, reply with only `[SILENT]` so no message is sent.
```

Agent 将此内容作为 prompt（提示词）的一部分，因此任何可以用自然语言描述的内容都可以——工具调用、shell 命令、发送消息、汇总文件。

#### 第二步：创建 hook

```text
~/.hermes/hooks/boot-md/
├── HOOK.yaml
└── handler.py
```

**`~/.hermes/hooks/boot-md/HOOK.yaml`**

```yaml
name: boot-md
description: Run ~/.hermes/BOOT.md on gateway startup
events:
  - gateway:startup
```

**`~/.hermes/hooks/boot-md/handler.py`**

```python
"""Run ~/.hermes/BOOT.md on every gateway startup."""

import logging
import threading
from pathlib import Path

logger = logging.getLogger("hooks.boot-md")

BOOT_FILE = Path.home() / ".hermes" / "BOOT.md"


def _build_prompt(content: str) -> str:
    return (
        "You are running a startup boot checklist. Follow the instructions "
        "below exactly.\n\n"
        "---\n"
        f"{content}\n"
        "---\n\n"
        "Execute each instruction. Use the send_message tool to deliver any "
        "messages to platforms like Discord or Slack.\n"
        "If nothing needs attention and there is nothing to report, reply "
        "with ONLY: [SILENT]"
    )


def _run_boot_agent(content: str) -> None:
    """Spawn a one-shot agent and execute the checklist.

    Uses the gateway's resolved model and runtime credentials so this works
    against custom endpoints, aggregators, and OAuth-based providers alike.
    """
    try:
        from gateway.run import _resolve_gateway_model, _resolve_runtime_agent_kwargs
        from run_agent import AIAgent

        agent = AIAgent(
            model=_resolve_gateway_model(),
            **_resolve_runtime_agent_kwargs(),
            platform="gateway",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=20,
        )
        result = agent.run_conversation(_build_prompt(content))
        response = result.get("final_response", "")
        if response and "[SILENT]" not in response:
            logger.info("boot-md completed: %s", response[:200])
        else:
            logger.info("boot-md completed (nothing to report)")
    except Exception as e:
        logger.error("boot-md agent failed: %s", e)


async def handle(event_type: str, context: dict) -> None:
    if not BOOT_FILE.exists():
        return
    content = BOOT_FILE.read_text(encoding="utf-8").strip()
    if not content:
        return

    logger.info("Running BOOT.md (%d chars)", len(content))

    # Background thread so gateway startup isn't blocked on a full agent turn.
    thread = threading.Thread(
        target=_run_boot_agent,
        args=(content,),
        name="boot-md",
        daemon=True,
    )
    thread.start()
```

两个关键行：

- `_resolve_gateway_model()` 读取 gateway 当前配置的模型。
- `_resolve_runtime_agent_kwargs()` 以与普通 gateway 轮次相同的方式解析 provider 凭据——包括 API 密钥、base URL、OAuth token 和凭据池。

若不使用这两行，裸 `AIAgent()` 会回退到内置默认值，并在任何非默认端点上返回 401。

#### 第三步：测试

重启 gateway：

```bash
hermes gateway restart
```

查看日志：

```bash
hermes logs --follow --level INFO | grep boot-md
```

你应该看到 `Running BOOT.md (N chars)`，随后是 `boot-md completed: ...`（agent 执行内容的摘要）或 `boot-md completed (nothing to report)`（agent 回复了 `[SILENT]`）。

删除 `~/.hermes/BOOT.md` 即可禁用检查清单——hook 保持加载状态，但在文件不存在时会静默跳过。

#### 扩展此模式

- **感知调度的检查清单：** 在 BOOT.md 指令中基于 `datetime.now().weekday()` 进行判断（"如果是周一，还需检查每周部署日志"）。指令是自由格式文本，agent 能推理的内容都可以使用。
- **多个检查清单：** 将 hook 指向不同文件（`STARTUP.md`、`MORNING.md` 等），并为每个文件注册独立的 hook 目录。
- **非 agent 变体：** 如果不需要完整的 agent 循环，完全跳过 `AIAgent`，直接通过 `httpx` 在处理器中发送固定通知。更轻量、更快速，且无 provider 依赖。

#### 为什么这不是内置功能

Hermes 早期版本将此作为内置 hook 发布，每次 gateway 启动时都会静默生成一个使用裸默认值的 agent。这让使用自定义端点的用户感到意外，也让不知道它在运行的用户无从察觉。将其作为文档化模式保留——由你在 hooks 目录中构建——意味着你能清楚地看到它的行为，并通过编写文件来选择启用。

### 工作原理

1. Gateway 启动时，`HookRegistry.discover_and_load()` 扫描 `~/.hermes/hooks/`
2. 每个包含 `HOOK.yaml` + `handler.py` 的子目录都会被动态加载
3. 处理器按其声明的事件注册
4. 在每个生命周期节点，`hooks.emit()` 触发所有匹配的处理器
5. 任何处理器中的错误都会被捕获并记录——损坏的 hook 永远不会导致 agent 崩溃

:::info
Gateway hooks 仅在 **gateway**（Telegram、Discord、Slack、WhatsApp、Teams）中触发。CLI 不加载 gateway hooks。如需在所有环境中生效的 hook，请使用 [plugin hooks](#plugin-hooks)。
:::

## Plugin Hooks

[插件](/user-guide/features/plugins)可以注册在 **CLI 和 gateway** 会话中均会触发的 hook。这些 hook 通过插件 `register()` 函数中的 `ctx.register_hook()` 以编程方式注册。

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", my_tool_observer)
    ctx.register_hook("post_tool_call", my_tool_logger)
    ctx.register_hook("pre_llm_call", my_memory_callback)
    ctx.register_hook("post_llm_call", my_sync_callback)
    ctx.register_hook("on_session_start", my_init_callback)
    ctx.register_hook("on_session_end", my_cleanup_callback)
```

**所有 hook 的通用规则：**

- 回调接收**关键字参数**；为保持向前兼容，请始终接受 `**kwargs`。
- 回调异常会被记录并跳过，后续回调仍会继续。
- 下表分类仅描述当前行为：**观察者**忽略返回值，**Transform** 接受第一个有效字符串替换，**指令/控制**消费已说明的返回结构。Plugin middleware 是独立的 registry/surface，不属于另一类 hook。
- `turn_id`、`api_request_id`、`task_id`、`session_id`、`api_call_count` 等关联字段因 hook 而异，可能不存在；应将这些 ID 视为 opaque 值。
- 运行时事件名以 `hermes_cli.plugins.VALID_HOOKS` 为准。`hermes hooks list` 只列出已配置的 shell/outbound hook，并非可用事件目录；只有 `hermes hooks test <event>` 收到无效事件时才会打印有效集合。

### 已发布的 plugin-hook 目录

下表列出每个 call site 实际传入的事件专属字段。为保持向后兼容，`PluginManager` 还会向每个 plugin-hook 回调加入 `telemetry_schema_version="hermes.observer.v1"`。这个旧版 envelope 标记并不表示所有 hook payload 共用同一套语义 schema；新的版本化 contract 应归属于具体事件或 capability family。

| Hook | 类别 | 精确时机与返回行为 | 显式 payload 字段 | 隐私/敏感性 |
|---|---|---|---|---|
| `pre_tool_call` | 指令/控制 | 执行前一次；第一个有效 `block` 或 `approve` 指令生效。 | `tool_name`, `args`, `task_id`, `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `middleware_trace` | 原始参数可能含用户内容、路径、命令或 secret。 |
| `post_tool_call` | 观察者 | 阻断、错误或成功结果产生后；忽略返回值。 | `tool_name`, `args`, `result`, `task_id`, `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `duration_ms`, `status`, `error_type`, `error_message`, `middleware_trace` | 结果/错误文本可能含任意工具或用户内容及 secret。 |
| `transform_tool_result` | Transform | `post_tool_call` 后、写入会话前；第一个字符串替换结果。 | `tool_name`, `args`, `result`, `task_id`, `session_id`, `tool_call_id`, `turn_id`, `api_request_id`, `duration_ms`, `status`, `error_type`, `error_message` | 暴露完整的 model-bound 结果和参数。 |
| `transform_terminal_output` | Transform | 前台进程输出完成有界捕获后、最终 output limit 前；第一个字符串替换输出。 | `command`, `output`, `returncode`, `task_id`, `env_type` | 命令/输出可能含凭据。 |
| `pre_llm_call` | 指令/控制 | 每轮 loop 前一次；所有有效字符串或 `{"context": ...}` 会拼接并注入用户消息。 | `session_id`, `task_id`, `turn_id`, `user_message`, `conversation_history`, `is_first_turn`, `model`, `platform`, `parent_session_id`, `sender_id` | 完整用户消息和会话历史。 |
| `post_llm_call` | 观察者 | 成功且未中断的轮次 finalize 时；忽略返回值。 | `session_id`, `task_id`, `turn_id`, `user_message`, `assistant_response`, `conversation_history`, `model`, `platform` | 完整 prompt、response 和 history。 |
| `transform_llm_output` | Transform | `post_llm_call` 和最终交付前；第一个非空字符串替换 response。 | `response_text`, `session_id`, `model`, `platform` | 完整最终 assistant 文本。 |
| `pre_verify` | 指令/控制 | 有界的代码编辑 verify gate；第一个有效 continue/block-stop 指令让轮次继续。 | `session_id`, `platform`, `model`, `coding`, `attempt`, `final_response`, `changed_paths` | 草稿 response 和变更路径。 |
| `pre_api_request` | 观察者 | 每次 provider attempt 发请求前；忽略返回值。 | `task_id`, `turn_id`, `api_request_id`, `session_id`, `user_message`, `conversation_history`, `platform`, `model`, `provider`, `base_url`, `api_mode`, `api_call_count`, `retry_count`, `request_messages`, `message_count`, `tool_count`, `approx_input_tokens`, `request_char_count`, `max_tokens`, `started_at`, `middleware_trace`, `request` | 高敏感：兼容字段 `user_message`、`conversation_history`、`request_messages` 故意保留原始值；新 consumer 应优先用已清理的 `request`。 |
| `post_api_request` | 观察者 | Provider success 归一化后；忽略返回值。 | `task_id`, `turn_id`, `api_request_id`, `session_id`, `platform`, `model`, `provider`, `base_url`, `api_mode`, `api_call_count`, `api_duration`, `started_at`, `ended_at`, `finish_reason`, `message_count`, `response_model`, `response`, `usage`, `assistant_message`, `assistant_content_chars`, `assistant_tool_call_count` | 可用已清理的 `response`，但原始归一化 `assistant_message` 可能含模型/用户内容；`usage` 是计费数据。 |
| `api_request_error` | 观察者 | 每次失败的 provider attempt；忽略返回值。 | `task_id`, `turn_id`, `api_request_id`, `session_id`, `platform`, `model`, `provider`, `base_url`, `api_mode`, `api_call_count`, `api_duration`, `started_at`, `ended_at`, `status_code`, `retry_count`, `max_retries`, `retryable`, `reason`, `error`, `request` | Error 文本可能含 provider/用户数据；`request` 设计为已清理。 |
| `on_session_start` | 观察者 | 新 session 第一轮；忽略返回值。 | `session_id`, `model`, `platform` | 仅标识符和 routing metadata。 |
| `on_session_end` | 观察者 | Canonical 路径在每轮 finalize；CLI/TUI 退出还有精简 legacy shape。 | Canonical：`session_id`, `task_id`, `turn_id`, `completed`, `failed`, `interrupted`, `turn_exit_reason`, `model`, `platform`；退出路径可能增加 `reason`/`api_request_id` 并省略字段。 | ID、model/platform 和结果；canonical payload 无消息正文。 |
| `on_session_finalize` | 观察者 | CLI/TUI/gateway 通过 `finalize_session` teardown；gateway 关闭或过期时可只 finalize 而不 reset。忽略返回值。 | 按 surface：`session_id`, `platform`，可选 `reason`, `old_session_id`, `new_session_id` | Session 和 routing 标识。 |
| `on_session_reset` | 观察者 | CLI/TUI session boundary，或 gateway 创建替代 session 后；忽略返回值。 | CLI：`session_id`, `platform`, `reason`；TUI：`session_id`, `platform`；gateway：另有 `reason`, `old_session_id`, `new_session_id` | Session 和 routing 标识。 |
| `on_skill_lifecycle` | 观察者 | 权威 skill 使用状态变更后；忽略返回值。 | `action`, `skill_name`, `provenance`, `task_id`, `session_id`, `use_count`, `reused`, `reuse_after_patch` | 暴露本地 skill 名和 provenance。 |
| `subagent_start` | 观察者 | 子 agent 已构造、即将运行；忽略返回值。 | `parent_session_id`, `parent_turn_id`, `parent_subagent_id`, `child_session_id`, `child_subagent_id`, `child_role`, `child_goal` | Child goal 可能含用户/项目内容。 |
| `subagent_stop` | 观察者 | 子 agent 退出；忽略返回值。 | `parent_session_id`, `parent_turn_id`, `child_session_id`, `child_role`, `child_summary`, `child_status`, `tool_call_history`, `duration_ms` | Summary 和已脱敏 tool-history metadata 仍可能暴露项目结构。 |
| `pre_gateway_dispatch` | 指令/控制 | 非 internal 入站消息在 auth/pairing/dispatch 前；第一个有效 `skip`、`rewrite` 或 `allow` 控制流程。 | `event`, `gateway`, `session_store` | 极高权限的进程内对象会暴露入站用户/routing 数据和 host handle。 |
| `pre_approval_request` | 观察者 | Prompted 或 smart approval 前；忽略返回值。 | `command`, `description`, `pattern_key`, `pattern_keys`, `session_key`, `surface`, `turn_id`, `tool_call_id` | 命令可能含 secret；smart observer preparation 会强制脱敏，但各 surface 并非完全相同。 |
| `post_approval_response` | 观察者 | 决策、timeout 或 gateway 通知失败后；忽略返回值。 | `command`, `description`, `pattern_key`, `pattern_keys`, `session_key`, `surface`, `turn_id`, `tool_call_id`, `choice`；smart 路径可增加 `decided_by` | 同样的命令敏感性，加决策 metadata。 |
| `kanban_task_claimed` | 观察者 | Claim commit 后，在 dispatcher 进程 spawn worker 前；忽略返回值。 | `task_id`, `profile_name`, `board`, `assignee`, `run_id` | Board/task/profile/assignee 标识。 |
| `kanban_task_completed` | 观察者 | Completion 和 cleanup 后，通常在 worker 进程；忽略返回值。 | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `summary` | Summary 可能含项目/用户内容。 |
| `kanban_task_blocked` | 观察者 | Blocked transition 后；dependency-wait 路径在 transaction 退出前触发。忽略返回值。 | `task_id`, `profile_name`, `board`, `assignee`, `run_id`, `reason` | Reason 可能含项目/用户内容。 |

---

### `pre_tool_call`

在每次工具执行**之前立即**触发——内置工具和插件工具均适用。

**回调签名：**

```python
def my_callback(tool_name: str, args: dict, task_id: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----|------|------|
| `tool_name` | `str` | 即将执行的工具名称（如 `"terminal"`、`"web_search"`、`"read_file"`） |
| `args` | `dict` | 模型传递给工具的参数 |
| `task_id` | `str` | 会话/任务标识符。未设置时为空字符串。 |

**触发位置：** `model_tools.py` 中的 `handle_function_call()` 内，工具处理器运行前。每次工具调用触发一次——若模型并行调用 3 个工具，则触发 3 次。

**返回值——阻断或要求审批：**

```python
return {"action": "block", "message": "Reason the tool call was blocked"}
# 或
return {"action": "approve", "message": "Why approval is required", "rule_key": "optional:scope"}
```

第一个有效指令生效。`block` 要求非空 `message`，并用该文本作为返回给模型的错误来短路工具。`approve` 将调用升级到现有的人类审批 gate；`message` 和 `rule_key` 可选，拒绝、timeout 或 gate error 都会 fail closed。其他返回值会被忽略。

**使用场景：** 日志记录、审计追踪、工具调用计数、阻断危险操作、速率限制、按用户策略执行。

**示例——工具调用审计日志：**

```python
import json, logging
from datetime import datetime

logger = logging.getLogger(__name__)

def audit_tool_call(tool_name, args, task_id, **kwargs):
    logger.info("TOOL_CALL session=%s tool=%s args=%s",
                task_id, tool_name, json.dumps(args)[:200])

def register(ctx):
    ctx.register_hook("pre_tool_call", audit_tool_call)
```

**示例——对危险工具发出警告：**

```python
DANGEROUS = {"terminal", "write_file", "patch"}

def warn_dangerous(tool_name, **kwargs):
    if tool_name in DANGEROUS:
        print(f"⚠ Executing potentially dangerous tool: {tool_name}")

def register(ctx):
    ctx.register_hook("pre_tool_call", warn_dangerous)
```

---

### `post_tool_call`

在每次工具执行返回**之后立即**触发。

**回调签名：**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str,
                duration_ms: int, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----|------|------|
| `tool_name` | `str` | 刚刚执行的工具名称 |
| `args` | `dict` | 模型传递给工具的参数 |
| `result` | `str` | 工具的返回值（始终为 JSON 字符串） |
| `task_id` | `str` | 会话/任务标识符。未设置时为空字符串。 |
| `duration_ms` | `int` | 工具分发耗时，单位毫秒（使用 `time.monotonic()` 在 `registry.dispatch()` 前后测量）。 |

**触发位置：** `model_tools.py` 中的 `handle_function_call()` 内，工具处理器返回后。每次工具调用触发一次。若工具抛出未处理异常，**不会**触发（错误被捕获并以错误 JSON 字符串返回，`post_tool_call` 以该错误字符串作为 `result` 触发）。

**返回值：** 忽略。

**使用场景：** 记录工具结果、指标采集、追踪工具成功/失败率、延迟仪表盘、按工具预算告警、特定工具完成时发送通知。

**示例——追踪工具使用指标：**

```python
from collections import Counter, defaultdict
import json

_tool_counts = Counter()
_error_counts = Counter()
_latency_ms = defaultdict(list)

def track_metrics(tool_name, result, duration_ms=0, **kwargs):
    _tool_counts[tool_name] += 1
    _latency_ms[tool_name].append(duration_ms)
    try:
        parsed = json.loads(result)
        if "error" in parsed:
            _error_counts[tool_name] += 1
    except (json.JSONDecodeError, TypeError):
        pass

def register(ctx):
    ctx.register_hook("post_tool_call", track_metrics)
```

---

### `pre_llm_call`

**每轮触发一次**，在工具调用循环开始前。所有有效回调返回值会按插件顺序聚合，并注入当前轮次的用户消息。

**回调签名：**

```python
def my_callback(session_id: str, user_message: str, conversation_history: list,
                is_first_turn: bool, model: str, platform: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----|------|------|
| `session_id` | `str` | 当前会话的唯一标识符 |
| `user_message` | `str` | 本轮用户的原始消息（技能注入前） |
| `conversation_history` | `list` | 完整消息列表的副本（OpenAI 格式：`[{"role": "user", "content": "..."}]`） |
| `is_first_turn` | `bool` | 新会话的第一轮为 `True`，后续轮次为 `False` |
| `model` | `str` | 模型标识符（如 `"anthropic/claude-sonnet-4.6"`） |
| `platform` | `str` | 会话运行环境：`"cli"`、`"telegram"`、`"discord"` 等 |

**触发位置：** `run_agent.py` 中的 `run_conversation()` 内，上下文压缩后、主 `while` 循环前。每次 `run_conversation()` 调用触发一次（即每个用户轮次一次），而非工具循环内每次 API 调用触发一次。

**返回值：** 若回调返回包含 `"context"` 键的字典，或非空的普通字符串，该文本会追加到当前轮次的用户消息。返回 `None` 表示不注入。

```python
# 注入上下文
return {"context": "Recalled memories:\n- User likes Python\n- Working on hermes-agent"}

# 普通字符串（等效）
return "Recalled memories:\n- User likes Python"

# 不注入
return None
```

**上下文注入位置：** 始终注入到**用户消息**，而非系统 prompt。这保留了 prompt 缓存——系统 prompt 在各轮次间保持不变，已缓存的 token 得以复用。系统 prompt 是 Hermes 的领域（模型指导、工具执行、个性、技能）。插件在用户输入旁边贡献上下文。

干净的用户消息 `content` 保持不变。为保证 replay 和 prompt cache 稳定，Hermes 可能把实际发送给 API 的消息（包括插件注入上下文）持久化到该行的 `api_content` sidecar。

当**多个插件**返回上下文时，其输出按插件发现顺序（按目录名字母顺序）以双换行符连接。

**使用场景：** 记忆召回、RAG 上下文注入、护栏、每轮分析。

**示例——记忆召回：**

```python
import httpx

MEMORY_API = "https://your-memory-api.example.com"

def recall(session_id, user_message, is_first_turn, **kwargs):
    try:
        resp = httpx.post(f"{MEMORY_API}/recall", json={
            "session_id": session_id,
            "query": user_message,
        }, timeout=3)
        memories = resp.json().get("results", [])
        if not memories:
            return None
        text = "Recalled context:\n" + "\n".join(f"- {m['text']}" for m in memories)
        return {"context": text}
    except Exception:
        return None

def register(ctx):
    ctx.register_hook("pre_llm_call", recall)
```

**示例——护栏：**

```python
POLICY = "Never execute commands that delete files without explicit user confirmation."

def guardrails(**kwargs):
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", guardrails)
```

---

### `post_llm_call`

**每轮触发一次**，在工具调用循环完成且 agent 产生最终响应后。仅在**成功**的轮次触发——若轮次被中断则不触发。

**回调签名：**

```python
def my_callback(session_id: str, user_message: str, assistant_response: str,
                conversation_history: list, model: str, platform: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----|------|------|
| `session_id` | `str` | 当前会话的唯一标识符 |
| `user_message` | `str` | 本轮用户的原始消息 |
| `assistant_response` | `str` | Agent 本轮的最终文本响应 |
| `conversation_history` | `list` | 轮次完成后完整消息列表的副本 |
| `model` | `str` | 模型标识符 |
| `platform` | `str` | 会话运行环境 |

**触发位置：** `run_agent.py` 中的 `run_conversation()` 内，工具循环以最终响应退出后。受 `if final_response and not interrupted` 保护——因此当用户在轮次中途中断，或 agent 在未产生响应的情况下达到迭代上限时，**不会**触发。

**返回值：** 忽略。

**使用场景：** 将对话数据同步到外部记忆系统、计算响应质量指标、记录轮次摘要、触发后续操作。

**示例——同步到外部记忆：**

```python
import httpx

MEMORY_API = "https://your-memory-api.example.com"

def sync_memory(session_id, user_message, assistant_response, **kwargs):
    try:
        httpx.post(f"{MEMORY_API}/store", json={
            "session_id": session_id,
            "user": user_message,
            "assistant": assistant_response,
        }, timeout=5)
    except Exception:
        pass  # best-effort

def register(ctx):
    ctx.register_hook("post_llm_call", sync_memory)
```

**示例——追踪响应长度：**

```python
import logging
logger = logging.getLogger(__name__)

def log_response_length(session_id, assistant_response, model, **kwargs):
    logger.info("RESPONSE session=%s model=%s chars=%d",
                session_id, model, len(assistant_response or ""))

def register(ctx):
    ctx.register_hook("post_llm_call", log_response_length)
```

---

### `on_session_start`

在全新会话创建时触发**一次**。在会话延续时**不会**触发（用户在已有会话中发送第二条消息时）。

**回调签名：**

```python
def my_callback(session_id: str, model: str, platform: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----|------|------|
| `session_id` | `str` | 新会话的唯一标识符 |
| `model` | `str` | 模型标识符 |
| `platform` | `str` | 会话运行环境 |

**触发位置：** `run_agent.py` 中的 `run_conversation()` 内，新会话第一轮期间——具体在系统 prompt 构建后、工具循环开始前。检查条件为 `if not conversation_history`（无历史消息 = 新会话）。

**返回值：** 忽略。

**使用场景：** 初始化会话级状态、预热缓存、向外部服务注册会话、记录会话开始。

**示例——初始化会话缓存：**

```python
_session_caches = {}

def init_session(session_id, model, platform, **kwargs):
    _session_caches[session_id] = {
        "model": model,
        "platform": platform,
        "tool_calls": 0,
        "started": __import__("datetime").datetime.now().isoformat(),
    }

def register(ctx):
    ctx.register_hook("on_session_start", init_session)
```

---

### `on_session_end`

在每次 `run_conversation()` 调用**结束时**触发，无论结果如何。若用户在 agent 处理过程中退出，也会从 CLI 的退出处理器触发。

**回调签名：**

```python
def my_callback(session_id: str, completed: bool, interrupted: bool,
                model: str, platform: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----|------|------|
| `session_id` | `str` | 会话的唯一标识符 |
| `completed` | `bool` | Agent 产生最终响应时为 `True`，否则为 `False` |
| `interrupted` | `bool` | 轮次被中断时为 `True`（用户发送新消息、`/stop` 或退出） |
| `model` | `str` | 模型标识符 |
| `platform` | `str` | 会话运行环境 |

**触发位置：** 两处：
1. **`run_agent.py`** — 每次 `run_conversation()` 调用结束时，所有清理完成后。始终触发，即使轮次出错。
2. **`cli.py`** — CLI 的 atexit 处理器中，但**仅当** agent 在退出时处于处理中状态（`_agent_running=True`）。这捕获了处理过程中的 Ctrl+C 和 `/exit`。此时 `completed=False`，`interrupted=True`。

**返回值：** 忽略。

**使用场景：** 刷新缓冲区、关闭连接、持久化会话状态、记录会话时长、清理 `on_session_start` 中初始化的资源。

**示例——刷新并清理：**

```python
_session_caches = {}

def cleanup_session(session_id, completed, interrupted, **kwargs):
    cache = _session_caches.pop(session_id, None)
    if cache:
        # Flush accumulated data to disk or external service
        status = "completed" if completed else ("interrupted" if interrupted else "failed")
        print(f"Session {session_id} ended: {status}, {cache['tool_calls']} tool calls")

def register(ctx):
    ctx.register_hook("on_session_end", cleanup_session)
```

**示例——会话时长追踪：**

```python
import time, logging
logger = logging.getLogger(__name__)

_start_times = {}

def on_start(session_id, **kwargs):
    _start_times[session_id] = time.time()

def on_end(session_id, completed, interrupted, **kwargs):
    start = _start_times.pop(session_id, None)
    if start:
        duration = time.time() - start
        logger.info("SESSION_DURATION session=%s seconds=%.1f completed=%s interrupted=%s",
                     session_id, duration, completed, interrupted)

def register(ctx):
    ctx.register_hook("on_session_start", on_start)
    ctx.register_hook("on_session_end", on_end)
```

---

### `on_session_finalize`

当 CLI 或 gateway **销毁**活跃会话时触发——例如用户执行 `/new`、gateway GC 了空闲会话，或 CLI 在 agent 活跃时退出。可用它刷新与旧 session ID 绑定的状态。Gateway reset 时，替代会话会先创建并持久化，然后才调用此回调。

**回调签名：**

```python
def my_callback(session_id: str | None, platform: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----|------|------|
| `session_id` | `str` 或 `None` | 即将销毁的会话 ID。若无活跃会话则可能为 `None`。 |
| `platform` | `str` | `"cli"` 或消息平台名称（`"telegram"`、`"discord"` 等）。 |

**触发位置：** CLI/TUI teardown，以及 gateway reset、关闭或空闲过期路径。Gateway 关闭和过期可只触发 finalize，而不触发对应的 `on_session_reset`。

**返回值：** 忽略。

**使用场景：** 在会话 ID 被丢弃前持久化最终会话指标、关闭每会话资源、发出最终遥测事件、排空队列写入。

---

### `on_session_reset`

在 CLI 或 TUI session boundary，或 gateway 为活跃聊天**换入新会话 key** 时触发。这让插件无需等待下一个 `on_session_start` 即可响应会话状态已被清除。

**回调签名：**

```python
def my_callback(session_id: str, platform: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----|------|------|
| `session_id` | `str` | 新会话的 ID（已轮换为新值）。 |
| `platform` | `str` | `"cli"`、`"tui"` 或消息平台名称。 |
| `reason` | `str`，可选 | CLI 和 gateway reset 路径提供。 |
| `old_session_id` | `str`，可选 | 仅 gateway，旧 session ID。 |
| `new_session_id` | `str`，可选 | 仅 gateway，新 session ID。 |

**触发位置：** CLI 提供 `session_id`、`platform`、`reason`；TUI 提供 `session_id`、`platform`；gateway 在分配新 key 后另加 `reason`、`old_session_id`、`new_session_id`。Gateway reset 顺序为：创建并持久化替代会话 → `on_session_finalize(old_id)` → `on_session_reset(new_id)` → 第一条入站消息时的 `on_session_start(new_id)`。

**返回值：** 忽略。

**使用场景：** 重置以 `session_id` 为键的每会话缓存、发出"会话已轮换"分析事件、初始化新状态桶。

---

参见 **[构建插件指南](/developer-guide/plugins)**，获取包含工具 schema、处理器和高级 hook 模式的完整演练。

---

### `subagent_stop`

`delegate_task` 完成后，**每个子 agent 触发一次**。无论你委托了单个任务还是三个任务的批次，此 hook 对每个子 agent 各触发一次，在父线程上串行执行。

**回调签名：**

```python
def my_callback(parent_session_id: str, child_role: str | None,
                child_summary: str | None, child_status: str,
                duration_ms: int, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----|------|------|
| `parent_session_id` | `str` | 委托父 agent 的会话 ID |
| `child_role` | `str \| None` | 子 agent 上设置的编排角色标签（若功能未启用则为 `None`） |
| `child_summary` | `str \| None` | 子 agent 返回给父 agent 的最终响应 |
| `child_status` | `str` | `"completed"`、`"failed"`、`"interrupted"` 或 `"error"` |
| `duration_ms` | `int` | 运行子 agent 的挂钟时间，单位毫秒 |

**触发位置：** `tools/delegate_tool.py` 中，`ThreadPoolExecutor.as_completed()` 排空所有子 future 后。触发被编排到父线程，因此 hook 作者无需考虑并发回调执行问题。

**返回值：** 忽略。

**使用场景：** 记录编排活动、为计费累计子 agent 时长、写入委托后审计记录。

**示例——记录编排器活动：**

```python
import logging
logger = logging.getLogger(__name__)

def log_subagent(parent_session_id, child_role, child_status, duration_ms, **kwargs):
    logger.info(
        "SUBAGENT parent=%s role=%s status=%s duration_ms=%d",
        parent_session_id, child_role, child_status, duration_ms,
    )

def register(ctx):
    ctx.register_hook("subagent_stop", log_subagent)
```

:::info
在大量委托场景下（如编排器角色 × 5 个叶节点 × 嵌套深度），`subagent_stop` 每轮会触发多次。保持回调快速执行；将耗时操作推送到后台队列。
:::

---

### `pre_gateway_dispatch`

在 gateway 中，**每条入站 `MessageEvent` 触发一次**，在内部事件守卫之后、认证/配对和 agent 分发**之前**。这是 gateway 级消息流策略（只听不回窗口、人工接管、按聊天路由等）的拦截点，这些策略不适合放在任何单一平台适配器中。

**回调签名：**

```python
def my_callback(event, gateway, session_store, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----|------|------|
| `event` | `MessageEvent` | 标准化的入站消息（包含 `.text`、`.source`、`.message_id`、`.internal` 等）。 |
| `gateway` | `GatewayRunner` | 活跃的 gateway 运行器，插件可调用 `gateway.adapters[platform].send(...)` 进行旁路回复（所有者通知等）。 |
| `session_store` | `SessionStore` | 用于通过 `session_store.append_to_transcript(...)` 静默摄入转录。 |

**触发位置：** `gateway/run.py` 中的 `GatewayRunner._handle_message()` 内，`is_internal` 计算后立即触发。**内部事件完全跳过此 hook**（它们是系统生成的——后台进程完成等——不得被面向用户的策略拦截）。

**返回值：** `None` 或字典。第一个被识别的 action 字典生效；其余插件结果被忽略。插件回调中的异常会被捕获并记录；gateway 在出错时始终回退到正常分发。

| 返回值 | 效果 |
|-------|------|
| `{"action": "skip", "reason": "..."}` | 丢弃消息——无 agent 回复、无配对流程、无认证。假定插件已处理（如静默摄入到转录）。 |
| `{"action": "rewrite", "text": "new text"}` | 替换 `event.text`，然后以修改后的事件继续正常分发。适用于将缓冲的环境消息合并为单个 prompt。 |
| `{"action": "allow"}` / `None` | 正常分发——运行完整的认证/配对/agent 循环链。 |

**使用场景：** 只听不回的群聊（仅在被 @ 时响应；将环境消息缓冲为上下文）；人工接管（所有者手动处理聊天时静默摄入客户消息）；按 profile 速率限制；策略驱动的路由。

**示例——静默丢弃未授权的私信，不触发配对代码：**

```python
def deny_unauthorized_dms(event, **kwargs):
    src = event.source
    if src.chat_type == "dm" and not _is_approved_user(src.user_id):
        return {"action": "skip", "reason": "unauthorized-dm"}
    return None

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", deny_unauthorized_dms)
```

**示例——在被提及时将环境消息缓冲重写为单个 prompt：**

```python
_buffers = {}

def buffer_or_rewrite(event, **kwargs):
    key = (event.source.platform, event.source.chat_id)
    buf = _buffers.setdefault(key, [])
    if _bot_mentioned(event.text):
        combined = "\n".join(buf + [event.text])
        buf.clear()
        return {"action": "rewrite", "text": combined}
    buf.append(event.text)
    return {"action": "skip", "reason": "ambient-buffered"}

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", buffer_or_rewrite)
```

---

### `pre_approval_request`

在审批请求向用户展示**之前立即**触发——覆盖所有界面：交互式 CLI、Ink TUI、gateway 平台（Telegram、Discord、Slack、WhatsApp、Matrix 等）以及 ACP 客户端（VS Code、Zed、JetBrains）。

这是接入自定义通知器的正确位置——例如弹出允许/拒绝通知的 macOS 菜单栏应用，或记录每个带上下文审批请求的审计日志。

**回调签名：**

```python
def my_callback(
    command: str,
    description: str,
    pattern_key: str,
    pattern_keys: list[str],
    session_key: str,
    surface: str,
    **kwargs,
):
```

| 参数 | 类型 | 描述 |
|-----|------|------|
| `command` | `str` | 等待审批的 shell 命令 |
| `description` | `str` | 命令被标记的人类可读原因（多个模式匹配时合并） |
| `pattern_key` | `str` | 触发审批的主要模式键（如 `"rm_rf"`、`"sudo"`） |
| `pattern_keys` | `list[str]` | 所有匹配的模式键 |
| `session_key` | `str` | 会话标识符，用于按聊天限定通知范围 |
| `surface` | `str` | 交互式 CLI/TUI 提示为 `"cli"`，异步平台审批为 `"gateway"` |

**返回值：** 忽略。此处的 hook 仅作观察用途；不能否决或预先回答审批。使用 [`pre_tool_call`](#pre_tool_call) 在工具到达审批系统前阻断它。

**使用场景：** 桌面通知、推送告警、审计日志、Slack webhook、升级路由、指标。

**示例——macOS 桌面通知：**

```python
import subprocess

def notify_approval(command, description, session_key, **kwargs):
    title = "Hermes needs approval"
    body = f"{description}: {command[:80]}"
    subprocess.Popen([
        "osascript", "-e",
        f'display notification "{body}" with title "{title}"',
    ])

def register(ctx):
    ctx.register_hook("pre_approval_request", notify_approval)
```

---

### `post_approval_response`

在用户响应审批提示、提示超时，或 gateway 无法发送审批通知**之后**触发。通知失败会在尚无审批决定时以 `choice="notify_failed"` 触发。

**回调签名：**

```python
def my_callback(
    command: str,
    description: str,
    pattern_key: str,
    pattern_keys: list[str],
    session_key: str,
    surface: str,
    choice: str,
    **kwargs,
):
```

与 `pre_approval_request` 相同的 kwargs，另加：

| 参数 | 类型 | 描述 |
|-----|------|------|
| `choice` | `str` | Prompted 路径使用 `"once"`、`"session"`、`"always"`、`"deny"`、`"timeout"` 或 `"notify_failed"`；smart 路径使用 `"smart_approve"` 或 `"smart_deny"` |
| `decided_by` | `str` | Smart 决策为 `"aux_llm"`；prompted 路径不存在。 |

**返回值：** 忽略。

**使用场景：** 关闭对应的桌面通知、在审计日志中记录最终决定、更新指标、推进速率限制器。

```python
def log_decision(command, choice, session_key, **kwargs):
    logger.info("approval %s: %s for session %s", choice, command[:60], session_key)

def register(ctx):
    ctx.register_hook("post_approval_response", log_decision)
```

---

### `transform_tool_result`

在工具返回**之后**、结果追加到对话**之前**触发。允许插件重写**任意**工具的结果字符串——不仅限于终端输出——在模型看到之前进行处理。

**回调签名：**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str, **kwargs) -> str | None:
```

完整 payload 还包括 `session_id`、`tool_call_id`、`turn_id`、`api_request_id`、`duration_ms`、`status`、`error_type`、`error_message`。`result` 是 tool dispatch 返回的最终结果；它和 `args` 都可能包含任意用户/工具内容及 secret。

**返回值：** 第一个 `str`（包括空字符串）替换结果，`None` 保持不变。

**使用场景：** 从 `web_extract` 输出中脱敏组织特定的 PII、为长 JSON 工具响应添加摘要头、向 `read_file` 结果注入检索增强提示、将 `delegate_task` 子 agent 报告重写为项目特定 schema。

```python
import re
SECRET = re.compile(r"sk-[A-Za-z0-9]{32,}")

def redact_secrets(tool_name, result, **kwargs):
    if SECRET.search(result):
        return SECRET.sub("[REDACTED]", result)
    return None

def register(ctx):
    ctx.register_hook("transform_tool_result", redact_secrets)
```

适用于所有工具。仅针对 terminal 的重写请参见下方 `transform_terminal_output`——它范围更窄，在 `transform_tool_result` 前运行，且替换内容仍受 terminal 工具的最终 output limit 限制。

---

### `transform_terminal_output`

在 `terminal` 工具完成前台进程的有界输出捕获后、最终 output limit 前触发。插件可替换已捕获的 stdout/stderr；替换内容仍会受最终 output limit 限制。

**回调签名：**

```python
def my_callback(
    command: str,
    output: str,
    returncode: int,
    task_id: str,
    env_type: str,
    **kwargs,
) -> str | None:
```

| 参数 | 类型 | 描述 |
|-----|------|------|
| `command` | `str` | 产生输出的 shell 命令。 |
| `output` | `str` | 有界进程捕获后的合并 stdout/stderr。 |
| `returncode` | `int` | 进程返回码。 |
| `task_id` | `str` | 有效 task ID，未设置时为空字符串。 |
| `env_type` | `str` | 执行环境类型。 |

**返回值：** 第一个 `str` 替换输出，`None` 保持不变。命令和输出可能包含凭据或其他敏感数据。

```python
def summarize_find(command, output, **kwargs):
    if command.startswith("find ") and len(output) > 50_000:
        lines = output.count("\n")
        head = "\n".join(output.splitlines()[:40])
        return f"{head}\n\n[summary: {lines} paths total, showing first 40]"
    return None

def register(ctx):
    ctx.register_hook("transform_terminal_output", summarize_find)
```

与随后运行的 `transform_tool_result` 配合使用；后者覆盖所有工具，包括 `terminal`。

---

### `transform_llm_output`

**每轮触发一次**，在工具调用循环完成且模型产生最终响应后、该响应交付给用户（CLI、gateway 或程序调用方）**之前**。允许插件使用经典编程方法重写 assistant 的最终文本——无需为 SOUL 风格文本或技能驱动的转换消耗额外推理 token。

**回调签名：**

```python
def my_callback(
    response_text: str,
    session_id: str,
    model: str,
    platform: str,
    **kwargs,
) -> str | None:
```

| 参数 | 类型 | 描述 |
|-----|------|------|
| `response_text` | `str` | 本轮 assistant 的最终响应文本。 |
| `session_id` | `str` | 本次对话的会话 ID（一次性运行时可能为空）。 |
| `model` | `str` | 产生响应的模型名称（如 `anthropic/claude-sonnet-4.6`）。 |
| `platform` | `str` | 交付平台（`cli`、`telegram`、`discord` 等；未设置时为空）。 |

**返回值：** 非空 `str` 替换响应文本，`None` 或空字符串保持不变。当多个插件注册时，**第一个非空字符串生效**。与 tool/terminal transform 不同，空字符串不会作为替换值。

**使用场景：** 应用个性/词汇转换（海盗腔、海绵宝宝体）、从最终文本中脱敏用户特定标识符、追加项目特定签名页脚、在不消耗 SOUL 指令 token 的情况下执行内部风格指南。

```python
import os, re

def spongebob(response_text, **kwargs):
    if os.environ.get("SPONGEBOB_MODE") != "on":
        return None  # pass through unchanged
    return re.sub(r"!", "!! Tartar sauce!", response_text)

def register(ctx):
    ctx.register_hook("transform_llm_output", spongebob)
```

此 hook 受非空、非中断响应保护——不会在停止按钮中断或空轮次时触发。异常会被记录为警告，不会中断 agent 执行。

### API request 观察者 hook

#### `pre_api_request`

每次 provider attempt 发出前触发，仅观察。兼容字段 `user_message`、`conversation_history`、`request_messages` 是原始且故意未清理的数据；新 consumer 应优先使用已清理的 `request` envelope。

#### `post_api_request`

Provider response 成功归一化后触发，仅观察。优先使用已清理的 `response`；`assistant_message` 是原始归一化消息，`usage` 是计费数据。

#### `api_request_error`

Provider attempt 失败时触发，包含 status/retry timing、`error` 对象和已清理的 `request`，仅观察。Error message 仍可能包含 provider 或用户数据。

### `on_skill_lifecycle`

权威 skill 使用状态变更后触发，仅观察；会暴露本地 `skill_name`、provenance、关联 ID、use count 和 reuse flag。

### Kanban 生命周期观察者

#### `kanban_task_claimed`

Claim commit 后，在 dispatcher 进程 spawn worker 前触发。

#### `kanban_task_completed`

Completion 和 cleanup 后触发，通常位于 worker 进程；`summary` 可能包含项目或用户内容。

#### `kanban_task_blocked`

普通 blocked transition 后触发；dependency-wait 路径在该 write transaction 退出前调用。`reason` 可能包含项目或用户内容。

三个 kanban hook 均仅观察，并携带 `task_id`、`profile_name`、`board`、`assignee`、`run_id`；completed 增加 `summary`，blocked 增加 `reason`。

---

## Shell Hooks

在 `cli-config.yaml` 中声明 shell 脚本 hook，Hermes 会在对应的插件 hook 事件触发时将其作为子进程运行——在 CLI 和 gateway 会话中均适用。无需编写 Python 插件。

当你希望用一个即插即用的单文件脚本（Bash、Python 或任何带 shebang 的脚本）来实现以下功能时，使用 shell hooks：

- **阻断工具调用** — 拒绝危险的 `terminal` 命令、执行按目录策略、要求对破坏性的 `write_file` / `patch` 操作进行审批。
- **工具调用后运行** — 自动格式化 agent 刚写入的 Python 或 TypeScript 文件、记录 API 调用、触发 CI 工作流。
- **向下一个 LLM 轮次注入上下文** — 在用户消息前追加 `git status` 输出、当前星期几或检索到的文档（参见 [`pre_llm_call`](#pre_llm_call)）。
- **观察生命周期事件** — 在子 agent 完成（`subagent_stop`）或会话开始（`on_session_start`）时写入日志行。

Shell hooks 通过在 CLI 启动（`hermes_cli/main.py`）和 gateway 启动（`gateway/run.py`）时调用 `agent.shell_hooks.register_from_config(cfg)` 来注册。它们与 Python 插件 hook 自然组合——两者都流经同一个分发器。

### 对比一览

| 维度 | Shell hooks | [Plugin hooks](#plugin-hooks) | [Gateway hooks](#gateway-event-hooks) |
|------|-------------|-------------------------------|---------------------------------------|
| 声明位置 | `~/.hermes/config.yaml` 中的 `hooks:` 块 | 插件 `plugin.yaml` 中的 `register()` | `HOOK.yaml` + `handler.py` 目录 |
| 存放位置 | `~/.hermes/agent-hooks/`（约定） | `~/.hermes/plugins/<name>/` | `~/.hermes/hooks/<name>/` |
| 语言 | 任意（Bash、Python、Go 二进制等） | 仅 Python | 仅 Python |
| 运行环境 | CLI + Gateway | CLI + Gateway | 仅 Gateway |
| 事件 | `VALID_HOOKS`（含 `subagent_stop`） | `VALID_HOOKS` | Gateway 生命周期（`gateway:startup`、`agent:*`、`command:*`） |
| 可阻断工具调用 | 是（`pre_tool_call`） | 是（`pre_tool_call`） | 否 |
| 可注入 LLM 上下文 | 是（`pre_llm_call`） | 是（`pre_llm_call`） | 否 |
| 授权 | 每个 `(event, command)` 对首次使用时提示 | 隐式（Python 插件信任） | 隐式（目录信任） |
| 进程间隔离 | 是（子进程） | 否（进程内） | 否（进程内） |

### 配置 schema

```yaml
hooks:
  <event_name>:                  # Must be in VALID_HOOKS
    - matcher: "<regex>"         # Optional; used for pre/post_tool_call only
      command: "<shell command>" # Required; runs via shlex.split, shell=False
      timeout: <seconds>         # Optional; default 60, capped at 300

hooks_auto_accept: false         # See "Consent model" below
```

事件名称必须是 [plugin hook 事件](#plugin-hooks)之一；拼写错误会产生"你是否想输入 X？"警告并被跳过。单个条目中的未知键会被忽略；缺少 `command` 会跳过并发出警告。`timeout > 300` 会被截断并发出警告。

### JSON 通信协议

每次事件触发时，Hermes 为每个匹配的 hook（在 matcher 允许的情况下）生成一个子进程，将 JSON 载荷通过 **stdin** 传入，并从 **stdout** 读取 JSON 响应。

**stdin——脚本接收的载荷：**

```json
{
  "hook_event_name": "pre_tool_call",
  "tool_name":       "terminal",
  "tool_input":      {"command": "rm -rf /"},
  "session_id":      "sess_abc123",
  "cwd":             "/home/user/project",
  "extra":           {"task_id": "...", "tool_call_id": "..."}
}
```

对于非工具事件（`pre_llm_call`、`subagent_stop`、会话生命周期），`tool_name` 和 `tool_input` 为 `null`。`extra` 字典携带所有事件特定的 kwargs（`user_message`、`conversation_history`、`child_role`、`duration_ms` 等）。不可序列化的值会被字符串化而非省略。

**stdout——可选响应：**

```jsonc
// Block a pre_tool_call (both shapes accepted; normalised internally):
{"decision": "block", "reason":  "Forbidden: rm -rf"}   // Claude-Code style
{"action":   "block", "message": "Forbidden: rm -rf"}   // Hermes-canonical

// Inject context for pre_llm_call:
{"context": "Today is Friday, 2026-04-17"}

// Silent no-op — any empty / non-matching output is fine:
```

格式错误的 JSON、非零退出码和超时会记录警告，但永远不会中止 agent 循环。

### 实际示例

#### 1. 每次写入后自动格式化 Python 文件

```yaml
# ~/.hermes/config.yaml
hooks:
  post_tool_call:
    - matcher: "write_file|patch"
      command: "~/.hermes/agent-hooks/auto-format.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/auto-format.sh
payload="$(cat -)"
path=$(echo "$payload" | jq -r '.tool_input.path // empty')
[[ "$path" == *.py ]] && command -v black >/dev/null && black "$path" 2>/dev/null
printf '{}\n'
```

Agent 的上下文内文件视图**不会**自动重新读取——重新格式化仅影响磁盘上的文件。后续的 `read_file` 调用会读取格式化后的版本。

#### 2. 阻断破坏性 `terminal` 命令

```yaml
hooks:
  pre_tool_call:
    - matcher: "terminal"
      command: "~/.hermes/agent-hooks/block-rm-rf.sh"
      timeout: 5
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/block-rm-rf.sh
payload="$(cat -)"
cmd=$(echo "$payload" | jq -r '.tool_input.command // empty')
if echo "$cmd" | grep -qE 'rm[[:space:]]+-rf?[[:space:]]+/'; then
  printf '{"decision": "block", "reason": "blocked: rm -rf / is not permitted"}\n'
else
  printf '{}\n'
fi
```

#### 3. 向每轮注入 `git status`（Claude-Code `UserPromptSubmit` 等效）

```yaml
hooks:
  pre_llm_call:
    - command: "~/.hermes/agent-hooks/inject-cwd-context.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/inject-cwd-context.sh
cat - >/dev/null   # discard stdin payload
if status=$(git status --porcelain 2>/dev/null) && [[ -n "$status" ]]; then
  jq --null-input --arg s "$status" \
     '{context: ("Uncommitted changes in cwd:\n" + $s)}'
else
  printf '{}\n'
fi
```

Claude Code 的 `UserPromptSubmit` 事件在 Hermes 中没有对应的独立事件——`pre_llm_call` 在相同位置触发，且已支持上下文注入。在此使用即可。

#### 4. 记录每次子 agent 完成

```yaml
hooks:
  subagent_stop:
    - command: "~/.hermes/agent-hooks/log-orchestration.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/log-orchestration.sh
log=~/.hermes/logs/orchestration.log
jq -c '{ts: now, parent: .session_id, extra: .extra}' < /dev/stdin >> "$log"
printf '{}\n'
```

### 授权模型

每个唯一的 `(event, command)` 对在 Hermes 首次遇到时会提示用户审批，然后将决定持久化到 `~/.hermes/shell-hooks-allowlist.json`。后续运行（CLI 或 gateway）跳过提示。

三种方式可绕过交互式提示——满足其一即可：

1. CLI 上的 `--accept-hooks` 标志（如 `hermes --accept-hooks chat`）
2. `HERMES_ACCEPT_HOOKS=1` 环境变量
3. `cli-config.yaml` 中的 `hooks_auto_accept: true`

非 TTY 运行（gateway、cron、CI）需要这三种方式之一——否则任何新添加的 hook 会静默保持未注册状态并记录警告。

**脚本编辑被静默信任。** 允许列表以精确的命令字符串为键，而非脚本的哈希值，因此编辑磁盘上的脚本不会使授权失效。`hermes hooks doctor` 会标记 mtime 漂移，以便你发现编辑并决定是否重新审批。

### `hermes hooks` CLI

| 命令 | 功能 |
|------|------|
| `hermes hooks list` | 列出已配置的 hook，包含 matcher、超时和授权状态 |
| `hermes hooks test <event> [--for-tool X] [--payload-file F]` | 对合成载荷触发所有匹配的 hook 并打印解析后的响应 |
| `hermes hooks revoke <command>` | 删除所有匹配 `<command>` 的允许列表条目（下次重启后生效） |
| `hermes hooks doctor` | 对每个已配置的 hook 检查：执行位、允许列表状态、mtime 漂移、JSON 输出有效性和大致执行时间 |

### 安全性

Shell hooks 以**你的完整用户凭据**运行——与 cron 条目或 shell 别名的信任边界相同。将 `config.yaml` 中的 `hooks:` 块视为特权配置：

- 只引用你自己编写或完整审查过的脚本。
- 将脚本保存在 `~/.hermes/agent-hooks/` 内，便于审计路径。
- 拉取共享配置后重新运行 `hermes hooks doctor`，在新添加的 hook 注册前发现它们。
- 如果你的 config.yaml 在团队中进行版本控制，审查修改 `hooks:` 部分的 PR 时应与审查 CI 配置一样严格。

### 顺序与优先级

Python 插件 hook 和 shell hook 都流经同一个 `invoke_hook()` 分发器。Python 插件先注册（`discover_and_load()`），shell hook 后注册（`register_from_config()`），因此在平局情况下 Python `pre_tool_call` 的 block 决定优先。第一个有效的 block 生效——聚合器在任何回调产生带非空 message 的 `{"action": "block", "message": str}` 时立即返回。

## Outbound Webhooks

在 `config.yaml` 的 `hooks.outbound:` 中为一个或多个已发布事件配置外部 HTTP endpoint。Outbound webhook 是**仅通知**的 consumer：回调会序列化事件并放入有界队列，由 daemon worker 发送，因此 endpoint 的返回值不能阻断、转换或引导 agent。可使用环境变量提供 HMAC secret；接收方应校验签名，并将 payload 视为与上表相同敏感度的数据。

```yaml
hooks:
  outbound:
    - url: "https://example.com/hermes-events"
      events: [post_tool_call, on_session_end]
      secret_env: HERMES_WEBHOOK_SECRET
```

配置中的事件名同样以 `hermes_cli.plugins.VALID_HOOKS` 为准。`hermes hooks list` 会列出这些已配置 target，但不会输出完整的可用事件目录。