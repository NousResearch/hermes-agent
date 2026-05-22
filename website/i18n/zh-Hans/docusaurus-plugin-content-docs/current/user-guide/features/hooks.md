---
sidebar_position: 6
title: "Event Hooks"
description: "在关键生命周期节点运行自定义代码 — 记录日志、发送告警、调用 webhook"
---

# Event Hooks

Hermes 拥有三套 hook 系统，可在关键生命周期节点运行自定义代码：

| 系统 | 注册方式 | 运行环境 | 用途 |
|--------|---------------|---------|----------|
| **[Gateway hooks](#gateway-event-hooks)** | `~/.hermes/hooks/` 目录下的 `HOOK.yaml` + `handler.py` | 仅 Gateway | 日志、告警、webhook |
| **[Plugin hooks](#plugin-hooks)** | [plugin](/user-guide/features/plugins) 中的 `ctx.register_hook()` | CLI + Gateway | 工具拦截、指标、护栏 |
| **[Shell hooks](#shell-hooks)** | `~/.hermes/config.yaml` 中 `hooks:` 块指向 shell 脚本 | CLI + Gateway | 即插即用脚本，用于拦截、自动格式化、上下文注入 |

三套系统均为非阻塞 — 任何 hook 中的错误都会被捕获并记录，不会导致 agent 崩溃。

## Gateway Event Hooks

Gateway hooks 在 gateway 运行期间（Telegram、Discord、Slack、WhatsApp、Teams）自动触发，不会阻塞主 agent 流水线。

### 创建 Hook

每个 hook 是 `~/.hermes/hooks/` 下的一个目录，包含两个文件：

```text
~/.hermes/hooks/
└── my-hook/
    ├── HOOK.yaml      # 声明监听哪些事件
    └── handler.py     # Python handler 函数
```

#### HOOK.yaml

```yaml
name: my-hook
description: 将所有 agent 活动记录到文件
events:
  - agent:start
  - agent:end
  - agent:step
```

`events` 列表决定哪些事件会触发你的 handler。你可以订阅任意组合的事件，包括通配符如 `command:*`。

#### handler.py

```python
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path.home() / ".hermes" / "hooks" / "my-hook" / "activity.log"

async def handle(event_type: str, context: dict):
    """每次订阅的事件触发时调用。函数名必须为 'handle'。"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        **context,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

**Handler 规则：**
- 函数名必须为 `handle`
- 接收 `event_type`（字符串）和 `context`（字典）
- 可以是 `async def` 或普通 `def` — 两者都支持
- 错误会被捕获并记录，不会导致 agent 崩溃

### 可用事件

| Event | 触发时机 | Context keys |
|-------|---------------|--------------|
| `gateway:startup` | Gateway 进程启动 | `platforms`（活跃平台名称列表） |
| `session:start` | 新建消息会话 | `platform`, `user_id`, `session_id`, `session_key` |
| `session:end` | 会话结束（重置前） | `platform`, `user_id`, `session_key` |
| `session:reset` | 用户执行 `/new` 或 `/reset` | `platform`, `user_id`, `session_key` |
| `agent:start` | Agent 开始处理消息 | `platform`, `user_id`, `session_id`, `message` |
| `agent:step` | 工具调用循环的每次迭代 | `platform`, `user_id`, `session_id`, `iteration`, `tool_names` |
| `agent:end` | Agent 完成处理 | `platform`, `user_id`, `session_id`, `message`, `response` |
| `command:*` | 执行任意斜杠命令 | `platform`, `user_id`, `command`, `args` |

#### 通配符匹配

注册为 `command:*` 的 handler 会对任意 `command:` 事件触发（`command:model`、`command:reset` 等）。通过单次订阅即可监控所有斜杠命令。

### 示例

#### 长任务 Telegram 告警

当 agent 运行超过 10 步时给自己发消息：

```yaml
# ~/.hermes/hooks/long-task-alert/HOOK.yaml
name: long-task-alert
description: 当 agent 执行过多步骤时告警
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

#### 命令使用记录器

追踪哪些斜杠命令被使用：

```yaml
# ~/.hermes/hooks/command-logger/HOOK.yaml
name: command-logger
description: 记录斜杠命令使用情况
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

#### 会话启动 Webhook

新会话时 POST 到外部服务：

```yaml
# ~/.hermes/hooks/session-webhook/HOOK.yaml
name: session-webhook
description: 新会话时通知外部服务
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

### 教程：BOOT.md — 每次 Gateway 启动时运行启动清单

社区中流行的模式：在 `~/.hermes/BOOT.md` 放置一个 Markdown 清单，让 agent 每次 gateway 启动时运行一次。适用于"每次启动时检查夜间 cron 失败并在有故障时通过 Discord 通知我"，或"汇总过去 24 小时的 deploy.log 并发布到 Slack #ops"。

本教程展示如何将其构建为用户自定义 hook。Hermes 不内置 BOOT.md hook — 你自行接入想要的行为。

#### 我们要构建什么

1. `~/.hermes/BOOT.md` 文件，包含自然语言启动指令。
2. 一个 gateway hook，在 `gateway:startup` 时触发，使用 gateway 解析的 model/credentials 生成一次性 agent，并运行 BOOT.md 指令。
3. `[SILENT]` 约定，让 agent 在没有事项报告时选择不发送消息。

#### 第一步：编写你的清单

创建 `~/.hermes/BOOT.md`。像给人类助手下指令一样编写：

```markdown
# Startup Checklist

1. 运行 `hermes cron list` 并检查是否有定时任务夜间失败。
2. 如果有失败的，使用 `send_message` 工具将摘要发送到 Discord #ops。
3. 检查 `/opt/app/deploy.log` 在过去 24 小时内是否有 ERROR 行。如果有，汇总并包含在同一条 Discord 消息中。
4. 如果一切正常，仅回复 `[SILENT]`，不发送任何消息。
```

Agent 会将此视为 prompt 的一部分，因此任何能用自然语言描述的内容都有效 — 工具调用、shell 命令、发送消息、汇总文件。

#### 第二步：创建 hook

```text
~/.hermes/hooks/boot-md/
├── HOOK.yaml
└── handler.py
```

**`~/.hermes/hooks/boot-md/HOOK.yaml`**

```yaml
name: boot-md
description: 在 gateway 启动时运行 ~/.hermes/BOOT.md
events:
  - gateway:startup
```

**`~/.hermes/hooks/boot-md/handler.py`**

```python
"""每次 gateway 启动时运行 ~/.hermes/BOOT.md。"""

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
    """生成一次性 agent 并执行清单。

    使用 gateway 解析的 model 和运行时 credentials，因此可兼容
    自定义 endpoint、aggregator 和基于 OAuth 的 provider。
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

    # 后台线程，避免 gateway 启动被完整的 agent 回合阻塞。
    thread = threading.Thread(
        target=_run_boot_agent,
        args=(content,),
        name="boot-md",
        daemon=True,
    )
    thread.start()
```

关键的两行：

- `_resolve_gateway_model()` 读取 gateway 当前配置的 model。
- `_resolve_runtime_agent_kwargs()` 以与普通 gateway 回合相同的方式解析 provider credentials — 包括 API keys、base URLs、OAuth tokens 和 credential pools。

没有它们，裸 `AIAgent()` 会回退到内置默认值，对任何非默认 endpoint 都会返回 401。

#### 第三步：测试

重启 gateway：

```bash
hermes gateway restart
```

查看日志：

```bash
hermes logs --follow --level INFO | grep boot-md
```

你应该看到 `Running BOOT.md (N chars)`，随后是 `boot-md completed: ...`（agent 执行摘要）或 `boot-md completed (nothing to report)`（当 agent 回复 `[SILENT]` 时）。

删除 `~/.hermes/BOOT.md` 可禁用清单 — hook 保持加载，但在文件不存在时静默跳过。

#### 扩展此模式

- **按日程的清单：** 在 BOOT.md 指令中利用 `datetime.now().weekday()`（"如果是周一，额外检查每周部署日志"）。指令是自由文本，任何 agent 能推理的内容都适用。
- **多个清单：** 指向不同文件（`STARTUP.md`、`MORNING.md` 等），并为每个注册单独的 hook 目录。
- **非 agent 变体：** 如果不需要完整的 agent 循环，完全跳过 `AIAgent`，直接通过 `httpx` POST 固定通知。更便宜、更快，且不依赖 provider。

#### 为什么这不是内置功能

Hermes 的早期版本将此作为内置 hook 发布，并在每次 gateway 启动时静默使用裸默认值生成 agent。这让使用自定义 endpoint 的用户感到意外，并且未写入文件的用户根本不知道它在运行。将其作为文档化模式 — 由你在 hooks 目录中自行构建 — 意味着你能确切看到它的行为，并通过编写文件来选择加入。

### 工作原理

1. Gateway 启动时，`HookRegistry.discover_and_load()` 扫描 `~/.hermes/hooks/`
2. 每个包含 `HOOK.yaml` + `handler.py` 的子目录被动态加载
3. Handler 按声明的事件注册
4. 在每个生命周期节点，`hooks.emit()` 触发所有匹配的 handler
5. 任何 handler 中的错误都会被捕获并记录 — 损坏的 hook 永远不会导致 agent 崩溃

:::info
Gateway hooks 仅在 **gateway**（Telegram、Discord、Slack、WhatsApp、Teams）中触发。CLI 不会加载 gateway hooks。如需在所有环境中生效的 hook，请使用 [plugin hooks](#plugin-hooks)。
:::

## Plugin Hooks

[Plugins](/user-guide/features/plugins) 可以注册在 **CLI 和 gateway** 会话中触发的 hook。这些通过 plugin 的 `register()` 函数中以 `ctx.register_hook()` 编程方式注册。

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

- 回调接收 **关键字参数**。始终接受 `**kwargs` 以保证向前兼容 — 未来版本可能新增参数而不会破坏你的 plugin。
- 如果回调 **崩溃**，会被记录并跳过。其他 hook 和 agent 正常运行。行为异常的 plugin 永远不会破坏 agent。
- 两个 hook 的返回值会影响行为：[`pre_tool_call`](#pre_tool_call) 可以 **拦截** 工具，[`pre_llm_call`](#pre_llm_call) 可以 **注入上下文** 到 LLM 调用。其他所有 hook 都是即发即弃的观察者。

### 快速参考

| Hook | 触发时机 | 返回值 |
|------|-----------|---------|
| [`pre_tool_call`](#pre_tool_call) | 任意工具执行前 | `{"action": "block", "message": str}` 否决调用 |
| [`post_tool_call`](#post_tool_call) | 任意工具返回后 | 忽略 |
| [`pre_llm_call`](#pre_llm_call) | 每轮一次，工具调用循环前 | `{"context": str}` 将上下文前置到用户消息 |
| [`post_llm_call`](#post_llm_call) | 每轮一次，工具调用循环后 | 忽略 |
| [`on_session_start`](#on_session_start) | 新会话创建时（仅首回合） | 忽略 |
| [`on_session_end`](#on_session_end) | 会话结束时 | 忽略 |
| [`on_session_finalize`](#on_session_finalize) | CLI/gateway 销毁活跃会话（刷新、保存、统计） | 忽略 |
| [`on_session_reset`](#on_session_reset) | Gateway 更换新 session key（如 `/new`、`/reset`） | 忽略 |
| [`subagent_stop`](#subagent_stop) | `delegate_task` 子 agent 退出后 | 忽略 |
| [`pre_gateway_dispatch`](#pre_gateway_dispatch) | Gateway 收到用户消息，在鉴权 + 分发前 | `{"action": "skip" \| "rewrite" \| "allow", ...}` 影响流程 |
| [`pre_approval_request`](#pre_approval_request) | 危险命令需要用户审批，在发送提示/通知前 | 忽略 |
| [`post_approval_response`](#post_approval_response) | 用户响应审批提示（或超时）后 | 忽略 |
| [`transform_tool_result`](#transform_tool_result) | 任意工具返回后，在结果交还给 model 前 | `str` 替换结果，`None` 保持不变 |
| [`transform_terminal_output`](#transform_terminal_output) | `terminal` 工具内部，在截断/ANSI 剥离/脱敏前 | `str` 替换原始输出，`None` 保持不变 |
| [`transform_llm_output`](#transform_llm_output) | 工具调用循环完成后，在最终响应交付前 | `str` 替换响应文本，`None`/空字符串 保持不变 |

---

### `pre_tool_call`

在每次工具执行 **之前** 立即触发 — 包括内置工具和 plugin 工具。

**回调签名：**

```python
def my_callback(tool_name: str, args: dict, task_id: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | `str` | 即将执行的工具名称（如 `"terminal"`、`"web_search"`、`"read_file"`） |
| `args` | `dict` | Model 传递给工具的参数 |
| `task_id` | `str` | Session/task 标识符。未设置时为空字符串。 |

**触发位置：** `model_tools.py` 的 `handle_function_call()` 中，在工具 handler 运行之前。每次工具调用触发一次 — 如果 model 并行调用 3 个工具，则触发 3 次。

**返回值 — 否决调用：**

```python
return {"action": "block", "message": "工具调用被拦截的原因"}
```

Agent 会将工具短路，并将 `message` 作为错误返回给 model。第一个匹配的 block 指令生效（Python plugin 优先注册，然后是 shell hooks）。任何其他返回值都会被忽略，因此现有的纯观察回调无需改动即可继续工作。

**使用场景：** 日志、审计追踪、工具调用计数、拦截危险操作、限流、按用户策略执行。

**示例 — 工具调用审计日志：**

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

**示例 — 危险工具告警：**

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

在每次工具执行返回 **之后** 立即触发。

**回调签名：**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str,
                duration_ms: int, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | `str` | 刚执行的工具名称 |
| `args` | `dict` | Model 传递给工具的参数 |
| `result` | `str` | 工具的返回值（始终是 JSON 字符串） |
| `task_id` | `str` | Session/task 标识符。未设置时为空字符串。 |
| `duration_ms` | `int` | 工具分发耗时，毫秒（在 `registry.dispatch()` 周围使用 `time.monotonic()` 测量）。 |

**触发位置：** `model_tools.py` 的 `handle_function_call()` 中，在工具 handler 返回之后。每次工具调用触发一次。如果工具抛出未处理的异常，则 **不会** 触发（错误被捕获并作为 error JSON 字符串返回，此时 `post_tool_call` 会以该 error 字符串作为 `result` 触发）。

**返回值：** 忽略。

**使用场景：** 记录工具结果、指标收集、追踪工具成功/失败率、延迟面板、按工具预算告警、特定工具完成时发送通知。

**示例 — 追踪工具使用指标：**

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

每轮 **一次**，在工具调用循环开始 **之前** 触发。这是 **唯一返回值会被使用** 的 hook — 它可以向当前回合的用户消息注入上下文。

**回调签名：**

```python
def my_callback(session_id: str, user_message: str, conversation_history: list,
                is_first_turn: bool, model: str, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | 当前会话的唯一标识符 |
| `user_message` | `str` | 本回合用户的原始消息（在任何 skill 注入之前） |
| `conversation_history` | `list` | 完整消息列表的副本（OpenAI 格式：`[{"role": "user", "content": "..."}]`） |
| `is_first_turn` | `bool` | 新会话的首回合为 `True`，后续回合为 `False` |
| `model` | `str` | Model 标识符（如 `"anthropic/claude-sonnet-4.6"`） |
| `platform` | `str` | 会话运行位置：`"cli"`、`"telegram"`、`"discord"` 等 |

**触发位置：** `run_agent.py` 的 `run_conversation()` 中，在上下文压缩之后、主 `while` 循环之前。每次 `run_conversation()` 调用触发一次（即每用户回合一次），而非工具循环内的每次 API 调用。

**返回值：** 如果回调返回包含 `"context"` 键的字典，或非空普通字符串，文本将追加到当前回合的用户消息。返回 `None` 表示不注入。

```python
# 注入上下文
return {"context": "Recalled memories:\n- User likes Python\n- Working on hermes-agent"}

# 普通字符串（等效）
return "Recalled memories:\n- User likes Python"

# 不注入
return None
```

**上下文注入位置：** 始终注入到 **用户消息**，而非 system prompt。这保留了 prompt cache — system prompt 在各回合间保持不变，因此缓存的 token 可被复用。System prompt 是 Hermes 的领域（model 指引、工具强制、个性、skills）。Plugin 的上下文与用户输入并列。

所有注入的上下文都是 **临时的** — 仅在 API 调用时添加。Conversation history 中的原始用户消息不会被修改，且不会持久化到会话数据库。

当 **多个 plugin** 返回上下文时，它们的输出按 plugin 发现顺序（按目录名字母顺序）以双换行连接。

**使用场景：** 记忆召回、RAG 上下文注入、护栏、每轮分析。

**示例 — 记忆召回：**

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

**示例 — 护栏：**

```python
POLICY = "Never execute commands that delete files without explicit user confirmation."

def guardrails(**kwargs):
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", guardrails)
```

---

### `post_llm_call`

每轮 **一次**，在工具调用循环完成且 agent 生成最终响应 **之后** 触发。仅在 **成功** 的回合触发 — 回合被中断时不触发。

**回调签名：**

```python
def my_callback(session_id: str, user_message: str, assistant_response: str,
                conversation_history: list, model: str, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | 当前会话的唯一标识符 |
| `user_message` | `str` | 本回合用户的原始消息 |
| `assistant_response` | `str` | 本回合 agent 的最终文本响应 |
| `conversation_history` | `list` | 回合完成后的完整消息列表副本 |
| `model` | `str` | Model 标识符 |
| `platform` | `str` | 会话运行位置 |

**触发位置：** `run_agent.py` 的 `run_conversation()` 中，在工具循环以最终响应退出之后。受 `if final_response and not interrupted` 保护 — 因此当用户中途打断回合，或 agent 达到迭代限制且未生成响应时，**不会** 触发。

**返回值：** 忽略。

**使用场景：** 将对话数据同步到外部记忆系统、计算响应质量指标、记录回合摘要、触发后续操作。

**示例 — 同步到外部记忆：**

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

**示例 — 追踪响应长度：**

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

在全新会话创建时 **触发一次**。在会话继续时（用户在现有会话中发送第二条消息）**不会** 触发。

**回调签名：**

```python
def my_callback(session_id: str, model: str, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | 新会话的唯一标识符 |
| `model` | `str` | Model 标识符 |
| `platform` | `str` | 会话运行位置 |

**触发位置：** `run_agent.py` 的 `run_conversation()` 中，新会话的首回合期间 — 具体在 system prompt 构建之后、工具循环开始之前。检查条件是 `if not conversation_history`（无先前消息 = 新会话）。

**返回值：** 忽略。

**使用场景：** 初始化会话级状态、预热缓存、向外部服务注册会话、记录会话启动。

**示例 — 初始化会话缓存：**

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

在每个 `run_conversation()` 调用的 **最末尾** 触发，无论结果如何。如果用户在 agent 处理中途退出，也会从 CLI 的退出 handler 触发。

**回调签名：**

```python
def my_callback(session_id: str, completed: bool, interrupted: bool,
                model: str, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | 会话的唯一标识符 |
| `completed` | `bool` | Agent 生成最终响应时为 `True`，否则为 `False` |
| `interrupted` | `bool` | 回合被中断（用户发送新消息、`/stop` 或退出）时为 `True` |
| `model` | `str` | Model 标识符 |
| `platform` | `str` | 会话运行位置 |

**触发位置：** 两个地方：
1. **`run_agent.py`** — 每个 `run_conversation()` 调用的末尾，在所有清理之后。始终触发，即使回合出错。
2. **`cli.py`** — CLI 的 atexit handler 中，但 **仅** 在退出时 agent 正在运行中（`_agent_running=True`）。这能捕获处理期间的 Ctrl+C 和 `/exit`。此时 `completed=False` 且 `interrupted=True`。

**返回值：** 忽略。

**使用场景：** 刷新缓冲区、关闭连接、持久化会话状态、记录会话时长、清理 `on_session_start` 中初始化的资源。

**示例 — 刷新与清理：**

```python
_session_caches = {}

def cleanup_session(session_id, completed, interrupted, **kwargs):
    cache = _session_caches.pop(session_id, None)
    if cache:
        # 将累积数据刷新到磁盘或外部服务
        status = "completed" if completed else ("interrupted" if interrupted else "failed")
        print(f"Session {session_id} ended: {status}, {cache['tool_calls']} tool calls")

def register(ctx):
    ctx.register_hook("on_session_end", cleanup_session)
```

**示例 — 会话时长追踪：**

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

当 CLI 或 gateway **销毁** 活跃会话时触发 — 例如用户运行 `/new`、gateway GC 了空闲会话，或 CLI 在有活跃 agent 时退出。这是在会话身份消失前，刷新与 outgoing session 关联的 state 的最后机会。

**回调签名：**

```python
def my_callback(session_id: str | None, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` or `None` | 正在退出的会话 ID。如果没有活跃会话，可能为 `None`。 |
| `platform` | `str` | `"cli"` 或消息平台名称（`"telegram"`、`"discord"` 等）。 |

**触发位置：** `cli.py`（`/new` / CLI 退出时）和 `gateway/run.py`（会话被重置或 GC 时）。在 gateway 侧始终与 `on_session_reset` 配对出现。

**返回值：** 忽略。

**使用场景：** 在会话 ID 被丢弃前持久化最终会话指标、关闭每会话资源、发送最终遥测事件、排空队列写入。

---

### `on_session_reset`

当 gateway **为活跃聊天更换新 session key** 时触发 — 用户调用了 `/new`、`/reset`、`/clear`，或 adapter 在空闲窗口后选择了新会话。这让 plugin 能在对话状态被清除时立即做出反应，而无需等待下一个 `on_session_start`。

**回调签名：**

```python
def my_callback(session_id: str, platform: str, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `session_id` | `str` | 新会话的 ID（已轮换为新值）。 |
| `platform` | `str` | 消息平台名称。 |

**触发位置：** `gateway/run.py` 中，新 session key 分配之后、下一条入站消息处理之前。在 gateway 上顺序为：`on_session_finalize(old_id)` → swap → `on_session_reset(new_id)` → 首回合的 `on_session_start(new_id)`。

**返回值：** 忽略。

**使用场景：** 重置以 `session_id` 为 key 的每会话缓存、发送"会话轮换"分析事件、初始化新的 state bucket。

---

查看 **[Build a Plugin guide](/guides/build-a-hermes-plugin)** 获取完整教程，包括工具 schema、handler 和高级 hook 模式。

---

### `subagent_stop`

在 `delegate_task` 子 agent 完成后 **每个子 agent 触发一次**。无论你是委派单个任务还是批量三个，此 hook 都会为每个子 agent 触发一次，在父线程上串行执行。

**回调签名：**

```python
def my_callback(parent_session_id: str, child_role: str | None,
                child_summary: str | None, child_status: str,
                duration_ms: int, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `parent_session_id` | `str` | 委派父 agent 的会话 ID |
| `child_role` | `str \| None` | 设置在子 agent 上的 orchestrator role 标签（如果功能未启用则为 `None`） |
| `child_summary` | `str \| None` | 子 agent 返回给父 agent 的最终响应 |
| `child_status` | `str` | `"completed"`、`"failed"`、`"interrupted"` 或 `"error"` |
| `duration_ms` | `int` | 子 agent 运行的 wall-clock 时间，毫秒 |

**触发位置：** `tools/delegate_tool.py` 中，在 `ThreadPoolExecutor.as_completed()` 排空所有子 future 之后。触发被调度到父线程，因此 hook 作者无需考虑并发回调执行。

**返回值：** 忽略。

**使用场景：** 记录编排活动、累积子 agent 时长用于计费、编写委派后审计记录。

**示例 — 记录编排器活动：**

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
在重度委派场景下（如 orchestrator roles × 5 leaves × 嵌套深度），`subagent_stop` 每回合会触发多次。保持回调轻量；将耗时操作推送到后台队列。
:::

---

### `pre_gateway_dispatch`

在 gateway 中每个入站 `MessageEvent` 触发 **一次**，在内部事件守卫 **之后**、鉴权/配对和 agent 分发 **之前**。这是 gateway 级消息流策略（仅监听窗口、人工交接、按聊天路由等）的拦截点，这些策略不适合放在单个平台 adapter 中。

**回调签名：**

```python
def my_callback(event, gateway, session_store, **kwargs):
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `event` | `MessageEvent` | 规范化的入站消息（具有 `.text`、`.source`、`.message_id`、`.internal` 等属性）。 |
| `gateway` | `GatewayRunner` | 活跃的 gateway runner，plugin 可通过 `gateway.adapters[platform].send(...)` 调用侧信道回复（如 owner 通知等）。 |
| `session_store` | `SessionStore` | 用于通过 `session_store.append_to_transcript(...)` 静默摄入 transcript。 |

**触发位置：** `gateway/run.py` 的 `GatewayRunner._handle_message()` 中，在 `is_internal` 计算之后。**内部事件完全跳过此 hook**（它们是系统生成的 — 后台进程完成等 — 不应被用户策略把关）。

**返回值：** `None` 或 dict。第一个被识别的 action dict 生效；剩余的 plugin 结果会被忽略。Plugin 回调中的异常会被捕获并记录；出错时 gateway 始终回退到正常分发。

| Return | Effect |
|--------|--------|
| `{"action": "skip", "reason": "..."}` | 丢弃消息 — 无 agent 回复、无配对流程、无鉴权。假设 plugin 已处理（如静默摄入到 transcript）。 |
| `{"action": "rewrite", "text": "new text"}` | 替换 `event.text`，然后继续正常分发修改后的事件。适用于将缓冲的环境消息折叠为单个 prompt。 |
| `{"action": "allow"}` / `None` | 正常分发 — 运行完整的鉴权 / 配对 / agent-loop 链。 |

**使用场景：** 仅监听群聊（仅在被 @ 时响应；将环境消息缓冲到上下文中）；人工交接（owner 手动处理聊天时静默摄入客户消息）；按 profile 限流；策略驱动路由。

**示例 — 静默丢弃未授权的 DM，不触发配对码：**

```python
def deny_unauthorized_dms(event, **kwargs):
    src = event.source
    if src.chat_type == "dm" and not _is_approved_user(src.user_id):
        return {"action": "skip", "reason": "unauthorized-dm"}
    return None

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", deny_unauthorized_dms)
```

**示例 — 在被 @ 时将环境消息缓冲区重写为单个 prompt：**

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

在审批请求展示给用户 **之前** 立即触发 — 覆盖所有界面：交互式 CLI、Ink TUI、gateway 平台（Telegram、Discord、Slack、WhatsApp、Matrix 等）和 ACP 客户端（VS Code、Zed、JetBrains）。

这是接入自定义通知器的合适位置 — 例如，一个在菜单栏弹出允许/拒绝通知的 macOS 应用，或一个记录每次审批请求及上下文的审计日志。

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

| Parameter | Type | Description |
|-----------|------|-------------|
| `command` | `str` | 等待审批的 shell 命令 |
| `description` | `str` | 命令被标记的人工可读原因（多个 pattern 匹配时合并） |
| `pattern_key` | `str` | 触发审批的主 pattern key（如 `"rm_rf"`、`"sudo"`） |
| `pattern_keys` | `list[str]` | 所有匹配的 pattern keys |
| `session_key` | `str` | 会话标识符，用于按聊天范围通知 |
| `surface` | `str` | `"cli"` 表示交互式 CLI/TUI 提示，`"gateway"` 表示异步平台审批 |

**返回值：** 忽略。此处的 hook 仅为观察者；不能否决或预回答审批。使用 [`pre_tool_call`](#pre_tool_call) 在请求到达审批系统前拦截工具。

**使用场景：** 桌面通知、推送告警、审计日志、Slack webhook、升级路由、指标。

**示例 — macOS 桌面通知：**

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

在用户响应审批提示（或提示超时）**之后** 触发。

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

与 `pre_approval_request` 的 kwargs 相同，外加：

| Parameter | Type | Description |
|-----------|------|-------------|
| `choice` | `str` | `"once"`、`"session"`、`"always"`、`"deny"` 或 `"timeout"` 之一 |

**返回值：** 忽略。

**使用场景：** 关闭匹配的桌面通知、在审计日志中记录最终决定、更新指标、推进限流器。

```python
def log_decision(command, choice, session_key, **kwargs):
    logger.info("approval %s: %s for session %s", choice, command[:60], session_key)

def register(ctx):
    ctx.register_hook("post_approval_response", log_decision)
```

---

### `transform_tool_result`

在工具返回 **之后**、结果追加到对话 **之前** 触发。让 plugin 能在 model 看到之前重写 **任意** 工具的结果字符串 — 不仅限于 terminal 输出。

**回调签名：**

```python
def my_callback(
    tool_name: str,
    arguments: dict,
    result: str,
    task_id: str | None,
    **kwargs,
) -> str | None:
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | `str` | 产生结果的工具（`read_file`、`web_extract`、`delegate_task` …）。 |
| `arguments` | `dict` | Model 调用工具时使用的参数。 |
| `result` | `str` | 工具的原始结果字符串，已截断和 ANSI 剥离后。 |
| `task_id` | `str \| None` | 在 RL/benchmark 环境中运行时的 task/session ID。 |

**返回值：** `str` 替换结果（model 看到的是返回的字符串），`None` 保持不变。

**使用场景：** 从 `web_extract` 输出中脱敏组织特定的 PII、为长 JSON 工具响应包装摘要 header、向 `read_file` 结果注入检索增强提示、将 `delegate_task` 子 agent 报告重写为项目特定的 schema。

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

适用于每个工具。对于仅 terminal 的重写，请参见下方的 `transform_terminal_output` — 它更窄，且在流水线中运行得更早（截断前、脱敏前）。

---

### `transform_terminal_output`

在 `terminal` 工具的前台输出流水线内部触发，在默认 50 KB 截断、ANSI 剥离和 secret 脱敏 **之前**。让 plugin 能在任何下游处理触及 shell 命令的原始 stdout/stderr 之前重写它。

**回调签名：**

```python
def my_callback(
    command: str,
    output: str,
    exit_code: int,
    cwd: str,
    task_id: str | None,
    **kwargs,
) -> str | None:
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `command` | `str` | 产生输出的 shell 命令。 |
| `output` | `str` | 原始合并 stdout/stderr（可能非常大 — 截断在 hook 之后发生）。 |
| `exit_code` | `int` | 进程退出码。 |
| `cwd` | `str` | 命令运行的工作目录。 |

**返回值：** `str` 替换输出，`None` 保持不变。

**使用场景：** 为产生巨量输出的命令注入摘要（`du -ah`、`find`、`tree`）、用项目特定的标记标记输出以便下游 hook 知道如何处理、剥离在运行间抖动并破坏 prompt cache 的时间噪声。

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

与 `transform_tool_result`（覆盖所有其他工具）配合使用效果更佳。

---

### `transform_llm_output`

每轮 **一次**，在工具调用循环完成且 model 生成最终响应 **之后**、该响应交付给用户（CLI、gateway 或程序化调用者）**之前** 触发。让 plugin 能用经典编程方法重写 assistant 的最终文本 — 无需在 SOUL 风格文本或 skill 驱动的 transform 上消耗额外的推理 token。

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

| Parameter | Type | Description |
|-----------|------|-------------|
| `response_text` | `str` | 本回合 assistant 的最终响应文本。 |
| `session_id` | `str` | 本次对话的会话 ID（单次运行可能为空）。 |
| `model` | `str` | 产生响应的 model 名称（如 `anthropic/claude-sonnet-4.6`）。 |
| `platform` | `str` | 交付平台（`cli`、`telegram`、`discord` …；未设置时为空）。 |

**返回值：** 非空 `str` 替换响应文本，`None` 或空字符串保持不变。多个 plugin 注册时 **第一个非空字符串生效** — 与 `transform_tool_result` 一致。

**使用场景：** 应用个性/词汇 transform（海盗语、海绵宝宝）、从最终文本中脱敏用户特定标识符、追加项目特定的签名 footer、在不消耗 SOUL 指令 token 的情况下强制执行 house style guide。

```python
import os, re

def spongebob(response_text, **kwargs):
    if os.environ.get("SPONGEBOB_MODE") != "on":
        return None  # 原样透传
    return re.sub(r"!", "!! Tartar sauce!", response_text)

def register(ctx):
    ctx.register_hook("transform_llm_output", spongebob)
```

此 hook 在非空、非中断响应时触发 — 在停止按钮中断或空回合时不会触发。异常会被记录为警告且不会中断 agent 执行。

---

## Shell Hooks

在 `cli-config.yaml` 中声明 shell-script hooks，Hermes 会在对应的 plugin-hook 事件触发时将其作为子进程运行 — 在 CLI 和 gateway 会话中均有效。无需编写 Python plugin。

当你希望用即插即用的单文件脚本（Bash、Python、任何带 shebang 的语言）实现以下功能时，使用 shell hooks：

- **拦截工具调用** — 拒绝危险的 `terminal` 命令、强制执行按目录策略、要求对破坏性的 `write_file` / `patch` 操作进行审批。
- **在工具调用后运行** — 自动格式化 agent 刚写入的 Python 或 TypeScript 文件、记录 API 调用、触发 CI 工作流。
- **向下一个 LLM 回合注入上下文** — 将 `git status` 输出、当前工作日或检索到的文档前置到用户消息（参见 [`pre_llm_call`](#pre_llm_call)）。
- **观察生命周期事件** — 在 subagent 完成（`subagent_stop`）或会话启动（`on_session_start`）时写入日志行。

Shell hooks 通过在 CLI 启动（`hermes_cli/main.py`）和 gateway 启动（`gateway/run.py`）时调用 `agent.shell_hooks.register_from_config(cfg)` 注册。它们与 Python plugin hooks 自然组合 — 两者都通过同一个分发器流转。

### 一览对比

| 维度 | Shell hooks | [Plugin hooks](#plugin-hooks) | [Gateway hooks](#gateway-event-hooks) |
|-----------|-------------|-------------------------------|---------------------------------------|
| 声明位置 | `~/.hermes/config.yaml` 中的 `hooks:` 块 | `plugin.yaml` plugin 中的 `register()` | `HOOK.yaml` + `handler.py` 目录 |
| 所在目录 | `~/.hermes/agent-hooks/`（约定） | `~/.hermes/plugins/<name>/` | `~/.hermes/hooks/<name>/` |
| 语言 | 任意（Bash、Python、Go 二进制等） | 仅 Python | 仅 Python |
| 运行环境 | CLI + Gateway | CLI + Gateway | 仅 Gateway |
| 事件 | `VALID_HOOKS`（含 `subagent_stop`） | `VALID_HOOKS` | Gateway 生命周期（`gateway:startup`、`agent:*`、`command:*`） |
| 可拦截工具调用 | 是（`pre_tool_call`） | 是（`pre_tool_call`） | 否 |
| 可注入 LLM 上下文 | 是（`pre_llm_call`） | 是（`pre_llm_call`） | 否 |
| 同意机制 | 每个 `(event, command)` 对首次使用时提示 | 隐式（信任 Python plugin） | 隐式（信任目录） |
| 进程间隔离 | 是（子进程） | 否（进程内） | 否（进程内） |

### 配置 schema

```yaml
hooks:
  <event_name>:                  # 必须在 VALID_HOOKS 中
    - matcher: "<regex>"         # 可选；仅用于 pre/post_tool_call
      command: "<shell command>" # 必需；通过 shlex.split 运行，shell=False
      timeout: <seconds>         # 可选；默认 60，上限 300

hooks_auto_accept: false         # 参见下方的"同意模型"
```

事件名称必须是 [plugin hook 事件](#plugin-hooks) 之一；拼写错误会产生 "Did you mean X?" 警告并被跳过。单条 entry 中的未知 key 会被忽略；缺少 `command` 会跳过并发出警告。`timeout > 300` 会被钳制并发出警告。

### JSON 通信协议

每次事件触发时，Hermes 会为每个匹配的 hook（matcher 允许时）生成一个子进程，将 JSON payload 通过 **stdin** 传入，并从 **stdout** 读取 JSON 返回。

**stdin — 脚本接收的 payload：**

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

对于非工具事件（`pre_llm_call`、`subagent_stop`、会话生命周期），`tool_name` 和 `tool_input` 为 `null`。`extra` dict 携带所有事件特定的 kwargs（`user_message`、`conversation_history`、`child_role`、`duration_ms` …）。不可序列化的值会被字符串化而非省略。

**stdout — 可选响应：**

```jsonc
// 拦截 pre_tool_call（两种格式均接受；内部统一标准化）：
{"decision": "block", "reason":  "Forbidden: rm -rf"}   // Claude-Code 风格
{"action":   "block", "message": "Forbidden: rm -rf"}   // Hermes 规范风格

// 为 pre_llm_call 注入上下文：
{"context": "Today is Friday, 2026-04-17"}

// 静默无操作 — 任何空/不匹配输出均可：
```

JSON 格式错误、非零退出码和超时会记录警告，但永远不会中止 agent 循环。

### 完整示例

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

Agent 的上下文内文件视图 **不会** 自动重新读取 — 重新格式化仅影响磁盘上的文件。后续的 `read_file` 调用会获取格式化后的版本。

#### 2. 拦截破坏性的 `terminal` 命令

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

#### 3. 每轮注入 `git status`（Claude-Code `UserPromptSubmit` 等效）

```yaml
hooks:
  pre_llm_call:
    - command: "~/.hermes/agent-hooks/inject-cwd-context.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/inject-cwd-context.sh
cat - >/dev/null   # 丢弃 stdin payload
if status=$(git status --porcelain 2>/dev/null) && [[ -n "$status" ]]; then
  jq --null-input --arg s "$status" \
     '{context: ("Uncommitted changes in cwd:\n" + $s)}'
else
  printf '{}\n'
fi
```

Claude Code 的 `UserPromptSubmit` 事件在 Hermes 中不单独存在 — `pre_llm_call` 在同一位置触发，且已支持上下文注入。在此使用即可。

#### 4. 记录每个 subagent 完成

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

### 同意模型

每个唯一的 `(event, command)` 对在 Hermes 首次遇到时会提示用户审批，随后将决定持久化到 `~/.hermes/shell-hooks-allowlist.json`。后续运行（CLI 或 gateway）会跳过提示。

三个逃生口可绕过交互式提示 — 满足任意一个即可：

1. CLI 上的 `--accept-hooks` 标志（如 `hermes --accept-hooks chat`）
2. `HERMES_ACCEPT_HOOKS=1` 环境变量
3. `cli-config.yaml` 中的 `hooks_auto_accept: true`

非 TTY 运行（gateway、cron、CI）需要三者之一 — 否则任何新添加的 hook 会静默保持未注册状态并记录警告。

**脚本编辑被静默信任。** allowlist 以精确的命令字符串为 key，而非脚本哈希，因此编辑磁盘上的脚本不会使同意失效。`hermes hooks doctor` 会标记 mtime 变更，以便你发现编辑并决定是否重新审批。

### `hermes hooks` CLI

| Command | 功能 |
|---------|--------------|
| `hermes hooks list` | 输出已配置的 hooks，含 matcher、timeout 和同意状态 |
| `hermes hooks test <event> [--for-tool X] [--payload-file F]` | 针对合成 payload 触发每个匹配的 hook 并打印解析后的响应 |
| `hermes hooks revoke <command>` | 移除所有匹配 `<command>` 的 allowlist 条目（下次重启生效） |
| `hermes hooks doctor` | 对每个已配置的 hook：检查执行权限、allowlist 状态、mtime 变更、JSON 输出有效性和大致执行时间 |

### 安全

Shell hooks 以 **你的完整用户凭证** 运行 — 与 cron 条目或 shell alias 处于同一信任边界。将 `config.yaml` 中的 `hooks:` 块视为特权配置：

- 仅引用你编写或完整审查过的脚本。
- 将脚本放在 `~/.hermes/agent-hooks/` 内，以便路径易于审计。
- 拉取共享配置后重新运行 `hermes hooks doctor`，以便在 hooks 注册前发现新增项。
- 如果 config.yaml 在团队间版本控制，审查更改 `hooks:` 部分的 PR，如同审查 CI 配置一样。

### 排序与优先级

Python plugin hooks 和 shell hooks 都通过同一个 `invoke_hook()` 分发器流转。Python plugin 先注册（`discover_and_load()`），shell hooks 后注册（`register_from_config()`），因此 Python 的 `pre_tool_call` block 决定在平票情况下优先。第一个有效的 block 获胜 — 一旦任何回调产生 `{"action": "block", "message": str}` 且消息非空，聚合器立即返回。
