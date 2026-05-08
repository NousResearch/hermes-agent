---
sidebar_position: 6
title: "事件钩子"
description: "在关键生命周期点运行自定义代码——记录活动、发送警报、发布到 webhook"
---

# 事件钩子

Hermes 有三个钩子系统，在关键生命周期点运行自定义代码：

| 系统 | 通过以下注册 | 运行位置 | 用例 |
|--------|---------------|---------|----------|
| **[网关事件钩子](#gateway-event-hooks)** | `~/.hermes/hooks/` 中的 `HOOK.yaml` + `handler.py` | 仅网关 | 记录、警报、webhook |
| **[插件钩子](#plugin-hooks)** | [插件](/docs/user-guide/features/plugins)中的 `ctx.register_hook()` | CLI + 网关 | 工具拦截、指标、护栏 |
| **[Shell 钩子](#shell-hooks)** | `~/.hermes/config.yaml` 中指向 shell 脚本的 `hooks:` 块 | CLI + 网关 | 即插即用脚本用于阻止、自动格式化、上下文注入 |

所有三个系统都是非阻塞的——任何钩子中的错误都会被捕获并记录，从不让 agent 崩溃。

## 网关事件钩子

网关钩子在网关运行期间自动触发（Telegram、Discord、Slack、WhatsApp、Teams），不阻塞主 agent 管道。

### 创建钩子

每个钩子是 `~/.hermes/hooks/` 下包含两个文件的一个目录：

```text
~/.hermes/hooks/
└── my-hook/
    ├── HOOK.yaml      # 声明监听哪些事件
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

`events` 列表决定哪些事件触发处理程序。您可以订阅任何事件组合，包括通配符如 `command:*`。

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

**处理程序规则：**
- 必须命名为 `handle`
- 接收 `event_type`（字符串）和 `context`（字典）
- 可以是 `async def` 或普通 `def`——两者都可用
- 错误被捕获并记录，从不让 agent 崩溃

### 可用事件

| 事件 | 触发时机 | 上下文键 |
|-------|---------------|--------------|
| `gateway:startup` | 网关进程启动 | `platforms`（活动平台名称列表） |
| `session:start` | 创建新消息会话 | `platform`、`user_id`、`session_id`、`session_key` |
| `session:end` | 会话结束（重置前） | `platform`、`user_id`、`session_key` |
| `session:reset` | 用户运行 `/new` 或 `/reset` | `platform`、`user_id`、`session_key` |
| `agent:start` | Agent 开始处理消息 | `platform`、`user_id`、`session_id`、`message` |
| `agent:step` | 工具调用循环的每次迭代 | `platform`、`user_id`、`session_id`、`iteration`、`tool_names` |
| `agent:end` | Agent 完成处理 | `platform`、`user_id`、`session_id`、`message`、`response` |
| `command:*` | 执行的任何斜杠命令 | `platform`、`user_id`、`command`、`args` |

#### 通配符匹配

为 `command:*` 注册的处理程序对任何 `command:` 事件触发（`command:model`、`command:reset` 等）。用单一订阅监控所有斜杠命令。

### 示例

#### 长时间任务的 Telegram 警报

当 agent 花费超过 10 步时给自己发送消息：

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

#### 命令使用记录器

跟踪使用了哪些斜杠命令：

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

#### 会话启动 Webhook

在新会话时 POST 到外部服务：

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

### 教程：BOOT.md — 在每个网关启动时运行启动检查清单

社区的一个流行模式：在 `~/.hermes/BOOT.md` 放置一个 Markdown 检查清单，让 agent 在网关每次启动时运行一次。对于"每次启动，检查隔夜 cron 失败，如果任何失败就在 Discord #ops 上 ping 我"，或"汇总 deploy.log 最近 24 小时的内容并发布到 Slack #ops"，很有用。

本教程展示如何作为用户定义的钩子自己构建。Hermes 不附带内置 BOOT.md 钩子——您连接您想要的精确行为。

#### 我们要构建的内容

1. `~/.hermes/BOOT.md` 中的自然语言启动说明文件。
2. 在 `gateway:startup` 时触发的网关钩子，使用您的网关解析的模型/凭证生成一次性 agent，并运行 BOOT.md 说明。
3. `[SILENT]` 约定，让 agent 在无需报告时选择退出发送消息。

#### 步骤 1：编写检查清单

创建 `~/.hermes/BOOT.md`。像给人类助手说明一样写：

```markdown
# Startup Checklist

1. Run `hermes cron list` and check if any scheduled jobs failed overnight.
2. If any failed, send a summary to Discord #ops using the `send_message` tool.
3. Check if `/opt/app/deploy.log` has any ERROR lines from the last 24 hours. If yes, summarize them and include in the same Discord message.
4. If nothing went wrong, reply with only `[SILENT]` so no message is sent.
```

Agent 将其作为提示的一部分看到，因此您可以用纯语言描述任何 agent 可以推理的内容——工具调用、shell 命令、发送消息、汇总文件。

#### 步骤 2：创建钩子

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

- `_resolve_gateway_model()` 读取网关当前配置的模型。
- `_resolve_runtime_agent_kwargs()` 以与正常网关轮次相同的方式解析提供商凭证——包括 API 密钥、base URL、OAuth 令牌和凭证池。

没有这些，裸 `AIAgent()` 回退到内置默认值，会对任何非默认端点返回 401。

#### 步骤 3：测试

重启网关：

```bash
hermes gateway restart
```

查看日志：

```bash
hermes logs --follow --level INFO | grep boot-md
```

您应该看到 `Running BOOT.md (N chars)`，然后是 `boot-md completed: ...`（agent 所做工作的摘要）或 `boot-md completed (nothing to report)`（agent 回复了 `[SILENT]`）。

删除 `~/.hermes/BOOT.md` 以禁用检查清单——钩子保持加载但当文件不存在时静默跳过。

#### 扩展模式

- **计划感知检查清单：** 在 BOOT.md 说明内按键 `datetime.now().weekday()`（"如果是周一，还要检查每周部署日志"）。说明是自由格式文本，因此 agent 可以推理的任何内容都可行。
- **多个检查清单：** 将钩子指向不同文件（`STARTUP.md`、`MORNING.md` 等）并为每个注册单独的钩子目录。
- **非 agent 变体：** 如果您不需要完整的 agent 循环，完全跳过 `AIAgent`，让处理程序直接通过 `httpx` 发布固定通知。更便宜、更快，零提供商依赖。

#### 为什么这不是内置

早期版本的 Hermes 将其作为内置钩子发货，在每次网关启动时静默使用裸默认值生成 agent。这让有自定义端点的用户惊讶，并让不知道它运行的用户看不见。保持为记录的模式——由您在自己的钩子目录中构建——意味着您确切看到它做什么并通过编写文件来选择加入。

### 工作原理

1. 在网关启动时，`HookRegistry.discover_and_load()` 扫描 `~/.hermes/hooks/`
2. 每个带有 `HOOK.yaml` + `handler.py` 的子目录被动态加载
3. 处理程序注册到其声明的事件
4. 在每个生命周期点，`hooks.emit()` 触发所有匹配的处理程序
5. 任何处理程序中的错误被捕获并记录——损坏的钩子从不让 agent 崩溃

:::info
网关钩子仅在**网关**中触发（Telegram、Discord、Slack、WhatsApp、Teams）。CLI 不加载网关钩子。对于各处都工作的钩子，使用[插件钩子](#plugin-hooks)。
:::

## 插件钩子

[插件](/docs/user-guide/features/plugins) 可以注册在 **CLI 和网关**会话中触发的钩子。这些通过插件 `register()` 函数中的 `ctx.register_hook()` 以编程方式注册。

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", my_tool_observer)
    ctx.register_hook("post_tool_call", my_tool_logger)
    ctx.register_hook("pre_llm_call", my_memory_callback)
    ctx.register_hook("post_llm_call", my_sync_callback)
    ctx.register_hook("on_session_start", my_init_callback)
    ctx.register_hook("on_session_end", my_cleanup_callback)
```

**所有钩子的一般规则：**

- 回调接收**关键字参数**。始终接受 `**kwargs` 以保持向前兼容——新参数可能在未来版本中添加而不会破坏您的插件。
- 如果回调**崩溃**，它被记录并跳过。其他钩子和 agent 正常继续。行为不当的插件永远不能破坏 agent。
- 两个钩子的返回值影响行为：[`pre_tool_call`](#pre_tool_call) 可以**阻止**工具，[`pre_llm_call`](#pre_llm_call) 可以**注入上下文**到 LLM 调用。所有其他钩子都是即发即忘观察者。

### 快速参考

| 钩子 | 触发时机 | 返回值 |
|------|-----------|---------|
| [`pre_tool_call`](#pre_tool_call) | 任何工具执行前 | `{"action": "block", "message": str}` 以否决调用 |
| [`post_tool_call`](#post_tool_call) | 任何工具返回后 | 忽略 |
| [`pre_llm_call`](#pre_llm_call) | 工具调用循环前每轮一次 | `{"context": str}` 以前置上下文到用户消息 |
| [`post_llm_call`](#post_llm_call) | 工具调用循环完成后每轮一次 | 忽略 |
| [`on_session_start`](#on_session_start) | 创建新会话时（仅首次轮次） | 忽略 |
| [`on_session_end`](#on_session_end) | 会话结束 | 忽略 |
| [`on_session_finalize`](#on_session_finalize) | CLI/网关拆除活动会话时（刷新、保存、统计） | 忽略 |
| [`on_session_reset`](#on_session_reset) | 网关交换新会话密钥时（如 `/new`、`/reset`） | 忽略 |
| [`subagent_stop`](#subagent_stop) | `delegate_task` 子代理已退出 | 忽略 |
| [`pre_gateway_dispatch`](#pre_gateway_dispatch) | 网关收到用户消息，在 auth + 分派前 | `{"action": "skip" \| "rewrite" \| "allow", ...}` 以影响流程 |
| [`pre_approval_request`](#pre_approval_request) | 危险命令需要用户批准，在发送提示/通知前 | 忽略 |
| [`post_approval_response`](#post_approval_response) | 用户响应批准提示（或超时）后 | 忽略 |
| [`transform_tool_result`](#transform_tool_result) | 任何工具返回后，在结果交给模型前 | `str` 替换结果，`None` 保持不变 |
| [`transform_terminal_output`](#transform_terminal_output) | 在 `terminal` 工具内，在截断/ANSI 剥离/编辑前 | `str` 替换原始输出，`None` 保持不变 |
| [`transform_llm_output`](#transform_llm_output) | 工具调用循环完成后，在最终响应传递给用户前 | `str` 替换响应文本，`None`/空 保持不变 |

---

### `pre_tool_call`

**立即在**每个工具执行前触发——内置工具和插件工具。

**回调签名：**

```python
def my_callback(tool_name: str, args: dict, task_id: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `tool_name` | `str` | 即将执行的工具名称（如 `"terminal"`、`"web_search"`、`"read_file"`） |
| `args` | `dict` | 模型传递给工具的参数 |
| `task_id` | `str` | 会话/任务标识符。未设置时为空字符串。 |

**触发：** 在 `model_tools.py` 中 `handle_function_call()` 内部，工具的处理程序运行前。每工具调用触发一次——如果模型并行调用 3 个工具，这触发 3 次。

**返回值——否决调用：**

```python
return {"action": "block", "message": "Reason the tool call was blocked"}
```

Agent 用 `message` 作为返回给模型的错误短路工具。第一个匹配的阻止指令优先（Python 插件先注册，然后是 shell 钩子）。任何其他返回值被忽略，因此现有纯观察者回调保持不变地工作。

**用例：** 记录、审计跟踪、工具调用计数器、阻止危险操作、速率限制、每用户策略执行。

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

**示例 — 危险工具警告：**

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

**立即在**每个工具执行返回后触发。

**回调签名：**

```python
def my_callback(tool_name: str, args: dict, result: str, task_id: str,
                duration_ms: int, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `tool_name` | `str` | 刚执行的工具名称 |
| `args` | `dict` | 模型传递给工具的参数 |
| `result` | `str` | 工具的返回值（始终是 JSON 字符串） |
| `task_id` | `str` | 会话/任务标识符。未设置时为空字符串。 |
| `duration_ms` | `int` | 工具调度花费的时间，毫秒（用 `time.monotonic()` 在 `registry.dispatch()` 周围测量）。 |

**触发：** 在 `model_tools.py` 中 `handle_function_call()` 内部，工具的处理程序返回后。每工具调用触发一次。如果工具抛出未处理异常**不**触发（错误被捕获并作为错误 JSON 字符串返回，`post_tool_call` 用该错误字符串作为 `result` 触发）。

**返回值：** 忽略。

**用例：** 记录工具结果、指标收集、跟踪工具成功率/失败率、延迟仪表板、每工具预算警报、在特定工具完成时发送通知。

**示例 — 跟踪工具使用指标：**

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

**每轮一次**，在工具调用循环开始前。这是**唯一使用返回值**的钩子——它可以将上下文注入当前轮次的用户消息。

**回调签名：**

```python
def my_callback(session_id: str, user_message: str, conversation_history: list,
                is_first_turn: bool, model: str, platform: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `session_id` | `str` | 当前会话的唯一标识符 |
| `user_message` | `str` | 本轮次用户的原始消息（任何技能注入前） |
| `conversation_history` | `list` | 完整消息列表的副本（OpenAI 格式：`[{"role": "user", "content": "..."}]`） |
| `is_first_turn` | `bool` | 如果是新会话的首次轮次则为 `True`，后续轮次为 `False` |
| `model` | `str` | 模型标识符（如 `"anthropic/claude-sonnet-4.6"`） |
| `platform` | `str` | 会话运行位置：`"cli"`、`"telegram"`、`"discord"` 等 |

**触发：** 在 `run_agent.py` 中 `run_conversation()` 内部，上下文压缩后但在主 `while` 循环前。每 `run_conversation()` 调用触发一次（每用户轮次一次），不是工具循环内每次 API 调用一次。

**返回值：** 如果回调返回带有 `"context"` 键的字典，或普通非空字符串，文本被追加到当前轮次的用户消息。返回 `None` 不注入。

```python
# 注入上下文
return {"context": "Recalled memories:\n- User likes Python\n- Working on hermes-agent"}

# 普通字符串（等效）
return "Recalled memories:\n- User likes Python"

# 不注入
return None
```

**注入位置：** 始终是**用户消息**，而不是系统提示。这保留提示缓存——系统提示在轮次之间保持相同，因此缓存的令牌被重用。系统提示是 Hermes 的领地（模型指导、工具执行、人格、技能）。插件与用户输入一起贡献上下文。

所有注入的上下文是**临时的**——仅在 API 调用时添加。对话历史中的原始用户消息从不变更，没有任何东西被持久化到会话数据库。

当**多个插件**返回上下文时，它们的输出按插件发现顺序（目录名字母顺序）用双换行连接。

**用例：** 记忆召回、RAG 上下文注入、护栏、每轮分析。

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

**每轮一次**，在工具调用循环完成且 agent 产生最终响应后。仅在**成功**的轮次触发——如果轮次被中断**不**触发。

**回调签名：**

```python
def my_callback(session_id: str, user_message: str, assistant_response: str,
                conversation_history: list, model: str, platform: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `session_id` | `str` | 当前会话的唯一标识符 |
| `user_message` | `str` | 用户的原始消息 |
| `assistant_response` | `str` | agent 本轮次的最终文本响应 |
| `conversation_history` | `list` | 轮次完成后完整消息列表的副本 |
| `model` | `str` | 模型标识符 |
| `platform` | `str` | 会话运行位置 |

**触发：** 在 `run_agent.py` 中 `run_conversation()` 内部，工具循环退出并产生最终响应后。受 `if final_response and not interrupted` 保护——因此当用户在轮次中途中断或 agent 达到迭代限制而未产生响应时**不**触发。

**返回值：** 忽略。

**用例：** 将对话数据同步到外部记忆系统、计算响应质量指标、记录轮次摘要、触发后续操作。

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
        pass  # 尽力而为

def register(ctx):
    ctx.register_hook("post_llm_call", sync_memory)
```

**示例 — 跟踪响应长度：**

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

**首次**创建全新会话时触发。**不**在会话延续时触发（当用户在现有会话中发送第二条消息时）。

**回调签名：**

```python
def my_callback(session_id: str, model: str, platform: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `session_id` | `str` | 新会话的唯一标识符 |
| `model` | `str` | 模型标识符 |
| `platform` | `str` | 会话运行位置 |

**触发：** 在 `run_agent.py` 中 `run_conversation()` 内部，新会话的首次轮次——具体是在系统提示构建后但工具循环开始前。检查是 `if not conversation_history`（无先前消息 = 新会话）。

**返回值：** 忽略。

**用例：** 初始化会话作用域状态、预热缓存、向外部服务注册会话、记录会话开始。

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

在每个 `run_conversation()` 调用的**最末尾**触发，无论结果如何。如果用户退出时 agent 正在轮次中，也从 CLI 的退出处理程序触发。

**回调签名：**

```python
def my_callback(session_id: str, completed: bool, interrupted: bool,
                model: str, platform: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `session_id` | `str` | 会话的唯一标识符 |
| `completed` | `bool` | 如果 agent 产生了最终响应则为 `True`，否则为 `False` |
| `interrupted` | `bool` | 如果轮次被中断（用户发送新消息、`/stop` 或退出）则为 `True` |
| `model` | `str` | 模型标识符 |
| `platform` | `str` | 会话运行位置 |

**触发：** 在两个地方：
1. **`run_agent.py`** — 在每个 `run_conversation()` 调用的末尾，所有清理之后。总是触发，即使轮次出错。
2. **`cli.py`** — 在 CLI 的 atexit 处理程序中，但**仅当**退出发生时 agent 正在轮次中（`_agent_running=True`）。这捕获 Ctrl+C 和处理期间 `/exit`。在这种情况下 `completed=False` 且 `interrupted=True`。

**返回值：** 忽略。

**用例：** 刷新缓冲区、关闭连接、持久化会话状态、记录会话持续时间、清理在 `on_session_start` 中初始化的资源。

**示例 — 刷新和清理：**

```python
_session_caches = {}

def cleanup_session(session_id, completed, interrupted, **kwargs):
    cache = _session_caches.pop(session_id, None)
    if cache:
        # 将累积的数据刷新到磁盘或外部服务
        status = "completed" if completed else ("interrupted" if interrupted else "failed")
        print(f"Session {session_id} ended: {status}, {cache['tool_calls']} tool calls")

def register(ctx):
    ctx.register_hook("on_session_end", cleanup_session)
```

**示例 — 会话持续时间跟踪：**

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

在 CLI 或网关**拆除**活动会话时触发——例如当用户运行 `/new`、网关 GC 清理空闲会话、或 CLI 在活动 agent 退出时。这是在传出会话的身份消失之前刷新与该会话绑定的状态的最后机会。

**回调签名：**

```python
def my_callback(session_id: str | None, platform: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `session_id` | `str` 或 `None` | 传出会话的 ID。如果没有活动会话，则为 `None`。 |
| `platform` | `str` | `"cli"` 或消息平台名称（`"telegram"`、`"discord"` 等）。 |

**触发：** 在 `cli.py`（`/new` / CLI 退出）和 `gateway/run.py`（会话重置或 GC）时。在网关端与 `on_session_reset` 配对。

**返回值：** 忽略。

**用例：** 在会话 ID 被丢弃前持久化最终会话指标、关闭每会话资源、发出最终遥测事件、排空排队写入。

---

### `on_session_reset`

当网关**交换新会话密钥**给活动聊天时触发——用户调用了 `/new`、`/reset`、`/clear`、或适配器在空闲窗口后选择了新会话。这让插件对会话状态已被清除这一事实做出反应，而不必等待下一个 `on_session_start`。

**回调签名：**

```python
def my_callback(session_id: str, platform: str, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `session_id` | `str` | 新会话的 ID（已轮换到新值）。 |
| `platform` | `str` | 消息平台名称。 |

**触发：** 在 `gateway/run.py` 中，新会话密钥分配后但在处理下一个入站消息前。在网关上，顺序是：`on_session_finalize(old_id)` → 交换 → `on_session_reset(new_id)` → 首个入站轮次上的 `on_session_start(new_id)`。

**返回值：** 忽略。

**用例：** 重置按 `session_id` 键控的每会话缓存、发出"会话轮换"分析、初始化新的状态桶。

---

参见**[构建插件指南](/docs/guides/build-a-hermes-plugin)**获取包括工具 schema、处理程序和高级钩子模式的完整演练。

---

### `subagent_stop`

在 `delegate_task` 完成后**每个子代理一次**触发。无论您委托了单个任务还是三个批次，此钩子对每个子代理触发一次，在父线程上串行化。

**回调签名：**

```python
def my_callback(parent_session_id: str, child_role: str | None,
                child_summary: str | None, child_status: str,
                duration_ms: int, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `parent_session_id` | `str` | 委托父代理的会话 ID |
| `child_role` | `str \| None` | 在子代理上设置的协调器角色标签（功能未启用时为 `None`） |
| `child_summary` | `str \| None` | 子代理返回给父代理的最终响应 |
| `child_status` | `str` | `"completed"`、`"failed"`、`"interrupted"` 或 `"error"` |
| `duration_ms` | `int` | 运行子代理的墙上时间，毫秒 |

**触发：** 在 `tools/delegate_tool.py` 中 `ThreadPoolExecutor.as_completed()` 排空所有子 futures 后。回调在父线程上编组，以便钩子作者不必推理并发回调执行。

**返回值：** 忽略。

**用例：** 记录协调活动、累积子持续时间用于计费、编写后委托审计记录。

**示例 — 记录协调器活动：**

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
对于重型委托（如协调器角色 × 5 叶 × 嵌套深度），`subagent_stop` 每轮触发很多次。保持回调快速；将昂贵的工作推到后台队列。
:::

---

### `pre_gateway_dispatch`

在网关中**每个传入 `MessageEvent` 一次**触发，在内部事件守卫之后但在 auth/配对和 agent 分派**之前**。这是网关级消息流策略（仅监听组聊、人工交接、每聊天路由等）的拦截点，这些不能干净地适配任何单一平台适配器。

**回调签名：**

```python
def my_callback(event, gateway, session_store, **kwargs):
```

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `event` | `MessageEvent` | 规范化入站消息（有 `.text`、`.source`、`.message_id`、`.internal` 等）。 |
| `gateway` | `GatewayRunner` | 活动网关运行器，让插件可以调用 `gateway.adapters[platform].send(...)` 进行侧通道回复（所有者通知等）。 |
| `session_store` | `SessionStore` | 用于通过 `session_store.append_to_transcript(...)` 静默摄入脚本。 |

**触发：** 在 `gateway/run.py` 中 `GatewayRunner._handle_message()` 内部，在计算 `is_internal` 后。**内部事件完全跳过钩子**（它们是系统生成的——后台进程完成等——不应被面向用户的策略把关）。

**返回值：** `None` 或字典。第一个识别的动作字典胜出；剩余插件结果被忽略。插件回调中的异常被捕获并记录；网关在错误时总是正常分派。

| 返回 | 效果 |
|--------|--------|
| `{"action": "skip", "reason": "..."}` | 丢弃消息——无 agent 回复，无配对流程，无 auth。假设插件已处理（如静默摄入到脚本中）。 |
| `{"action": "rewrite", "text": "new text"}` | 替换 `event.text`，然后用修改后的事件继续正常分派。对于将缓冲的环境消息折叠为提及时的单一提示很有用。 |
| `{"action": "allow"}` / `None` | 正常分派——运行完整 auth / 配对 / agent 循环链。 |

**用例：** 仅监听组聊（仅在提及时回复；将环境消息缓冲到上下文）；人工交接（静默摄入客户消息而所有者手动处理聊天）；按画像速率限制；策略驱动路由。

**示例 — 静默丢弃未授权 DM 而不触发配对代码：**

```python
def deny_unauthorized_dms(event, **kwargs):
    src = event.source
    if src.chat_type == "dm" and not _is_approved_user(src.user_id):
        return {"action": "skip", "reason": "unauthorized-dm"}
    return None

def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", deny_unauthorized_dms)
```

**示例 — 将环境消息缓冲重写为提及时的单一提示：**

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

**立即在**向用户显示批准请求**前**触发——覆盖每个表面：交互式 CLI、Ink TUI、网关平台（Telegram、Discord、Slack、WhatsApp、Matrix 等）和 ACP 客户端（VS Code、Zed、JetBrains）。

这是连接自定义通知器的好地方——例如弹出允许/拒绝通知的 macOS 菜单栏应用，或记录每个带有上下文的批准请求的审计日志。

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
|-----------|------|-------------|
| `command` | `str` | 等待批准的 shell 命令 |
| `description` | `str` | 命令被标记的人类可读原因（多个模式匹配时合并） |
| `pattern_key` | `str` | 触发批准的主要模式键（如 `"rm_rf"`、`"sudo"`） |
| `pattern_keys` | `list[str]` | 所有匹配的模式键 |
| `session_key` | `str` | 会话标识符，可用于按聊天范围通知 |
| `surface` | `str` | `"cli"` 表示交互式 CLI/TUI 提示，`"gateway"` 表示异步平台批准 |

**返回值：** 忽略。钩子在这里是纯观察者；它们不能否决或预回答批准。使用 [`pre_tool_call`](#pre_tool_call) 在工具到达批准系统前阻止。

**用例：** 桌面通知、推送警报、审计日志、Slack webhook、升级路由、指标。

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

在用户响应批准提示**后**触发（或提示超时）。

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

与 `pre_approval_request` 相同的 kwargs，加上：

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `choice` | `str` | `"once"`、`"session"`、`"always"`、`"deny"` 或 `"timeout"` 之一 |

**返回值：** 忽略。

**用例：** 关闭匹配桌面通知、在审计日志中记录最终决定、更新指标、向前滚动速率限制器。

```python
def log_decision(command, choice, session_key, **kwargs):
    logger.info("approval %s: %s for session %s", choice, command[:60], session_key)

def register(ctx):
    ctx.register_hook("post_approval_response", log_decision)
```

---

### `transform_tool_result`

**在工具返回后**且**结果被追加到对话前**触发。让插件重写**任何**工具的结果字符串——不仅是终端输出——在模型看到之前。

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

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `tool_name` | `str` | 生成结果的工具（`read_file`、`web_extract`、`delegate_task`、……）。 |
| `arguments` | `dict` | 模型调用工具时使用的参数。 |
| `result` | `str` | 工具的原始结果字符串，截断和 ANSI 剥离后。 |
| `task_id` | `str \| None` | 在 RL/基准环境中运行时的任务/会话 ID。 |

**返回值：** `str` 替换结果（返回的字符串是模型看到的），`None` 保持不变。

**用例：** 从 `web_extract` 输出中编辑组织特定 PII、在长 JSON 工具响应前注入摘要头、向 `read_file` 结果注入 RAG 提示、将 `delegate_task` 子代理报告重写为项目特定 schema。

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

适用于每个工具。纯终端重写见下方 `transform_terminal_output`——它更窄且在管道中更早运行（截断前、编辑前）。

---

### `transform_terminal_output`

在 `terminal` 工具的前景输出管道内触发，**在**默认 50 KB 截断、ANSI 剥离和密钥编辑**之前**。让插件在下游处理之前重写 shell 命令的原始 stdout/stderr。

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

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `command` | `str` | 生成输出的 shell 命令。 |
| `output` | `str` | 原始组合 stdout/stderr（可能非常大——截断发生在钩子之后）。 |
| `exit_code` | `int` | 进程退出码。 |
| `cwd` | `str` | 命令运行的工作目录。 |

**返回值：** `str` 替换输出，`None` 保持不变。

**用例：** 为产生大量输出的命令注入摘要（`du -ah`、`find`、`tree`）、用项目特定标记标记输出以便下游钩子知道如何处理、剥离在运行之间跳动并破坏提示缓存的时间噪声。

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

与 `transform_tool_result`（覆盖每个其他工具）配合良好。

---

### `transform_llm_output`

在工具调用循环完成且模型产生最终响应后**每轮一次**触发，在该响应传递给用户（CLI、网关或编程调用者）**前**。让插件使用经典编程方法重写助手的最终文本——不燃烧额外推理令牌在 SOUL 风格文本或技能驱动的转换上。

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
|-----------|------|-------------|
| `response_text` | `str` | 本轮次助手的最终响应文本。 |
| `session_id` | `str` | 此对话的会话 ID（一次性运行时可能为空）。 |
| `model` | `str` | 生成响应的模型名（如 `anthropic/claude-sonnet-4.6`）。 |
| `platform` | `str` | 传递平台（`cli`、`telegram`、`discord`、……；未设置时为空）。 |

**返回值：** 非空 `str` 替换响应文本，`None` 或空字符串保持不变。当多个插件注册时**第一个非空字符串胜出**——与 `transform_tool_result` 一致。

**用例：** 应用人格/词汇转换（海盗语、海绵宝宝）、从最终文本中编辑用户特定标识符、追加项目特定签名页脚、强制执行房屋风格指南而不燃烧令牌用于 SOUL 说明。

```python
import os, re

def spongebob(response_text, **kwargs):
    if os.environ.get("SPONGEBOB_MODE") != "on":
        return None  # 原样传递
    return re.sub(r"!", "!! Tartar sauce!", response_text)

def register(ctx):
    ctx.register_hook("transform_llm_output", spongebob)
```

钩子受非空、非中断响应保护——在停止按钮中断或空轮次上不会触发。异常记录为警告，不破坏 agent 执行。

---

## Shell 钩子

在 `cli-config.yaml` 中声明 shell 脚本钩子，Hermes 会在相应插件钩子事件触发时作为子进程运行它们——在 CLI 和网关会话中。无需编写 Python 插件。

当您想要即插即用的单文件脚本（Bash、Python、任何带 shebang 的东西）时使用 shell 钩子：

- **阻止工具调用** — 拒绝危险的 `terminal` 命令、执行每目录策略、要求对破坏性 `write_file` / `patch` 操作进行批准。
- **在工具调用后运行** — 自动格式化 agent 刚写入的 Python 或 TypeScript 文件、记录 API 调用、触发 CI 工作流。
- **将上下文注入下一个 LLM 轮次** — 将 `git status` 输出、当前工作日或检索到的文档前置到用户消息（见 [`pre_llm_call`](#pre_llm_call)）。
- **观察生命周期事件** — 在子代理完成时（`subagent_stop`）或会话开始时（`on_session_start`）写入日志行。

Shell 钩子通过在 CLI 启动（`hermes_cli/main.py`）和网关启动（`gateway/run.py`）时调用 `agent.shell_hooks.register_from_config(cfg)` 注册。它们与 Python 插件钩子自然组合——两者都通过同一调度器流动。

### 一览比较

| 维度 | Shell 钩子 | [插件钩子](#plugin-hooks) | [网关钩子](#gateway-event-hooks) |
|-----------|-------------|-------------------------------|---------------------------------------|
| 声明位置 | `~/.hermes/config.yaml` 中的 `hooks:` 块 | `plugin.yaml` 插件中的 `register()` | `HOOK.yaml` + `handler.py` 目录 |
| 位置 | `~/.hermes/agent-hooks/`（约定） | `~/.hermes/plugins/<name>/` | `~/.hermes/hooks/<name>/` |
| 语言 | 任意（Bash、Python、Go 二进制等） | 仅 Python | 仅 Python |
| 运行位置 | CLI + 网关 | CLI + 网关 | 仅网关 |
| 事件 | `VALID_HOOKS`（含 `subagent_stop`） | `VALID_HOOKS` | 网关生命周期（`gateway:startup`、`agent:*`、`command:*`） |
| 可阻止工具调用 | 是（`pre_tool_call`） | 是（`pre_tool_call`） | 否 |
| 可注入 LLM 上下文 | 是（`pre_llm_call`） | 是（`pre_llm_call`） | 否 |
| 同意 | 每个 `(event, command)` 对首次使用提示 | 隐式（Python 插件信任） | 隐式（目录信任） |
| 进程间隔离 | 是（子进程） | 否（进程内） | 否（进程内） |

### 配置模式

```yaml
hooks:
  <event_name>:                  # 必须在 VALID_HOOKS 中
    - matcher: "<regex>"         # 可选；仅用于 pre/post_tool_call
      command: "<shell command>" # 必需；通过 shlex.split 运行，shell=False
      timeout: <seconds>         # 可选；默认 60，上限 300

hooks_auto_accept: false         # 见下方"同意模型"
```

事件名称必须是[插件钩子事件](#plugin-hooks)之一；拼写错误产生"Did you mean X?"警告并被跳过。单个条目内未知键被忽略；缺少 `command` 是跳过并警告。`timeout > 300` 被限制并警告。

### JSON 线协议

每次事件触发时，Hermes 为每个匹配钩子（matcher 允许时）生成子进程，将 JSON 负载通过 **stdin** 传递，读取 **stdout** 作为 JSON 返回。

**stdin — 脚本接收的负载：**

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

`tool_name` 和 `tool_input` 对于非工具事件（`pre_llm_call`、`subagent_stop`、会话生命周期）为 `null`。`extra` 字典携带所有事件特定 kwargs（`user_message`、`conversation_history`、`child_role`、`duration_ms`、……）。不可序列化的值被字符串化而非省略。

**stdout — 可选响应：**

```jsonc
// 阻止 pre_tool_call（接受两种形状；内部规范化）：
{"decision": "block", "reason":  "Forbidden: rm -rf"}   // Claude-Code 风格
{"action":   "block", "message": "Forbidden: rm -rf"}   // Hermes 规范

// 为 pre_llm_call 注入上下文：
{"context": "Today is Friday, 2026-04-17"}

// 静默无操作——任何空输出或无匹配输出都可以：
```

格式错误的 JSON、非零退出码和超时记录警告但从不中止 agent 循环。

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

Agent 的文件上下文中视图**不会**自动重新读取——重新格式化仅影响磁盘上的文件。后续 `read_file` 调用会获取格式化版本。

#### 2. 阻止破坏性 `terminal` 命令

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

#### 3. 将 `git status` 注入每个轮次（Claude-Code `UserPromptSubmit` 等效）

```yaml
hooks:
  pre_llm_call:
    - command: "~/.hermes/agent-hooks/inject-cwd-context.sh"
```

```bash
#!/usr/bin/env bash
# ~/.hermes/agent-hooks/inject-cwd-context.sh
cat - >/dev/null   # 丢弃 stdin 负载
if status=$(git status --porcelain 2>/dev/null) && [[ -n "$status" ]]; then
  jq --null-input --arg s "$status" \
     '{context: ("Uncommitted changes in cwd:\n" + $s)}'
else
  printf '{}\n'
fi
```

Claude Code 的 `UserPromptSubmit` 事件故意不是单独的 Hermes 事件——`pre_llm_call` 在同一位置触发，已支持上下文注入。在此使用它。

#### 4. 记录每个子代理完成

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

每个唯一的 `(event, command)` 对在 Hermes 首次看到时提示用户批准，然后持久化决定到 `~/.hermes/shell-hooks-allowlist.json`。后续运行（CLI 或网关）跳过提示。

三个逃生舱口绕过交互式提示——任一都足够：

1. CLI 上的 `--accept-hooks` 标志（如 `hermes --accept-hooks chat`）
2. `HERMES_ACCEPT_HOOKS=1` 环境变量
3. `cli-config.yaml` 中的 `hooks_auto_accept: true`

非 TTY 运行（网关、cron、CI）需要三者之一——否则任何新添加的钩子静默保持未注册并记录警告。

**脚本编辑被静默信任。** 允许列表按键的确切命令字符串索引，而非脚本哈希，因此编辑磁盘上的脚本不会使同意无效。`hermes hooks doctor` 标记 mtime 漂移，以便您可以发现编辑并决定是否重新批准。

### `hermes hooks` CLI

| 命令 | 功能 |
|---------|--------------|
| `hermes hooks list` | 转储配置的钩子，含 matcher、timeout 和同意状态 |
| `hermes hooks test <event> [--for-tool X] [--payload-file F]` | 用合成负载触发每个匹配钩子并打印解析的响应 |
| `hermes hooks revoke <command>` | 移除每个匹配 `<command>` 的允许列表条目（下次重启生效） |
| `hermes hooks doctor` | 对每个配置的钩子：检查 exec 位、允许列表状态、mtime 漂移、JSON 输出有效性和粗略执行时间 |

### 安全性

Shell 钩子使用**您的完整用户凭证**运行——与 cron 条目或 shell 别名相同的信任边界。将 `config.yaml` 中的 `hooks:` 块视为特权配置：

- 仅引用您编写或完全审核的脚本。
- 将脚本保留在 `~/.hermes/agent-hooks/` 中以便审核路径。
- 在拉取共享配置后重新运行 `hermes hooks doctor` 以发现新添加的钩子后再注册。
- 如果您的 config.yaml 在团队间版本控制，请像审核 CI 配置一样审核 PR 中的 `hooks:` 部分。

### 排序和优先级

Python 插件钩子和 shell 钩子都通过同一 `invoke_hook()` 调度器流动。Python 插件先注册（`disconnect_and_load()`），shell 钩子后注册（`register_from_config()`），因此在平局情况下 Python `pre_tool_call` 阻止决定优先。第一个产生带非空消息的 `{"action": "block", "message": str}` 的回调胜出——聚合器在任何回调产生后立即返回。
