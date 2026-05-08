---
sidebar_position: 9
sidebar_label: "Build a Plugin"
title: "Build a Hermes Plugin"
description: "分步指南：构建具有工具、钩子、数据文件和技能的完整 Hermes 插件"
---

# 构建 Hermes 插件

本指南将引导你从头开始构建完整的 Hermes 插件。到最后，你将拥有一个包含多个工具、生命周期钩子、附带数据文件和捆绑技能的工作插件 — 插件系统支持的所有内容。

:::info 不确定需要哪个指南？
Hermes 有几种不同的可插拔接口 — 有些使用 Python `register_*` API，其他是配置驱动或拖放目录。首先使用此映射：

| 如果你想添加… | 阅读 |
|---|---|
| 自定义工具、钩子、斜杠命令、技能或 CLI 子命令 | **本指南**（通用插件接口） |
| **LLM / 推理后端**（新提供商） | [模型提供商插件](/docs/developer-guide/model-provider-plugin) |
| 一个 **gateway 频道**（Discord/Telegram/IRC/Teams/等） | [添加平台适配器](/docs/developer-guide/adding-platform-adapters) |
| 一个 **记忆后端**（Honcho/Mem0/Supermemory/等） | [记忆提供商插件](/docs/developer-guide/memory-provider-plugin) |
| 一个 **上下文压缩引擎** | [上下文引擎插件](/docs/developer-guide/context-engine-plugin) |
| 一个 **图像生成后端** | [图像生成提供商插件](/docs/developer-guide/image-gen-provider-plugin) |
| 一个 **TTS 后端**（任何 CLI — Piper、VoxCPM、Kokoro、语音克隆等） | [TTS 自定义命令提供商](/docs/user-guide/features/tts#custom-command-providers) — 配置驱动，无需 Python |
| 一个 **STT 后端**（自定义 whisper / ASR CLI） | [语音消息转录](/docs/user-guide/features/tts#voice-message-transcription-stt) — 将 `HERMES_LOCAL_STT_COMMAND` 设置为 shell 模板 |
| **通过 MCP 的外部工具**（文件系统、GitHub、Linear、任何 MCP 服务器） | [MCP](/docs/user-guide/features/mcp) — 在 `config.yaml` 中声明 `mcp_servers.<name>` |
| **Gateway 事件钩子**（在启动时、会话事件、命令上触发） | [事件钩子](/docs/user-guide/features/hooks#gateway-event-hooks) — 将 `HOOK.yaml` + `handler.py` 拖放到 `~/.hermes/hooks/<name>/` |
| **Shell 钩子**（在事件上运行 shell 命令） | [Shell 钩子](/docs/user-guide/features/hooks#shell-hooks) — 在 `config.yaml` 中的 `hooks:` 下声明 |
| **额外技能来源**（自定义 GitHub 仓库、私有技能索引） | [技能](/docs/user-guide/features/skills) — `hermes skills tap add <repo>` · [发布 tap](/docs/user-guide/features/skills#publishing-a-custom-skill-tap) |
| 一流 **核心** 推理提供商（不是插件） | [添加提供商](/docs/developer-guide/adding-providers) |

有关每个扩展接口（包括配置驱动（TTS、STT、MCP、shell 钩子）和拖放目录（gateway 钩子）样式的合并视图，请参阅完整的[可插拔接口表](/docs/user-guide/features/plugins#pluggable-interfaces--where-to-go-for-each)。
:::

## 你要构建什么

一个具有两个工具的**计算器**插件：
- `calculate` — 计算数学表达式（`2**16`、`sqrt(144)`、`pi * 5**2`）
- `unit_convert` — 在单元之间转换（`100 F → 37.78 C`、`5 km → 3.11 mi`）

另外还有一个记录每个工具调用的钩子，以及一个捆绑的技能文件。

## 步骤 1：创建插件目录

```bash
mkdir -p ~/.hermes/plugins/calculator
cd ~/.hermes/plugins/calculator
```

## 步骤 2：编写清单

创建 `plugin.yaml`：

```yaml
name: calculator
version: 1.0.0
description: Math calculator — evaluate expressions and convert units
provides_tools:
  - calculate
  - unit_convert
provides_hooks:
  - post_tool_call
```

这告诉 Hermes："我是一个名为 calculator 的插件，我提供工具和钩子。"`provides_tools` 和 `provides_hooks` 字段是插件注册的内容列表。

你可以添加的可选字段：
```yaml
author: Your Name
requires_env:          # 基于环境变量网关加载；在安装期间提示
  - SOME_API_KEY       # 简单格式 — 如果缺少则插件被禁用
  - name: OTHER_KEY    # 丰富格式 — 在安装期间显示说明/url
    description: "Key for the Other service"
    url: "https://other.com/keys"
    secret: true
```

## 步骤 3：编写工具模式

创建 `schemas.py` — 这是 LLM 读取以决定何时调用你的工具的内容：

```python
"""Tool schemas — what the LLM sees."""

CALCULATE = {
    "name": "calculate",
    "description": (
        "Evaluate a mathematical expression and return the result. "
        "Supports arithmetic (+, -, *, /, **), functions (sqrt, sin, cos, "
        "log, abs, round, floor, ceil), and constants (pi, e). "
        "Use this for any math the user asks about."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate (e.g., '2**10', 'sqrt(144)')",
            },
        },
        "required": ["expression"],
    },
}

UNIT_CONVERT = {
    "name": "unit_convert",
    "description": (
        "Convert a value between units. Supports length (m, km, mi, ft, in), "
        "weight (kg, lb, oz, g), temperature (C, F, K), data (B, KB, MB, GB, TB), "
        "and time (s, min, hr, day)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "The numeric value to convert",
            },
            "from_unit": {
                "type": "string",
                "description": "Source unit (e.g., 'km', 'lb', 'F', 'GB')",
            },
            "to_unit": {
                "type": "string",
                "description": "Target unit (e.g., 'mi', 'kg', 'C', 'MB')",
            },
        },
        "required": ["value", "from_unit", "to_unit"],
    },
}
```

**为什么模式很重要：**`description` 字段是 LLM 决定何时使用你的工具的方式。具体说明它的作用以及何时使用它。`parameters` 定义 LLM 传递的参数。

## 步骤 4：编写工具处理程序

创建 `tools.py` — 这是 LLM 调用你的工具时实际执行的代码：

```python
"""Tool handlers — the code that runs when the LLM calls each tool."""

import json
import math

# Safe globals for expression evaluation — no file/network access
_SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "pow": pow, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log2": math.log2, "log10": math.log10,
    "floor": math.floor, "ceil": math.ceil,
    "pi": math.pi, "e": math.e,
    "factorial": math.factorial,
}

def calculate(args: dict, **kwargs) -> str:
    """Evaluate a math expression safely.

    Rules for handlers:
    1. Receive args (dict) — the parameters the LLM passed
    2. Do the work
    3. Return a JSON string — ALWAYS, even on error
    4. Accept **kwargs for forward compatibility
    """
    expression = args.get("expression", "").strip()
    if not expression:
        return json.dumps({"error": "No expression provided"})

    try:
        result = eval(expression, {"__builtins__": {}}, _SAFE_MATH)
        return json.dumps({"expression": expression, "result": result})
    except ZeroDivisionError:
        return json.dumps({"expression": expression, "error": "Division by zero"})
    except Exception as e:
        return json.dumps({"expression": expression, "error": f"Invalid: {e}"})

# Conversion tables — values are in base units
_LENGTH = {"m": 1, "km": 1000, "mi": 1609.34, "ft": 0.3048, "in": 0.0254, "cm": 0.01}
_WEIGHT = {"kg": 1, "g": 0.001, "lb": 0.453592, "oz": 0.0283495}
_DATA = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
_TIME = {"s": 1, "ms": 0.001, "min": 60, "hr": 3600, "day": 86400}

def _convert_temp(value, from_u, to_u):
    # Normalize to Celsius
    c = {"F": (value - 32) * 5/9, "K": value - 273.15}.get(from_u, value)
    # Convert to target
    return {"F": c * 9/5 + 32, "K": c + 273.15}.get(to_u, c)

def unit_convert(args: dict, **kwargs) -> str:
    """Convert between units."""
    value = args.get("value")
    from_unit = args.get("from_unit", "").strip()
    to_unit = args.get("to_unit", "").strip()

    if value is None or not from_unit or not to_unit:
        return json.dumps({"error": "Need value, from_unit, and to_unit"})

    try:
        # Temperature
        if from_unit.upper() in {"C","F","K"} and to_unit.upper() in {"C","F","K"}:
            result = _convert_temp(float(value), from_unit.upper(), to_unit.upper())
            return json.dumps({"input": f"{value} {from_unit}", "result": round(result, 4),
                             "output": f"{round(result, 4)} {to_unit}"})

        # Ratio-based conversions
        for table in (_LENGTH, _WEIGHT, _DATA, _TIME):
            lc = {k.lower(): v for k, v in table.items()}
            if from_unit.lower() in lc and to_unit.lower() in lc:
                result = float(value) * lc[from_unit.lower()] / lc[to_unit.lower()]
                return json.dumps({"input": f"{value} {from_unit}",
                                 "result": round(result, 6),
                                 "output": f"{round(result, 6)} {to_unit}"})

        return json.dumps({"error": f"Cannot convert {from_unit} → {to_unit}"})
    except Exception as e:
        return json.dumps({"error": f"Conversion failed: {e}"})
```

**处理程序的关键规则：**
1. **签名：**`def my_handler(args: dict, **kwargs) -> str`
2. **返回：** 始终是 JSON 字符串。成功和错误都一样。
3. **永远不要引发：** 捕获所有异常，改为返回错误 JSON。
4. **接受 `**kwargs`:** Hermes 未来可能会传递额外上下文。

## 步骤 5：编写注册

创建 `__init__.py` — 这将模式连接到处理程序：

```python
"""Calculator plugin — registration."""

import logging

from . import schemas, tools

logger = logging.getLogger(__name__)

# Track tool usage via hooks
_call_log = []

def _on_post_tool_call(tool_name, args, result, task_id, **kwargs):
    """Hook: runs after every tool call (not just ours)."""
    _call_log.append({"tool": tool_name, "session": task_id})
    if len(_call_log) > 100:
        _call_log.pop(0)
    logger.debug("Tool called: %s (session %s)", tool_name, task_id)

def register(ctx):
    """Wire schemas to handlers and register hooks."""
    ctx.register_tool(name="calculate",    toolset="calculator",
                      schema=schemas.CALCULATE,    handler=tools.calculate)
    ctx.register_tool(name="unit_convert", toolset="calculator",
                      schema=schemas.UNIT_CONVERT, handler=tools.unit_convert)

    # This hook fires for ALL tool calls, not just ours
    ctx.register_hook("post_tool_call", _on_post_tool_call)
```

**`register()` 的作用：**
- 在启动时恰好调用一次
- `ctx.register_tool()` 将你的工具放入注册表 — 模型立即看到它
- `ctx.register_hook()` 订阅生命周期事件
- `ctx.register_cli_command()` 注册 CLI 子命令（例如 `hermes my-plugin <subcommand>`）
- `ctx.register_command()` 注册会话内斜杠命令（例如 CLI / gateway 聊天中的 `/myplugin <args>`）— 请参阅下面的[注册斜杠命令](#register-slash-commands)
- `ctx.dispatch_tool(name, arguments)` — 使用父 agent 的上下文（审批、凭据、task_id）自动连接，调用任何其他工具（内置的或来自其他插件的）。从需要调用 `terminal`、`read_file` 或任何其他工具的斜杠命令处理程序使用时非常有用，就像模型直接调用它一样。
- 如果此函数崩溃，插件被禁用但 Hermes 继续正常运行

**`dispatch_tool` 示例 — 调用工具的斜杠命令：**

```python
def handle_scan(ctx, argstr):
    """Implement /scan by invoking the terminal tool through the registry."""
    result = ctx.dispatch_tool("terminal", {"command": f"find . -name '{argstr}'"})
    return result  # returned to the caller's chat UI

def register(ctx):
    ctx.register_command("scan", handle_scan, help="Find files matching a glob")
```

被分派的工具通过正常的审批、编辑和预算管道 — 这是一个真正的工具调用，而不是绕过它们的快捷方式。

## 步骤 6：测试它

启动 Hermes：

```bash
hermes
```

你应该在横幅的工具列表中看到 `calculator: calculate, unit_convert`。

尝试这些提示：
```
What's 2 to the power of 16?
Convert 100 fahrenheit to celsius
What's the square root of 2 times pi?
How many gigabytes is 1.5 terabytes?
```

检查插件状态：
```
/plugins
```

输出：
```
Plugins (1):
  ✓ calculator v1.0.0 (2 tools, 1 hooks)
```

## 插件的最终结构

```
~/.hermes/plugins/calculator/
├── plugin.yaml      # "I'm calculator, I provide tools and hooks"
├── __init__.py      # Wiring: schemas → handlers, register hooks
├── schemas.py       # What the LLM reads (descriptions + parameter specs)
└── tools.py         # What runs (calculate, unit_convert functions)
```

四个文件，清晰的分离：
- **清单** 声明插件是什么
- **模式** 为 LLM 描述工具
- **处理程序** 实现实际逻辑
- **注册** 连接所有内容

## 插件还能做什么？

### 附带数据文件

将任何文件放在你的插件目录中并在导入时读取它们：

```python
# In tools.py or __init__.py
from pathlib import Path

_PLUGIN_DIR = Path(__file__).parent
_DATA_FILE = _PLUGIN_DIR / "data" / "languages.yaml"

with open(_DATA_FILE) as f:
    _DATA = yaml.safe_load(f)
```

### 捆绑技能

插件可以附带技能文件，agent 通过 `skill_view("plugin:skill")` 加载这些文件。在 `__init__.py` 中注册它们：

```
~/.hermes/plugins/my-plugin/
├── __init__.py
├── plugin.yaml
└── skills/
    ├── my-workflow/
    │   └── SKILL.md
    └── my-checklist/
        └── SKILL.md
```

```python
from pathlib import Path

def register(ctx):
    skills_dir = Path(__file__).parent / "skills"
    for child in sorted(skills_dir.iterdir()):
        skill_md = child / "SKILL.md"
        if child.is_dir() and skill_md.exists():
            ctx.register_skill(child.name, skill_md)
```

Agent 现在可以使用它们的命名空间名称加载你的技能：

```python
skill_view("my-plugin:my-workflow")   # → 插件的版本
skill_view("my-workflow")              # → 内置版本（未更改）
```

**关键属性：**
- 插件技能是**只读的** — 它们不会进入 `~/.hermes/skills/` 并且无法通过 `skill_manage` 编辑。
- 插件技能**不**列在系统提示词的 `<available_skills>` 索引中 — 它们是选择加入显式加载。
- 裸技能名称不受影响 — 命名空间防止与内置技能发生冲突。
- 当 agent 加载插件技能时，会预置一个捆绑上下文横幅，列出来自同一插件的其他技能。

:::tip 旧模式
旧的 `shutil.copy2` 模式（将技能复制到 `~/.hermes/skills/`）仍然有效，但会与内置技能产生名称冲突风险。对于新插件，首选 `ctx.register_skill()`。
:::

### 基于环境变量网关

如果你的插件需要 API 密钥：

```yaml
# plugin.yaml — 简单格式（向后兼容）
requires_env:
  - WEATHER_API_KEY
```

如果未设置 `WEATHER_API_KEY`，插件将被禁用并显示明确的消息。没有崩溃，agent 中没有错误 — 只是"Plugin weather disabled (missing: WEATHER_API_KEY)"。

当用户运行 `hermes plugins install` 时，他们会**以交互方式提示**任何缺少的 `requires_env` 变量。值会自动保存到 `.env`。

为了更好的安装体验，请使用带有说明和注册 URL 的丰富格式：

```yaml
# plugin.yaml — 丰富格式
requires_env:
  - name: WEATHER_API_KEY
    description: "API key for OpenWeather"
    url: "https://openweathermap.org/api"
    secret: true
```

| 字段 | 必需 | 说明 |
|-------|----------|-------------|
| `name` | 是 | 环境变量名称 |
| `description` | 否 | 在安装提示期间显示给用户 |
| `url` | 否 | 获取凭据的位置 |
| `secret` | 否 | 如果为 `true`，则输入被隐藏（如密码字段） |

两种格式可以在同一列表中混合。已经设置的变量会被静默跳过。

### 条件工具可用性

对于依赖于可选库的工具：

```python
ctx.register_tool(
    name="my_tool",
    schema={...},
    handler=my_handler,
    check_fn=lambda: _has_optional_lib(),  # False = 工具对模型隐藏
)
```

### 注册多个钩子

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", before_any_tool)
    ctx.register_hook("post_tool_call", after_any_tool)
    ctx.register_hook("pre_llm_call", inject_memory)
    ctx.register_hook("on_session_start", on_new_session)
    ctx.register_hook("on_session_end", on_session_end)
```

### 钩子参考

每个钩子在**[事件钩子参考](/docs/user-guide/features/hooks#plugin-hooks)**中有完整记录 — 回调签名、参数表、每个触发的时间和示例。以下是摘要：

| 钩子 | 触发时间 | 回调签名 | 返回 |
|------|-----------|-------------------|---------|
| [`pre_tool_call`](/docs/user-guide/features/hooks#pre_tool_call) | 在任何工具执行之前 | `tool_name: str, args: dict, task_id: str` | 忽略 |
| [`post_tool_call`](/docs/user-guide/features/hooks#post_tool_call) | 在任何工具返回之后 | `tool_name: str, args: dict, result: str, task_id: str, duration_ms: int` | 忽略 |
| [`pre_llm_call`](/docs/user-guide/features/hooks#pre_llm_call) | 每轮一次，在工具调用循环之前 | `session_id: str, user_message: str, conversation_history: list, is_first_turn: bool, model: str, platform: str` | [上下文注入](#pre_llm_call-context-injection) |
| [`post_llm_call`](/docs/user-guide/features/hooks#post_llm_call) | 每轮一次，在工具调用循环之后（仅成功轮次） | `session_id: str, user_message: str, assistant_response: str, conversation_history: list, model: str, platform: str` | 忽略 |
| [`on_session_start`](/docs/user-guide/features/hooks#on_session_start) | 创建新会话（仅第一轮） | `session_id: str, model: str, platform: str` | 忽略 |
| [`on_session_end`](/docs/user-guide/features/hooks#on_session_end) | 每个 `run_conversation` 调用 + CLI 退出的结束 | `session_id: str, completed: bool, interrupted: bool, model: str, platform: str` | 忽略 |
| [`on_session_finalize`](/docs/user-guide/features/hooks#on_session_finalize) | CLI/gateway 拆除活动会话 | `session_id: str \| None, platform: str` | 忽略 |
| [`on_session_reset`](/docs/user-guide/features/hooks#on_session_reset) | Gateway 换入新会话密钥（`/new`、`/reset`） | `session_id: str, platform: str` | 忽略 |

大多数钩子是触发即忘的观察者 — 它们的返回值被忽略。例外是 `pre_llm_call`，它可以向对话中注入上下文。

所有回调都应接受 `**kwargs` 以实现前向兼容性。如果钩子回调崩溃，它会被记录并跳过。其他钩子和 agent 继续正常运行。

### `pre_llm_call` 上下文注入

这是唯一返回值重要的钩子。当 `pre_llm_call` 回调返回带有 `"context"` 键（或纯字符串）的字典时，Hermes 将该文本注入到**当前轮次的用户消息**中。这是记忆插件、RAG 集成、guardrails 以及任何需要为模型提供额外上下文的插件的机制。

#### 返回格式

```python
# 带有 context 键的字典
return {"context": "Recalled memories:\n- User prefers dark mode\n- Last project: hermes-agent"}

# 纯字符串（等同于上面的字典形式）
return "Recalled memories:\n- User prefers dark mode"

# 返回 None 或不返回 → 无注入（仅观察者）
return None
```

任何非 None、非空返回（带有 `"context"` 键或纯非空字符串）都会被收集并附加到当前轮次的用户消息。

#### 注入如何工作

注入的上下文被附加到**用户消息**，而不是系统提示词。这是一个深思熟虑的设计选择：

- **提示词缓存保留** — 系统提示词在各轮次中保持相同。Anthropic 和 OpenRouter 缓存系统提示词前缀，因此保持其稳定可以节省多轮对话中 75%+ 的输入 token。如果插件修改了系统提示词，每一轮都将是缓存未命中。
- **临时性** — 注入仅发生在 API 调用时。对话历史中的原始用户消息永远不会突变，并且没有任何内容持久化到会话数据库中。
- **系统提示词是 Hermes 的领域** — 它包含模型特定的指导、工具强制执行规则、个性指令和缓存的技能内容。插件与用户的输入一起贡献上下文，而不是通过改变 agent 的核心指令。

#### 示例：记忆回忆插件

```python
"""Memory plugin — recalls relevant context from a vector store."""

import httpx

MEMORY_API = "https://your-memory-api.example.com"

def recall_context(session_id, user_message, is_first_turn, **kwargs):
    """Called before each LLM turn. Returns recalled memories."""
    try:
        resp = httpx.post(f"{MEMORY_API}/recall", json={
            "session_id": session_id,
            "query": user_message,
        }, timeout=3)
        memories = resp.json().get("results", [])
        if not memories:
            return None  # nothing to inject

        text = "Recalled context from previous sessions:\n"
        text += "\n".join(f"- {m['text']}" for m in memories)
        return {"context": text}
    except Exception:
        return None  # fail silently, don't break the agent

def register(ctx):
    ctx.register_hook("pre_llm_call", recall_context)
```

#### 示例：Guardrails 插件

```python
"""Guardrails plugin — enforces content policies."""

POLICY = """You MUST follow these content policies for this session:
- Never generate code that accesses the filesystem outside the working directory
- Always warn before executing destructive operations
- Refuse requests involving personal data extraction"""

def inject_guardrails(**kwargs):
    """Injects policy text into every turn."""
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", inject_guardrails)
```

#### 示例：仅观察者钩子（无注入）

```python
"""Analytics plugin — tracks turn metadata without injecting context."""

import logging
logger = logging.getLogger(__name__)

def log_turn(session_id, user_message, model, is_first_turn, **kwargs):
    """Fires before each LLM call. Returns None — no context injected."""
    logger.info("Turn: session=%s model=%s first=%s msg_len=%d",
                session_id, model, is_first_turn, len(user_message or ""))
    # No return → no injection

def register(ctx):
    ctx.register_hook("pre_llm_call", log_turn)
```

#### 多个插件返回上下文

当多个插件从 `pre_llm_call` 返回上下文时，它们的输出用双换行符连接并一起附加到用户消息。顺序遵循插件发现顺序（按插件目录名称字母顺序）。

### 注册 CLI 命令

插件可以添加自己的 `hermes <plugin>` 子命令树：

```python
def _my_command(args):
    """Handler for hermes my-plugin <subcommand>."""
    sub = getattr(args, "my_command", None)
    if sub == "status":
        print("All good!")
    elif sub == "config":
        print("Current config: ...")
    else:
        print("Usage: hermes my-plugin <status|config>")

def _setup_argparse(subparser):
    """Build the argparse tree for hermes my-plugin."""
    subs = subparser.add_subparsers(dest="my_command")
    subs.add_parser("status", help="Show plugin status")
    subs.add_parser("config", help="Show plugin config")
    subparser.set_defaults(func=_my_command)

def register(ctx):
    ctx.register_tool(...)
    ctx.register_cli_command(
        name="my-plugin",
        help="Manage my plugin",
        setup_fn=_setup_argparse,
        handler_fn=_my_command,
    )
```

注册后，用户可以运行 `hermes my-plugin status`、`hermes my-plugin config` 等。

**记忆提供商插件** 使用基于约定的方法：将 `register_cli(subparser)` 函数添加到插件的 `cli.py` 文件中。记忆插件发现系统会自动找到它 — 无需 `ctx.register_cli_command()` 调用。有关详细信息，请参阅[记忆提供商插件指南](/docs/developer-guide/memory-provider-plugin#adding-cli-commands)。

**活跃提供商网关：** 记忆插件 CLI 命令仅在其提供商是 config 中的活跃 `memory.provider` 时出现。如果用户尚未设置你的提供商，你的 CLI 命令不会弄乱帮助输出。

### 注册斜杠命令

插件可以注册会话内斜杠命令 — 用户在对话期间键入的命令（如 `/lcm status` 或 `/ping`）。这些在 CLI 和 gateway（Telegram、Discord 等）中都有效。

```python
def _handle_status(raw_args: str) -> str:
    """Handler for /mystatus — called with everything after the command name."""
    if raw_args.strip() == "help":
        return "Usage: /mystatus [help|check]"
    return "Plugin status: all systems nominal"

def register(ctx):
    ctx.register_command(
        "mystatus",
        handler=_handle_status,
        description="Show plugin status",
    )
```

注册后，用户可以在任何会话中键入 `/mystatus`。该命令出现在自动完成、`/help` 输出和 Telegram 机器人菜单中。

**签名：**`ctx.register_command(name: str, handler: Callable, description: str = "")`

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `name` | `str` | 命令名称，不带前导斜杠（例如 `"lcm"`、`"mystatus"`） |
| `handler` | `Callable[[str], str \| None]` | 使用原始参数字符串调用。也可以是 `async`。 |
| `description` | `str` | 显示在 `/help`、自动完成和 Telegram 机器人菜单中 |

**与 `register_cli_command()` 的主要区别：**

| | `register_command()` | `register_cli_command()` |
|---|---|---|
| 作为 | 会话中的 `/name` | 终端中的 `hermes name` |
| 工作位置 | CLI 会话、Telegram、Discord 等 | 仅终端 |
| 处理程序接收 | 原始参数字符串 | argparse `Namespace` |
| 使用场景 | 诊断、状态、快速操作 | 复杂的子命令树、设置向导 |

**冲突保护：** 如果插件尝试注册与内置命令（`help`、`model`、`new` 等）冲突的名称，注册会被静默拒绝并显示日志警告。内置命令始终优先。

**异步处理程序：** Gateway 调度会自动检测并等待异步处理程序，因此你可以使用同步或异步函数：

```python
async def _handle_check(raw_args: str) -> str:
    result = await some_async_operation()
    return f"Check result: {result}"

def register(ctx):
    ctx.register_command("check", handler=_handle_check, description="Run async check")
```

### 从斜杠命令分派工具

需要编排工具（通过 `delegate_task` 生成子 agent、调用 `file_edit` 等）的斜杠命令处理程序应使用 `ctx.dispatch_tool()` 而不是深入框架内部。父 agent 上下文（工作区提示、旋转器、模型继承）会自动连接。

```python
def register(ctx):
    def _handle_deliver(raw_args: str):
        result = ctx.dispatch_tool(
            "delegate_task",
            {
                "goal": raw_args,
                "toolsets": ["terminal", "file", "web"],
            },
        )
        return result

    ctx.register_command(
        "deliver",
        handler=_handle_deliver,
        description="Delegate a goal to a subagent",
    )
```

**签名：**`ctx.dispatch_tool(name: str, args: dict, *, parent_agent=None) -> str`

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `name` | `str` | 在工具注册表中注册的工具名称（例如 `"delegate_task"`、`"file_edit"`） |
| `args` | `dict` | 工具参数，与模型将发送的相同形状 |
| `parent_agent` | `Agent \| None` | 可选覆盖。省略时，从当前 CLI agent 解析（或在 gateway 模式下正常降级） |

**运行时行为：**

- **CLI 模式：**`parent_agent` 从活动 CLI agent 解析，因此工作区提示、旋转器和模型选择按预期继承。
- **Gateway 模式：** 没有 CLI agent，因此工具正常降级 — 从 `TERMINAL_CWD` 读取工作区，不显示旋转器。
- **显式覆盖：** 如果调用者显式传递 `parent_agent=`，则会被尊重且不会被覆盖。

这是用于从插件命令进行工具分派的公共、稳定接口。插件不应深入 `ctx._cli_ref.agent` 或类似的私有状态。

:::tip
本指南涵盖**通用插件**（工具、钩子、斜杠命令、CLI 命令）。下面的部分勾勒了每种专用插件类型的编写模式；每个都链接到其完整指南以获取字段参考和示例。
:::

## 专用插件类型

Hermes 除了通用接口外，还有五种专用插件类型。每个都作为 `plugins/<category>/<name>/`（捆绑）或 `~/.hermes/plugins/<category>/<name>/`（用户）下的目录提供。合同因类别而异 — 选择你需要的，然后阅读其完整指南。

### 模型提供商插件 — 添加 LLM 后端

将配置文件拖放到 `plugins/model-providers/<name>/`：

```python
# plugins/model-providers/acme/__init__.py
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(
    name="acme",
    aliases=("acme-inference",),
    display_name="Acme Inference",
    env_vars=("ACME_API_KEY", "ACME_BASE_URL"),
    base_url="https://api.acme.example.com/v1",
    auth_type="api_key",
    default_aux_model="acme-small-fast",
    fallback_models=("acme-large-v3", "acme-medium-v3"),
))
```

```yaml
# plugins/model-providers/acme/plugin.yaml
name: acme-provider
kind: model-provider
version: 1.0.0
description: Acme Inference — OpenAI-compatible direct API
```

首次调用 `get_provider_profile()` 或 `list_providers()` 时的延迟发现 — `auth.py`、`config.py`、`doctor.py`、`models.py`、`runtime_provider.py` 和 chat_completions 传输会自动连接到它。用户插件按名称覆盖捆绑的插件。

**完整指南：** [模型提供商插件](/docs/developer-guide/model-provider-plugin) — 字段参考、可覆盖的钩子（`prepare_messages`、`build_extra_body`、`build_api_kwargs_extras`、`fetch_models`）、api_mode 选择、身份验证类型、测试。

### 平台插件 — 添加 Gateway 频道

将适配器拖放到 `plugins/platforms/<name>/`：

```python
# plugins/platforms/myplatform/adapter.py
from gateway.platforms.base import BasePlatformAdapter

class MyPlatformAdapter(BasePlatformAdapter):
    async def connect(self): ...
    async def send(self, chat_id, text): ...
    async def disconnect(self): ...

def check_requirements():
    import os
    return bool(os.environ.get("MYPLATFORM_TOKEN"))

def _env_enablement():
    import os
    tok = os.getenv("MYPLATFORM_TOKEN", "").strip()
    if not tok:
        return None
    return {"token": tok}

def register(ctx):
    ctx.register_platform(
        name="myplatform",
        label="MyPlatform",
        adapter_factory=lambda cfg: MyPlatformAdapter(cfg),
        check_fn=check_requirements,
        required_env=["MYPLATFORM_TOKEN"],
        # Auto-populate PlatformConfig.extra from env so env-only setups
        # show up in `hermes gateway status` without SDK instantiation.
        env_enablement_fn=_env_enablement,
        # Opt in to cron delivery: `deliver=myplatform` routes to this var.
        cron_deliver_env_var="MYPLATFORM_HOME_CHANNEL",
        emoji="💬",
        platform_hint="You are chatting via MyPlatform. Keep responses concise.",
    )
```

```yaml
# plugins/platforms/myplatform/plugin.yaml
name: myplatform-platform
label: MyPlatform
kind: platform
version: 1.0.0
description: MyPlatform gateway adapter
requires_env:
  - name: MYPLATFORM_TOKEN
    description: "Bot token from the MyPlatform console"
    password: true
optional_env:
  - name: MYPLATFORM_HOME_CHANNEL
    description: "Default channel for cron delivery"
    password: false
```

**完整指南：** [添加平台适配器](/docs/developer-guide/adding-platform-adapters) — 完整的 `BasePlatformAdapter` 合同、消息路由、身份验证网关、设置向导集成。查看 `plugins/platforms/irc/` 以获取仅使用标准库的可用示例。

### 记忆提供商插件 — 添加跨会话知识后端

将 `MemoryProvider` 的实现拖放到 `plugins/memory/<name>/`：

```python
# plugins/memory/my-memory/__init__.py
from agent.memory_provider import MemoryProvider

class MyMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "my-memory"

    def is_available(self) -> bool:
        import os
        return bool(os.environ.get("MY_MEMORY_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id

    def sync_turn(self, user_message, assistant_response, **kwargs) -> None:
        ...

    def prefetch(self, query: str, **kwargs) -> str | None:
        ...

def register(ctx):
    ctx.register_memory_provider(MyMemoryProvider())
```

记忆提供商是单选 — 一次只有一个处于活动状态，通过 `config.yaml` 中的 `memory.provider` 选择。

**完整指南：** [记忆提供商插件](/docs/developer-guide/memory-provider-plugin) — 完整的 `MemoryProvider` ABC、线程合同、配置文件隔离、通过 `cli.py` 注册 CLI 命令。

### 上下文引擎插件 — 替换上下文压缩器

```python
# plugins/context_engine/my-engine/__init__.py
from agent.context_engine import ContextEngine

class MyContextEngine(ContextEngine):
    @property
    def name(self) -> str:
        return "my-engine"

    def should_compress(self, messages, model) -> bool: ...
    def compress(self, messages, model) -> list[dict]: ...

def register(ctx):
    ctx.register_context_engine(MyContextEngine())
```

上下文引擎是单选 — 通过 `config.yaml` 中的 `context.engine` 选择。

**完整指南：** [上下文引擎插件](/docs/developer-guide/context-engine-plugin)。

### 图像生成后端

将提供商拖放到 `plugins/image_gen/<name>/`：

```python
# plugins/image_gen/my-imggen/__init__.py
from agent.image_gen_provider import ImageGenProvider

class MyImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "my-imggen"

    def is_available(self) -> bool: ...
    def generate(self, prompt: str, **kwargs) -> str: ...   # returns image path

def register(ctx):
    ctx.register_image_gen_provider(MyImageGenProvider())
```

```yaml
# plugins/image_gen/my-imggen/plugin.yaml
name: my-imggen
kind: backend
version: 1.0.0
description: Custom image generation backend
```

**完整指南：** [图像生成提供商插件](/docs/developer-guide/image-gen-provider-plugin) — 完整的 `ImageGenProvider` ABC、`list_models()` / `get_setup_schema()` 元数据、`success_response()`/`error_response()` 辅助程序、base64 与 URL 输出、用户覆盖、pip 分发。

**参考示例：**`plugins/image_gen/openai/`（通过 OpenAI SDK 的 DALL-E / GPT-Image）、`plugins/image_gen/openai-codex/`、`plugins/image_gen/xai/`（Grok 图像生成）。

## 非 Python 扩展接口

Hermes 还接受根本不是 Python 插件的扩展。这些显示在[可插拔接口表](/docs/user-guide/features/plugins#pluggable-interfaces--where-to-go-for-each)中；以下部分简要勾勒每种编写风格。

### MCP 服务器 — 注册外部工具

模型上下文协议 (MCP) 服务器将他们自己的工具注册到 Hermes，无需任何 Python 插件。在 `~/.hermes/config.yaml` 中声明它们：

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
    timeout: 120

  linear:
    url: "https://mcp.linear.app/sse"
    auth:
      type: "oauth"
```

Hermes 在启动时连接到每个服务器，列出其工具，并将它们与内置工具一起注册。LLM 看到它们就像任何其他工具一样。**完整指南：** [MCP](/docs/user-guide/features/mcp)。

### Gateway 事件钩子 — 在生命周期事件上触发

将清单 + 处理程序拖放到 `~/.hermes/hooks/<name>/`：

```yaml
# ~/.hermes/hooks/long-task-alert/HOOK.yaml
name: long-task-alert
description: Send a push notification when a long task finishes
events:
  - agent:end
```

```python
# ~/.hermes/hooks/long-task-alert/handler.py
async def handle(event_type: str, context: dict) -> None:
    if context.get("duration_seconds", 0) > 120:
        # send notification …
        pass
```

事件包括 `gateway:startup`、`session:start`、`session:end`、`session:reset`、`agent:start`、`agent:step`、`agent:end` 和通配符 `command:*`。钩子中的错误会被捕获并记录 — 它们永远不会阻塞主管道。

**完整指南：** [Gateway 事件钩子](/docs/user-guide/features/hooks#gateway-event-hooks)。

### Shell 钩子 — 在工具调用时运行 shell 命令

如果你只是想在工具触发时运行脚本（通知、审计日志、桌面警报、自动格式化程序），请在 `config.yaml` 中使用 shell 钩子 — 无需 Python：

```yaml
hooks:
  - event: post_tool_call
    command: "notify-send 'Tool ran: {tool_name}'"
    when:
      tools: [terminal, patch, write_file]
```

支持与 Python 插件钩子相同的所有事件（`pre_tool_call`、`post_tool_call`、`pre_llm_call`、`post_llm_call`、`on_session_start`、`on_session_end`、`pre_gateway_dispatch`）以及用于 `pre_tool_call` 阻止决策的结构化 JSON 输出。

**完整指南：** [Shell 钩子](/docs/user-guide/features/hooks#shell-hooks)。

### 技能来源 — 添加自定义技能注册表

如果你维护技能的 GitHub 仓库（或者想从内置来源之外的社区索引中提取），请将其添加为 **tap**：

```bash
hermes skills tap add myorg/skills-repo
hermes skills search my-workflow --source myorg/skills-repo
hermes skills install myorg/skills-repo/my-workflow
```

发布你自己的 tap 只是一个带有 `skills/<skill-name>/SKILL.md` 目录的 GitHub 仓库 — 无需服务器或注册表注册。

**完整指南：** [技能中心](/docs/user-guide/features/skills#skills-hub) · [发布自定义 tap](/docs/user-guide/features/skills#publishing-a-custom-skill-tap)（仓库布局、最小示例、非默认路径、信任级别）。

### 通过命令模板的 TTS / STT

任何读取/写入音频或文本的 CLI 都可以通过 `config.yaml` 插入 — 无需 Python 代码：

```yaml
tts:
  provider: voxcpm
  providers:
    voxcpm:
      type: command
      command: "voxcpm --ref ~/voice.wav --text-file {input_path} --out {output_path}"
      output_format: mp3
      voice_compatible: true
```

对于 STT，将 `HERMES_LOCAL_STT_COMMAND` 指向 shell 模板。支持的占位符：`{input_path}`、`{output_path}`、`{format}`、`{voice}`、`{model}`、`{speed}` (TTS)；`{input_path}`、`{output_dir}`、`{language}`、`{model}` (STT)。任何路径交互 CLI 都会自动成为插件。

**完整指南：** [TTS 自定义命令提供商](/docs/user-guide/features/tts#custom-command-providers) · [STT](/docs/user-guide/features/tts#voice-message-transcription-stt)。

## 通过 pip 分发

为了公开分享插件，将入口点添加到你的 Python 包：

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-plugin = "my_plugin_package"
```

```bash
pip install hermes-plugin-calculator
# Plugin auto-discovered on next hermes startup
```

## 通过 NixOS 分发

如果你提供带有入口点的 `pyproject.toml`，NixOS 用户可以声明性地安装你的插件：

**入口点插件**（推荐用于分发）：
```nix
# User's configuration.nix
services.hermes-agent.extraPythonPackages = [
  (pkgs.python312Packages.buildPythonPackage {
    pname = "my-plugin";
    version = "1.0.0";
    src = pkgs.fetchFromGitHub {
      owner = "you";
      repo = "hermes-my-plugin";
      rev = "v1.0.0";
      hash = "sha256-...";  # nix-prefetch-url --unpack
    };
    format = "pyproject";
    build-system = [ pkgs.python312Packages.setuptools ];
  })
];
```

**目录插件**（不需要 `pyproject.toml`）：
```nix
services.hermes-agent.extraPlugins = [
  (pkgs.fetchFromGitHub {
    owner = "you";
    repo = "hermes-my-plugin";
    rev = "v1.0.0";
    hash = "sha256-...";
  })
];
```

有关完整文档（包括叠加使用和冲突检查），请参阅 [Nix 设置指南](/docs/getting-started/nix-setup#plugins)。

## 常见错误

**处理程序不返回 JSON 字符串：**
```python
# Wrong — returns a dict
def handler(args, **kwargs):
    return {"result": 42}

# Right — returns a JSON string
def handler(args, **kwargs):
    return json.dumps({"result": 42})
```

**处理程序签名中缺少 `**kwargs`：**
```python
# Wrong — will break if Hermes passes extra context
def handler(args):
    ...

# Right
def handler(args, **kwargs):
    ...
```

**处理程序引发异常：**
```python
# Wrong — exception propagates, tool call fails
def handler(args, **kwargs):
    result = 1 / int(args["value"])  # ZeroDivisionError!
    return json.dumps({"result": result})

# Right — catch and return error JSON
def handler(args, **kwargs):
    try:
        result = 1 / int(args.get("value", 0))
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

**模式描述太模糊：**
```python
# Bad — model doesn't know when to use it
"description": "Does stuff"

# Good — model knows exactly when and how
"description": "Evaluate a mathematical expression. Use for arithmetic, trig, logarithms. Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e."
```
