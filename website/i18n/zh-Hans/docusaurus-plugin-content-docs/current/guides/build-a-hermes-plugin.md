---
sidebar_position: 9
sidebar_label: "Build a Plugin"
title: "构建 Hermes 插件"
description: "逐步指南：构建包含工具、钩子、数据文件和技能的完整 Hermes 插件"
---

# 构建 Hermes 插件

本指南从零开始构建一个完整的 Hermes 插件。完成后，你将拥有一个包含多个工具、生命周期钩子（hook）、随附数据文件和捆绑技能的可用插件——涵盖插件系统支持的所有功能。

:::info 不确定需要哪份指南？
Hermes 有多种不同的可插拔接口——有些使用 Python `register_*` API，另一些是配置驱动或放入指定目录即可生效。请先查阅下表：

| 如果你想添加… | 请阅读 |
|---|---|
| 自定义工具、钩子、斜杠命令、技能或 CLI 子命令 | **本指南**（通用插件接口） |
| **LLM / 推理后端**（新提供商） | [模型提供商插件](/developer-guide/model-provider-plugin) |
| **网关频道**（Discord/Telegram/IRC/Teams 等） | [添加平台适配器](/developer-guide/adding-platform-adapters) |
| **记忆后端**（Honcho/Mem0/Supermemory 等） | [记忆提供商插件](/developer-guide/memory-provider-plugin) |
| **上下文压缩引擎** | [上下文引擎插件](/developer-guide/context-engine-plugin) |
| **图像生成后端** | [图像生成提供商插件](/developer-guide/image-gen-provider-plugin) |
| **视频生成后端** | [视频生成提供商插件](/developer-guide/video-gen-provider-plugin) |
| **TTS 后端**（任意 CLI——Piper、VoxCPM、Kokoro、声音克隆等） | [TTS 自定义命令提供商](/user-guide/features/tts#custom-command-providers)——配置驱动，无需 Python |
| **STT 后端**（自定义 whisper / ASR CLI） | [语音消息转录](/user-guide/features/tts#voice-message-transcription-stt)——将 `HERMES_LOCAL_STT_COMMAND` 设置为 shell 模板 |
| **通过 MCP 接入外部工具**（文件系统、GitHub、Linear、任意 MCP 服务器） | [MCP](/user-guide/features/mcp)——在 `config.yaml` 中声明 `mcp_servers.<name>` |
| **网关事件钩子**（在启动、会话事件、命令时触发） | [事件钩子](/user-guide/features/hooks#gateway-event-hooks)——将 `HOOK.yaml` + `handler.py` 放入 `~/.hermes/hooks/<name>/` |
| **Shell 钩子**（在事件发生时运行 shell 命令） | [Shell 钩子](/user-guide/features/hooks#shell-hooks)——在 `config.yaml` 的 `hooks:` 下声明 |
| **额外技能来源**（自定义 GitHub 仓库、私有技能索引） | [技能](/user-guide/features/skills)——`hermes skills tap add <repo>` · [发布 tap](/user-guide/features/skills#publishing-a-custom-skill-tap) |
| 一流的**核心**推理提供商（非插件） | [添加提供商](/developer-guide/adding-providers) |

查看完整的[可插拔接口表](/user-guide/features/plugins#pluggable-interfaces--where-to-go-for-each)，获取每种扩展接口的汇总视图，包括配置驱动（TTS、STT、MCP、shell 钩子）和放入目录（网关钩子）两种方式。
:::

## 你将构建什么

一个**计算器**插件，包含两个工具：
- `calculate`——计算数学表达式（`2**16`、`sqrt(144)`、`pi * 5**2`）
- `unit_convert`——在单位之间转换（`100 F → 37.78 C`、`5 km → 3.11 mi`）

另外还有一个记录每次工具调用的钩子，以及一个捆绑的技能文件。

## 第一步：创建插件目录

```bash
mkdir -p ~/.hermes/plugins/calculator
cd ~/.hermes/plugins/calculator
```

## 第二步：编写清单文件

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

这告诉 Hermes："我是一个名为 calculator 的插件，我提供工具和钩子。" `provides_tools` 和 `provides_hooks` 字段是插件注册内容的列表。

可选字段示例：
```yaml
author: Your Name
requires_env:          # 根据环境变量决定是否加载；安装时会提示用户
  - SOME_API_KEY       # 简单格式——缺失时插件禁用
  - name: OTHER_KEY    # 富格式——安装时显示描述/URL
    description: "Key for the Other service"
    url: "https://other.com/keys"
    secret: true
```

## 第三步：编写工具 schema

创建 `schemas.py`——这是 LLM 读取以决定何时调用你的工具的内容：

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

**schema 为何重要：** `description` 字段决定了 LLM 何时使用你的工具。请明确说明工具的功能和使用时机。`parameters` 定义了 LLM 传入的参数。

## 第四步：编写工具处理器

创建 `tools.py`——这是 LLM 调用工具时实际执行的代码：

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

**处理器的关键规则：**
1. **签名：** `def my_handler(args: dict, **kwargs) -> str`
2. **返回值：** 始终返回 JSON 字符串。成功和错误均如此。
3. **不要抛出异常：** 捕获所有异常，改为返回错误 JSON。
4. **接受 `**kwargs`：** Hermes 未来可能传入额外上下文。

## 第五步：编写注册代码

创建 `__init__.py`——将 schema 与处理器连接起来：

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
- `ctx.register_tool()` 将你的工具放入注册表——模型立即可见
- `ctx.register_hook()` 订阅生命周期事件
- `ctx.register_cli_command()` 注册 CLI 子命令（例如 `hermes my-plugin <subcommand>`）
- `ctx.register_command()` 注册会话内斜杠命令（例如在 CLI / 网关聊天中输入 `/myplugin <args>`）——详见下方[注册斜杠命令](#register-slash-commands)
- `ctx.dispatch_tool(name, arguments)` ——以父代理的上下文（审批、凭证、task_id 自动连接）调用任意其他工具（内置或来自其他插件）。适用于需要直接调用 `terminal`、`read_file` 或其他工具的斜杠命令处理器，效果等同于模型直接调用。
- 如果此函数崩溃，插件将被禁用，但 Hermes 继续正常运行

**`dispatch_tool` 示例——执行工具的斜杠命令：**

```python
def handle_scan(ctx, argstr):
    """Implement /scan by invoking the terminal tool through the registry."""
    result = ctx.dispatch_tool("terminal", {"command": f"find . -name '{argstr}'"})
    return result  # returned to the caller's chat UI

def register(ctx):
    ctx.register_command("scan", handle_scan, help="Find files matching a glob")
```

被分发的工具会经过正常的审批、脱敏和预算流程——这是真实的工具调用，而非绕过这些流程的捷径。

## 第六步：测试

启动 Hermes：

```bash
hermes
```

你应该在启动横幅的工具列表中看到 `calculator: calculate, unit_convert`。

尝试以下提示词（prompt）：
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

### 调试插件发现问题

如果你的插件没有出现，或出现了但未加载——设置 `HERMES_PLUGINS_DEBUG=1` 可在 stderr 获取详细的发现日志：

```bash
HERMES_PLUGINS_DEBUG=1 hermes plugins list
```

你将看到每个插件来源（内置、用户、项目、entry-points）的以下信息：

- 扫描了哪些目录，每个目录产出了多少个清单
- 每个清单：解析后的键、名称、类型、来源、磁盘路径
- 跳过原因：`disabled via config`、`not enabled in config`、`exclusive plugin`、`no plugin.yaml, depth cap reached`
- 加载时：正在导入的插件，以及 `register(ctx)` 注册内容的单行摘要（工具、钩子、斜杠命令、CLI 命令）
- 解析失败时：异常的完整堆栈跟踪（YAML 扫描器错误等）
- `register()` 失败时：指向 `__init__.py` 中抛出异常的行的完整堆栈跟踪

同样的日志始终写入 `~/.hermes/logs/agent.log`，失败时为 WARNING 级别，设置环境变量时为 DEBUG 级别（全部内容）。如果无法使用环境变量运行（例如从网关内部），可以改为追踪日志文件：

```bash
hermes logs --level WARNING | grep -i plugin
```

插件未出现的常见原因：

- **未在配置中启用**——插件需要手动启用。运行 `hermes plugins enable <name>`（名称来自 `plugins list` 输出，嵌套布局下可能是 `<category>/<plugin>`）。
- **目录结构错误**——必须是 `~/.hermes/plugins/<plugin-name>/plugin.yaml`（扁平）或 `~/.hermes/plugins/<category>/<plugin-name>/plugin.yaml`（一级分类嵌套，最多）。更深层的目录会被忽略。
- **缺少 `__init__.py`**——插件目录需要同时包含 `plugin.yaml` 和带有 `register(ctx)` 函数的 `__init__.py`。
- **`kind` 错误**——网关适配器需要在清单中设置 `kind: platform`。记忆提供商会被自动检测为 `kind: exclusive`，并通过 `memory.provider` 配置路由，而非 `plugins.enabled`。

## 插件的最终结构

```
~/.hermes/plugins/calculator/
├── plugin.yaml      # "我是 calculator，我提供工具和钩子"
├── __init__.py      # 连接：schema → 处理器，注册钩子
├── schemas.py       # LLM 读取的内容（描述 + 参数规格）
└── tools.py         # 实际运行的代码（calculate、unit_convert 函数）
```

四个文件，职责清晰：
- **清单**声明插件是什么
- **Schema** 向 LLM 描述工具
- **处理器**实现实际逻辑
- **注册**将一切连接起来

## 插件还能做什么？

### 随附数据文件

将任意文件放入插件目录，并在导入时读取：

```python
# In tools.py or __init__.py
from pathlib import Path

_PLUGIN_DIR = Path(__file__).parent
_DATA_FILE = _PLUGIN_DIR / "data" / "languages.yaml"

with open(_DATA_FILE) as f:
    _DATA = yaml.safe_load(f)
```

### 捆绑技能

插件可以随附技能文件，代理通过 `skill_view("plugin:skill")` 加载。在 `__init__.py` 中注册：

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

代理现在可以通过命名空间名称加载你的技能：

```python
skill_view("my-plugin:my-workflow")   # → 插件版本
skill_view("my-workflow")              # → 内置版本（不受影响）
```

**关键特性：**
- 插件技能是**只读**的——它们不会进入 `~/.hermes/skills/`，也无法通过 `skill_manage` 编辑。
- 插件技能**不会**列在系统提示词的 `<available_skills>` 索引中——需要显式加载。
- 裸技能名称不受影响——命名空间防止与内置技能冲突。
- 代理加载插件技能时，会在前面添加一个捆绑上下文横幅，列出同一插件的兄弟技能。

:::tip 旧版模式
旧的 `shutil.copy2` 模式（将技能复制到 `~/.hermes/skills/`）仍然有效，但存在与内置技能名称冲突的风险。新插件请优先使用 `ctx.register_skill()`。
:::

### 根据环境变量决定是否启用

如果你的插件需要 API 密钥：

```yaml
# plugin.yaml — 简单格式（向后兼容）
requires_env:
  - WEATHER_API_KEY
```

如果 `WEATHER_API_KEY` 未设置，插件将被禁用并显示清晰的提示信息。不会崩溃，代理中也不会报错——只会显示"Plugin weather disabled (missing: WEATHER_API_KEY)"。

用户运行 `hermes plugins install` 时，会**交互式提示**输入任何缺失的 `requires_env` 变量。值会自动保存到 `.env`。

为了获得更好的安装体验，使用带有描述和注册 URL 的富格式：

```yaml
# plugin.yaml — 富格式
requires_env:
  - name: WEATHER_API_KEY
    description: "API key for OpenWeather"
    url: "https://openweathermap.org/api"
    secret: true
```

| 字段 | 必填 | 描述 |
|-------|----------|-------------|
| `name` | 是 | 环境变量名称 |
| `description` | 否 | 安装提示时显示给用户 |
| `url` | 否 | 获取凭证的地址 |
| `secret` | 否 | 若为 `true`，输入时隐藏（类似密码字段） |

两种格式可在同一列表中混用。已设置的变量会被静默跳过。

### 懒加载可选 Python 依赖

如果你的插件封装了一个并非所有用户都会安装的 SDK（供应商 SDK、重型 ML 库、平台特定包），不要在模块顶部 `import` 它。在工具处理器内部使用 `tools.lazy_deps.ensure(...)` 辅助函数——Hermes 会在首次使用时安装该包，并受用户 `security.allow_lazy_installs` 配置的控制。

```python
# tools.py
from tools.lazy_deps import ensure, FeatureUnavailable

def my_tool_handler(args, **kwargs):
    try:
        ensure("my-plugin.my-backend")   # key must be in LAZY_DEPS
    except FeatureUnavailable as exc:
        return {"error": str(exc)}

    import my_backend_sdk   # safe now
    ...
```

来自 `tools/lazy_deps.py` 安全模型的两条规则：

| 规则 | 原因 |
|---|---|
| 你的功能键必须出现在内置的 `LAZY_DEPS` 允许列表中 | 防止恶意配置诱使 Hermes 安装任意包——只有 Hermes 自身随附的规格才符合条件 |
| 规格仅限 PyPI 包名 | 不允许 `--index-url`、`git+https://` 或 `file:` 路径。在允许列表条目中使用 PEP 440 固定版本（`"my-sdk>=1.2,<2"`） |

对于通过 pip 分发的第三方插件，在你自己的 `pyproject.toml` 中将可选依赖声明为 `[project.optional-dependencies]` extras，并告知用户执行 `pip install your-plugin[backend]`——该路径不经过 `lazy_deps`。懒加载安装最适合**内置**插件，因为对每次安装都强制依赖会增加 Hermes 基础安装的体积。

当全局设置 `security.allow_lazy_installs: false` 时，`ensure()` 会立即抛出 `FeatureUnavailable` 并附带修复提示——你的插件应捕获该异常并优雅降级（返回错误结果，而非让工具循环崩溃）。

### 条件工具可用性

对于依赖可选库的工具：

```python
ctx.register_tool(
    name="my_tool",
    schema={...},
    handler=my_handler,
    check_fn=lambda: _has_optional_lib(),  # False = 工具对模型隐藏
)
```

### 覆盖内置工具

要用你自己的实现替换内置工具（例如将默认浏览器工具替换为有头 Chrome CDP 后端，或将 `web_search` 替换为自定义企业索引），传入 `override=True`：

```python
def register(ctx):
    ctx.register_tool(
        name="browser_navigate",             # 与内置工具同名
        toolset="plugin_my_browser",         # 你自己的 toolset 命名空间
        schema={...},
        handler=my_custom_navigate,
        override=True,                       # 显式启用覆盖
    )
```

不加 `override=True` 时，注册表会拒绝任何会遮蔽来自不同 toolset 的已有工具的注册——这防止了意外覆盖。覆盖操作会以 INFO 级别记录日志，可在 `~/.hermes/logs/agent.log` 中审计。插件在内置工具之后加载，因此注册顺序是正确的：你的处理器会替换内置处理器。

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

每个钩子的完整文档见**[事件钩子参考](/user-guide/features/hooks#plugin-hooks)**——回调签名、参数表、触发时机和示例。以下是摘要：

| 钩子 | 触发时机 | 回调签名 | 返回值 |
|------|-----------|-------------------|---------|
| [`pre_tool_call`](/user-guide/features/hooks#pre_tool_call) | 任意工具执行前 | `tool_name: str, args: dict, task_id: str` | 忽略 |
| [`post_tool_call`](/user-guide/features/hooks#post_tool_call) | 任意工具返回后 | `tool_name: str, args: dict, result: str, task_id: str, duration_ms: int` | 忽略 |
| [`pre_llm_call`](/user-guide/features/hooks#pre_llm_call) | 每轮一次，工具调用循环前 | `session_id: str, user_message: str, conversation_history: list, is_first_turn: bool, model: str, platform: str` | [上下文注入](#pre_llm_call-context-injection) |
| [`post_llm_call`](/user-guide/features/hooks#post_llm_call) | 每轮一次，工具调用循环后（仅成功轮次） | `session_id: str, user_message: str, assistant_response: str, conversation_history: list, model: str, platform: str` | 忽略 |
| [`on_session_start`](/user-guide/features/hooks#on_session_start) | 新会话创建（仅第一轮） | `session_id: str, model: str, platform: str` | 忽略 |
| [`on_session_end`](/user-guide/features/hooks#on_session_end) | 每次 `run_conversation` 调用结束 + CLI 退出 | `session_id: str, completed: bool, interrupted: bool, model: str, platform: str` | 忽略 |
| [`on_session_finalize`](/user-guide/features/hooks#on_session_finalize) | CLI/网关销毁活跃会话 | `session_id: str \| None, platform: str` | 忽略 |
| [`on_session_reset`](/user-guide/features/hooks#on_session_reset) | 网关切换新会话键（`/new`、`/reset`） | `session_id: str, platform: str` | 忽略 |

大多数钩子是即发即忘的观察者——其返回值被忽略。例外是 `pre_llm_call`，它可以向对话中注入上下文。

所有回调都应接受 `**kwargs` 以保持向前兼容性。如果钩子回调崩溃，会被记录日志并跳过。其他钩子和代理继续正常运行。

### `pre_llm_call` 上下文注入

这是唯一一个返回值有意义的钩子。当 `pre_llm_call` 回调返回包含 `"context"` 键的字典（或纯字符串）时，Hermes 会将该文本注入**当前轮次的用户消息**中。这是记忆插件、RAG 集成、护栏以及任何需要向模型提供额外上下文的插件所使用的机制。

#### 返回格式

```python
# 包含 context 键的字典
return {"context": "Recalled memories:\n- User prefers dark mode\n- Last project: hermes-agent"}

# 纯字符串（等同于上面的字典形式）
return "Recalled memories:\n- User prefers dark mode"

# 返回 None 或不返回 → 不注入（仅观察）
return None
```

任何非 None、非空的返回值，只要包含 `"context"` 键（或为非空纯字符串），都会被收集并追加到当前轮次的用户消息中。

#### 注入的工作原理

注入的上下文追加到**用户消息**，而非系统提示词（system prompt）。这是有意为之的设计：

- **保留提示词缓存**——系统提示词在各轮次之间保持不变。Anthropic 和 OpenRouter 会缓存系统提示词前缀，保持其稳定可在多轮对话中节省 75% 以上的输入 token。如果插件修改系统提示词，每轮都会缓存未命中。
- **临时性**——注入仅在 API 调用时发生。会话历史中的原始用户消息不会被修改，也不会持久化到会话数据库。
- **系统提示词是 Hermes 的领地**——它包含模型特定的指导、工具执行规则、个性指令和缓存的技能内容。插件在用户输入旁边贡献上下文，而非修改代理的核心指令。

#### 示例：记忆召回插件

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

#### 示例：护栏插件

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

#### 示例：仅观察钩子（不注入）

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

当多个插件从 `pre_llm_call` 返回上下文时，它们的输出以双换行符连接，一起追加到用户消息中。顺序遵循插件发现顺序（按插件目录名称字母排序）。

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

注册后，用户可以运行 `hermes my-plugin status`、`hermes my-plugin config` 等命令。

**记忆提供商插件**使用基于约定的方式：在插件的 `cli.py` 文件中添加 `register_cli(subparser)` 函数。记忆插件发现系统会自动找到它——无需调用 `ctx.register_cli_command()`。详见[记忆提供商插件指南](/developer-guide/memory-provider-plugin#adding-cli-commands)。

**活跃提供商限制：** 记忆插件 CLI 命令仅在其提供商是配置中活跃的 `memory.provider` 时才会出现。如果用户尚未设置你的提供商，你的 CLI 命令不会出现在帮助输出中。

### 注册斜杠命令

插件可以注册会话内斜杠命令——用户在对话中输入的命令（如 `/lcm status` 或 `/ping`）。这些命令在 CLI 和网关（Telegram、Discord 等）中均可使用。

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

注册后，用户可以在任意会话中输入 `/mystatus`。该命令会出现在自动补全、`/help` 输出和 Telegram 机器人菜单中。

**签名：** `ctx.register_command(name: str, handler: Callable, description: str = "")`

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `name` | `str` | 不含前导斜杠的命令名称（例如 `"lcm"`、`"mystatus"`） |
| `handler` | `Callable[[str], str \| None]` | 以原始参数字符串调用。也可以是 `async`。 |
| `description` | `str` | 显示在 `/help`、自动补全和 Telegram 机器人菜单中 |

**与 `register_cli_command()` 的主要区别：**

| | `register_command()` | `register_cli_command()` |
|---|---|---|
| 调用方式 | 会话中的 `/name` | 终端中的 `hermes name` |
| 适用范围 | CLI 会话、Telegram、Discord 等 | 仅终端 |
| 处理器接收 | 原始参数字符串 | argparse `Namespace` |
| 使用场景 | 诊断、状态查询、快速操作 | 复杂子命令树、设置向导 |

**冲突保护：** 如果插件尝试注册与内置命令（`help`、`model`、`new` 等）冲突的名称，注册会被静默拒绝并记录警告日志。内置命令始终优先。

**异步处理器：** 网关分发会自动检测并 await 异步处理器，因此可以使用同步或异步函数：

```python
async def _handle_check(raw_args: str) -> str:
    result = await some_async_operation()
    return f"Check result: {result}"

def register(ctx):
    ctx.register_command("check", handler=_handle_check, description="Run async check")
```

### 从斜杠命令分发工具

需要编排工具的斜杠命令处理器（生成子代理 `delegate_task`、调用 `file_edit` 等）应使用 `ctx.dispatch_tool()`，而非深入框架内部。父代理上下文（工作区提示、spinner、模型继承）会自动连接。

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

**签名：** `ctx.dispatch_tool(name: str, args: dict, *, parent_agent=None) -> str`

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `name` | `str` | 工具注册表中的工具名称（例如 `"delegate_task"`、`"file_edit"`） |
| `args` | `dict` | 工具参数，与模型发送的格式相同 |
| `parent_agent` | `Agent \| None` | 可选覆盖。省略时从当前 CLI 代理解析（网关模式下优雅降级） |

**运行时行为：**

- **CLI 模式：** `parent_agent` 从活跃的 CLI 代理解析，工作区提示、spinner 和模型选择按预期继承。
- **网关模式：** 没有 CLI 代理，工具优雅降级——工作区从 `TERMINAL_CWD` 读取，不显示 spinner。
- **显式覆盖：** 如果调用者显式传入 `parent_agent=`，则尊重该值，不会被覆盖。

这是从插件命令分发工具的公开稳定接口。插件不应访问 `ctx._cli_ref.agent` 或类似的私有状态。

:::tip
本指南涵盖**通用插件**（工具、钩子、斜杠命令、CLI 命令）。以下各节简要介绍每种专用插件类型的编写模式；每节均链接到其完整指南以获取字段参考和示例。
:::

## 专用插件类型

Hermes 在通用接口之外还有五种专用插件类型。每种都以目录形式存放在 `plugins/<category>/<name>/`（内置）或 `~/.hermes/plugins/<category>/<name>/`（用户）下。各类别的约定不同——选择你需要的类型，然后阅读其完整指南。

### 模型提供商插件——添加 LLM 后端

在 `plugins/model-providers/<name>/` 下放置一个配置文件：

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

在任何调用 `get_provider_profile()` 或 `list_providers()` 的地方首次使用时懒加载发现——`auth.py`、`config.py`、`doctor.py`、`models.py`、`runtime_provider.py` 和 chat_completions 传输层会自动连接。用户插件按名称覆盖内置插件。

**完整指南：** [模型提供商插件](/developer-guide/model-provider-plugin)——字段参考、可覆盖钩子（`prepare_messages`、`build_extra_body`、`build_api_kwargs_extras`、`fetch_models`）、api_mode 选择、认证类型、测试。

### 平台插件——添加网关频道

在 `plugins/platforms/<name>/` 下放置适配器：

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
        # 从环境变量自动填充 PlatformConfig.extra，使仅环境变量的设置
        # 在 `hermes gateway status` 中显示，无需 SDK 实例化。
        env_enablement_fn=_env_enablement,
        # 启用 cron 投递：`deliver=myplatform` 路由到此变量。
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

**完整指南：** [添加平台适配器](/developer-guide/adding-platform-adapters)——完整的 `BasePlatformAdapter` 约定、消息路由、认证限制、设置向导集成。参考 `plugins/platforms/irc/` 获取仅使用标准库的可用示例。

### 记忆提供商插件——添加跨会话知识后端

在 `plugins/memory/<name>/` 下实现 `MemoryProvider`：

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

记忆提供商是单选的——同一时间只有一个处于活跃状态，通过 `config.yaml` 中的 `memory.provider` 选择。

**完整指南：** [记忆提供商插件](/developer-guide/memory-provider-plugin)——完整的 `MemoryProvider` ABC、线程约定、配置文件隔离、通过 `cli.py` 注册 CLI 命令。

### 上下文引擎插件——替换上下文压缩器

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

上下文引擎是单选的——通过 `config.yaml` 中的 `context.engine` 选择。

**完整指南：** [上下文引擎插件](/developer-guide/context-engine-plugin)。

### 图像生成后端

在 `plugins/image_gen/<name>/` 下放置提供商：

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

**完整指南：** [图像生成提供商插件](/developer-guide/image-gen-provider-plugin)——完整的 `ImageGenProvider` ABC、`list_models()` / `get_setup_schema()` 元数据、`success_response()`/`error_response()` 辅助函数、base64 与 URL 输出、用户覆盖、pip 分发。

**参考示例：** `plugins/image_gen/openai/`（DALL-E / GPT-Image via OpenAI SDK）、`plugins/image_gen/openai-codex/`、`plugins/image_gen/xai/`（Grok 图像生成）。

## 非 Python 扩展接口

Hermes 也接受完全不是 Python 插件的扩展。这些在[可插拔接口表](/user-guide/features/plugins#pluggable-interfaces--where-to-go-for-each)中有所展示；以下各节简要介绍每种编写方式。

### MCP 服务器——注册外部工具

Model Context Protocol（MCP）服务器无需任何 Python 插件即可将自己的工具注册到 Hermes。在 `~/.hermes/config.yaml` 中声明：

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

Hermes 在启动时连接到每个服务器，列出其工具，并与内置工具一起注册。LLM 看到它们的方式与其他工具完全相同。**完整指南：** [MCP](/user-guide/features/mcp)。

### 网关事件钩子——在生命周期事件时触发

将清单和处理器放入 `~/.hermes/hooks/<name>/`：

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

事件包括 `gateway:startup`、`session:start`、`session:end`、`session:reset`、`agent:start`、`agent:step`、`agent:end` 以及通配符 `command:*`。钩子中的错误会被捕获并记录日志——它们不会阻塞主流程。

**完整指南：** [网关事件钩子](/user-guide/features/hooks#gateway-event-hooks)。

### Shell 钩子——在工具调用时运行 shell 命令

如果你只想在工具触发时运行脚本（通知、审计日志、桌面提醒、自动格式化），在 `config.yaml` 中使用 shell 钩子——无需 Python：

```yaml
hooks:
  - event: post_tool_call
    command: "notify-send 'Tool ran: {tool_name}'"
    when:
      tools: [terminal, patch, write_file]
```

支持与 Python 插件钩子相同的所有事件（`pre_tool_call`、`post_tool_call`、`pre_llm_call`、`post_llm_call`、`on_session_start`、`on_session_end`、`pre_gateway_dispatch`），以及用于 `pre_tool_call` 阻断决策的结构化 JSON 输出。

**完整指南：** [Shell 钩子](/user-guide/features/hooks#shell-hooks)。

### 技能来源——添加自定义技能注册表

如果你维护了一个技能 GitHub 仓库（或想从内置来源之外的社区索引拉取），将其添加为 **tap**：

```bash
hermes skills tap add myorg/skills-repo
hermes skills search my-workflow --source myorg/skills-repo
hermes skills install myorg/skills-repo/my-workflow
```

发布你自己的 tap 只需一个包含 `skills/<skill-name>/SKILL.md` 目录的 GitHub 仓库——无需服务器或注册表注册。

**完整指南：** [技能中心](/user-guide/features/skills#skills-hub) · [发布自定义 tap](/user-guide/features/skills#publishing-a-custom-skill-tap)（仓库结构、最小示例、非默认路径、信任级别）。

### 通过命令模板接入 TTS / STT

任何读写音频或文本的 CLI 都可以通过 `config.yaml` 接入——无需 Python 代码：

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

对于 STT，将 `HERMES_LOCAL_STT_COMMAND` 指向一个 shell 模板。支持的占位符：`{input_path}`、`{output_path}`、`{format}`、`{voice}`、`{model}`、`{speed}`（TTS）；`{input_path}`、`{output_dir}`、`{language}`、`{model}`（STT）。任何与路径交互的 CLI 都自动成为插件。

**完整指南：** [TTS 自定义命令提供商](/user-guide/features/tts#custom-command-providers) · [STT](/user-guide/features/tts#voice-message-transcription-stt)。

## 通过 pip 分发

如需公开分享插件，在你的 Python 包中添加 entry point：

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-plugin = "my_plugin_package"
```

```bash
pip install hermes-plugin-calculator
# 下次 hermes 启动时自动发现插件
```

## 为 NixOS 分发

如果你提供了带有 entry points 的 `pyproject.toml`，NixOS 用户可以声明式安装你的插件：

**Entry-point 插件**（推荐用于分发）：
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

**目录插件**（无需 `pyproject.toml`）：
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

完整文档（包括 overlay 用法和冲突检查）见 [Nix 设置指南](/getting-started/nix-setup#plugins)。

## 常见错误

**处理器未返回 JSON 字符串：**
```python
# 错误——返回了字典
def handler(args, **kwargs):
    return {"result": 42}

# 正确——返回 JSON 字符串
def handler(args, **kwargs):
    return json.dumps({"result": 42})
```

**处理器签名缺少 `**kwargs`：**
```python
# 错误——Hermes 传入额外上下文时会报错
def handler(args):
    ...

# 正确
def handler(args, **kwargs):
    ...
```

**处理器抛出异常：**
```python
# 错误——异常传播，工具调用失败
def handler(args, **kwargs):
    result = 1 / int(args["value"])  # ZeroDivisionError!
    return json.dumps({"result": result})

# 正确——捕获异常并返回错误 JSON
def handler(args, **kwargs):
    try:
        result = 1 / int(args.get("value", 0))
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

**Schema 描述过于模糊：**
```python
# 差——模型不知道何时使用
"description": "Does stuff"

# 好——模型清楚地知道何时以及如何使用
"description": "Evaluate a mathematical expression. Use for arithmetic, trig, logarithms. Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e."
```