---
sidebar_position: 10
---

# Build a Hermes Plugin

This guide walks through building a complete Hermes plugin from scratch. By the end you'll have a working plugin with multiple tools, lifecycle hooks, shipped data files, and a bundled skill — everything the plugin system supports.

## What you're building

A **calculator** plugin with two tools:
- `calculate` — evaluate math expressions (`2**16`, `sqrt(144)`, `pi * 5**2`)
- `unit_convert` — convert between units (`100 F → 37.78 C`, `5 km → 3.11 mi`)

Plus a hook that logs every tool call, and a bundled skill file.

## Step 1: Create the plugin directory

```bash
mkdir -p ~/.hermes/plugins/calculator
cd ~/.hermes/plugins/calculator
```

## Step 2: Write the manifest

Create `plugin.yaml`:

```yaml
name: calculator
version: 1.0.0
description: Math calculator — evaluate expressions and convert units
provides:
  tools: true
  hooks: true
```

This tells Hermes: "I'm a plugin called calculator, I provide tools and hooks." That's all the manifest needs.

Optional fields you could add:
```yaml
author: Your Name
requires_env:          # gate loading on env vars
  - SOME_API_KEY       # plugin disabled if missing
```

## Step 3: Write the tool schemas

Create `schemas.py` — this is what the LLM reads to decide when to call your tools:

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

**Why schemas matter:** The `description` field is how the LLM decides when to use your tool. Be specific about what it does and when to use it. The `parameters` define what arguments the LLM passes.

## Step 4: Write the tool handlers

Create `tools.py` — this is the code that actually executes when the LLM calls your tools:

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

**Key rules for handlers:**
1. **Signature:** `def my_handler(args: dict, **kwargs) -> str`
2. **Return:** Always a JSON string. Success and errors alike.
3. **Never raise:** Catch all exceptions, return error JSON instead.
4. **Accept `**kwargs`:** Hermes may pass additional context in the future.

## Step 5: Write the registration

Create `__init__.py` — this wires schemas to handlers:

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

**What `register()` does:**
- Called exactly once at startup
- `ctx.register_tool()` puts your tool in the registry — the model sees it immediately
- `ctx.register_hook()` subscribes to lifecycle events
- If this function crashes, the plugin is disabled but Hermes continues fine

## Step 6: Test it

Start Hermes:

```bash
hermes
```

You should see `calculator: calculate, unit_convert` in the banner's tool list.

Try these prompts:
```
What's 2 to the power of 16?
Convert 100 fahrenheit to celsius
What's the square root of 2 times pi?
How many gigabytes is 1.5 terabytes?
```

Check plugin status:
```
/plugins
```

Output:
```
Plugins (1):
  ✓ calculator v1.0.0 (2 tools, 1 hooks)
```

## Your plugin's final structure

```
~/.hermes/plugins/calculator/
├── plugin.yaml      # "I'm calculator, I provide tools and hooks"
├── __init__.py      # Wiring: schemas → handlers, register hooks
├── schemas.py       # What the LLM reads (descriptions + parameter specs)
└── tools.py         # What runs (calculate, unit_convert functions)
```

Four files, clear separation:
- **Manifest** declares what the plugin is
- **Schemas** describe tools for the LLM
- **Handlers** implement the actual logic
- **Registration** connects everything

## What else can plugins do?

### Ship data files

Put any files in your plugin directory and read them at import time:

```python
# In tools.py or __init__.py
from pathlib import Path

_PLUGIN_DIR = Path(__file__).parent
_DATA_FILE = _PLUGIN_DIR / "data" / "languages.yaml"

with open(_DATA_FILE) as f:
    _DATA = yaml.safe_load(f)
```

### Bundle a skill

Include a `skill.md` file and install it during registration:

```python
import shutil
from pathlib import Path

def _install_skill():
    """Copy our skill to ~/.hermes/skills/ on first load."""
    try:
        from hermes_cli.config import get_hermes_home
        dest = get_hermes_home() / "skills" / "my-plugin" / "SKILL.md"
    except Exception:
        dest = Path.home() / ".hermes" / "skills" / "my-plugin" / "SKILL.md"

    if dest.exists():
        return  # don't overwrite user edits

    source = Path(__file__).parent / "skill.md"
    if source.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

def register(ctx):
    ctx.register_tool(...)
    _install_skill()
```

### Gate on environment variables

If your plugin needs an API key:

```yaml
# plugin.yaml
requires_env:
  - WEATHER_API_KEY
```

If `WEATHER_API_KEY` isn't set, the plugin is disabled with a clear message. No crash, no error in the agent — just "Plugin weather disabled (missing: WEATHER_API_KEY)".

### Conditional tool availability

For tools that depend on optional libraries:

```python
ctx.register_tool(
    name="my_tool",
    schema={...},
    handler=my_handler,
    check_fn=lambda: _has_optional_lib(),  # False = tool hidden from model
)
```

### Register multiple hooks

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", before_any_tool)
    ctx.register_hook("post_tool_call", after_any_tool)
    ctx.register_hook("on_session_start", on_new_session)
    ctx.register_hook("on_session_end", on_session_end)
```

Available hooks:

| Hook | When | Arguments |
|------|------|-----------|
| `pre_tool_call` | Before any tool runs | `tool_name`, `args`, `task_id` |
| `post_tool_call` | After any tool returns | `tool_name`, `args`, `result`, `task_id` |
| `pre_llm_call` | Before LLM API call | `messages`, `model` |
| `post_llm_call` | After LLM response | `messages`, `response`, `model` |
| `on_session_start` | Session begins | `session_id`, `platform` |
| `on_session_end` | Session ends | `session_id`, `platform` |

Hooks are observers — they can't modify arguments or return values. If a hook crashes, it's logged and skipped; other hooks and the tool continue normally.

### Distribute via pip

For sharing plugins publicly, add an entry point to your Python package:

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-plugin = "my_plugin_package"
```

```bash
pip install hermes-plugin-calculator
# Plugin auto-discovered on next hermes startup
```

## Common mistakes

**Handler doesn't return JSON string:**
```python
# Wrong — returns a dict
def handler(args, **kwargs):
    return {"result": 42}

# Right — returns a JSON string
def handler(args, **kwargs):
    return json.dumps({"result": 42})
```

**Missing `**kwargs` in handler signature:**
```python
# Wrong — will break if Hermes passes extra context
def handler(args):
    ...

# Right
def handler(args, **kwargs):
    ...
```

**Handler raises exceptions:**
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

**Schema description too vague:**
```python
# Bad — model doesn't know when to use it
"description": "Does stuff"

# Good — model knows exactly when and how
"description": "Evaluate a mathematical expression. Use for arithmetic, trig, logarithms. Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e."
```
