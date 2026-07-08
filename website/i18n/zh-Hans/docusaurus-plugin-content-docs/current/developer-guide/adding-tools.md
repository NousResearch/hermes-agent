---
sidebar_position: 2
title: "添加工具"
description: "如何向 Hermes Agent 添加新工具——schema、handler、注册与 toolset"
---

# 添加工具

在编写工具之前，先问自己：**这是否应该是一个 [skill](creating-skills.md)？**

:::warning 仅限内置核心工具
本页面用于向仓库本身添加 **Hermes 内置工具**。
如果你想要个人专用、项目本地或其他自定义工具，而不修改 Hermes 核心，请使用插件方式：

- [插件](/user-guide/features/plugins)
- [构建 Hermes 插件](/guides/build-a-hermes-plugin)

大多数自定义工具创建场景默认使用插件。只有当你明确希望在 `tools/` 和 `toolsets.py` 中发布新的内置工具时，才遵循本页面。
:::

以下情况应创建 **Skill**：该能力可以通过指令 + shell 命令 + 现有工具来实现（如 arXiv 搜索、git 工作流、Docker 管理、PDF 处理）。

以下情况应创建 **Tool**：需要与 API 密钥进行端到端集成、自定义处理逻辑、二进制数据处理或流式传输（如浏览器自动化、TTS、视觉分析）。

## 概述

添加一个工具涉及 **2 个文件**：

1. **`tools/your_tool.py`** — handler、schema、check 函数、`registry.register()` 调用
2. **`toolsets.py`** — 将工具名称添加到 `_HERMES_CORE_TOOLS`（或特定 toolset）

任何包含顶层 `registry.register()` 调用的 `tools/*.py` 文件都会在启动时被自动发现——无需手动维护导入列表。

## 第一步：创建内置工具文件

每个工具文件遵循相同的结构：

```python
# tools/weather_tool.py
"""Weather Tool -- look up current weather for a location."""

import json
import os
import logging

logger = logging.getLogger(__name__)


# --- Availability check ---

def check_weather_requirements() -> bool:
    """Return True if the tool's dependencies are available."""
    return bool(os.getenv("WEATHER_API_KEY"))


# --- Handler ---

def weather_tool(location: str, units: str = "metric") -> str:
    """Fetch weather for a location. Returns JSON string."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return json.dumps({"error": "WEATHER_API_KEY not configured"})
    try:
        # ... call weather API ...
        return json.dumps({"location": location, "temp": 22, "units": units})
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- Schema ---

WEATHER_SCHEMA = {
    "name": "weather",
    "description": "Get current weather for a location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or coordinates (e.g. 'London' or '51.5,-0.1')"
            },
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "Temperature units (default: metric)",
                "default": "metric"
            }
        },
        "required": ["location"]
    }
}


# --- Registration ---

from tools.registry import registry

registry.register(
    name="weather",
    toolset="weather",
    schema=WEATHER_SCHEMA,
    handler=lambda args, **kw: weather_tool(
        location=args.get("location", ""),
        units=args.get("units", "metric")),
    check_fn=check_weather_requirements,
    requires_env=["WEATHER_API_KEY"],
)
```

### 关键规则

:::danger 重要
- Handler **必须**返回 JSON 字符串（通过 `json.dumps()`），不得返回原始 dict
- 错误**必须**以 `{"error": "message"}` 形式返回，不得抛出异常
- `check_fn` 在构建工具定义时被调用——若返回 `False`，该工具将被静默排除
- `handler` 接收 `(args: dict, **kwargs)`，其中 `args` 是 LLM 的工具调用参数
:::

## 第二步：将内置工具添加到 Toolset

在 `toolsets.py` 中添加工具名称：

```python
# If it should be available on all platforms (CLI + messaging):
_HERMES_CORE_TOOLS = [
    ...
    "weather",  # <-- add here
]

# Or create a new standalone toolset:
"weather": {
    "description": "Weather lookup tools",
    "tools": ["weather"],
    "includes": []
},
```

## ~~第三步：添加发现导入~~（不再需要）

包含顶层 `registry.register()` 调用的工具模块会由 `tools/registry.py` 中的 `discover_builtin_tools()` 自动发现。无需手动维护导入列表——只需在 `tools/` 中创建文件，启动时即可自动加载。

## 异步 Handler

如果你的 handler 需要异步代码，使用 `is_async=True` 标记：

```python
async def weather_tool_async(location: str) -> str:
    async with aiohttp.ClientSession() as session:
        ...
    return json.dumps(result)

registry.register(
    name="weather",
    toolset="weather",
    schema=WEATHER_SCHEMA,
    handler=lambda args, **kw: weather_tool_async(args.get("location", "")),
    check_fn=check_weather_requirements,
    is_async=True,  # registry calls _run_async() automatically
)
```

registry 会透明地处理异步桥接——你无需自己调用 `asyncio.run()`。

## 需要 task_id 的 Handler

管理每个会话状态的工具通过 `**kwargs` 接收 `task_id`：

```python
def _handle_weather(args, **kw):
    task_id = kw.get("task_id")
    return weather_tool(args.get("location", ""), task_id=task_id)

registry.register(
    name="weather",
    ...
    handler=_handle_weather,
)
```

## Agent 循环拦截工具

某些工具（`todo`、`memory`、`session_search`、`delegate_task`）需要访问每个会话的 agent 状态。这些工具在到达 registry 之前会被 `run_agent.py` 拦截。registry 仍然保存它们的 schema，但如果绕过拦截，`dispatch()` 会返回一个回退错误。

## 可选：Setup Wizard 集成

如果你的工具需要 API 密钥，将其添加到 `hermes_cli/config.py`：

```python
OPTIONAL_ENV_VARS = {
    ...
    "WEATHER_API_KEY": {
        "description": "Weather API key for weather lookup",
        "prompt": "Weather API key",
        "url": "https://weatherapi.com/",
        "tools": ["weather"],
        "password": True,
    },
}
```

## 检查清单

- [ ] 已创建包含 handler、schema、check 函数和注册调用的工具文件
- [ ] 已在 `toolsets.py` 中添加到适当的 toolset
- [ ] 已确认该工具确实应为内置/核心工具而非插件
- [ ] Handler 返回 JSON 字符串，错误以 `{"error": "..."}` 形式返回
- [ ] 可选：已将 API 密钥添加到 `hermes_cli/config.py` 的 `OPTIONAL_ENV_VARS`
- [ ] 可选：已添加到 `toolset_distributions.py` 以支持批量处理
- [ ] 已通过 `hermes chat -q "Use the weather tool for London"` 测试