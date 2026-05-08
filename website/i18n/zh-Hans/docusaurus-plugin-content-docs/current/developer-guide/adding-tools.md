---
sidebar_position: 2
title: "添加工具"
description: "如何向 Hermes Agent 添加新工具 — 模式、处理器、注册和工具集"
---

# 添加工具

在编写工具之前，问问自己：**这应该是一个 [skill](creating-skills.md) 吗？**

:::warning 仅内置核心工具
此页面用于将**内置 Hermes 工具**添加到仓库本身。
如果您想要个人的、项目本地的或其他自定义工具而不修改 Hermes 核心，请使用插件路径：

- [插件](/docs/user-guide/features/plugins)
- [构建 Hermes 插件](/docs/guides/build-a-hermes-plugin)

对于大多数自定义工具创建，默认使用插件。只有当您明确想要在 `tools/` 和 `toolsets.py` 中发布新的内置工具时，才遵循此页面。
:::

当功能可以表达为指令 + shell 命令 + 现有工具（arXiv 搜索、git 工作流、Docker 管理、PDF 处理）时，将其制作为 **Skill**。

当它需要与 API 密钥、自定义处理逻辑、二进制数据处理或流式传输（浏览器自动化、TTS、视觉分析）的端到端集成时，将其制作为 **Tool**。

## 概述

添加工具涉及 **2 个文件**：

1. **`tools/your_tool.py`** — 处理器、模式、检查函数、`registry.register()` 调用
2. **`toolsets.py`** — 将工具名称添加到 `_HERMES_CORE_TOOLS`（或特定工具集）

任何带有顶层 `registry.register()` 调用的 `tools/*.py` 文件都会在启动时自动被发现 — 不需要手动导入列表。

## 步骤 1：创建内置工具文件

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
- 处理器**必须**返回 JSON 字符串（通过 `json.dumps()`），永远不是原始字典
- 错误**必须**返回为 `{"error": "message"}`，永远不作为异常抛出
- 构建工具定义时调用 `check_fn` — 如果返回 `False`，工具将被静默排除
- `handler` 接收 `(args: dict, **kwargs)`，其中 `args` 是 LLM 的工具调用参数
:::

## 步骤 2：将内置工具添加到工具集

在 `toolsets.py` 中，添加工具名称：

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

## ~~步骤 3：添加发现导入~~（不再需要）

带有顶层 `registry.register()` 调用的工具模块由 `tools/registry.py` 中的 `discover_builtin_tools()` 自动发现。不需要维护手动导入列表 — 只需在 `tools/` 中创建您的文件，它就会在启动时被拾取。

## 异步处理器

如果您的处理器需要异步代码，使用 `is_async=True` 标记：

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

注册表透明地处理异步桥接 — 您永远不需要自己调用 `asyncio.run()`。

## 需要 task_id 的处理器

管理每会话状态的工具通过 `**kwargs` 接收 `task_id`：

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

## 代理循环拦截工具

某些工具（`todo`、`memory`、`session_search`、`delegate_task`）需要访问每会话代理状态。这些工具由 `run_agent.py` 在到达注册表之前拦截。注册表仍然保存其模式，但如果拦截被绕过，`dispatch()` 返回后备错误。

## 可选：设置向导集成

如果您的工具需要 API 密钥，将其添加到 `hermes_cli/config.py`：

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

## 清单

- [ ] 工具文件已创建，包含处理器、模式、检查函数和注册
- [ ] 添加到 `toolsets.py` 中的适当工具集
- [ ] 确认这真的应该是内置/核心工具而不是插件
- [ ] 处理器返回 JSON 字符串，错误作为 `{"error": "..."}` 返回
- [ ] 可选：API 密钥添加到 `hermes_cli/config.py` 中的 `OPTIONAL_ENV_VARS`
- [ ] 可选：添加到 `toolset_distributions.py` 用于批处理
- [ ] 使用 `hermes chat -q "Use the weather tool for London"` 测试
