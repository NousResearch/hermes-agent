---
sidebar_position: 12
title: "网页搜索提供商插件"
description: "如何为 Hermes Agent 构建网页搜索/提取/爬取后端插件"
---

# 构建网页搜索提供商插件

网页搜索提供商插件注册一个后端，用于处理 `web_search`、`web_extract` 以及（可选的）深度爬取工具调用。内置提供商——Firecrawl、SearXNG、Tavily、Exa、Parallel、Brave Search（免费层）和 DDGS——均以插件形式存放于 `plugins/web/<name>/` 目录下。你可以在该目录旁新建一个目录来添加新提供商，或覆盖已有的内置提供商。

:::tip
网页搜索是 Hermes 支持的多种**后端插件**之一。其他插件（各有其 ABC）包括：[图像生成提供商插件](/developer-guide/image-gen-provider-plugin)、[视频生成提供商插件](/developer-guide/video-gen-provider-plugin)、[记忆提供商插件](/developer-guide/memory-provider-plugin)、[上下文引擎插件](/developer-guide/context-engine-plugin)和[模型提供商插件](/developer-guide/model-provider-plugin)。通用工具/hook/CLI 插件请参阅[构建 Hermes 插件](/guides/build-a-hermes-plugin)。
:::

## 发现机制

Hermes 在三个位置扫描网页搜索后端：

1. **内置** — `<repo>/plugins/web/<name>/`（以 `kind: backend` 自动加载，始终可用）
2. **用户** — `~/.hermes/plugins/web/<name>/`（通过 `plugins.enabled` 或 `hermes plugins enable <name>` 按需启用）
3. **Pip** — 声明了 `hermes_agent.plugins` 入口点的包

每个插件的 `register(ctx)` 函数调用 `ctx.register_web_search_provider(...)` ——将实例注册到 `agent/web_search_registry.py` 中的注册表。各能力的活跃提供商由配置决定：

| 能力 | 配置键 | 回退至 |
|---|---|---|
| `web_search` | `web.search_backend` | `web.backend` |
| `web_extract` | `web.extract_backend` | `web.backend` |
| `web_extract` 内的深度爬取模式 | `web.extract_backend` | `web.backend` |

若两个键均未设置，Hermes 将根据环境中存在的 API key/URL 自动检测后端。`hermes tools` 会引导用户完成选择。

## 目录结构

```
plugins/web/my-backend/
├── __init__.py     # register() 入口点
├── provider.py     # WebSearchProvider 子类
└── plugin.yaml     # 包含 kind: backend 和 provides_web_providers 的清单文件
```

`brave_free/` 和 `ddgs/` 是代码库中最小的参考实现——`brave_free` 是需要 API key 的纯搜索提供商，`ddgs` 是无需 key 且懒加载 SDK 的提供商。

## WebSearchProvider ABC

继承 `agent.web_search_provider.WebSearchProvider`。唯一必须实现的成员是 `name`、`is_available()`，以及你所实现的 `search()` / `extract()` / `crawl()` 中的相应方法。

```python
# plugins/web/my-backend/provider.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider


class MyBackendWebSearchProvider(WebSearchProvider):
    """Minimal search-only provider against the My Backend HTTP API."""

    @property
    def name(self) -> str:
        # Stable id used in web.search_backend / web.extract_backend / web.backend
        # config keys. Lowercase, no spaces; hyphens permitted.
        return "my-backend"

    @property
    def display_name(self) -> str:
        # Human label shown in `hermes tools`. Defaults to `name`.
        return "My Backend"

    def is_available(self) -> bool:
        # Cheap check — env var present, optional dep importable, etc.
        # MUST NOT make network calls (runs on every `hermes tools` paint).
        return bool(os.getenv("MY_BACKEND_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx

        api_key = os.environ["MY_BACKEND_API_KEY"]
        try:
            resp = httpx.get(
                "https://api.example.com/search",
                params={"q": query, "count": max(1, min(int(limit), 20))},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            return {"success": False, "error": str(exc)}

        # Response shape is fixed — see "Response shape" below.
        return {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "description": item.get("snippet", ""),
                        "position": idx + 1,
                    }
                    for idx, item in enumerate(data.get("results", []))
                ],
            },
        }
```

```python
# plugins/web/my-backend/__init__.py
from plugins.web.my_backend.provider import MyBackendWebSearchProvider


def register(ctx) -> None:
    """Plugin entry point — called once at load time."""
    ctx.register_web_search_provider(MyBackendWebSearchProvider())
```

## plugin.yaml

```yaml
name: web-my-backend
version: 1.0.0
description: "My Backend web search — Bearer-auth REST API"
author: Your Name
kind: backend
provides_web_providers:
  - my-backend
requires_env:
  - MY_BACKEND_API_KEY
```

| 键 | 用途 |
|---|---|
| `kind: backend` | 将插件路由至后端加载路径 |
| `provides_web_providers` | 该插件注册的提供商 `name` 列表——在 `register()` 运行之前，加载器即可通过此字段在 `hermes tools` 中公示插件 |
| `requires_env` | 在 `hermes plugins install` 期间进行交互式凭据提示（富格式说明参见[构建 Hermes 插件](/guides/build-a-hermes-plugin#gate-on-environment-variables)） |

## ABC 参考

完整契约位于 `agent/web_search_provider.py`。可覆盖的方法如下：

| 成员 | 必须 | 默认值 | 用途 |
|---|---|---|---|
| `name` | ✅ | — | 在 `web.*_backend` 配置中使用的稳定 id |
| `display_name` | — | `name` | 在 `hermes tools` 中显示的标签 |
| `is_available()` | ✅ | — | 轻量可用性检查——环境变量、可选依赖等 |
| `supports_search()` | — | `True` | `web_search` 路由的能力标志 |
| `supports_extract()` | — | `False` | `web_extract` 路由的能力标志 |
| `search(query, limit)` | 条件必须 | 抛出异常 | 当 `supports_search()` 返回 `True` 时必须实现 |
| `extract(urls, **kwargs)` | 条件必须 | 抛出异常 | 当 `supports_extract()` 返回 `True` 时必须实现 |

提供商可以在单个类中声明多种能力——Firecrawl、Tavily、Exa 和 Parallel 均实现了搜索和提取两种能力。Brave Search 和 DDGS 仅支持搜索；SearXNG 也仅支持搜索，并有文档说明的"与提取提供商配对使用"工作流。

## 响应格式

工具包装器期望固定的响应信封（envelope），以避免在不同后端之间进行转换。

**搜索成功：**

```python
{
    "success": True,
    "data": {
        "web": [
            {"title": str, "url": str, "description": str, "position": int},
            ...
        ],
    },
}
```

**提取成功：**

```python
{
    "success": True,
    "data": [
        {
            "url": str,
            "title": str,
            "content": str,
            "raw_content": str,
            "metadata": dict,    # optional
            "error": str,        # optional, only on per-URL failure
        },
        ...
    ],
}
```

**任意能力，失败时：**

```python
{"success": False, "error": "human-readable message"}
```

`search()` 和 `extract()` 均可定义为 `async def`——调度器通过 `inspect.iscoroutinefunction` 检测协程函数并相应地进行 await。对于小型后端，执行阻塞 I/O（HTTP、SDK 调用）的同步实现也完全可行；调度器会处理线程调度。

## 能力标志

Hermes 根据 `supports_*` 标志将调用路由至正确的提供商。一种常见的多提供商配置：

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "brave-free"     # 纯搜索，速度快，每月免费 2k 次
  extract_backend: "firecrawl"     # 提取 + 爬取，付费配额
```

当 `web.search_backend` 或 `web.extract_backend` 未设置时，均回退至 `web.backend`。若该项也未设置，Hermes 将根据环境变量的存在情况，选取第一个支持所请求能力的可用提供商。

如果你的提供商只支持一种能力，将其他标志保持默认值（`False`）即可，注册表会在对应工具调用时跳过它——当用户仅将 X 用于搜索而要求 agent 进行提取时，不会看到误导性的"提供商 X 失败"错误。

## Hermes 如何将其接入工具

`web_search` 和 `web_extract` 工具位于 `tools/web_tools.py`。调用时执行以下步骤：

1. 读取相关配置键（`web_search` 对应 `web.search_backend`，`web_extract` 对应 `web.extract_backend`）
2. 向注册表查询具有该 `name` 的提供商
3. 检查 `is_available()` 及对应的 `supports_*()` 标志
4. 调度至 `search()` / `extract()` / `crawl()`，若方法为协程则进行 await
5. 将响应信封 JSON 序列化后返回给 LLM

错误以工具结果的形式呈现；LLM 决定如何解释。若没有提供商被注册（或所有可用提供商均未通过能力检查），工具将返回一条指向 `hermes tools` 的友好错误信息。

## 懒加载可选依赖

如果你的提供商封装了第三方 SDK（如 DDGS 封装了 `ddgs` 包），请勿在模块顶层 `import`。在 `is_available()` 或 `search()` 内部使用 `tools.lazy_deps.ensure(...)` ——Hermes 将在首次使用时安装该包，并受 `security.allow_lazy_installs` 控制。安全模型详见[构建 Hermes 插件 → 懒加载](/guides/build-a-hermes-plugin#lazy-install-optional-python-dependencies)。

## 参考实现

- **`plugins/web/brave_free/`** — 小型、需要 API key 的纯搜索 HTTP 提供商。适合作为起始模板。
- **`plugins/web/ddgs/`** — 无需 key、懒加载 SDK 的提供商。适用于封装 Python 包的后端。
- **`plugins/web/firecrawl/`** — 完整的多能力提供商（搜索 + 提取 + 爬取），支持多种格式模式。
- **`plugins/web/searxng/`** — 自托管、通过 URL 配置、无需认证的后端。
- **`plugins/web/xai/`** — 通过 Grok 服务端 `web_search` 工具实现的 LLM 驱动搜索。展示了如何复用现有的 OAuth/环境变量凭据（`tools/xai_http.py`）而无需新增环境变量，以及如何编写遵守无网络调用约定的轻量 `is_available()`。

## 通过 pip 分发

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-backend-web = "my_backend_web_package"
```

`my_backend_web_package` 必须暴露顶层 `register` 函数。完整配置说明参见通用插件指南中的[通过 pip 分发](/guides/build-a-hermes-plugin#distribute-via-pip)。

## 相关页面

- [网页搜索](/user-guide/features/web-search) — 面向用户的功能文档及各后端配置说明
- [插件概览](/user-guide/features/plugins) — 所有插件类型一览
- [构建 Hermes 插件](/guides/build-a-hermes-plugin) — 通用工具/hook/斜杠命令指南