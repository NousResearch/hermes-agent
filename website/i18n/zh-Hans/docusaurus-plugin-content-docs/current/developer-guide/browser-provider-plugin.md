---
sidebar_position: 13
title: "浏览器 Provider 插件"
description: "如何为 Hermes Agent 构建云端浏览器后端插件"
---

# 构建浏览器 Provider 插件

浏览器 Provider 插件负责注册一个**云端浏览器后端**，用于处理云端模式下的 `browser_*` 工具调用（导航、点击、截图等）。内置的 Provider —— Browserbase、Browser Use 和 Firecrawl —— 都以插件形式存在于 `plugins/browser/<name>/` 目录下。你可以新增一个 Provider，也可以在它们旁边放一个目录来覆盖已有的 Provider。

:::tip
浏览器后端只是 Hermes 支持的多种**后端插件**之一。其他插件（各有自己的 ABC）包括 [Web Search Provider 插件](/developer-guide/web-search-provider-plugin)（本 ABC 刻意与其保持一致）、[图片生成](/developer-guide/image-gen-provider-plugin)、[视频生成](/developer-guide/video-gen-provider-plugin)、[Memory Provider](/developer-guide/memory-provider-plugin)、[Context Engine](/developer-guide/context-engine-plugin)、[Secret Source](/developer-guide/secret-source-plugin) 以及 [Model Provider](/developer-guide/model-provider-plugin)。通用的工具/钩子/CLI 插件请参考[构建 Hermes 插件](/developer-guide/plugins)。
:::

## 工作原理

浏览器 Provider **并不**实现浏览功能。它实现的是**会话生命周期**：创建远程浏览器会话、返回 CDP websocket URL、以及拆除会话。Hermes 自身的浏览器栈（`agent-browser` + `tools/browser_tool.py`）会连接你返回的 CDP URL，并从那里驱动页面 —— 每个 Provider 都能免费获得完整的 `browser_*` 工具集。

当前激活的 Provider 由 `config.yaml` 中的 `browser.cloud_provider` 配置项决定；`tools/browser_tool.py` 中的调度器是一个纯粹的注册表查找，没有任何针对特定 Provider 的条件逻辑。

## 发现机制

Hermes 会在三个位置扫描浏览器后端：

1. **内置** —— `<repo>/plugins/browser/<name>/`（带 `kind: backend` 标记的自动加载）
2. **用户** —— `~/.hermes/plugins/browser/<name>/`（通过 `plugins.enabled` 或 `hermes plugins enable <name>` 手动启用）
3. **Pip** —— 声明了 `hermes_agent.plugins` 入口点的 Python 包

每个插件的 `register(ctx)` 会调用 `ctx.register_browser_provider(...)`，将实例注册到 `agent/browser_registry.py` 的注册表中。

## 目录结构

```
plugins/browser/my-backend/
├── __init__.py     # register() 入口点
├── provider.py     # BrowserProvider 子类
└── plugin.yaml     # 清单文件，包含 kind: backend 和 provides_browser_providers
```

`plugin.yaml`：

```yaml
name: browser-my-backend
version: 1.0.0
description: "My cloud browser backend. Requires MY_BACKEND_API_KEY."
author: you
kind: backend
provides_browser_providers:
  - my-backend
```

`__init__.py`：

```python
from plugins.browser.my_backend.provider import MyBackendProvider


def register(ctx) -> None:
    ctx.register_browser_provider(MyBackendProvider())
```

## BrowserProvider ABC

实现 `agent.browser_provider.BrowserProvider`。需要实现三个生命周期方法加一个身份属性：

```python
from agent.browser_provider import BrowserProvider


class MyBackendProvider(BrowserProvider):
    @property
    def name(self) -> str:
        return "my-backend"          # the browser.cloud_provider config value

    @property
    def display_name(self) -> str:
        return "My Backend"          # shown in `hermes tools`

    def is_available(self) -> bool:
        """Cheap check only — env var present, dep importable.
        NO network calls: runs at tool-registration time and on every
        `hermes tools` paint."""
        return bool(os.environ.get("MY_BACKEND_API_KEY"))

    def create_session(self, task_id: str) -> dict:
        """Create a remote browser session; return the session-metadata contract."""
        session = my_api.create_browser(...)
        return {
            "session_name": f"my-backend-{task_id}",  # unique agent-browser session name
            "bb_session_id": session.id,              # provider session ID (for cleanup)
            "cdp_url": session.cdp_ws_url,            # CDP websocket URL
            "features": {"stealth": True},            # feature flags you enabled
        }

    def close_session(self, session_id: str) -> bool:
        """Terminate by provider session ID. Log-and-return-False on error —
        never raise, so the dispatcher's cleanup loop keeps moving."""
        ...

    def emergency_cleanup(self, session_id: str) -> None:
        """Best-effort teardown from atexit/signal handlers. Must not raise."""
        ...
```

### 会话元数据契约

`create_session()` 至少需要返回 `session_name`、`bb_session_id`、`cdp_url` 和 `features`。有两个值得了解的特殊之处：

- **`bb_session_id` 是一个历史遗留的键名**，为保持与 `tools/browser_tool.py` 的向后兼容而保留原样 —— 它存放的是*你的* Provider 的会话 ID，与厂商无关。不要重命名它。
- `create_session()` **可以抛出异常** —— 凭据缺失时抛出 `ValueError`，网络/API 故障时抛出 `RuntimeError`。调度器会将这些错误呈现给用户。这与 `close_session`/`emergency_cleanup` 不同，后者绝不能抛出异常。

可选的 `external_call_id` 键用于支持托管网关的计费。

### `get_setup_schema()` —— `hermes tools` 选择界面的行

重写此方法可以让你在浏览器自动化选择器中以一等选项的身份出现，包含 API 密钥提示和安装钩子：

```python
def get_setup_schema(self) -> dict:
    return {
        "name": "My Backend",
        "badge": "paid",
        "tag": "Cloud browser with stealth and proxies",
        "env_vars": [
            {"key": "MY_BACKEND_API_KEY",
             "prompt": "My Backend API key",
             "url": "https://mybackend.example"},
        ],
        "post_setup": "agent_browser",   # auto-installs the agent-browser npm dep
    }
```

根据项目的工具后端标准：如果一个后端不能通过 `hermes tools` 选择和配置，那就不算完成 —— "手动设置这个环境变量"不算是集成。

## 用户配置

```yaml
browser:
  cloud_provider: my-backend
```

## 参考实现

`plugins/browser/` 下的三个内置 Provider 是标准示例，按复杂度递增排列：`firecrawl`（最简单）、`browser_use`，以及 `browserbase`（隐身/代理/保活等特性标记，在付费功能不可用时能优雅降级）。复制最接近你需求的那个即可。

## 检查清单

- [ ] `name` 是小写且稳定的（它是用户编写的配置值）
- [ ] `is_available()` 不做任何网络调用
- [ ] `create_session()` 返回完整的元数据契约（`bb_session_id` 键名保持不变）
- [ ] `close_session()` / `emergency_cleanup()` 绝不抛出异常
- [ ] `get_setup_schema()` 暴露你的环境变量，让 `hermes tools` 能配置后端
- [ ] `plugin.yaml` 声明了 `kind: backend` + `provides_browser_providers`
