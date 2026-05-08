---
sidebar_position: 8
title: "内存 Provider 插件"
description: "如何为 Hermes Agent 构建内存 provider 插件"
---

# 构建内存 Provider 插件

内存 provider 插件为 Hermes Agent 提供超越内置 MEMORY.md 和 USER.md 的持久化跨会话知识。本指南涵盖如何构建一个。

:::tip
内存 provider 是两种**provider 插件**类型之一。另一种是[上下文引擎插件](/docs/developer-guide/context-engine-plugin)，它替换内置上下文压缩器。两者遵循相同的模式：单选、由配置驱动、通过 `hermes plugins` 管理。
:::

## 目录结构

每个内存 provider 位于 `plugins/memory/<name>/`：

```
plugins/memory/my-provider/
├── __init__.py      # MemoryProvider implementation + register() entry point
├── plugin.yaml      # Metadata (name, description, hooks)
└── README.md        # Setup instructions, config reference, tools
```

## MemoryProvider ABC

您的插件实现来自 `agent/memory_provider.py` 的 `MemoryProvider` 抽象基类：

```python
from agent.memory_provider import MemoryProvider

class MyMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "my-provider"

    def is_available(self) -> bool:
        """Check if this provider can activate. NO network calls."""
        return bool(os.environ.get("MY_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        """Called once at agent startup.

        kwargs always includes:
          hermes_home (str): Active HERMES_HOME path. Use for storage.
        """
        self._api_key = os.environ.get("MY_API_KEY", "")
        self._session_id = session_id

    # ... implement remaining methods
```

## 必需方法

### 核心生命周期

| 方法 | 调用时机 | 必须实现？ |
|--------|-----------|-----------------|
| `name` (property) | 始终 | **是** |
| `is_available()` | 代理初始化，激活前 | **是** — 无网络调用 |
| `initialize(session_id, **kwargs)` | 代理启动 | **是** |
| `get_tool_schemas()` | 初始化后，用于工具注入 | **是** |
| `handle_tool_call(name, args)` | 代理使用您的工具时 | **是**（如果您有工具） |

### 配置

| 方法 | 目的 | 必须实现？ |
|--------|---------|-----------------|
| `get_config_schema()` | 为 `hermes memory setup` 声明配置字段 | **是** |
| `save_config(values, hermes_home)` | 将非 secret 配置写入原生位置 | **是**（除非仅 env 变量） |

### 可选钩子

| 方法 | 调用时机 | 用例 |
|--------|-----------|----------|
| `system_prompt_block()` | 系统提示词组装 | 静态 provider 信息 |
| `prefetch(query)` | 每个 API 调用之前 | 返回回忆的上下文 |
| `queue_prefetch(query)` | 每轮之后 | 为下一轮预热 |
| `sync_turn(user, assistant)` | 每个完成的轮次之后 | 持久化对话 |
| `on_session_end(messages)` | 对话结束 | 最终提取/刷新 |
| `on_pre_compress(messages)` | 上下文压缩之前 | 在丢弃前保存洞察 |
| `on_memory_write(action, target, content)` | 内置内存写入 | 镜像到您的后端 |
| `shutdown()` | 进程退出 | 清理连接 |

## 配置模式

`get_config_schema()` 返回字段描述符列表，供 `hermes memory setup` 使用：

```python
def get_config_schema(self):
    return [
        {
            "key": "api_key",
            "description": "My Provider API key",
            "secret": True,           # → written to .env
            "required": True,
            "env_var": "MY_API_KEY",   # explicit env var name
            "url": "https://my-provider.com/keys",  # where to get it
        },
        {
            "key": "region",
            "description": "Server region",
            "default": "us-east",
            "choices": ["us-east", "eu-west", "ap-south"],
        },
        {
            "key": "project",
            "description": "Project identifier",
            "default": "hermes",
        },
    ]
```

`secret: True` 和 `env_var` 的字段到 `.env`。非 secret 字段传递给 `save_config()`。

:::tip 最小 vs 完整模式
`get_config_schema()` 中的每个字段在 `hermes memory setup` 期间提示。拥有许多选项的 provider 应保持模式最小 — 仅包括用户**必须**配置的字段（API 密钥、必需凭据）。在配置文件中记录可选设置（例如 `$HERMES_HOME/myprovider.json`）而不是在设置期间提示所有设置。这保持设置向导快速，同时仍支持高级配置。参见 Supermemory provider 作为示例 — 它仅提示 API 密钥；所有其他选项位于 `supermemory.json`。
:::

## 保存配置

```python
def save_config(self, values: dict, hermes_home: str) -> None:
    """Write non-secret config to your native location."""
    import json
    from pathlib import Path
    config_path = Path(hermes_home) / "my-provider.json"
    config_path.write_text(json.dumps(values, indent=2))
```

对于仅 env 变量的 provider，保留默认 no-op。

## 插件入口点

```python
def register(ctx) -> None:
    """Called by the memory plugin discovery system."""
    ctx.register_memory_provider(MyMemoryProvider())
```

## plugin.yaml

```yaml
name: my-provider
version: 1.0.0
description: "Short description of what this provider does."
hooks:
  - on_session_end    # list hooks you implement
```

## 线程约定

**`sync_turn()` 必须是非阻塞的。** 如果您的后端有延迟（API 调用、LLM 处理），在守护线程中运行工作：

```python
def sync_turn(self, user_content, assistant_content):
    def _sync():
        try:
            self._api.ingest(user_content, assistant_content)
        except Exception as e:
            logger.warning("Sync failed: %s", e)

    if self._sync_thread and self._sync_thread.is_alive():
        self._sync_thread.join(timeout=5.0)
    self._sync_thread = threading.Thread(target=_sync, daemon=True)
    self._sync_thread.start()
```

## 配置文件隔离

所有存储路径**必须**使用 `initialize()` 中的 `hermes_home` kwarg，而不是硬编码 `~/.hermes`：

```python
# CORRECT — profile-scoped
from hermes_constants import get_hermes_home
data_dir = get_hermes_home() / "my-provider"

# WRONG — shared across all profiles
data_dir = Path("~/.hermes/my-provider").expanduser()
```

## 测试

参见 `tests/agent/test_memory_plugin_e2e.py` 获取使用真实 SQLite provider 的完整 E2E 测试模式。

```python
from agent.memory_manager import MemoryManager

mgr = MemoryManager()
mgr.add_provider(my_provider)
mgr.initialize_all(session_id="test-1", platform="cli")

# Test tool routing
result = mgr.handle_tool_call("my_tool", {"action": "add", "content": "test"})

# Test lifecycle
mgr.sync_all("user msg", "assistant msg")
mgr.on_session_end([])
mgr.shutdown_all()
```

## 添加 CLI 命令

内存 provider 插件可以注册自己的 CLI 子命令树（例如 `hermes my-provider status`、`hermes my-provider config`）。这使用基于约定的发现系统 — 无需更改核心文件。

### 工作原理

1. 在插件目录添加 `cli.py` 文件
2. 定义构建 argparse 树的 `register_cli(subparser)` 函数
3. 内存插件系统通过 `discover_plugin_cli_commands()` 在启动时发现它
4. 您的命令出现在 `hermes <provider-name> <subcommand>` 下

**活动 provider 门控：** 您的 CLI 命令仅在 provider 是配置中活动的 `memory.provider` 时显示。如果用户未配置您的 provider，您的命令不会显示在 `hermes --help` 中。

### 示例

```python
# plugins/memory/my-provider/cli.py

def my_command(args):
    """Handler dispatched by argparse."""
    sub = getattr(args, "my_command", None)
    if sub == "status":
        print("Provider is active and connected.")
    elif sub == "config":
        print("Showing config...")
    else:
        print("Usage: hermes my-provider <status|config>")

def register_cli(subparser) -> None:
    """Build the hermes my-provider argparse tree.

    Called by discover_plugin_cli_commands() at argparse setup time.
    """
    subs = subparser.add_subparsers(dest="my_command")
    subs.add_parser("status", help="Show provider status")
    subs.add_parser("config", help="Show provider config")
    subparser.set_defaults(func=my_command)
```

### 参考实现

参见 `plugins/memory/honcho/cli.py` 获取具有 13 个子命令、跨配置文件管理（`--target-profile`）以及配置读/写的完整示例。

### 带 CLI 的目录结构

```
plugins/memory/my-provider/
├── __init__.py      # MemoryProvider implementation + register()
├── plugin.yaml      # Metadata
├── cli.py           # register_cli(subparser) — CLI commands
└── README.md        # Setup instructions
```

## 单 Provider 规则

一次只能激活**一个**外部内存 provider。如果用户尝试注册第二个，MemoryManager 会拒绝并发出警告。这防止工具模式膨胀和冲突后端。
