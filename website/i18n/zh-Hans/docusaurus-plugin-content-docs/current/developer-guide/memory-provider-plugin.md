---
sidebar_position: 8
title: "Memory Provider 插件"
description: "如何为 Hermes Agent 构建 memory provider 插件"
---

# 构建 Memory Provider 插件

Memory provider 插件为 Hermes Agent 提供跨会话的持久化知识，超越内置的 MEMORY.md 和 USER.md。本指南介绍如何构建一个 memory provider 插件。

:::tip
Memory provider 是两种 **provider 插件**类型之一。另一种是 [Context Engine 插件](/developer-guide/context-engine-plugin)，用于替换内置的上下文压缩器。两者遵循相同的模式：单选、配置驱动、通过 `hermes plugins` 管理。
:::

## 目录结构

每个 memory provider 位于 `plugins/memory/<name>/`：

```
plugins/memory/my-provider/
├── __init__.py      # MemoryProvider 实现 + register() 入口点
├── plugin.yaml      # 元数据（name、description、hooks）
└── README.md        # 配置说明、配置参考、工具
```

## MemoryProvider 抽象基类

你的插件需要实现 `agent/memory_provider.py` 中的 `MemoryProvider` 抽象基类（ABC）：

```python
from agent.memory_provider import MemoryProvider

class MyMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "my-provider"

    def is_available(self) -> bool:
        """检查此 provider 是否可以激活。禁止发起网络请求。"""
        return bool(os.environ.get("MY_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        """在 agent 启动时调用一次。

        kwargs 始终包含：
          hermes_home (str): 当前活跃的 HERMES_HOME 路径。用于存储数据。
        """
        self._api_key = os.environ.get("MY_API_KEY", "")
        self._session_id = session_id

    # ... 实现其余方法
```

## 必须实现的方法

### 核心生命周期

| 方法 | 调用时机 | 是否必须实现？ |
|--------|-----------|-----------------|
| `name`（property） | 始终 | **是** |
| `is_available()` | agent 初始化，激活前 | **是** — 禁止网络请求 |
| `initialize(session_id, **kwargs)` | agent 启动 | **是** |
| `get_tool_schemas()` | 初始化后，用于注入工具 | **是** |
| `handle_tool_call(name, args)` | agent 调用你的工具时 | **是**（如果有工具） |

### 配置

| 方法 | 用途 | 是否必须实现？ |
|--------|---------|-----------------|
| `get_config_schema()` | 为 `hermes memory setup` 声明配置字段 | **是** |
| `save_config(values, hermes_home)` | 将非敏感配置写入原生位置 | **是**（除非仅使用环境变量） |

### 可选 Hook

| 方法 | 调用时机 | 使用场景 |
|--------|-----------|----------|
| `system_prompt_block()` | 系统 prompt 组装时 | 静态 provider 信息 |
| `prefetch(query)` | 每次 API 调用前 | 返回召回的上下文 |
| `queue_prefetch(query)` | 每轮对话结束后 | 为下一轮预热 |
| `sync_turn(user, assistant)` | 每轮对话完成后 | 持久化对话内容 |
| `on_session_end(messages)` | 对话结束时 | 最终提取/刷新 |
| `on_pre_compress(messages)` | 上下文压缩前 | 在丢弃前保存关键信息 |
| `on_memory_write(action, target, content)` | 内置 memory 写入时 | 同步到你的后端 |
| `shutdown()` | 进程退出时 | 清理连接 |

## 配置 Schema

`get_config_schema()` 返回一个字段描述符列表，供 `hermes memory setup` 使用：

```python
def get_config_schema(self):
    return [
        {
            "key": "api_key",
            "description": "My Provider API key",
            "secret": True,           # → 写入 .env
            "required": True,
            "env_var": "MY_API_KEY",   # 显式指定环境变量名
            "url": "https://my-provider.com/keys",  # 获取密钥的地址
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

`secret: True` 且带有 `env_var` 的字段写入 `.env`。非敏感字段传递给 `save_config()`。

:::tip 最简 Schema 与完整 Schema
`get_config_schema()` 中的每个字段都会在 `hermes memory setup` 期间提示用户输入。选项较多的 provider 应保持 schema 精简——只包含用户**必须**配置的字段（API key、必要凭证）。可选配置请在配置文件参考文档中说明（例如 `$HERMES_HOME/myprovider.json`），而不是在 setup 向导中逐一提示。这样既能保持 setup 流程简洁，又支持高级配置。可参考 Supermemory provider 的实现——它只提示输入 API key，其余选项均位于 `supermemory.json` 中。
:::

## 保存配置

```python
def save_config(self, values: dict, hermes_home: str) -> None:
    """将非敏感配置写入原生位置。"""
    import json
    from pathlib import Path
    config_path = Path(hermes_home) / "my-provider.json"
    config_path.write_text(json.dumps(values, indent=2))
```

对于仅使用环境变量的 provider，保留默认的空实现即可。

## 插件入口点

```python
def register(ctx) -> None:
    """由 memory 插件发现系统调用。"""
    ctx.register_memory_provider(MyMemoryProvider())
```

## plugin.yaml

```yaml
name: my-provider
version: 1.0.0
description: "此 provider 功能的简短描述。"
hooks:
  - on_session_end    # 列出你实现的 hook
```

## 线程约定

**`sync_turn()` 必须是非阻塞的。** 如果你的后端存在延迟（API 调用、LLM 处理），请在守护线程中执行：

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

## Profile 隔离

所有存储路径**必须**使用 `initialize()` 中的 `hermes_home` kwarg，而不是硬编码的 `~/.hermes`：

```python
# 正确 — 按 profile 隔离
from hermes_constants import get_hermes_home
data_dir = get_hermes_home() / "my-provider"

# 错误 — 所有 profile 共享
data_dir = Path("~/.hermes/my-provider").expanduser()
```

## 测试

完整的端到端测试模式（使用真实 SQLite provider）请参见 `tests/agent/test_memory_plugin_e2e.py`。

```python
from agent.memory_manager import MemoryManager

mgr = MemoryManager()
mgr.add_provider(my_provider)
mgr.initialize_all(session_id="test-1", platform="cli")

# 测试工具路由
result = mgr.handle_tool_call("my_tool", {"action": "add", "content": "test"})

# 测试生命周期
mgr.sync_all("user msg", "assistant msg")
mgr.on_session_end([])
mgr.shutdown_all()
```

## 添加 CLI 命令

Memory provider 插件可以注册自己的 CLI 子命令树（例如 `hermes my-provider status`、`hermes my-provider config`）。这套系统基于约定发现，无需修改核心文件。

### 工作原理

1. 在插件目录中添加 `cli.py` 文件
2. 定义 `register_cli(subparser)` 函数来构建 argparse 树
3. memory 插件系统在启动时通过 `discover_plugin_cli_commands()` 自动发现
4. 你的命令以 `hermes <provider-name> <subcommand>` 的形式出现

**仅对活跃 provider 开放：** 你的 CLI 命令只在你的 provider 是配置中活跃的 `memory.provider` 时才会出现。如果用户尚未配置你的 provider，你的命令不会显示在 `hermes --help` 中。

### 示例

```python
# plugins/memory/my-provider/cli.py

def my_command(args):
    """由 argparse 分发的处理函数。"""
    sub = getattr(args, "my_command", None)
    if sub == "status":
        print("Provider is active and connected.")
    elif sub == "config":
        print("Showing config...")
    else:
        print("Usage: hermes my-provider <status|config>")

def register_cli(subparser) -> None:
    """构建 hermes my-provider 的 argparse 树。

    在 argparse 初始化时由 discover_plugin_cli_commands() 调用。
    """
    subs = subparser.add_subparsers(dest="my_command")
    subs.add_parser("status", help="Show provider status")
    subs.add_parser("config", help="Show provider config")
    subparser.set_defaults(func=my_command)
```

### 参考实现

完整示例请参见 `plugins/memory/honcho/cli.py`，包含 13 个子命令、跨 profile 管理（`--target-profile`）以及配置读写。

### 含 CLI 的目录结构

```
plugins/memory/my-provider/
├── __init__.py      # MemoryProvider 实现 + register()
├── plugin.yaml      # 元数据
├── cli.py           # register_cli(subparser) — CLI 命令
└── README.md        # 配置说明
```

## 单 Provider 规则

同一时间只能有**一个**外部 memory provider 处于活跃状态。如果用户尝试注册第二个，MemoryManager 会拒绝并发出警告。这可以防止工具 schema 膨胀和后端冲突。