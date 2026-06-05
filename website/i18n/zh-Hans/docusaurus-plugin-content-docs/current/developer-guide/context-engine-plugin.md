---
sidebar_position: 9
title: "Context Engine 插件"
description: "如何构建替换内置 ContextCompressor 的 context engine 插件"
---

# 构建 Context Engine 插件

Context engine 插件用于替换内置的 `ContextCompressor`，以实现管理对话上下文的替代策略。例如，无损上下文管理（LCM）引擎通过构建知识 DAG 来替代有损摘要。

## 工作原理

Agent 的上下文管理基于 `ContextEngine` ABC（`agent/context_engine.py`）构建。内置的 `ContextCompressor` 是默认实现。插件引擎必须实现相同的接口。

同一时间只能有**一个** context engine 处于激活状态。选择由配置驱动：

```yaml
# config.yaml
context:
  engine: "compressor"    # 默认内置
  engine: "lcm"           # 激活名为 "lcm" 的插件引擎
```

插件引擎**永远不会自动激活** — 用户必须显式将 `context.engine` 设置为插件名称。

## 目录结构

每个 context engine 位于 `plugins/context_engine/<name>/`：

```
plugins/context_engine/lcm/
├── __init__.py      # 导出 ContextEngine 子类
├── plugin.yaml      # 元数据（name、description、version）
└── ...              # 引擎所需的其他模块
```

## ContextEngine ABC

你的引擎必须实现以下**必需**方法：

```python
from agent.context_engine import ContextEngine

class LCMEngine(ContextEngine):

    @property
    def name(self) -> str:
        """短标识符，例如 'lcm'。必须与 config.yaml 中的值匹配。"""
        return "lcm"

    def update_from_response(self, usage: dict) -> None:
        """每次 LLM 调用后，以 usage dict 为参数调用。

        从响应中更新 self.last_prompt_tokens、self.last_completion_tokens、
        self.last_total_tokens。
        """

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """若本轮应触发压缩则返回 True。"""

    def compress(self, messages: list, current_tokens: int = None,
                 focus_topic: str = None) -> list:
        """压缩消息列表并返回新的（可能更短的）列表。

        返回的列表必须是有效的 OpenAI 格式消息序列。

        ``focus_topic`` 是来自手动 ``/compress <focus>`` 的可选主题字符串；
        支持引导式压缩的引擎应优先保留与其相关的信息，其他引擎可忽略。
        """
```

### 引擎必须维护的类属性

Agent 直接读取这些属性用于显示和日志记录：

```python
last_prompt_tokens: int = 0
last_completion_tokens: int = 0
last_total_tokens: int = 0
threshold_tokens: int = 0        # 触发压缩的阈值
context_length: int = 0          # 模型的完整上下文窗口
compression_count: int = 0       # compress() 已运行的次数
```

### 可选方法

这些方法在 ABC 中有合理的默认实现，按需覆盖：

| 方法 | 默认行为 | 何时覆盖 |
|--------|---------|--------------|
| `on_session_start(session_id, **kwargs)` | 空操作 | 需要加载持久化状态（DAG、DB）时 |
| `on_session_end(session_id, messages)` | 空操作 | 需要刷新状态、关闭连接时 |
| `on_session_reset()` | 重置 token 计数器 | 有需要清除的会话级状态时 |
| `update_model(model, context_length, ...)` | 更新 context_length 和阈值 | 需要在切换模型时重新计算预算时 |
| `get_tool_schemas()` | 返回 `[]` | 引擎提供 agent 可调用的工具时（例如 `lcm_grep`） |
| `handle_tool_call(name, args, **kwargs)` | 返回错误 JSON | 实现工具处理器时 |
| `should_compress_preflight(messages)` | 返回 `False` | 可在 API 调用前进行低成本预估时 |
| `get_status()` | 标准 token/阈值字典 | 有自定义指标需要暴露时 |

## 引擎工具

Context engine 可以暴露 agent 直接调用的工具。从 `get_tool_schemas()` 返回 schema，并在 `handle_tool_call()` 中处理调用：

```python
def get_tool_schemas(self):
    return [{
        "name": "lcm_grep",
        "description": "Search the context knowledge graph",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    }]

def handle_tool_call(self, name, args, **kwargs):
    if name == "lcm_grep":
        results = self._search_dag(args["query"])
        return json.dumps({"results": results})
    return json.dumps({"error": f"Unknown tool: {name}"})
```

引擎工具在启动时注入到 agent 的工具列表中并自动分发 — 无需注册到注册表。

## 注册

### 通过目录（推荐）

将引擎放置于 `plugins/context_engine/<name>/`。`__init__.py` 必须导出一个 `ContextEngine` 子类。发现系统会自动找到并实例化它。

### 通过通用插件系统

通用插件也可以注册 context engine：

```python
def register(ctx):
    engine = LCMEngine(context_length=200000)
    ctx.register_context_engine(engine)
```

只能注册一个引擎。第二个尝试注册的插件将被拒绝并发出警告。

## 生命周期

```
1. 引擎实例化（插件加载或目录发现）
2. on_session_start() — 对话开始
3. update_from_response() — 每次 API 调用后
4. should_compress() — 每轮检查
5. compress() — 当 should_compress() 返回 True 时调用
6. on_session_end() — 会话边界（CLI 退出、/reset、gateway 过期）
```

`on_session_reset()` 在 `/new` 或 `/reset` 时调用，用于清除会话级状态而不完全关闭。

## 配置

用户通过 `hermes plugins` → Provider Plugins → Context Engine 选择引擎，或直接编辑 `config.yaml`：

```yaml
context:
  engine: "lcm"   # 必须与引擎的 name 属性匹配
```

`compression` 配置块（`compression.threshold`、`compression.protect_last_n` 等）专属于内置的 `ContextCompressor`。如有需要，你的引擎应定义自己的配置格式，并在初始化期间从 `config.yaml` 读取。

## 测试

```python
from agent.context_engine import ContextEngine

def test_engine_satisfies_abc():
    engine = YourEngine(context_length=200000)
    assert isinstance(engine, ContextEngine)
    assert engine.name == "your-name"

def test_compress_returns_valid_messages():
    engine = YourEngine(context_length=200000)
    msgs = [{"role": "user", "content": "hello"}]
    result = engine.compress(msgs)
    assert isinstance(result, list)
    assert all("role" in m for m in result)
```

完整的 ABC 契约测试套件请参见 `tests/agent/test_context_engine.py`。

## 另请参阅

- [上下文压缩与缓存](/developer-guide/context-compression-and-caching) — 内置压缩器的工作原理
- [Memory Provider 插件](/developer-guide/memory-provider-plugin) — 类似的单选插件系统（用于内存）
- [插件](/user-guide/features/plugins) — 通用插件系统概述