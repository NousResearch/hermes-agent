---
sidebar_position: 9
title: "工具运行时"
description: "工具注册表、toolset、调度及终端环境的运行时行为"
---

# 工具运行时

Hermes 工具是自注册函数，按 toolset（工具集）分组，并通过中央注册表/调度系统执行。

主要文件：

- `tools/registry.py`
- `model_tools.py`
- `toolsets.py`
- `agent/tool_executor.py`
- `tools/approval.py`
- `tools/terminal_tool.py`
- `tools/environments/*`

## 工具注册模型

每个工具模块在导入时调用 `registry.register(...)`。

`model_tools.py` 负责导入/发现工具模块，并构建供模型使用的 schema 列表。

### `registry.register()` 的工作原理

`tools/` 中每个自注册工具模块都会在模块级别调用 `registry.register()` 来声明自身。注册形式如下：

```python
registry.register(
    name="terminal",               # 唯一工具名称（用于 API schema）
    toolset="terminal",            # 该工具所属的 toolset
    schema={...},                  # OpenAI function-calling schema（描述、参数）
    handler=handle_terminal,       # 工具被调用时执行的函数
    check_fn=check_terminal,       # 可选：返回 True/False 表示是否可用
    requires_env=["SOME_VAR"],     # 可选：所需的环境变量（用于 UI 显示）
    is_async=False,                # handler 是否为异步协程
    description="Run commands",    # 人类可读的描述
    emoji="💻",                    # 用于 spinner/进度显示的 emoji
    max_result_size_chars=None,    # 可选的单工具输出预算
    dynamic_schema_overrides=None, # 可选的运行时 schema 字段回调
    override=False,                # 显式替换其他 toolset 中的工具
)
```

每次调用都会创建一个 `ToolEntry`，以工具名称为键存储在单例 `ToolRegistry._tools` 字典中。如果该名称已被注册：

- 在同一 toolset 中重新注册会静默替换已有条目，用于重连和刷新流程。
- 如果新旧 toolset 均以 `mcp-` 开头，新的 MCP 条目会替换旧条目，并以 `DEBUG` 级别记录冲突。这用于 MCP 服务器刷新，以及处理不同 MCP 服务器间的同名工具。
- 对于其他跨 toolset 替换，设置 `override=True` 会显式允许新条目替换旧条目，并以 `INFO` 级别记录覆盖。插件覆盖还要求操作者在 `config.yaml` 中启用 `plugins.entries.<plugin_id>.allow_tool_override`。
- 否则，注册表会记录错误、拒绝新注册并保留已有条目。默认绝不允许意外的跨 toolset 遮蔽。

### 发现机制：`discover_builtin_tools()`

当 `model_tools.py` 被导入时，会调用 `tools/registry.py` 中的 `discover_builtin_tools()`。该函数使用 AST 解析扫描所有 `tools/*.py` 文件，找出包含顶层 `registry.register()` 调用的模块，然后导入它们：

```python
# tools/registry.py（简化版）
def discover_builtin_tools(tools_dir=None):
    tools_path = Path(tools_dir) if tools_dir else Path(__file__).parent
    for path in sorted(tools_path.glob("*.py")):
        if path.name in {"__init__.py", "registry.py", "mcp_tool.py"}:
            continue
        if _module_registers_tools(path):  # AST 检查顶层 registry.register()
            importlib.import_module(f"tools.{path.stem}")
```

这种自动发现机制意味着新工具文件会被自动识别——无需手动维护列表。AST 检查只匹配顶层的 `registry.register()` 调用（不匹配函数内部的调用），因此 `tools/` 中的辅助模块不会被导入。

每次导入都会触发模块的 `registry.register()` 调用。可选工具模块的导入错误会被捕获并记录，不会阻止其他工具加载。

核心工具发现完成后，还会发现 MCP 工具和插件工具：

1. **MCP 工具** — `tools.mcp_tool.discover_mcp_tools()` 读取 MCP 服务器配置，并注册来自外部服务器的工具。
2. **插件工具** — `hermes_cli.plugins.discover_plugins()` 加载用户/项目/pip 插件，这些插件可能注册额外的工具。

## 工具可用性检查（`check_fn`）

每个工具可以选择性地提供一个 `check_fn`——一个可调用对象，在工具可用时返回 `True`，否则返回 `False`。典型的检查包括：

- **API 密钥是否存在** — 例如，`lambda: bool(os.environ.get("SERP_API_KEY"))` 用于网络搜索
- **服务是否运行** — 例如，检查 Honcho 服务器是否已配置
- **二进制文件是否已安装** — 例如，验证浏览器工具的 `playwright` 是否可用

当 `registry.get_definitions()` 为模型构建 schema 列表时，会通过 `_check_fn_cached()` 判断可用性：

```python
# 简化自 registry.py
if entry.check_fn and not _check_fn_cached(entry.check_fn):
    continue  # 本次 schema 构建跳过该工具
```

关键行为：

- 单次调用缓存确保多个工具共享同一个 `check_fn` 时，在一次 `get_definitions()` 处理中只执行一次。
- 结果还会跨调用缓存约 30 秒。相关配置变更后，可调用 `invalidate_check_fn_cache()` 立即清空该缓存。
- 如果某个函数最近一次成功检查发生在约 60 秒内，则其 `False` 结果或异常会被视为瞬时故障：工具继续保持可用，本次失败不写入缓存，下次调用会重新探测。如果近期没有成功记录，则缓存失败结果并隐藏该工具。
- `is_toolset_available()` 会逐一判断已注册工具；只要 toolset 中至少有一个工具可暴露，就返回 `True`。混合 toolset 不会因为其中一个工具不可用而整体隐藏。

## Toolset 解析

Toolset 是工具的命名集合。Hermes 通过以下方式解析它们：

- 显式启用/禁用的 toolset 列表
- 平台预设（`hermes-cli`、`hermes-telegram` 等）
- 动态 MCP toolset
- 精选的特殊用途集合，如 `hermes-acp`

### `get_tool_definitions()` 如何过滤工具

主入口点为 `model_tools.get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode)`：

1. **若提供了 `enabled_toolsets`** — 仅包含这些 toolset 中的工具。每个 toolset 名称通过 `resolve_toolset()` 解析，将复合 toolset 展开为单个工具名称。

2. **若提供了 `disabled_toolsets`** — 无论初始集合来自 `enabled_toolsets` 还是默认集合，最后都会减去这些 toolset。对于平台 bundle 和 posture toolset，Hermes 会保留共享核心工具，只减去该 bundle 的非核心增量。

3. **若省略 `enabled_toolsets`** — 先从所有已知 toolset 开始，再应用禁用项减法。

4. **注册表过滤** — 解析后的工具名称集合传递给 `registry.get_definitions()`，后者应用 `check_fn` 过滤并返回 OpenAI 格式的 schema。

5. **动态 schema 修补** — `registry.get_definitions()` 首先应用各条目可选的 `dynamic_schema_overrides`。随后，`model_tools` 会根据可用沙箱工具重建 `execute_code`，根据检测到的能力和配置重建 `discord` / `discord_admin`，并从 `browser_navigate` 中移除对不可用 Web 工具的引用。

### 旧版 toolset 名称

带有 `_tools` 后缀的旧版 toolset 名称（例如 `web_tools`、`terminal_tools`）通过 `_LEGACY_TOOLSET_MAP` 映射到其现代工具名称，以保持向后兼容性。

## 调度

运行时，注册表工具通过中央注册表调度。需要实时 agent 状态、回调或 provider 自有状态的工具则由 agent 运行时路由。

### 调度流程：模型 tool_call → handler 执行

当模型返回 `tool_call` 时，流程如下：

```
模型响应包含 tool_call
    ↓
agent/tool_executor.py
    ↓
[工具请求中间件]
    ↓
[插件 pre-hook + 工具循环 guardrail]
    ↓
[Agent 自有工具？] → 使用 agent/context/provider 状态执行
[注册表工具？] → model_tools.handle_function_call(...)
    ↓
[工具执行中间件]
    ↓
registry.dispatch(name, args, **kwargs)
    ↓
按名称查找 ToolEntry
[异步 handler？] → 通过 _run_async() 桥接
[同步 handler？]  → 直接调用
    ↓
规范化为字符串或受支持的多模态 envelope
    ↓
[插件 post-hook]
    ↓
[仅注册表工具：可选的 transform_tool_result hook]
```

### 错误包装

注册表工具执行会在两层进行错误处理：

1. **`registry.dispatch()`** — 捕获 handler 异常、清理错误文本，并返回 JSON 错误字符串。它接受普通字符串结果和受支持的多模态 envelope；不支持的结果类型会转换为 `tool_result_contract` JSON 错误。

2. **`handle_function_call()`** — 用第二层 try/except 包装注册表调度周边的编排，并在失败时返回清理后的 JSON 错误字符串。

Agent 自有路径会在工具执行器中自行处理异常。成功的工具结果不要求是 JSON；普通 handler 返回字符串，上述注册表和编排错误包装器才使用 JSON 错误结构。

### Agent 自有工具

部分内置工具会在注册表调度前被拦截，因为它们需要 agent 级状态或运行时回调：

- `todo` — 规划/任务跟踪
- `memory` — 持久化 memory 写入
- `session_search` — 跨会话召回
- `delegate_task` — 生成子 agent 会话
- `clarify` — 使用当前澄清回调
- `read_terminal` — 通过当前终端 UI 回调读取内容

Context engine 工具和 memory provider 工具也由各自的运行时组件路由，而不经过中央注册表。上述内置 Agent 自有工具仍会注册 schema 供 `get_tool_definitions` 使用；正常执行时所需的状态和回调由 agent 运行时提供。

### 异步桥接

当工具 handler 为异步时，`_run_async()` 将其桥接到同步调度路径：

- **CLI 路径（无运行中的事件循环）** — 使用持久化事件循环以保持缓存的异步客户端存活
- **Gateway 路径（有运行中的事件循环）** — 使用 `asyncio.run()` 启动一个一次性线程
- **工作线程（并行工具）** — 使用存储在线程本地存储中的每线程持久化循环

## DANGEROUS_PATTERNS 审批流程

终端工具集成了定义在 `tools/approval.py` 中的危险命令审批系统：

1. **模式检测** — `DANGEROUS_PATTERNS` 是一个 `(regex, description)` 元组列表，涵盖破坏性操作：
   - 递归删除（`rm -rf`）
   - 文件系统格式化（`mkfs`、`dd`）
   - SQL 破坏性操作（`DROP TABLE`、不带 `WHERE` 的 `DELETE FROM`）
   - 系统配置覆写（`> /etc/`）
   - 服务操控（`systemctl stop`）
   - 远程代码执行（`curl | sh`）
   - Fork bomb、进程终止等

2. **检测** — 在受 guard 保护的终端环境中执行命令前，`check_all_command_guards()` 会综合 hardline 阻断、用户 deny 规则、Tirith 发现和 `detect_dangerous_command(command)`。隔离容器后端会跳过这些 guard；Docker 挂载宿主机路径后则不再跳过。

3. **审批提示** — 若发现匹配：
   - **CLI 模式** — 交互式提示要求用户批准、拒绝或永久允许
   - **Gateway 模式** — 异步审批回调将请求发送至消息平台
   - **智能审批** — 可选地，辅助 LLM 可自动批准匹配模式但风险较低的命令（例如，`rm -rf node_modules/` 是安全的，但匹配"递归删除"模式）

4. **会话状态** — 审批按会话跟踪。在当前会话中批准“递归删除”后，后续匹配该模式的命令不会再次提示。灾难性 hardline 命令和用户 deny 规则仍会在审批绕过前被阻断。

5. **永久允许列表** — "永久允许"选项会将该模式写入 `config.yaml` 的 `command_allowlist`，跨会话持久化。

## 终端/运行时环境

终端系统支持多种后端：

- local
- docker
- ssh
- singularity
- modal
- daytona

还支持：

- 按任务的 cwd 覆盖
- 后台进程管理
- PTY 模式
- 危险命令的审批回调

## 并发

工具调用可以顺序执行，也可以并发执行，具体取决于工具组合和交互需求。

## 相关文档

- [Toolsets 参考](../reference/toolsets-reference.md)
- [内置工具参考](../reference/tools-reference.md)
- [Agent 循环内部机制](./agent-loop.md)
- [ACP 内部机制](./acp-internals.md)
