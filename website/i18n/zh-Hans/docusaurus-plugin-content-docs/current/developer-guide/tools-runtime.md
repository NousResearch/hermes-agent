---
sidebar_position: 9
title: "工具运行时"
description: "工具注册表、工具集、调度和终端环境的运行时行为"
---

# 工具运行时

Hermes 工具是自注册函数，按工具集分组，通过中央注册表/调度系统执行。

主要文件：

- `tools/registry.py`
- `model_tools.py`
- `toolsets.py`
- `tools/terminal_tool.py`
- `tools/environments/*`

## 工具注册模型

每个工具模块在导入时调用 `registry.register(...)`。

`model_tools.py` 负责导入/发现工具模块，并构建模型使用的 schema 列表。

### `registry.register()` 的工作方式

`tools/` 中的每个工具文件在模块级别调用 `registry.register()` 来声明自己。函数签名如下：

```python
registry.register(
    name="terminal",               # 唯一工具名（用于 API schema）
    toolset="terminal",            # 所属工具集
    schema={...},                  # OpenAI function-calling schema（description、parameters）
    handler=handle_terminal,       # 工具被调用时执行的函数
    check_fn=check_terminal,       # 可选：返回 True/False 表示可用性
    requires_env=["SOME_VAR"],     # 可选：所需环境变量（用于 UI 显示）
    is_async=False,                # 处理函数是否为异步协程
    description="Run commands",    # 人类可读的描述
    emoji="💻",                    # 用于 spinner/进度显示的 emoji
)
```

每次调用会创建一个 `ToolEntry`，存储在以工具名为键的单例 `ToolRegistry._tools` 字典中。如果跨工具集出现名称冲突，会记录警告，后注册的覆盖先注册的。

### 发现：`discover_builtin_tools()`

当 `model_tools.py` 被导入时，它会调用 `tools/registry.py` 中的 `discover_builtin_tools()`。该函数使用 AST 解析扫描每个 `tools/*.py` 文件，找到包含顶层 `registry.register()` 调用的模块，然后导入它们：

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

这种自动发现意味着新工具文件会被自动拾取——无需维护手动列表。AST 检查只匹配顶层的 `registry.register()` 调用（不包括函数内部的调用），因此 `tools/` 中的辅助模块不会被导入。

每次导入会触发模块的 `registry.register()` 调用。可选工具中的错误（例如缺少图片生成的 `fal_client`）会被捕获并记录——不会阻止其他工具加载。

核心工具发现完成后，MCP 工具和插件工具也会被发现：

1. **MCP 工具** — `tools.mcp_tool.discover_mcp_tools()` 读取 MCP 服务器配置并注册来自外部服务器的工具。
2. **插件工具** — `hermes_cli.plugins.discover_plugins()` 加载用户/项目/pip 插件，它们可能会注册额外的工具。

## 工具可用性检查（`check_fn`）

每个工具可以可选地提供一个 `check_fn`——一个在工具可用时返回 `True`、否则返回 `False` 的可调用对象。典型检查包括：

- **API key 存在** — 例如 `lambda: bool(os.environ.get("SERP_API_KEY"))` 用于网络搜索
- **服务运行中** — 例如检查 Honcho 服务器是否已配置
- **二进制文件已安装** — 例如验证 `playwright` 是否可用于浏览器工具

当 `registry.get_definitions()` 为模型构建 schema 列表时，它会运行每个工具的 `check_fn()`：

```python
# 来自 registry.py 的简化版
if entry.check_fn:
    try:
        available = bool(entry.check_fn())
    except Exception:
        available = False   # 异常 = 不可用
    if not available:
        continue            # 完全跳过此工具
```

关键行为：
- 检查结果**按调用缓存**——如果多个工具共享同一个 `check_fn`，它只运行一次。
- `check_fn()` 中的异常被视为"不可用"（故障安全）。
- `is_toolset_available()` 方法检查工具集的 `check_fn` 是否通过，用于 UI 显示和工具集解析。

## 工具集解析

工具集是命名的工具包。Hermes 通过以下方式解析它们：

- 显式的启用/禁用工具集列表
- 平台预设（`hermes-cli`、`hermes-telegram` 等）
- 动态 MCP 工具集
- 精选的特殊用途集合，如 `hermes-acp`

### `get_tool_definitions()` 如何过滤工具

主入口点是 `model_tools.get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode)`：

1. **如果提供了 `enabled_toolsets`** — 只包含这些工具集中的工具。每个工具集名称通过 `resolve_toolset()` 解析，将复合工具集展开为单独的工具名。

2. **如果提供了 `disabled_toolsets`** — 从所有工具集开始，然后减去禁用的工具集。

3. **如果两者都未提供** — 包含所有已知工具集。

4. **注册表过滤** — 解析后的工具名集合传递给 `registry.get_definitions()`，它应用 `check_fn` 过滤并返回 OpenAI 格式的 schema。

5. **动态 schema 修补** — 过滤后，`execute_code` 和 `browser_navigate` schema 会被动态调整，只引用实际通过过滤的工具（防止模型幻觉出不可用的工具）。

### 旧版工具集名称

带有 `_tools` 后缀的旧工具集名称（例如 `web_tools`、`terminal_tools`）通过 `_LEGACY_TOOLSET_MAP` 映射到其现代工具名，以保持向后兼容。

## 调度

在运行时，工具通过中央注册表调度，某些 agent 级别的工具（如 memory/todo/session-search 处理）在 agent 循环中被拦截。

### 调度流程：模型 tool_call → 处理函数执行

当模型返回 `tool_call` 时，流程如下：

```
Model response with tool_call
    ↓
run_agent.py agent loop
    ↓
model_tools.handle_function_call(name, args, task_id, user_task)
    ↓
[Agent-loop tools?] → 由 agent loop 直接处理（todo、memory、session_search、delegate_task）
    ↓
[Plugin pre-hook] → invoke_hook("pre_tool_call", ...)
    ↓
registry.dispatch(name, args, **kwargs)
    ↓
Look up ToolEntry by name
    ↓
[Async handler?] → 通过 _run_async() 桥接
[Sync handler?]  → 直接调用
    ↓
Return result string (or JSON error)
    ↓
[Plugin post-hook] → invoke_hook("post_tool_call", ...)
```

### 错误包装

所有工具执行都被两层错误处理包装：

1. **`registry.dispatch()`** — 捕获处理函数中的任何异常，返回 `{"error": "Tool execution failed: ExceptionType: message"}` JSON。

2. **`handle_function_call()`** — 将整个调度包装在第二层 try/except 中，返回 `{"error": "Error executing tool_name: message"}`。

这确保模型始终收到格式良好的 JSON 字符串，而不会收到未处理的异常。

### Agent 循环工具

四个工具在注册表调度之前被拦截，因为它们需要 agent 级别的状态（TodoStore、MemoryStore 等）：

- `todo` — 规划/任务跟踪
- `memory` — 持久化记忆写入
- `session_search` — 跨会话召回
- `delegate_task` — 生成子 agent 会话

这些工具的 schema 仍然注册在注册表中（用于 `get_tool_definitions`），但如果调度直接到达它们的处理函数，会返回一个存根错误。

### 异步桥接

当工具处理函数是异步的，`_run_async()` 将其桥接到同步调度路径：

- **CLI 路径（无运行中的循环）** — 使用持久事件循环来保持缓存的异步客户端存活
- **Gateway 路径（有运行中的循环）** — 使用 `asyncio.run()` 启动一次性线程
- **工作线程（并行工具）** — 使用存储在线程本地存储中的每线程持久循环

## DANGEROUS_PATTERNS 审批流程

终端工具集成了定义在 `tools/approval.py` 中的危险命令审批系统：

1. **模式检测** — `DANGEROUS_PATTERNS` 是一个 `(regex, description)` 元组列表，涵盖破坏性操作：
   - 递归删除（`rm -rf`）
   - 文件系统格式化（`mkfs`、`dd`）
   - SQL 破坏性操作（`DROP TABLE`、不带 `WHERE` 的 `DELETE FROM`）
   - 系统配置覆盖（`> /etc/`）
   - 服务操作（`systemctl stop`）
   - 远程代码执行（`curl | sh`）
   - Fork 炸弹、进程终止等

2. **检测** — 在执行任何终端命令之前，`detect_dangerous_command(command)` 会检查所有模式。

3. **审批提示** — 如果找到匹配：
   - **CLI 模式** — 交互式提示要求用户批准、拒绝或永久允许
   - **Gateway 模式** — 异步审批回调将请求发送到消息平台
   - **智能审批** — 可选地，辅助 LLM 可以自动批准匹配模式但风险较低的命令（例如 `rm -rf node_modules/` 是安全的但匹配"递归删除"模式）

4. **会话状态** — 审批按会话跟踪。一旦你在某个会话中批准了"递归删除"，后续的 `rm -rf` 命令不会再次提示。

5. **永久允许列表** — "永久允许"选项将模式写入 `config.yaml` 的 `command_allowlist`，跨会话持久化。

## 终端/运行时环境

终端系统支持多个后端：

- local
- docker
- ssh
- singularity
- modal
- daytona
- vercel_sandbox

它还支持：

- 按任务覆盖 cwd
- 后台进程管理
- PTY 模式
- 危险命令的审批回调

## 并发

工具调用可能顺序或并发执行，具体取决于工具组合和交互要求。

## 相关文档

- [工具集参考](../reference/toolsets-reference.md)
- [内置工具参考](../reference/tools-reference.md)
- [Agent 循环内部机制](./agent-loop.md)
- [ACP 内部机制](./acp-internals.md)
