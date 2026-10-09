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
- `tools/terminal_tool.py`
- `tools/environments/*`

## 工具注册模型

每个工具模块在导入时调用 `registry.register(...)`。

`model_tools.py` 负责导入/发现工具模块，并构建供模型使用的 schema 列表。

### `registry.register()` 的工作原理

`tools/` 中的每个工具文件在模块级别调用 `registry.register()` 来声明自身。函数签名如下：

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
)
```

每次调用都会创建一个 `ToolEntry`，以工具名称为键存储在单例 `ToolRegistry._tools` 字典中。若不同 toolset 之间出现名称冲突，会记录警告，后注册的条目覆盖前者。

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

每次导入都会触发模块的 `registry.register()` 调用。可选工具中的错误（例如图像生成工具缺少 `fal_client`）会被捕获并记录——不会阻止其他工具加载。

核心工具发现完成后，还会发现 MCP 工具和插件工具：

1. **MCP 工具** — `tools.mcp_tool.discover_mcp_tools()` 读取 MCP 服务器配置，并注册来自外部服务器的工具。
2. **插件工具** — `hermes_cli.plugins.discover_plugins()` 加载用户/项目/pip 插件，这些插件可能注册额外的工具。

## 工具可用性检查（`check_fn`）

每个工具可以选择性地提供一个 `check_fn`——一个可调用对象，在工具可用时返回 `True`，否则返回 `False`。典型的检查包括：

- **API 密钥是否存在** — 例如，`lambda: bool(os.environ.get("SERP_API_KEY"))` 用于网络搜索
- **服务是否运行** — 例如，检查 Honcho 服务器是否已配置
- **二进制文件是否已安装** — 例如，验证浏览器工具的 `playwright` 是否可用

当 `registry.get_definitions()` 为模型构建 schema 列表时，会运行每个工具的 `check_fn()`：

```python
# 简化自 registry.py
if entry.check_fn:
    try:
        available = bool(entry.check_fn())
    except Exception:
        available = False   # 异常 = 不可用
    if not available:
        continue            # 完全跳过该工具
```

关键行为：
- 检查结果**按调用缓存**——若多个工具共享同一个 `check_fn`，只运行一次。
- `check_fn()` 中的异常被视为"不可用"（故障安全）。
- `is_toolset_available()` 方法检查某个 toolset 的 `check_fn` 是否通过，用于 UI 显示和 toolset 解析。

## Toolset 解析

Toolset 是工具的命名集合。Hermes 通过以下方式解析它们：

- 显式启用/禁用的 toolset 列表
- 平台预设（`hermes-cli`、`hermes-telegram` 等）
- 动态 MCP toolset
- 精选的特殊用途集合，如 `hermes-acp`

### `get_tool_definitions()` 如何过滤工具

主入口点为 `model_tools.get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode)`：

1. **若提供了 `enabled_toolsets`** — 仅包含这些 toolset 中的工具。每个 toolset 名称通过 `resolve_toolset()` 解析，将复合 toolset 展开为单个工具名称。

2. **若提供了 `disabled_toolsets`** — 从所有 toolset 开始，减去已禁用的。

3. **若两者均未提供** — 包含所有已知 toolset。

4. **注册表过滤** — 解析后的工具名称集合传递给 `registry.get_definitions()`，后者应用 `check_fn` 过滤并返回 OpenAI 格式的 schema。

5. **动态 schema 修补** — 过滤后，`execute_code` 和 `browser_navigate` 的 schema 会被动态调整，仅引用实际通过过滤的工具（防止模型幻觉出不可用的工具）。

### 旧版 toolset 名称

带有 `_tools` 后缀的旧版 toolset 名称（例如 `web_tools`、`terminal_tools`）通过 `_LEGACY_TOOLSET_MAP` 映射到其现代工具名称，以保持向后兼容性。

## 调度

运行时，工具通过中央注册表调度，但部分 agent 级别的工具（如 memory/todo/session-search 处理）由 agent 循环直接处理。

### 调度流程：模型 tool_call → handler 执行

当模型返回 `tool_call` 时，流程如下：

```
模型响应包含 tool_call
    ↓
run_agent.py agent 循环
    ↓
model_tools.handle_function_call(name, args, task_id, user_task)
    ↓
[Agent 循环工具？] → 由 agent 循环直接处理（todo、memory、session_search、delegate_task）
    ↓
[插件 pre-hook] → invoke_hook("pre_tool_call", ...)
    ↓
registry.dispatch(name, args, **kwargs)
    ↓
按名称查找 ToolEntry
    ↓
[异步 handler？] → 通过 _run_async() 桥接
[同步 handler？]  → 直接调用
    ↓
返回结果字符串（或 JSON 错误）
    ↓
[插件 post-hook] → invoke_hook("post_tool_call", ...)
```

### 错误包装

所有工具执行在两个层级进行错误处理：

1. **`registry.dispatch()`** — 捕获 handler 抛出的任何异常，并以 JSON 形式返回 `{"error": "Tool execution failed: ExceptionType: message"}`。

2. **`handle_function_call()`** — 将整个调度包裹在次级 try/except 中，返回 `{"error": "Error executing tool_name: message"}`。

这确保模型始终收到格式正确的 JSON 字符串，而不会遇到未处理的异常。

### Agent 循环工具

以下四个工具在注册表调度之前被拦截，因为它们需要 agent 级别的状态（TodoStore、MemoryStore 等）：

- `todo` — 规划/任务跟踪
- `memory` — 持久化 memory 写入
- `session_search` — 跨会话召回
- `delegate_task` — 生成子 agent 会话

这些工具的 schema 仍在注册表中注册（供 `get_tool_definitions` 使用），但若调度以某种方式直接到达它们，其 handler 会返回一个存根错误。

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

2. **检测** — 在执行任何终端命令之前，`detect_dangerous_command(command)` 会对所有模式进行检查。

3. **审批提示** — 若发现匹配：
   - **CLI 模式** — 交互式提示要求用户批准、拒绝或永久允许
   - **Gateway 模式** — 异步审批回调将请求发送至消息平台
   - **智能审批** — 可选地，辅助 LLM 可自动批准匹配模式但风险较低的命令（例如，`rm -rf node_modules/` 是安全的，但匹配"递归删除"模式）

4. **会话状态** — 审批按会话跟踪。一旦在某个会话中批准了"递归删除"，后续的 `rm -rf` 命令不会再次提示。

5. **永久允许列表** — "永久允许"选项会将该模式写入 `config.yaml` 的 `command_allowlist`，跨会话持久化。

## 终端/运行时环境

终端系统支持多种后端：

- local
- docker
- ssh
- singularity
- modal
- daytona
- vercel_sandbox

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