---
sidebar_position: 7
title: "子智能体委派"
description: "使用 delegate_task 为并行工作流生成隔离的子智能体"
---

# 子智能体委派

`delegate_task` 工具会生成具有隔离上下文、受限工具集和独立终端会话的子 AIAgent 实例。每个子智能体获得全新的对话并独立运行——只有其最终摘要会进入父智能体的上下文。

## 单任务

```python
delegate_task(
    goal="Debug why tests fail",
    context="Error: assertion in test_foo.py line 42",
    toolsets=["terminal", "file"],
)
```

## 并行批处理

默认最多 3 个并发子智能体（可配置，无硬性上限）：

```python
delegate_task(tasks=[
    {"goal": "Research topic A", "toolsets": ["web"]},
    {"goal": "Research topic B", "toolsets": ["web"]},
    {"goal": "Fix the build", "toolsets": ["terminal", "file"]}
])
```

## 运行时模型和提供商选择

子智能体通常继承父智能体的模型/提供商，或使用 `delegation.model` 和
`delegation.provider` 中配置的默认值。对于一次性任务，智能体可以在
`delegate_task` 调用中直接指定运行时：

```python
delegate_task(
    goal="Summarize these logs and identify the likely root cause",
    context="Logs are in /tmp/service.log. Return only findings and next steps.",
    toolsets=["terminal", "file"],
    model="openai/gpt-5-nano",
    provider="openrouter",
)
```

批处理模式下，可在顶层设置 `model` 和 `provider` 应用于所有子智能体，
也可在每个 task 对象中分别设置，让不同子任务使用不同模型：

```python
delegate_task(tasks=[
    {
        "goal": "Cheap mechanical scan of the changed files",
        "context": "Use git diff and report obvious mistakes only.",
        "toolsets": ["terminal", "file"],
        "model": "openai/gpt-5-nano",
        "provider": "openrouter",
    },
    {
        "goal": "Careful design review of the public API change",
        "context": "Focus on backwards compatibility and hidden edge cases.",
        "toolsets": ["file"],
        "model": "anthropic/claude-sonnet-4",
        "provider": "openrouter",
    },
])
```

当目标模型不在父智能体同一 provider 上时，请传入 `provider`。省略
`model`/`provider`，或传入 `"self"`、`"inherit"`、`"parent"`，表示继承父智能体或默认委派运行时。

## 子智能体上下文的工作方式

:::warning 关键：子智能体一无所知
子智能体以**全新对话**启动。它们对父智能体的对话历史、之前的工具调用或委派前讨论的任何内容一无所知。子智能体的唯一上下文来自父智能体调用 `delegate_task` 时填写的 `goal` 和 `context` 字段。
:::

这意味着父智能体必须在调用中传递子智能体所需的**一切**信息：

```python
# BAD - subagent has no idea what "the error" is
delegate_task(goal="Fix the error")

# GOOD - subagent has all context it needs
delegate_task(
    goal="Fix the TypeError in api/handlers.py",
    context="""The file api/handlers.py has a TypeError on line 47:
    'NoneType' object has no attribute 'get'.
    The function process_request() receives a dict from parse_body(),
    but parse_body() returns None when Content-Type is missing.
    The project is at /home/user/myproject and uses Python 3.11."""
)
```

子智能体会收到一个基于你的 goal 和 context 构建的专注系统 prompt（提示词），指示其完成任务并提供结构化摘要，包括所做的事情、发现的内容、修改的文件以及遇到的问题。

## 实际示例

### 并行研究

同时研究多个主题并收集摘要：

```python
delegate_task(tasks=[
    {
        "goal": "Research the current state of WebAssembly in 2025",
        "context": "Focus on: browser support, non-browser runtimes, language support",
        "toolsets": ["web"]
    },
    {
        "goal": "Research the current state of RISC-V adoption in 2025",
        "context": "Focus on: server chips, embedded systems, software ecosystem",
        "toolsets": ["web"]
    },
    {
        "goal": "Research quantum computing progress in 2025",
        "context": "Focus on: error correction breakthroughs, practical applications, key players",
        "toolsets": ["web"]
    }
])
```

### 代码审查 + 修复

将审查并修复的工作流委派给全新上下文：

```python
delegate_task(
    goal="Review the authentication module for security issues and fix any found",
    context="""Project at /home/user/webapp.
    Auth module files: src/auth/login.py, src/auth/jwt.py, src/auth/middleware.py.
    The project uses Flask, PyJWT, and bcrypt.
    Focus on: SQL injection, JWT validation, password handling, session management.
    Fix any issues found and run the test suite (pytest tests/auth/).""",
    toolsets=["terminal", "file"]
)
```

### 多文件重构

将会大量占用父智能体上下文的大型重构任务委派出去：

```python
delegate_task(
    goal="Refactor all Python files in src/ to replace print() with proper logging",
    context="""Project at /home/user/myproject.
    Use the 'logging' module with logger = logging.getLogger(__name__).
    Replace print() calls with appropriate log levels:
    - print(f"Error: ...") -> logger.error(...)
    - print(f"Warning: ...") -> logger.warning(...)
    - print(f"Debug: ...") -> logger.debug(...)
    - Other prints -> logger.info(...)
    Don't change print() in test files or CLI output.
    Run pytest after to verify nothing broke.""",
    toolsets=["terminal", "file"]
)
```

## 批处理模式详情

当你提供 `tasks` 数组时，子智能体会使用线程池**并行**运行：

- **最大并发数：** 默认 3 个任务（可通过 `delegation.max_concurrent_children` 或环境变量 `DELEGATION_MAX_CONCURRENT_CHILDREN` 配置；最低为 1，无硬性上限）。超出限制的批次会返回工具错误，而不是被静默截断。
- **线程池：** 使用 `ThreadPoolExecutor`，以配置的并发限制作为最大工作线程数
- **进度显示：** 在 CLI 模式下，树形视图会实时显示每个子智能体的工具调用，并附带每个任务的完成行。在 gateway 模式下，进度会被批量汇总并转发给父智能体的进度回调
- **结果排序：** 结果按任务索引排序，与输入顺序一致，不受完成顺序影响
- **中断传播：** 中断父智能体（例如发送新消息）会中断所有活跃的子智能体

单任务委派直接运行，无线程池开销。

## 默认委派模型

你也可以通过 `config.yaml` 为所有子智能体配置默认模型——适用于默认将常规子任务路由到更便宜/更快的模型，除非某次 `delegate_task` 调用显式覆盖：

```yaml
# In ~/.hermes/config.yaml
delegation:
  model: "google/gemini-flash-2.0"    # Cheaper model for subagents
  provider: "openrouter"              # Optional: route subagents to a different provider
```

如果省略，子智能体将使用与父智能体相同的模型。某次调用中的 `model` 或
`provider` 会优先于这些配置默认值。

## 工具集选择建议

`toolsets` 参数控制子智能体可以访问的工具。根据任务选择：

| 工具集模式 | 使用场景 |
|----------------|----------|
| `["terminal", "file"]` | 代码工作、调试、文件编辑、构建 |
| `["web"]` | 研究、事实核查、文档查阅 |
| `["terminal", "file", "web"]` | 全栈任务（默认） |
| `["file"]` | 只读分析、无需执行的代码审查 |
| `["terminal"]` | 系统管理、进程管理 |

无论你指定什么，某些工具集对子智能体始终被屏蔽：
- `delegation` — 对叶子子智能体屏蔽（默认）。`role="orchestrator"` 的子智能体可保留，受 `max_spawn_depth` 约束——参见下方[深度限制与嵌套编排](#depth-limit-and-nested-orchestration)。
- `clarify` — 子智能体无法与用户交互
- `memory` — 不可写入共享持久内存
- `code_execution` — 子智能体应逐步推理
- `send_message` — 无跨平台副作用（例如发送 Telegram 消息）

## 最大迭代次数

每个子智能体都有迭代次数限制（默认：50），控制其可进行的工具调用轮次：

```python
delegate_task(
    goal="Quick file check",
    context="Check if /etc/nginx/nginx.conf exists and print its first 10 lines",
    max_iterations=10  # Simple task, don't need many turns
)
```

## 子智能体超时

默认情况下，子智能体**没有挂钟超时限制**。子智能体只会因其实际执行的操作而失败——API 错误、工具错误或达到迭代预算上限——而不会被委派层面的计时器终止。早期版本曾设有硬性上限（300 秒，后为 600 秒），但这会在任务执行过程中误杀正常工作的子智能体：深度代码审查、大规模研究分发以及慢速推理模型经常需要超过 10 分钟，而它们全程都在稳定推进。

真正卡死的子智能体仍会被检测到：当子智能体没有任何进展（无 API 调用、无工具启动）时，心跳陈旧度监控会停止刷新父智能体的活动状态，从而让网关的不活动超时机制对真正卡死的工作进程生效。

如果仍需要硬性上限（例如对无人值守的 cron 驱动委派进行成本控制），可按安装实例选择启用：

```yaml
delegation:
  child_timeout_seconds: 0     # 默认：0 = 无超时
  # child_timeout_seconds: 1800  # 选择启用的硬性上限（下限 30 秒）
```

正值会对每个子智能体强制执行挂钟时间硬限制；`0` 或负值表示禁用。

:::tip 零调用超时时的诊断转储
在配置了硬性上限的情况下，如果子智能体在**零次** API 调用的情况下超时（通常原因：provider 不可达、认证失败或工具 schema 被拒绝），`delegate_task` 会将结构化诊断信息写入 `~/.hermes/logs/subagent-timeout-<session>-<timestamp>.log`，其中包含子智能体的配置快照、凭据解析追踪以及早期错误消息。比之前的静默超时行为更易于定位根因。
:::

## 监控运行中的子智能体（`/agents`）

TUI 提供 `/agents` 浮层（别名 `/tasks`），将递归 `delegate_task` 扇出转化为一级审计界面：

- 运行中和最近完成的子智能体的实时树形视图，按父智能体分组
- 每个分支的费用、token 和已触及文件的汇总
- 终止和暂停控制——可在不中断其兄弟智能体的情况下取消特定子智能体
- 事后回顾：即使子智能体已返回父智能体，也可逐轮查看其历史记录

经典 CLI 仅将 `/agents` 打印为文本摘要；TUI 才是浮层真正发挥作用的地方。参见 [TUI — 斜杠命令](/user-guide/tui#slash-commands)。

## 深度限制与嵌套编排 {#depth-limit-and-nested-orchestration}

默认情况下，委派是**扁平的**：父智能体（深度 0）生成子智能体（深度 1），而这些子智能体无法进一步委派。这可防止失控的递归委派。

对于多阶段工作流（研究 → 综合，或对子问题进行并行编排），父智能体可以生成**编排者**子智能体，这些子智能体*可以*委派自己的工作线程：

```python
delegate_task(
    goal="Survey three code review approaches and recommend one",
    role="orchestrator",  # Allows this child to spawn its own workers
    context="...",
)
```

- `role="leaf"`（默认）：子智能体无法进一步委派——与扁平委派行为相同。
- `role="orchestrator"`：子智能体保留 `delegation` 工具集。受 `delegation.max_spawn_depth` 约束（默认 **1** = 扁平，因此在默认设置下 `role="orchestrator"` 无效）。将 `max_spawn_depth` 提高到 2 可允许编排者子智能体生成叶子孙智能体；设为 3 则允许三层（上限）。
- `delegation.orchestrator_enabled: false`：全局开关，无论 `role` 参数如何，强制所有子智能体为 `leaf`。

**费用警告：** 在 `max_spawn_depth: 3` 和 `max_concurrent_children: 3` 的情况下，树可达到 3×3×3 = 27 个并发叶子智能体。每增加一层都会成倍增加开销——请谨慎提高 `max_spawn_depth`。

## 生命周期与持久性

:::warning delegate_task 是同步的——不具备持久性
`delegate_task` 在**父智能体的当前轮次内**运行。它会阻塞父智能体，直到所有子智能体完成（或被取消）。它**不是**后台任务队列：

- 如果父智能体被中断（用户发送新消息、`/stop`、`/new`），所有活跃的子智能体都会被取消并返回 `status="interrupted"`。其进行中的工作将被丢弃。
- 子智能体在父智能体轮次结束后**不会**继续运行。
- 被取消的子智能体会返回结构化结果（`status="interrupted"`，`exit_reason="interrupted"`），但由于父智能体也被中断，该结果通常不会出现在用户可见的回复中。

对于必须在中断后存活或超出当前轮次的**持久长时间运行工作**，请使用：

- `cronjob`（action=`create`）——调度独立的智能体运行；不受父智能体轮次中断影响。
- `terminal(background=True, notify_on_complete=True)`——长时间运行的 shell 命令，在智能体执行其他操作时持续运行。
:::

## 关键特性

- 每个子智能体获得其**独立的终端会话**（与父智能体分离）
- **嵌套委派为可选项**——只有 `role="orchestrator"` 的子智能体可以进一步委派，且仅在 `max_spawn_depth` 从默认值 1（扁平）提高后才生效。可通过 `orchestrator_enabled: false` 全局禁用。
- 叶子子智能体**不能**调用：`delegate_task`、`clarify`、`memory`、`send_message`、`execute_code`。编排者子智能体保留 `delegate_task`，但仍不能使用其他四个。
- **中断传播**——中断父智能体会中断所有活跃的子智能体（包括编排者下的孙智能体）
- 只有最终摘要进入父智能体的上下文，保持 token 使用高效
- 子智能体继承父智能体的 **API 密钥、provider 配置和凭据池**（支持在速率限制时轮换密钥）

## delegate_task 与 execute_code 对比

| 因素 | delegate_task | execute_code |
|--------|--------------|-------------|
| **推理** | 完整 LLM 推理循环 | 仅 Python 代码执行 |
| **上下文** | 全新隔离对话 | 无对话，仅脚本 |
| **工具访问** | 所有非屏蔽工具，具备推理能力 | 通过 RPC 访问 7 个工具，无推理 |
| **并行性** | 默认 3 个并发子智能体（可配置） | 单脚本 |
| **最适合** | 需要判断力的复杂任务 | 机械式多步骤流水线 |
| **Token 费用** | 较高（完整 LLM 循环） | 较低（仅返回 stdout） |
| **用户交互** | 无（子智能体无法澄清） | 无 |

**经验法则：** 当子任务需要推理、判断或多步骤问题解决时，使用 `delegate_task`。当需要机械式数据处理或脚本化工作流时，使用 `execute_code`。

## 配置

```yaml
# In ~/.hermes/config.yaml
delegation:
  max_iterations: 50                        # Max turns per child (default: 50)
  # max_concurrent_children: 3              # Parallel children per batch (default: 3)
  # max_spawn_depth: 1                      # Tree depth (1-3, default 1 = flat). Raise to 2 to allow orchestrator children to spawn leaves; 3 for three levels.
  # orchestrator_enabled: true              # Disable to force all children to leaf role.
  model: "google/gemini-3-flash-preview"             # Optional provider/model override
  provider: "openrouter"                             # Optional built-in provider
  api_mode: anthropic_messages                       # optional; auto-detected from base_url for anthropic_messages endpoints

# Or use a direct custom endpoint instead of provider:
delegation:
  model: "qwen2.5-coder"
  base_url: "http://localhost:1234/v1"
  api_key: "local-key"
  # api_mode: "anthropic_messages"  # Optional. Wire protocol override for base_url ("chat_completions", "codex_responses", or "anthropic_messages"). Empty = auto-detect from URL (e.g. /anthropic suffix). Set explicitly for endpoints the heuristic can't classify (Azure AI Foundry, MiniMax, Zhipu GLM, LiteLLM proxies, …).
```

当 `base_url` 指向 Anthropic 兼容端点时——例如路径以 `/anthropic` 结尾、Azure Foundry Claude 路由或 MiniMax `/anthropic` 代理——`api_mode` 会被自动检测为 `anthropic_messages`，子智能体无需任何配置即可使用正确的传输格式。当自动检测结果有误时（罕见），请显式设置 `api_mode`。

:::tip
智能体会根据任务复杂度自动处理委派。你无需明确要求它进行委派——它会在合适时自行决定。
:::