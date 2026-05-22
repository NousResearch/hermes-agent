---
sidebar_position: 7
title: "子智能体委派"
description: "使用 delegate_task 生成隔离的子智能体以进行并行工作流"
---

# 子智能体委派

`delegate_task` 工具生成具有隔离上下文、受限工具集和自己的终端会话的子 AIAgent 实例。每个子智能体获得一个全新的对话并独立工作 —— 只有其最终摘要进入父智能体的上下文。

## 单个任务

```python
delegate_task(
    goal="Debug why tests fail",
    context="Error: assertion in test_foo.py line 42",
    toolsets=["terminal", "file"]
)
```

## 并行批处理

默认最多 3 个并发子智能体（可配置，无硬上限）：

```python
delegate_task(tasks=[
    {"goal": "Research topic A", "toolsets": ["web"]},
    {"goal": "Research topic B", "toolsets": ["web"]},
    {"goal": "Fix the build", "toolsets": ["terminal", "file"]}
])
```

## 子智能体上下文如何工作

:::warning 关键：子智能体一无所知
子智能体以**完全全新的对话**开始。它们对父智能体的对话历史、先前的工具调用或委派之前讨论的任何内容一无所知。子智能体的唯一上下文来自父智能体调用 `delegate_task` 时填充的 `goal` 和 `context` 字段。
:::

这意味着父智能体必须在调用中传递子智能体需要的**所有内容**：

```python
# 不好 - 子智能体不知道"错误"是什么
delegate_task(goal="Fix the error")

# 好 - 子智能体拥有所需的所有上下文
delegate_task(
    goal="Fix the TypeError in api/handlers.py",
    context="""The file api/handlers.py has a TypeError on line 47:
    'NoneType' object has no attribute 'get'.
    The function process_request() receives a dict from parse_body(),
    but parse_body() returns None when Content-Type is missing.
    The project is at /home/user/myproject and uses Python 3.11."""
)
```

子智能体接收一个由您的目标和上下文构建的聚焦系统提示，指示它完成任务并提供其操作的结构化摘要、发现、修改的任何文件以及遇到的任何问题。

## 实用示例

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

将审查和修复工作流委派给全新上下文：

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

委派会淹没父智能体上下文的大型重构任务：

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

当您提供 `tasks` 数组时，子智能体使用线程池**并行**运行：

- **最大并发数：** 默认 3 个任务（可通过 `delegation.max_concurrent_children` 或 `DELEGATION_MAX_CONCURRENT_CHILDREN` 环境变量配置；下限为 1，无硬上限）。超过限制的批处理返回工具错误，而非静默截断。
- **线程池：** 使用配置好的并发限制作为最大工作器的 `ThreadPoolExecutor`
- **进度显示：** 在 CLI 模式下，树视图实时显示每个子智能体的工具调用，带有每任务完成行。在网关模式下，进度被批处理并中继到父智能体的进度回调
- **结果排序：** 结果按任务索引排序以匹配输入顺序，无论完成顺序如何
- **中断传播：** 中断父智能体（例如发送新消息）会中断所有活动子智能体

单任务委派直接运行，没有线程池开销。

## 模型覆盖

您可以通过 `config.yaml` 为子智能体配置不同的模型 —— 用于将简单任务委派给更便宜/更快的模型：

```yaml
# 在 ~/.hermes/config.yaml 中
delegation:
  model: "google/gemini-flash-2.0"    # 用于子智能体的更便宜模型
  provider: "openrouter"              # 可选：将子智能体路由到不同的提供商
```

如果省略，子智能体使用与父智能体相同的模型。

## 工具集选择技巧

`toolsets` 参数控制子智能体可以访问哪些工具。根据任务选择：

| 工具集模式 | 用例 |
|----------------|----------|
| `["terminal", "file"]` | 代码工作、调试、文件编辑、构建 |
| `["web"]` | 研究、事实核查、文档查找 |
| `["terminal", "file", "web"]` | 全栈任务（默认） |
| `["file"]` | 只读分析、不执行的代码审查 |
| `["terminal"]` | 系统管理、进程管理 |

无论您指定什么，某些工具集对子智能体都是阻止的：
- `delegation` —— 对叶子子智能体阻止（默认）。为 `role="orchestrator"` 子智能体保留，受 `max_spawn_depth` 限制 —— 请参阅下面的[深度限制和嵌套编排](#depth-limit-and-nested-orchestration)。
- `clarify` —— 子智能体无法与用户交互
- `memory` —— 不写入共享持久内存
- `code_execution` —— 子智能体应该逐步推理
- `send_message` —— 无跨平台副作用（例如发送 Telegram 消息）

## 最大迭代次数

每个子智能体都有一个迭代限制（默认：50），控制它可以进行多少工具调用轮次：

```python
delegate_task(
    goal="Quick file check",
    context="Check if /etc/nginx/nginx.conf exists and print its first 10 lines",
    max_iterations=10  # 简单任务，不需要很多轮次
)
```

## 子智能体超时

如果子智能体安静超过 `delegation.child_timeout_seconds` 挂钟秒数，则会被杀死。默认值为 **600**（10 分钟）—— 从早期版本中的 300 秒增加，因为高推理模型在非平凡研究任务上被中途杀死。按安装调整：

```yaml
delegation:
  child_timeout_seconds: 600   # 默认
```

对于快速本地模型降低它；对于难题上的慢速推理模型提高它。计时器在子智能体每次进行 API 调用或工具调用时重置 —— 只有真正空闲的工作器才会触发杀死。

:::tip 零调用超时诊断转储
如果子智能体在**零**次 API 调用后超时（通常是：提供商不可达、认证失败或工具 schema 拒绝），`delegate_task` 会将结构化诊断写入 `~/.hermes/logs/subagent-timeout-<session>-<timestamp>.log`，包含子智能体的配置快照、凭证解析跟踪和任何早期错误消息。比之前的静默超时行为更容易根因分析。
:::

## 监控运行中的子智能体 (`/agents`)

TUI 附带 `/agents` 覆盖层（别名 `/tasks`），将递归 `delegate_task` 扇出转变为一流审计界面：

- 运行中和最近完成的子智能体的实时树视图，按父智能体分组
- 每分支成本、token 和文件触摸汇总
- 终止和暂停控制 —— 取消特定子智能体而不中断其同级
- 事后审查：即使子智能体返回父智能体后，也可以逐步查看每个子智能体的逐轮历史

经典 CLI 仅将 `/agents` 打印为文本摘要；TUI 是覆盖层闪耀的地方。请参阅 [TUI —— 斜杠命令](/user-guide/tui#slash-commands)。

## 深度限制和嵌套编排 {#depth-limit-and-nested-orchestration}

默认情况下，委派是**扁平的**：父智能体（深度 0）生成子智能体（深度 1），这些子智能体不能进一步委派。这防止了失控的递归委派。

对于多阶段工作流（研究 → 综合，或子问题上的并行编排），父智能体可以生成可以委派自己工作器的**编排器**子智能体：

```python
delegate_task(
    goal="Survey three code review approaches and recommend one",
    role="orchestrator",  # 允许此子智能体生成自己的工作器
    context="...",
)
```

- `role="leaf"`（默认）：子智能体不能进一步委派 —— 与扁平委派行为相同。
- `role="orchestrator"`：子智能体保留 `delegation` 工具集。受 `delegation.max_spawn_depth` 限制（默认 **1** = 扁平，因此在默认情况下 `role="orchestrator"` 是无操作的）。将 `max_spawn_depth` 提高到 2 以允许编排器子智能体生成叶子孙智能体；3 表示三级（上限）。
- `delegation.orchestrator_enabled: false`：全局关闭开关，强制每个子智能体为 `leaf`，无论 `role` 参数如何。

**成本警告：** 使用 `max_spawn_depth: 3` 和 `max_concurrent_children: 3`，树可以达到 3×3×3 = 27 个并发叶子智能体。每增加一级都会成倍增加花费 —— 有意识地提高 `max_spawn_depth`。

## 生命周期和持久性

:::warning delegate_task 是同步的 —— 非持久的
`delegate_task` 在**父智能体的当前轮次内**运行。它会阻塞父智能体，直到每个子智能体完成（或被取消）。它**不是**后台作业队列：

- 如果父智能体被中断（用户发送新消息、`/stop`、`/new`），所有活动子智能体会被取消并返回 `status="interrupted"`。它们进行中的工作被丢弃。
- 子智能体在父智能体轮次结束后**不会**继续运行。
- 取消的子智能体返回结构化结果（`status="interrupted"`、`exit_reason="interrupted"`），但由于父智能体也被中断了，该结果通常永远不会进入用户可见的回复。

对于必须在中断中存活或比当前轮次更持久的**持久长期运行工作**，请使用：

- `cronjob`（action=`create`）—— 安排单独的代理运行；免疫父轮次中断。
- `terminal(background=True, notify_on_complete=True)` —— 在智能体执行其他操作时继续运行的长期 shell 命令。
:::

## 关键属性

- 每个子智能体获得其**自己的终端会话**（与父智能体分离）
- **嵌套委派是可选的** —— 只有 `role="orchestrator"` 子智能体可以进一步委派，并且只有在 `max_spawn_depth` 从其默认值 1（扁平）提高时。使用 `orchestrator_enabled: false` 全局禁用。
- 叶子子智能体**不能**调用：`delegate_task`、`clarify`、`memory`、`send_message`、`execute_code`。编排器子智能体保留 `delegate_task` 但仍然不能使用其他四个。
- **中断传播** —— 中断父智能体会中断所有活动子智能体（包括编排器下的孙智能体）
- 只有最终摘要进入父智能体的上下文，保持 token 使用高效
- 子智能体继承父智能体的**API 密钥、提供商配置和凭证池**（在速率限制上启用密钥轮换）

## 委派 vs execute_code

| 因素 | delegate_task | execute_code |
|--------|--------------|-------------|
| **推理** | 完整 LLM 推理循环 | 仅 Python 代码执行 |
| **上下文** | 全新隔离对话 | 无对话，仅脚本 |
| **工具访问** | 所有非阻止工具带推理 | 通过 RPC 的 7 个工具，无推理 |
| **并行性** | 默认 3 个并发子智能体（可配置） | 单个脚本 |
| **最适合** | 需要判断的复杂任务 | 机械多步流水线 |
| **Token 成本** | 更高（完整 LLM 循环） | 更低（仅返回 stdout） |
| **用户交互** | 无（子智能体无法澄清） | 无 |

**经验法则：** 当子任务需要推理、判断或多步问题解决时使用 `delegate_task`。当您需要机械数据处理或脚本化工作流时使用 `execute_code`。

## 配置

```yaml
# 在 ~/.hermes/config.yaml 中
delegation:
  max_iterations: 50                        # 每个子智能体的最大轮次（默认：50）
  # max_concurrent_children: 3              # 每批并行子智能体（默认：3）
  # max_spawn_depth: 1                      # 树深度（1-3，默认 1 = 扁平）。提高到 2 以允许编排器子智能体生成叶子；3 表示三级。
  # orchestrator_enabled: true              # 禁用以强制所有子智能体为叶子角色。
  model: "google/gemini-3-flash-preview"             # 可选提供商/模型覆盖
  provider: "openrouter"                             # 可选内置提供商

# 或使用直接自定义端点而非提供商：
delegation:
  model: "qwen2.5-coder"
  base_url: "http://localhost:1234/v1"
  api_key: "local-key"
```

:::tip
智能体根据任务复杂性自动处理委派。您不需要明确要求它委派 —— 它在有意义时会这样做。
:::
