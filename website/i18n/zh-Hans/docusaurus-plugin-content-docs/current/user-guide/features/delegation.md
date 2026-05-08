---
sidebar_position: 7
title: "子代理委托"
description: "使用 delegate_task 生成具有隔离上下文、限制工具集的孤立子代理以进行并行工作流"
---

# 子代理委托

`delegate_task` 工具生成具有隔离上下文、限制工具集和独立终端会话的子 AIAgent 实例。每个子代理获得新的对话并独立工作——只有其最终摘要进入父代理的上下文。

## 单个任务

```python
delegate_task(
    goal="Debug why tests fail",
    context="Error: assertion in test_foo.py line 42",
    toolsets=["terminal", "file"]
)
```

## 并行批次

默认最多 3 个并发子代理（可配置，无硬性上限）：

```python
delegate_task(tasks=[
    {"goal": "Research topic A", "toolsets": ["web"]},
    {"goal": "Research topic B", "toolsets": ["web"]},
    {"goal": "Fix the build", "toolsets": ["terminal", "file"]}
])
```

## 子代理上下文如何工作

:::warning 关键：子代理一无所知
子代理从**完全新的对话**开始。它们对父代理的对话历史、先前的工具调用或委托之前讨论的任何内容都没有了解。子代理的唯一上下文来自父代理在调用 `delegate_task` 时填充的 `goal` 和 `context` 字段。
:::

这意味着父代理必须传递子代理**需要的一切**：

```python
# 错误 - 子代理不知道"错误"是什么
delegate_task(goal="Fix the error")

# 正确 - 子代理有其需要的所有上下文
delegate_task(
    goal="Fix the TypeError in api/handlers.py",
    context="""The file api/handlers.py has a TypeError on line 47:
    'NoneType' object has no attribute 'get'.
    The function process_request() receives a dict from parse_body(),
    but parse_body() returns None when Content-Type is missing.
    The project is at /home/user/myproject and uses Python 3.11."""
)
```

子代理收到一个从你的 goal 和 context 构建的专注系统提示，指导它完成任务并提供它做了什么、发现了什么、修改了哪些文件以及遇到的问题的结构化摘要。

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

将审查和修复工作流委托给全新上下文：

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

委托会淹没父代理上下文的大型重构任务：

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

## 批次模式详情

当你提供 `tasks` 数组时，子代理使用线程池**并行**运行：

- **最大并发：** 默认 3 个任务（可通过 `delegation.max_concurrent_children` 或 `DELEGATION_MAX_CONCURRENT_CHILDREN` 环境变量配置；下限为 1，无硬性上限）。超过限制的批次返回工具错误，而不是被静默截断。
- **线程池：** 使用配置并发限制作为最大工作线程数的 `ThreadPoolExecutor`
- **进度显示：** 在 CLI 模式下，树视图实时显示每个子代理的工具调用并带有每任务完成行。在 gateway 模式下，进度被批量处理并中继到父代理的进度回调
- **结果排序：** 按任务索引排序以匹配输入顺序，无论完成顺序如何
- **中断传播：** 中断父代理（例如发送新消息）会中断所有活动子代理

单任务委托直接运行，无线程池开销。

## 模型覆盖

你可以通过 `config.yaml` 为子代理配置不同的模型——适合将简单任务委托给更便宜/更快的模型：

```yaml
# In ~/.hermes/config.yaml
delegation:
  model: "google/gemini-flash-2.0"    # 子代理的更便宜模型
  provider: "openrouter"              # 可选：将子代理路由到不同的提供商
```

如果省略，子代理使用与父代理相同的模型。

## 工具集选择

`toolsets` 参数控制子代理可以访问哪些工具。根据任务选择：

| 工具集模式 | 用例 |
|----------------|----------|
| `["terminal", "file"]` | 代码工作、调试、文件编辑、构建 |
| `["web"]` | 研究、事实核查、文档查找 |
| `["terminal", "file", "web"]` | 全栈任务（默认） |
| `["file"]` | 只读分析、代码审查但不执行 |
| `["terminal"]` | 系统管理、进程管理 |

某些工具集无论你怎么指定都会被阻止：
- `delegation` — 叶子子代理被阻止（默认行为）。对于 `role="orchestrator"` 子代理会保留，受 `max_spawn_depth` 限制——见下方[深度限制和嵌套编排](#深度限制和嵌套编排)。
- `clarify` — 子代理不能与用户交互
- `memory` — 不写入共享持久内存
- `code_execution` — 子代理应该逐步推理
- `send_message` — 无跨平台副作用（例如发送 Telegram 消息）

## 最大迭代次数

每个子代理有迭代限制（默认：50），控制它可以进行的工具调用轮次：

```python
delegate_task(
    goal="Quick file check",
    context="Check if /etc/nginx/nginx.conf exists and print its first 10 lines",
    max_iterations=10  # 简单任务，不需要很多轮次
)
```

## 子代理超时

如果子代理静默超过 `delegation.child_timeout_seconds` 秒，被视为卡住并被杀死。默认是 **600**（10 分钟）——从早期版本的 300 秒提高了，因为高推理模型在非简单研究任务中会在思考中途被杀死。按安装需求调整：

```yaml
delegation:
  child_timeout_seconds: 600   # 默认
```

对于快速本地模型可以降低；对于处理难题的慢速推理模型可以提高。每次子代理发出 API 调用或工具调用时计时器重置——只有真正空闲的工作线程才会触发杀死。

:::tip 零调用超时的诊断转储
如果子代理在发出**零次** API 调用的情况下超时（通常原因：提供商不可达、认证失败或工具 schema 拒绝），`delegate_task` 会将结构化诊断写入 `~/.hermes/logs/subagent-timeout-<session>-<timestamp>.log`，包含子代理的配置快照、凭据解析跟踪和任何早期错误消息。这比以前的静默超时行为更容易排查根本原因。
:::

## 监控运行中的子代理（`/agents`）

TUI 提供了一个 `/agents` 覆盖层（别名 `/tasks`），将递归 `delegate_task` 扇出转化为一级审计界面：

- 运行中和最近完成的子代理的实时树视图，按父代理分组
- 每个分支的成本、令牌数和文件访问汇总
- 终止和暂停控制——可以中途取消特定子代理而不中断其兄弟子代理
- 事后审查：即使子代理已返回父代理，仍可逐步查看其逐轮历史

经典 CLI 只将 `/agents` 打印为文本摘要；TUI 才是覆盖层大放异彩的地方。参见 [TUI — 斜杠命令](/docs/user-guide/tui#slash-commands)。

## 深度限制和嵌套编排

默认情况下，委托是**扁平的**：父代理（深度 0）生成子代理（深度 1），这些子代理不能进一步委托。这防止了失控的递归委托。

对于多阶段工作流（研究 → 综合，或对子问题的并行编排），父代理可以生成**编排器**子代理，这些子代理*可以*委托自己的工作线程：

```python
delegate_task(
    goal="Survey three code review approaches and recommend one",
    role="orchestrator",  # 允许此子代理生成自己的工作线程
    context="...",
)
```

- `role="leaf"`（默认）：子代理不能进一步委托——与扁平委托行为相同。
- `role="orchestrator"`：子代理保留 `delegation` 工具集。受 `delegation.max_spawn_depth` 限制（默认 **1** = 扁平，所以 `role="orchestrator"` 在默认设置下无效）。将 `max_spawn_depth` 提高到 2 以允许编排器子代理生成叶子孙代理；3 为三级（上限）。
- `delegation.orchestrator_enabled: false`：全局开关，强制所有子代理为 `leaf`，无论 `role` 参数如何。

**成本警告：** 在 `max_spawn_depth: 3` 和 `max_concurrent_children: 3` 的情况下，树可以达到 3×3×3 = 27 个并发叶子代理。每增加一级都会倍增支出——请谨慎提高 `max_spawn_depth`。

## 生命周期和持久性

:::warning delegate_task 是同步的——不是持久的
`delegate_task` 在父代理的当前轮次内运行。它阻塞父代理直到每个子代理完成（或被取消）。它**不是**后台作业队列：

- 如果父代理被中断（用户发送新消息、`/stop`、`/new`），所有活动子代理被取消并返回 `status="interrupted"`。它们的进行中工作被丢弃。
- 子代理在父代理轮次结束后**不会**继续运行。
- 取消的子代理返回结构化结果（`status="interrupted"`、`exit_reason="interrupted"`），但因为父代理也被中断了，该结果通常永远不会进入用户可见的回复。

对于**持久的长期工作**（必须存活中断或超出当前轮次），使用：

- `cronjob`（action=`create`）——调度单独的 agent 运行；不受父轮次中断影响。
- `terminal(background=True, notify_on_complete=True)` — 长时间运行的 shell 命令在 agent 做其他事情时继续运行。
:::

## 关键属性

- 每个子代理获得自己的**终端会话**（与父代理分离）
- **嵌套委托是可选加入的** — 只有 `role="orchestrator"` 子代理可以进一步委托，并且只有在 `max_spawn_depth` 从默认值 1（扁平）提高时。使用 `orchestrator_enabled: false` 全局禁用。
- 叶子子代理**不能**调用：`delegate_task`、`clarify`、`memory`、`send_message`、`execute_code`。编排器子代理保留 `delegate_task` 但仍不能使用其他四个。
- **中断传播** — 中断父代理会中断所有活动子代理（包括编排器下的孙代理）
- 只有最终摘要进入父代理的上下文，保持令牌使用高效
- 子代理继承父代理的 **API 密钥、提供商配置和凭据池**（支持限速时的密钥轮换）

## 委托与 execute_code 对比

| 因素 | delegate_task | execute_code |
|--------|--------------|-------------|
| **推理** | 完整 LLM 推理循环 | 仅 Python 代码执行 |
| **上下文** | 全新隔离对话 | 无对话，只有脚本 |
| **工具访问** | 所有非阻止工具加推理 | 7 个工具通过 RPC，无推理 |
| **并行性** | 默认 3 个并发子代理（可配置） | 单个脚本 |
| **最适合** | 需要判断的复杂任务 | 机械多步骤流水线 |
| **令牌成本** | 更高（完整 LLM 循环） | 更低（仅返回 stdout） |
| **用户交互** | 无（子代理不能澄清） | 无 |

**经验法则：** 当子任务需要推理、判断或多步骤问题解决时使用 `delegate_task`。当你需要机械数据处理或脚本工作流时使用 `execute_code`。

## 配置

```yaml
# In ~/.hermes/config.yaml
delegation:
  max_iterations: 50                        # 每个子代理的最大轮次（默认：50）
  # max_concurrent_children: 3              # 每批次并行子代理（默认：3）
  # max_spawn_depth: 1                      # 树深度（1-3，默认 1 = 扁平）。提高到 2 允许编排器子代理生成叶子；3 为三级。
  # orchestrator_enabled: true              # 设为 false 强制所有子代理为叶子角色。
  model: "google/gemini-3-flash-preview"             # 可选的提供商/模型覆盖
  provider: "openrouter"                             # 可选的内置提供商

# 或者使用直接的自定义端点代替提供商：
delegation:
  model: "qwen2.5-coder"
  base_url: "http://localhost:1234/v1"
  api_key: "local-key"
```

:::tip
代理会根据任务复杂度自动处理委托。你不需要显式要求它委托——它会在合理时自动进行。
:::
