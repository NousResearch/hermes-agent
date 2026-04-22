---
sidebar_position: 7
title: "Subagent Delegation"
description: "Spawn isolated child agents for parallel workstreams with delegate_task"
---

# Subagent Delegation

The `delegate_task` tool spawns child AIAgent instances with isolated context, restricted toolsets, and their own terminal sessions. Each child gets a fresh conversation and works independently — only its final summary enters the parent's context.

## Single Task

```python
delegate_task(
    goal="Debug why tests fail",
    context="Error: assertion in test_foo.py line 42",
    toolsets=["terminal", "file"]
)
```

## Parallel Batch

Up to 3 concurrent subagents by default (configurable, no hard ceiling):

```python
delegate_task(tasks=[
    {"goal": "Research topic A", "toolsets": ["web"]},
    {"goal": "Research topic B", "toolsets": ["web"]},
    {"goal": "Fix the build", "toolsets": ["terminal", "file"]}
])
```

## How Subagent Context Works

:::warning Critical: Subagents Know Nothing
Subagents start with a **completely fresh conversation**. They have zero knowledge of the parent's conversation history, prior tool calls, or anything discussed before delegation. The subagent's only context comes from the `goal` and `context` fields the parent agent populates when it calls `delegate_task`.
:::

This means the parent agent must pass **everything** the subagent needs in the call:

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

The subagent receives a focused system prompt built from your goal and context, instructing it to complete the task and provide a structured summary of what it did, what it found, any files modified, and any issues encountered.

## Practical Examples

### Parallel Research

Research multiple topics simultaneously and collect summaries:

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

### Code Review + Fix

Delegate a review-and-fix workflow to a fresh context:

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

### Multi-File Refactoring

Delegate a large refactoring task that would flood the parent's context:

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

## Batch Mode Details

When you provide a `tasks` array, subagents run in **parallel** using a thread pool:

- **Maximum concurrency:** 3 tasks by default (configurable via `delegation.max_concurrent_children` or the `DELEGATION_MAX_CONCURRENT_CHILDREN` env var; floor of 1, no hard ceiling). Batches larger than the limit return a tool error rather than being silently truncated.
- **Thread pool:** Uses `ThreadPoolExecutor` with the configured concurrency limit as max workers
- **Progress display:** In CLI mode, a tree-view shows tool calls from each subagent in real-time with per-task completion lines. In gateway mode, progress is batched and relayed to the parent's progress callback
- **Result ordering:** Results are sorted by task index to match input order regardless of completion order
- **Interrupt propagation:** Interrupting the parent (e.g., sending a new message) interrupts all active children

Single-task delegation runs directly without thread pool overhead.

## Model Override

You can configure a different model for subagents via `config.yaml` — useful for delegating simple tasks to cheaper/faster models:

```yaml
# In ~/.hermes/config.yaml
delegation:
  model: "google/gemini-flash-2.0"    # Cheaper model for subagents
  provider: "openrouter"              # Optional: route subagents to a different provider
```

If omitted, subagents use the same model as the parent.

## Toolset Selection Tips

The `toolsets` parameter controls what tools the subagent has access to. Choose based on the task:

| Toolset Pattern | Use Case |
|----------------|----------|
| `["terminal", "file"]` | Code work, debugging, file editing, builds |
| `["web"]` | Research, fact-checking, documentation lookup |
| `["terminal", "file", "web"]` | Full-stack tasks (default) |
| `["file"]` | Read-only analysis, code review without execution |
| `["terminal"]` | System administration, process management |

Certain toolsets are blocked for subagents regardless of what you specify:
- `delegation` — blocked for leaf subagents (the default). Retained for `role="orchestrator"` children, bounded by `max_spawn_depth` — see [Depth Limit and Nested Orchestration](#depth-limit-and-nested-orchestration) below.
- `clarify` — subagents cannot interact with the user
- `memory` — no writes to shared persistent memory
- `code_execution` — children should reason step-by-step
- `send_message` — no cross-platform side effects (e.g., sending Telegram messages)

## Max Iterations

Each subagent has an iteration limit (default: 50) that controls how many tool-calling turns it can take:

```python
delegate_task(
    goal="Quick file check",
    context="Check if /etc/nginx/nginx.conf exists and print its first 10 lines",
    max_iterations=10  # Simple task, don't need many turns
)
```

## Depth Limit and Nested Orchestration

By default, delegation is **flat**: a parent (depth 0) spawns children (depth 1), and those children cannot delegate further. This prevents runaway recursive delegation.

For multi-stage workflows (research → synthesis, or parallel orchestration over sub-problems), a parent can spawn **orchestrator** children that *can* delegate their own workers:

```python
delegate_task(
    goal="Survey three code review approaches and recommend one",
    role="orchestrator",  # Allows this child to spawn its own workers
    context="...",
)
```

- `role="leaf"` (default): child cannot delegate further — identical to the flat-delegation behavior.
- `role="orchestrator"`: child retains the `delegation` toolset. Gated by `delegation.max_spawn_depth` (default **1** = flat, so `role="orchestrator"` is a no-op at defaults). Raise `max_spawn_depth` to 2 to allow orchestrator children to spawn leaf grandchildren; 3 for three levels (cap).
- `delegation.orchestrator_enabled: false`: global kill switch that forces every child to `leaf` regardless of the `role` parameter.

**Cost warning:** With `max_spawn_depth: 3` and `max_concurrent_children: 3`, the tree can reach 3×3×3 = 27 concurrent leaf agents. Each extra level multiplies spend — raise `max_spawn_depth` intentionally.

## Key Properties

- Each subagent gets its **own terminal session** (separate from the parent)
- **Nested delegation is opt-in** — only `role="orchestrator"` children can delegate further, and only when `max_spawn_depth` is raised from its default of 1 (flat). Disable globally with `orchestrator_enabled: false`.
- Leaf subagents **cannot** call: `delegate_task`, `clarify`, `memory`, `send_message`, `execute_code`. Orchestrator subagents retain `delegate_task` but still cannot use the other four.
- **Interrupt propagation** — interrupting the parent interrupts all active children (including grandchildren under orchestrators)
- Only the final summary enters the parent's context, keeping token usage efficient
- Subagents inherit the parent's **API key, provider configuration, and credential pool** (enabling key rotation on rate limits)

## Delegation vs execute_code

| Factor | delegate_task | execute_code |
|--------|--------------|-------------|
| **Reasoning** | Full LLM reasoning loop | Just Python code execution |
| **Context** | Fresh isolated conversation | No conversation, just script |
| **Tool access** | All non-blocked tools with reasoning | 7 tools via RPC, no reasoning |
| **Parallelism** | 3 concurrent subagents by default (configurable) | Single script |
| **Best for** | Complex tasks needing judgment | Mechanical multi-step pipelines |
| **Token cost** | Higher (full LLM loop) | Lower (only stdout returned) |
| **User interaction** | None (subagents can't clarify) | None |

**Rule of thumb:** Use `delegate_task` when the subtask requires reasoning, judgment, or multi-step problem solving. Use `execute_code` when you need mechanical data processing or scripted workflows.

## Configuration

```yaml
# In ~/.hermes/config.yaml
delegation:
  max_iterations: 50                        # Max turns per child (default: 50)
  # max_concurrent_children: 3              # Parallel children per batch (default: 3)
  # max_spawn_depth: 1                      # Tree depth (1-3, default 1 = flat). Raise to 2 to allow orchestrator children to spawn leaves; 3 for three levels.
  # orchestrator_enabled: true              # Disable to force all children to leaf role.
  model: "google/gemini-3-flash-preview"             # Optional provider/model override
  provider: "openrouter"                             # Optional built-in provider

# Or use a direct custom endpoint instead of provider:
delegation:
  model: "qwen2.5-coder"
  base_url: "http://localhost:1234/v1"
  api_key: "local-key"
```

:::tip
The agent handles delegation automatically based on the task complexity. You don't need to explicitly ask it to delegate — it will do so when it makes sense.
:::
