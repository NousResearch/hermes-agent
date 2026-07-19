---
sidebar_position: 7
title: "Subagent Delegation"
description: "Spawn isolated child agents for parallel workstreams with delegate_task"
---

# Subagent Delegation

The `delegate_task` tool spawns child AIAgent instances with isolated context, inherited tool access, and their own terminal sessions. Each child gets a fresh conversation and works independently — only its final summary enters the parent's context.

Top-level model calls run in the background automatically. Hermes returns a handle immediately so the conversation can continue, then posts the result back as a new message. An orchestrator subagent waits for its own workers so it can synthesize their results before returning.

## Single Task

```python
delegate_task(
    goal="Debug why tests fail",
    context="Error: assertion in test_foo.py line 42"
)
```

## Parallel Batch

Up to 3 concurrent subagents by default (configurable, no hard ceiling):

```python
delegate_task(tasks=[
    {"goal": "Research topic A", "context": "Focus on recent primary sources"},
    {"goal": "Research topic B", "context": "Compare the leading explanations"},
    {"goal": "Fix the build", "context": "Project root: /home/user/project"}
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
        "context": "Focus on: browser support, non-browser runtimes, language support"
    },
    {
        "goal": "Research the current state of RISC-V adoption in 2025",
        "context": "Focus on: server chips, embedded systems, software ecosystem"
    },
    {
        "goal": "Research quantum computing progress in 2025",
        "context": "Focus on: error correction breakthroughs, practical applications, key players"
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
    Fix any issues found and run the test suite (pytest tests/auth/)."""
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
    Run pytest after to verify nothing broke."""
)
```

## Batch Mode Details

When a top-level agent provides a `tasks` array, Hermes returns one background handle, runs the subagents in parallel, and posts one consolidated result after every child finishes. An orchestrator subagent waits for its batch in the current turn so it can synthesize the results.

- **Maximum concurrency:** 3 tasks by default (configurable via `delegation.max_concurrent_children` or the `DELEGATION_MAX_CONCURRENT_CHILDREN` env var; floor of 1, no hard ceiling). Batches larger than the limit return a tool error rather than being silently truncated.
- **Thread pool:** Uses `ThreadPoolExecutor` with the configured concurrency limit as max workers
- **Progress display:** In CLI mode, a tree-view shows tool calls from each subagent in real-time with per-task completion lines. In gateway mode, progress is batched and relayed to the parent's progress callback
- **Result ordering:** Results are sorted by task index to match input order regardless of completion order
- **Cancellation:** Follow-up messages do not cancel a top-level background batch. `/stop` or closing/resetting the owning session cancels its active children. Synchronous orchestrator children still follow their parent's interrupt state

Synchronous single-task delegation from an orchestrator runs directly without thread pool overhead.

### Durable background completions

When a background delegation finishes, Hermes stores its completion event in
the active profile's `state.db` before publishing it to the normal fresh-turn
queue. If Hermes restarts after completion but before delivery, the pending
event is restored and routed through the same ownership checks. Competing
consumers use a durable claim, so only the consumer that successfully accepts
the synthetic turn acknowledges delivery; failed attempts release the claim for
retry.

This does not resume child execution after a crash. A delegation whose owner
process disappears while it is still running is recorded as `unknown`, because
Hermes cannot prove whether its external side effects happened. Pending and
delivered records are bounded and profile-local.

## Model Override

You can configure a different model for subagents via `config.yaml` — useful for delegating simple tasks to cheaper/faster models:

```yaml
# In ~/.hermes/config.yaml
delegation:
  model: "google/gemini-flash-2.0"    # Cheaper model for subagents
  provider: "openrouter"              # Optional: route subagents to a different provider
```

If omitted, subagents use the same model as the parent.

## Inherited Tool Access

`delegate_task` does not accept a model-facing `toolsets` parameter. Each subagent inherits the parent's enabled toolsets so the model cannot grant a child capabilities that the parent does not have. Configure the parent's tools before starting the conversation if delegated work needs additional capabilities.

Certain tools are blocked for subagents even when the parent has them:
- `delegation` — blocked for leaf subagents (the default). Retained for `role="orchestrator"` children, bounded by `max_spawn_depth` — see [Depth Limit and Nested Orchestration](#depth-limit-and-nested-orchestration) below.
- `clarify` — subagents cannot interact with the user
- `memory` — no writes to shared persistent memory
- `code_execution` — children should reason step-by-step

## Max Iterations

Each subagent has an iteration limit (default: 50) that controls how many tool-calling turns it can take:

```python
delegate_task(
    goal="Quick file check",
    context="Check if /etc/nginx/nginx.conf exists and print its first 10 lines",
    max_iterations=10  # Simple task, don't need many turns
)
```

## Child Timeout

By default there is **no wall-clock timeout** on subagents. Children fail only from what they're actually doing — API errors, tool errors, or hitting their iteration budget — never from a delegation-level stopwatch. Earlier releases shipped a hard cap (300s, later 600s), which kept killing legitimately busy children mid-task: deep code reviews, large research fan-outs, and slow reasoning models routinely need more than 10 minutes while making steady progress the whole time.

Genuinely stuck children are still detected: the heartbeat staleness monitor stops refreshing the parent's activity when a child makes no progress (no API calls, no tool starts), letting the gateway inactivity timeout fire on a truly wedged worker.

If you want a hard cap anyway (e.g. cost control on unattended cron-driven delegation), opt in per-install:

```yaml
delegation:
  child_timeout_seconds: 0     # default: 0 = no timeout
  # child_timeout_seconds: 1800  # opt-in hard cap (floor 30s)
```

A positive value enforces a hard wall-clock limit on each child; `0` or a negative value disables it.

:::tip Diagnostic dump on zero-call timeout
With a hard cap configured, if a subagent times out having made **zero** API calls (usually: provider unreachable, auth failure, or tool-schema rejection), `delegate_task` writes a structured diagnostic to `~/.hermes/logs/subagent-timeout-<session>-<timestamp>.log` containing the subagent's config snapshot, credential-resolution trace, and any early error messages. Much easier to root-cause than the previous silent-timeout behavior.
:::

## Monitoring Running Subagents (`/agents`)

The TUI ships a `/agents` overlay (alias `/tasks`) that turns recursive `delegate_task` fan-out into a first-class audit surface:

- Live tree view of running and recently-finished subagents, grouped by parent
- Per-branch cost, token, and file-touched rollups
- Kill and pause controls — cancel a specific subagent mid-flight without interrupting its siblings
- Post-hoc review: step through each subagent's turn-by-turn history even after they've returned to the parent

The classic CLI just prints `/agents` as a text summary; the TUI is where the overlay shines. See [TUI — Slash commands](/user-guide/tui#slash-commands).

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
- `role="orchestrator"`: child retains the `delegation` toolset. Gated by `delegation.max_spawn_depth` (default **1** = flat, so `role="orchestrator"` is a no-op at defaults). Raise `max_spawn_depth` to 2 to allow orchestrator children to spawn leaf grandchildren; 3+ for deeper trees. There is no upper ceiling — cost is the practical limit.
- `delegation.orchestrator_enabled: false`: global kill switch that forces every child to `leaf` regardless of the `role` parameter.

**Cost warning:** With `max_spawn_depth: 3` and `max_concurrent_children: 3`, the tree can reach 3×3×3 = 27 concurrent leaf agents. Each extra level multiplies spend — raise `max_spawn_depth` intentionally.

## Lifetime and Durability

:::warning Background completion durability is not durable execution
Top-level model-facing `delegate_task` calls run in the background automatically where the session supports later delivery. Hermes returns a handle immediately, and the result re-enters the conversation after the child or batch finishes. Orchestrator subagents wait for their workers in the current turn because they must synthesize those results before returning. Stateless request/response endpoints fall back to synchronous execution when they cannot deliver a detached result later.

- Normal follow-up messages do not cancel background children. `/stop` cancels running background delegations, and closing or resetting the owning session discards its active children.
- Explicit session close/reset interrupts that session's background children. Closing a TUI viewer of a gateway-owned session does not kill the gateway's work.
- A Hermes process restart does **not** resume a running child. Its attempt becomes `unknown` because Hermes cannot prove which side effects happened.
- A child that completed before restart but whose result was not delivered is restored and routed back through the owning session's normal checks.
- Cancelled children return a structured result (`status="interrupted"`, `exit_reason="interrupted"`), but because the parent was interrupted too, that result often never makes it into a user-visible reply.

For **durable execution** that must survive session closure or process restart, use:

- `cronjob` (action=`create`) — schedules a separate agent run; immune to parent-turn interrupts.
- `terminal(background=True, notify_on_complete=True)` — long-running shell commands that keep running while the agent does other things.
:::

## Key Properties

- Each subagent gets its **own terminal session** (separate from the parent)
- Subagents inherit the parent's enabled toolsets; the model cannot select or widen them per call
- **Nested delegation is opt-in** — only `role="orchestrator"` children can delegate further, and only when `max_spawn_depth` is raised from its default of 1 (flat). Disable globally with `orchestrator_enabled: false`.
- Leaf subagents **cannot** call: `delegate_task`, `clarify`, `memory`, `execute_code`. Orchestrator subagents retain `delegate_task` but still cannot use the other three.
- **Cancellation follows ownership** — `/stop` or closing/resetting the owning session cancels its background children; synchronous descendants under orchestrators follow their parent's interrupt state
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
  # max_spawn_depth: 1                      # Tree depth (floor 1, no ceiling, default 1 = flat). Raise to 2 to allow orchestrator children to spawn leaves; 3+ for deeper trees.
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

When `base_url` points at an Anthropic-compatible endpoint — for example a path ending in `/anthropic`, an Azure Foundry Claude route, or a MiniMax `/anthropic` proxy — `api_mode` is auto-detected as `anthropic_messages` so the subagent uses the right wire format without you setting anything. Set `api_mode` explicitly when the auto-detection guess is wrong (rare).

:::tip
The agent handles delegation automatically based on the task complexity. You don't need to explicitly ask it to delegate — it will do so when it makes sense.
:::
