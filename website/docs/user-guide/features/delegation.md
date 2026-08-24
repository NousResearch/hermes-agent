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

### Cost strategy: frontier planner, inexpensive workers

Decomposing a problem into well-specified subtasks takes frontier-level judgment; executing a subtask that already comes with a clear goal, full context, and an output contract usually doesn't. Meanwhile the children are where the tokens go — a parallel batch of subagents typically burns the large majority of a run's total tokens, so the worker model is where the cost actually lives. Pinning `delegation.model` to an inexpensive model while your main session stays on a frontier model keeps the planning quality where it matters and cuts spend where the volume is:

```yaml
# ~/.hermes/config.yaml
model:
  default: "your-frontier-model"     # parent (planner) stays on the frontier model
delegation:
  model: "your-inexpensive-model"    # all delegate_task children run on this
  provider: "openrouter"             # optional: route children to a different provider
```

Resolution order: `delegation.base_url` (direct endpoint) takes precedence, then `delegation.provider` (full credential bundle resolved via the runtime provider system), and when neither is set children inherit the parent's provider and credentials; `delegation.model` applies in all cases, and when it is empty children inherit the parent's model.

Note that the pin is global: `delegate_task` has no per-task model parameter, so every child in a batch runs on the configured delegation model. For quality-sensitive subtasks that need a stronger model, either leave `delegation.model` unset for that session or hand the task to the [kanban board](kanban.md#per-task-model-override), which does support a per-task model override.

## The `/review` Command

`/review` spawns an independent, full-privilege background subagent whose only job is to review the work your conversation just produced — a PR, a diff, code, documentation, a design. It works on every surface: CLI, TUI, the Desktop app, and every gateway messaging platform.

```
/review                       # review whatever the last 10 messages presented
/review focus on security     # add extra instructions for the reviewer
```

What happens:

1. The last 10 user/assistant messages are snapshotted as the reviewer's starting evidence (tool output and system messages are excluded).
2. A reviewer subagent is dispatched on the same background delegation rail as `delegate_task` — it gets the full normal subagent toolset (terminal, web, files, browser...), so it actually opens the PR, reads the diff, and runs code rather than judging from the excerpt.
3. The reviewer inherits the primary agent's working context: any skills the primary agent had loaded (launch-preloaded or via `skill_view` during the session) are named in its briefing with an instruction to load them and judge the work against their conventions, and the workspace's project context files (AGENTS.md / CLAUDE.md / .cursorrules — same discovery as the main agent's system prompt) are embedded in the briefing as binding review standards.
4. When it finishes, its full review re-enters the same session as a normal background-subagent completion — your primary agent sees it and can act on it (fix the findings, push follow-ups, reply to you).

The canonical flow: your main agent opens a PR, you type `/review`, and a second pair of eyes investigates it while you keep working; the review lands back in the chat addressed to the agent that created the PR.

### Review model

By default the reviewer runs on your main model. To pin a dedicated review model, set `auxiliary.review` in `config.yaml`:

```yaml
auxiliary:
  review:
    provider: openrouter               # or nous, anthropic, a direct base_url, ...
    model: anthropic/claude-opus-4.6   # a strong reviewer model
```

Credentials resolve exactly like a `delegation.provider` pin (full runtime-provider bundle: base_url, api key, api_mode). `provider: auto` with an empty `model` means "inherit the main agent's model" — the default.

`/review` is deliberately separate from `/refine`: `/refine` reviews the conversation to update memory and skills, `/review` reviews the *work product* the conversation created.

## Inherited Tool Access

`delegate_task` does not accept a model-facing `toolsets` parameter. Each subagent inherits the parent's enabled toolsets so the model cannot grant a child capabilities that the parent does not have. Configure the parent's tools before starting the conversation if delegated work needs additional capabilities.

Certain tools are blocked for subagents even when the parent has them:
- `delegate_task` — blocked for leaf subagents (the default). Retained for `role="orchestrator"` children, bounded by `max_spawn_depth` — see [Depth Limit and Nested Orchestration](#depth-limit-and-nested-orchestration) below.
- `clarify` — subagents cannot interact with the user
- `memory` — no writes to shared persistent memory
- `send_message` — no cross-platform side effects
- `cronjob` — no scheduling more work in the parent's name

Both roles retain `execute_code` (programmatic tool calling) so children can batch mechanical work.

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

Genuinely stuck children are still detected: the heartbeat staleness monitor stops refreshing the parent's activity when a child makes no progress (no API calls, no tool starts, and no activity-timestamp ticks), letting the gateway inactivity timeout fire on a truly wedged worker. An in-flight model wait still counts as progress — subagents refresh the activity clock while waiting on the provider, so a slow local / long-prefill completion is not treated as stalled.

If you want a hard cap anyway (e.g. cost control on unattended cron-driven delegation), opt in per-install:

```yaml
delegation:
  child_timeout_seconds: 0     # default: 0 = no timeout
  # child_timeout_seconds: 1800  # opt-in hard cap (floor 30s)
```

A positive value enforces a hard wall-clock limit on each child; `0` or a negative value disables it.

When a configured cap fires, the child's result carries structured timeout
metadata alongside the error message so parents and hooks can distinguish a
stopwatch kill from other failures without parsing text: `timeout_seconds`
(the configured cap), `timed_out_after_seconds` (actual wall clock), and
`timeout_phase` (`before_first_llm_call` when the child never reached its
first request, `after_llm_calls` otherwise). All three are `null` on
non-timeout errors.

:::tip Diagnostic dump on zero-call timeout
With a hard cap configured, if a subagent times out having made **zero** API calls (usually: provider unreachable, auth failure, or tool-schema rejection), `delegate_task` writes a structured diagnostic to `~/.hermes/logs/subagent-timeout-<session>-<timestamp>.log` containing the subagent's config snapshot, credential-resolution trace, any early error messages, and stack traces for **all** live threads (not just the child's own) — a child parked waiting on a nested helper thread is indistinguishable from a slow provider without the full picture.
:::

## Stall Detection for Background Subagents

Background delegations (`delegate_task(background=true)`) are watched by a
**progress-based stall monitor** — on by default, zero config. Unlike a
wall-clock timeout, it never touches a child that is making progress, no
matter how long it runs.

The monitor samples each detached child's progress signals — API-call count,
current tool, and last-activity timestamp (which ticks on **every streamed
token**, tool transition, and API-call boundary, so a child mid-stream on a
long response always counts as alive):

1. **Progressing children are never touched.** Any advancing signal resets
   the clock.
2. A child whose progress is completely frozen past the stale threshold
   (450s idle, 1200s while inside a tool — legitimately slow terminal
   commands and web fetches get the higher ceiling) is **interrupted** and
   given a 120s grace window. A child that unwinds in time delivers its
   partial results through the normal completion path.
3. A child that never returns is force-finalized with a terminal `stalled`
   completion event, so the owning session hears an outcome instead of
   going silent, and the async slot frees for new work.

The `stalled` event carries structured metadata mirroring the sync-path
timeout fields: `stalled_after_quiet_seconds`, `stall_threshold_seconds`,
`stall_phase` (`idle` / `in_tool`), and `stall_grace_seconds`.

This closed a long-standing failure mode where a wedged background child
left its session looking dead until a process restart. The underlying wedge
(children hanging at their first API call after multi-day gateway uptime)
was also fixed at the root: delegated children now run their OpenAI-wire
API requests inline on their own conversation thread instead of a nested
worker thread — the layer where the wedge lived. The stall monitor remains
as the safety net for anything else.


## Monitoring Running Subagents (`/agents`)

The TUI ships a `/agents` overlay (alias `/tasks`) that turns recursive `delegate_task` fan-out into a first-class audit surface:

- Live tree view of running and recently-finished subagents, grouped by parent
- Per-branch cost, token, and file-touched rollups
- Kill and pause controls — cancel a specific subagent mid-flight without interrupting its siblings
- Post-hoc review: step through each subagent's turn-by-turn history even after they've returned to the parent

The classic CLI just prints `/agents` as a text summary; the TUI is where the overlay shines. See [TUI — Slash commands](/user-guide/tui#slash-commands).

On the classic CLI and every gateway platform (Telegram, Discord, Slack, ...),
`/agents` also lists **background delegations with live per-child activity**,
sampled directly from each running child:

```
Background delegations: 1 running
- deleg_ab12cd34 · running · research the delegation stall monitor
  - child 1: 4 api calls · in web_search · active 12s ago
  - child 2: 7 api calls · between turns · active 3s ago
```

A delegation the stall monitor has flagged shows as
`stalling · no progress 450s — interrupting`, and long-quiet-but-healthy
children show their quiet time so you can tell "slow" from "stuck" at a
glance.

## Steering a Running Subagent

Interrupting a child throws away its in-flight work; often you just want to redirect it.

### From the parent agent (model-facing)

The parent agent orchestrates its own running children with the same `delegate_task` tool it spawned them with — no separate control tool:

```json
{"action": "list"}
{"action": "steer", "subagent_id": "sa-0-1a2b3c4d", "message": "focus on pricing instead"}
{"action": "stop",  "subagent_id": "sa-0-1a2b3c4d"}
```

- **`list`** returns the conversation's live children: `subagent_id`, goal, status, `running_seconds`, `accepting_steer`, and the live transcript path. Ids also come back in the spawn dispatch response as `subagent_ids`.
- **`steer`** queues a course correction into a running child without stopping it (delivery semantics below).
- **`stop`** ends a child early at its next iteration boundary; the partial result still re-enters the conversation as a normal completion message.

Control actions run synchronously in-turn (never backgrounded), are scoped to the caller's own spawn tree — a conversation can never see or control another session's children — and never consume the per-turn subagent spawn cap, so `stop` keeps working even after the cap is hit.

### From the TUI / gateway (session-facing)

`steer_subagent(subagent_id, text)` in `tools/delegate_tool.py` is the redirection-side mirror of `interrupt_subagent()`: it queues text into a live child through the same mechanism as [`/steer`](/reference/slash-commands) — the text is appended to the child's last tool result at its next iteration boundary, the in-flight tool call is never cut, and the child sees it as an out-of-band user message. Programmatic hosts reach it through the session-scoped `subagent.steer` gateway RPC, which sits beside `subagent.interrupt`:

```json
{"method": "subagent.steer", "params": {"session_id": "owning-ui-session", "subagent_id": "sa-0-1a2b3c4d", "text": "focus on pricing instead"}}
```

Subagent ids come from `delegation.status` (or `list_active_subagents()`) — the same place `subagent.interrupt` gets them. The gateway accepts steering only from the exact live UI/gateway session that spawned the child. A missing, foreign, ambiguous, or stale/recycled session identity is rejected; knowing a global subagent id is not authority. Direct in-process callers retain the unscoped helper contract deliberately.

**Queued is not delivered, but it is never synthetic success.** A `"queued"` response means the text was accepted before the child's completion boundary, not necessarily that the child has seen it. Acceptance and completion are synchronized: either the child can still consume the text, or its exact text is drained into the result as `pending_steer`. Calls after closure return `"rejected"`. If a child accepted the steer but had already produced its final answer, the completion entry the parent receives retains it as `missed_steer`, with a note appended to the summary:

```
[steer did not land — the subagent finished before it could be delivered: focus on pricing instead]
```

So the parent (or the operator driving it) can tell a steered child from one that finished on the old instructions, and re-issue the guidance as a follow-up instead of trusting that it landed.

## Live Transcripts

Every `delegate_task` dispatch also creates one **append-only, human-readable log per task** so you (or the parent agent) can watch a subagent work in real time instead of waiting for the consolidated summary:

```
<hermes_home>/cache/delegation/live/<delegation_id>/task-<n>.log
```

The dispatch response includes the paths as `live_transcripts`, and the files are pre-created at dispatch time, so this works immediately:

```bash
tail -f ~/.hermes/cache/delegation/live/deleg_ab12cd34/task-0.log
```

Each line is timestamped and shows the child's assistant text, thinking snippets, tool calls (`-> tool_name({args})`), tool results, and a final status marker. A `manifest.json` in the same directory describes the batch (goals, task count, per-task status). The logs persist after completion — they double as the full-fidelity operational record alongside the summary — and directories older than 7 days are pruned automatically on new dispatches. Because they live under `cache/delegation`, they are also readable from remote terminal backends (Docker/Modal/SSH).

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
- Leaf subagents **cannot** call: `delegate_task`, `clarify`, `memory`, `send_message`, `cronjob`. Orchestrator subagents retain `delegate_task` but keep the other blocks. Both roles retain `execute_code` (programmatic tool calling) so children can batch mechanical work instead of burning reasoning iterations.
- **Cancellation follows ownership** — `/stop` or closing/resetting the owning session cancels its background children; synchronous descendants under orchestrators follow their parent's interrupt state
- Only the final summary enters the parent's context, keeping token usage efficient
- Subagents inherit the parent's **API key, provider configuration, and credential pool** (enabling key rotation on rate limits)

## Worktree Isolation

By default, subagents share the parent's working directory — fine for research
and read-heavy work, but parallel children editing the same repo can collide.
Set `delegation.worktree_isolation: true` to give each child its own git
worktree, branched from the repo's current `HEAD` (inspired by Muse Code's
`--subagent-worktree-isolation`):

```yaml
delegation:
  worktree_isolation: true   # default: false
```

With isolation on:

- Each child starts its terminal in `<repo>/.worktrees/subagent-<id>` on its
  own branch `hermes-subagent/subagent-<id>`, and its goal message tells it to
  work and commit there.
- The parent's checkout stays untouched; children can't clobber each other's
  edits.
- When a child finishes, its result entry gains a `worktree` field reporting
  `path`, `branch`, `commits` (ahead of the base), and `dirty`. The parent
  reviews or merges each branch (`git log <branch>`, `git merge <branch>`).
- A worktree left with **no commits and a clean tree is pruned automatically**
  (`pruned: true`); anything holding work is kept.
- Pruning requires proof. If a git inspection probe fails — or finalization
  itself errors — the worktree and branch are kept and the entry carries
  `inspection_failed: true` plus a `note` — `commits`/`dirty` are then
  defaults, not measurements, so inspect the worktree rather than assuming
  the child produced nothing.

Scope: opt-in, git-only, and local-terminal-backend-only. In a non-git
directory, on docker/ssh/modal backends, or if worktree creation fails, the
setting degrades silently to today's shared-workspace behavior — never an
error.

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
  # worktree_isolation: false               # Give each child its own git worktree (see Worktree Isolation above)
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
