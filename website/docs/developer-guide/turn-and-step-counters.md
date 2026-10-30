---
sidebar_position: 19
title: "Turn and Step Counters"
description: "Where per-turn and per-API-call counters live, how they reach hooks, and what a session persists"
---

# Turn and Step Counters

Hermes counts two different things that are easy to conflate: a **turn** (one user
message and everything the agent does in response) and a **step** (one API call
inside a turn's tool loop). They have different owners, different lifetimes, and
only one of them is persisted with a dedicated column. This page maps each counter
to its update point, its consumers, and the surfaces that expose it.

## At a glance

| Counter | Scope | Owner | Written at | Persisted |
|---|---|---|---|---|
| `_user_turn_count` | Session (per agent process) | `AIAgent` | `agent/turn_context.py:1132` | No — rehydrated from history |
| `api_call_count` (loop local) | Turn | `TurnState` / `IterationBudget` | `agent/turn_iteration_prep.py:406-407` | No |
| `agent._api_call_count` | Agent, mirrored for status surfaces | `AIAgent` | Same increment (+ refunds) | No |
| `agent.session_api_calls` | Session | `AIAgent` | `agent/turn_usage.py:85` | Only via the DB delta below |
| `sessions.api_call_count` | Session, durable | SessionDB | Per-call delta queue (`agent/turn_usage.py:264-281`) | Yes |
| `session_model_usage.api_call_count` | Session × model route | SessionDB | Upsert (`hermes_state_usage.py:55-62`) | Yes |

## User turns: `agent._user_turn_count`

`_user_turn_count` counts user messages the agent has processed in this
conversation. It starts at zero in the constructor (`run_agent.py:406`, also
listed in `agent/agent_init.py:2120`) and is incremented exactly once per turn,
after the user row is staged and appended to the transcript and before
compression or the tool loop runs (`agent/turn_context.py:1132`).

It is process-local. On a resumed session the first turn rehydrates it from the
persisted transcript by counting `role == "user"` rows (`agent/turn_context.py:729-762`),
which also restores the memory-nudge cadence (`_turns_since_memory`).

Consumers:

- The memory nudge counter `_turns_since_memory` (`agent/turn_context.py:770`).
- External memory providers, through `on_turn_start(turn_number, message, ...)`
  (`agent/turn_context.py:928` → `agent/memory_manager.py:613`). This is the only
  place the numeric turn index leaves the process.

There is no `turn_count` column in the session database and no turn number on
the hook bus. Anything that needs a turn total derives it from the transcript;
`hermes_cli/session_recap.py:66-70` (`_count_visible_turns`) is the canonical
example.

## Steps: the turn-local `api_call_count`

`api_call_count` is the number of provider calls the current turn has started. It
lives on the loop state (`agent/conversation_loop.py:1484`), is checked against
`max_iterations` in the loop condition (`agent/conversation_loop.py:1729`), and
is incremented in `begin_iteration`, immediately before the request is assembled
(`agent/turn_iteration_prep.py:406-407`). That same statement mirrors the value
onto `agent._api_call_count` for status surfaces (`run_agent.py:863-864`).

The loop cap has two halves: the local call count and the per-turn
`IterationBudget`, rebuilt at every turn start
(`agent/turn_context.py:664`). `begin_iteration` consumes one budget slot
(`agent/turn_iteration_prep.py:416`). Iterations that never reached the provider
— preflight compactions, provider-switch hops — are refunded, which decrements
both the local count and the budget in lockstep
(`agent/turn_context_compaction.py:103-118`).

When the cap is hit, the turn exits with a
`max_iterations_reached(N/M)` reason and, unless a pending answer is preserved,
asks the model for a summary (`agent/turn_finalizer.py:130-163`).

The count is 1-based: by the time `step_callback` fires, `begin_iteration` has
already run, so the first step is 1 (phase order in
`agent/conversation_loop.py:1731-1733`).

## What hooks see

**Steps are exposed; turn counts are not.**

| Surface | Event | Field | Semantics |
|---|---|---|---|
| Gateway shell/outbound hooks | `agent:step` | `iteration` | Current turn's step, 1-based (`gateway/run_turn_progress.py:883-899`) |
| Plugin lifecycle | `pre_api_request`, `post_api_request`, `api_request_error` | `api_call_count` | Current turn's step |
| Plugin streaming observers | `on_stream_start` / `on_stream_delta` / `on_stream_end` / `on_interim_message` | `iteration` | `agent._api_call_count` (`agent/stream_delivery.py:316-324`) |
| Plugin lifecycle | `pre_auxiliary_call`, `post_auxiliary_call` | `api_call_count` | **That auxiliary call's own attempt number**, not the parent turn's step (`agent/auxiliary_hooks.py:106-107`) |
| Plugin lifecycle | `pre_llm_call`, `post_llm_call`, `on_session_end` | `turn_id`, `is_first_turn` | Opaque identity and a boolean; no numeric turn count |
| Gateway shell/outbound hooks | `agent:start`, `agent:end` | — | No counters |

`agent.step_callback(api_call_count, prev_tools)` is the internal callback behind
`agent:step` (`agent/turn_iteration_prep.py:158-161`); the gateway binds it at
`gateway/run_turn_runner.py:472`, the ACP adapter at `acp_adapter/events.py:259`.
The exact payload contract is documented in the hooks reference
(`user-guide/features/hooks`).

## What a session persists

One column stores the durable step total: `sessions.api_call_count`
(`hermes_state_common.py:507`). It is accumulated by a per-call delta —
`api_call_count=1` queued by `queue_token_counts`
(`agent/turn_usage.py:264-281`) and applied through
`_update_token_counts_in_transaction` using the delta SQL that does
`api_call_count = COALESCE(api_call_count, 0) + ?`
(`hermes_state_usage.py:36-48`). Per-route attribution lands in `session_model_usage`
(`hermes_state_schema.py:85`). Main-loop rows carry `task = ''`; the latest route is
read with `get_recent_session_model_route` (`hermes_state_sessions.py:855-877`).
Auxiliary calls are a separate ledger: they are written only to
`session_model_usage` under a non-empty `task` and never touch the `sessions` row
(`hermes_state_usage.py:392-424`), then summed per task by
`auxiliary_usage_by_task` (`hermes_state_usage.py:427-449`, `WHERE task != ''`).

The in-memory counterpart is `agent.session_api_calls`
(`agent/agent_init.py:2131`, incremented at `agent/turn_usage.py:85` and
`agent/codex_runtime.py:155`). It is what `/usage` renders as "API calls"
(`gateway/slash_commands_status.py:215`), what the status bar tracks, and what
the local viewer receives as `session.usage.api_calls`
(`gateway/session_usage.py:20`).

### The two step totals can diverge

The chat-completions path queues its DB delta only when the response actually
carries usage: a usage-less response increments `session_api_calls` and then
returns before the queue call (`agent/turn_usage.py:85-104` vs `:264-281`). The
codex app-server path is the opposite — a usage-less turn still queues
`api_call_count=1` (`agent/codex_runtime.py:166-168`). So on the main path the
persisted `sessions.api_call_count` can lag the in-memory counter by the number
of usage-less attempts; on the codex path it does not. Failed and interrupted
calls count in neither place.

## Practical notes

- Use `iteration` / `api_call_count` for progress; do not treat
  `sessions.api_call_count` as "tool iterations" — it counts provider attempts
  that returned a response, not loop budget consumption.
- There is no persisted turn total. Count `role == "user"` rows, or keep your own
  counter; `message_count` includes assistant and tool rows and is not a turn
  count.
- Treat `turn_id` as opaque. It is an identity (and Relay/lease key), not an
  ordinal.

The scheduling design that consumes these counters (turn/step task engine, step
ledger) builds on this page: see [Scheduled Tasks](./scheduled-tasks.md).
