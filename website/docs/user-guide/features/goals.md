---
sidebar_position: 16
title: "Persistent Goals"
description: "Set a standing goal and let Hermes keep working across turns until it's done. Our take on the Ralph loop."
---

# Persistent Goals (`/goal`)

`/goal` gives Hermes a standing objective that survives across turns. The same primary model that performs the work records an exact structured `goal_outcome` before it ends each turn. Hermes consumes that record mechanically and feeds a continuation prompt back into the same session when more work remains — until the model reports verified completion, exhausts every safe approach and reports a genuine blocker, you pause or clear the goal, or the turn budget runs out.

It's our take on the **Ralph loop**, directly inspired by [Codex CLI 0.128.0's `/goal`](https://github.com/openai/codex) by Eric Traut (OpenAI). The core idea — keep a goal alive across turns and don't stop until it's achieved — is theirs. The implementation here is independent and adapted to Hermes' architecture.

## When to use it

Use `/goal` for tasks where you want Hermes to iterate on its own without you re-prompting every turn:

- "Fix every lint error in `src/` and verify `ruff check` passes"
- "Port feature X from repo Y, including tests, and get CI green"
- "Investigate why session IDs sometimes drift on mid-run compression and write up a report"
- "Build a small CLI to rename files by their EXIF dates, then test it against the photos/ folder"

Tasks where the agent does one turn and stops don't need `/goal`. Tasks where *you'd otherwise have to say "keep going" three times* are where this shines.

## Quick start

```
/goal Fix every failing test in tests/hermes_cli/ and make sure scripts/run_tests.sh passes for that directory
```

What you'll see:

1. **Goal accepted** — `⊙ Goal set (20-turn budget): <your goal>`
2. **Turn 1 runs** — Hermes starts working as if you'd sent the goal as a normal message.
3. **The primary model records the outcome** — `continue`, `complete`, or `blocked`, with its reason and evidence, through the `todo` tool.
4. **Loop fires if needed** — if the exact-turn outcome is `continue` (or is missing/invalid), you'll see `↻ Continuing toward goal (1/20): <model's reason>` and Hermes takes the next step automatically.
5. **Terminates** — eventually you see either `✓ Goal achieved: <reason>` or `⏸ Goal paused — N/20 turns used`.

## Commands

| Command | What it does |
|---|---|
| `/goal <text>` | Set (or replace) the standing goal. Kicks off the first turn immediately so you don't need to send a separate message. |
| `/goal draft <text>` | Draft a structured completion contract from a plain-language objective, then set it. See [Completion contracts](#completion-contracts). |
| `/goal show` | Print the active goal's completion contract. |
| `/goal` or `/goal status` | Show the current goal, its status, and turns used. |
| `/goal pause` | Stop the auto-continuation loop without clearing the goal. |
| `/goal resume` | Resume the loop (resets the turn counter back to zero). |
| `/goal clear` | Drop the goal entirely. |
| `/goal wait <pid> [reason]` | Park the loop on a background process — it stops re-poking the agent every turn while the process runs, and auto-resumes when it exits. |
| `/goal unwait` | Drop the wait barrier and resume the loop immediately. |

Works identically on the CLI and every gateway platform (Telegram, Discord, Slack, Matrix, Signal, WhatsApp, SMS, iMessage, Webhook, API server, and the web dashboard).

## Completion contracts

A bare `/goal <text>` works fine, but a *vague* goal gives the primary model a vague completion boundary. Codex's `/goal` guidance makes the same point: a durable objective works best when it names **what done means, how to prove it, what not to break, what's in scope, and when to stop**. Hermes adapts this as an optional **completion contract** layered on top of the existing goal loop.

A contract has five fields, all optional:

| Field | Meaning |
|---|---|
| `outcome` | The single end state that must be true when done. |
| `verification` | The specific test / command / artifact that *proves* the outcome. |
| `constraints` | What must not change or regress. |
| `boundaries` | Which files, dirs, tools, or systems are in scope. |
| `stop_when` | The condition under which Hermes should stop and ask for input. |

When a contract is set, the kickoff and continuation prompts tell the primary model to target the verification surface and respect the constraints. The model must record `complete` *only when the verification criterion is met with concrete evidence* (a command result, file excerpt, test output) — not a loose "looks done" claim. This directly tightens the most common `/goal` failure mode (premature completion or endless over-continuation on an underspecified objective).

### Two ways to set a contract

**1. Let Hermes draft it** (recommended — adapted from Codex's "let the agent draft the goal" tip):

```
/goal draft Migrate the auth service from session cookies to JWT
```

Hermes asks the same primary model to author the full contract through the `todo` tool, create a concrete plan, and begin the first safe step immediately. There is no auxiliary planner or completion judge: the model with the full task context remains the sole semantic authority.

**2. Write it inline** with `field: value` lines:

```
/goal Migrate auth to JWT
verify: pytest tests/auth passes
constraints: keep the /login response shape unchanged
boundaries: only touch services/auth and its tests
stop when: a DB schema migration is required
```

The first non-field line(s) are the goal headline; recognized field prefixes (`verify:`, `verified by:`, `constraints:`, `preserve:`, `boundaries:`, `scope:`, `stop when:`, `blocked:`, …) populate the contract. A plain goal with an incidental colon (`Fix bug: the parser drops commas`) is **not** mangled — only known field prefixes are pulled out.

Use `/goal show` to review the active contract. Contracts persist in `SessionDB.state_meta` alongside the goal, so they survive `/resume`. Old goals from before this feature load unchanged (no contract). Contracts and `/subgoal` criteria compose: subgoals become extra criteria the primary model must satisfy before it records completion.

## Adding criteria mid-goal: `/subgoal`

While a goal is active you can append extra acceptance criteria with `/subgoal <text>` without resetting the loop. Each call adds one numbered item to the goal's subgoal list; the **continuation prompt** the agent sees on the next turn includes the original goal plus an "Additional criteria the user added mid-loop" block. The primary model must consider every subgoal before recording `complete` — the goal isn't marked done until the original objective **and** every subgoal are met.

| Command | What it does |
|---|---|
| `/subgoal <text>` | Append a new criterion to the active goal. Requires an active `/goal`. |
| `/subgoal` (no args) | Show the current numbered subgoal list. |
| `/subgoal remove <N>` | Remove the Nth subgoal (1-based). |
| `/subgoal clear` | Drop every subgoal but keep the original goal intact. |

Subgoals are persisted alongside the goal in `SessionDB.state_meta`, so they survive `/resume`. Setting a new `/goal <text>` replaces the goal and clears the subgoal list; `/goal clear` does the same.

Use this when you start a loop ("fix the failing tests") and notice partway through that you also want it to "and add a regression test for the bug you just patched" — `/subgoal add a regression test` tightens the success criteria without breaking the running loop.

## Parking on a background process

Some goals are gated on something that takes minutes and runs on its own — CI on a pushed PR, a long build, a test matrix, or a deploy. Use an explicit wait barrier instead of spending continuation turns polling it. The barrier is mechanical control-plane state; Hermes does not infer waiting from response text or ask another model to classify the situation.

| Command | What it does |
|---|---|
| `/goal wait <pid> [reason]` | Park the loop until the process with that PID exits. |
| `/goal unwait` | Clear the wait barrier and resume immediately. |

The barrier is persisted with the goal in `SessionDB.state_meta`, so it survives `/resume`. `/goal pause`, `/goal resume`, and `/goal clear` all drop it. If the PID is already dead when the barrier is set (or dies while parked), the barrier clears on the next check — a stale barrier cannot wedge the loop.

Typical flow: Hermes pushes a PR, starts a CI watcher in the background, records its PID, and parks the goal on that PID. The loop stays quiet until the watcher exits, then the same primary model resumes with the actual result.

## Behavior details

### Model-authored outcome

Before ending each goal turn, the primary model records one structured outcome through the existing `todo` tool:

- `continue` — more work remains; Hermes feeds the next continuation prompt.
- `complete` — the objective and all criteria are satisfied with concrete verification.
- `blocked` — every safe available approach was exhausted and specific user or external input is genuinely required.

Hermes accepts only an outcome bound to the exact active model turn and goal generation. The runtime validates and applies that record mechanically; it never parses prose, searches for keywords, or invokes an auxiliary classifier to decide whether the task is done.

### Fail-open semantics

If the exact-turn model outcome is missing, stale, malformed, or invalid, Hermes treats it as `continue`. Bookkeeping therefore cannot invent completion or silently wedge progress. The **turn budget** is the backstop.

### Turn budget

Default is 20 continuation turns (`goals.max_turns` in `config.yaml`). When the budget is hit, Hermes auto-pauses and tells you exactly how to proceed:

```
⏸ Goal paused — 20/20 turns used. Use /goal resume to keep going, or /goal clear to stop.
```

`/goal resume` resets the counter to zero, so you can keep going in measured chunks.

### User messages always preempt

Any real message you send while a goal is active takes priority over the continuation loop. On the CLI your message lands in `_pending_input` ahead of the queued continuation; on the gateway it goes through the adapter FIFO the same way. The primary model can record the updated outcome in that same turn.

### Mid-run safety (gateway)

While an agent is already running, `/goal status`, `/goal pause`, `/goal clear`, `/goal wait`, and `/goal unwait` are safe to run — they only touch control-plane state and don't interrupt the current turn. Setting a **new** goal mid-run (`/goal <new text>`) is rejected with a message telling you to `/stop` first, so the old continuation can't race the new one.

### Persistence

Goal state lives in `SessionDB.state_meta` keyed by `goal:<session_id>`. That means `/resume` picks up right where you left off — set a goal, close your laptop, come back tomorrow, `/resume`, and the goal is still standing exactly as you left it (active, paused, or done).

### Prompt cache

The continuation prompt is a plain user-role message appended to history. It does **not** mutate the system prompt, swap toolsets, or touch the conversation in any way that invalidates Hermes' prompt cache. Running a 20-turn goal costs the same cache-wise as 20 turns of normal conversation.

## Configuration

Add to `~/.hermes/config.yaml`:

```yaml
goals:
  # Max continuation turns before Hermes auto-pauses and asks you to
  # /goal resume. Default 20. Lower this if you want tighter loops;
  # raise it for long-running refactors.
  max_turns: 20
```

## Example walkthrough

```
You: /goal Create four files /tmp/note_{1..4}.txt, one per turn, each containing its number as text

  ⊙ Goal set (20-turn budget): Create four files /tmp/note_{1..4}.txt, one per turn, each containing its number as text

Hermes: Creating /tmp/note_1.txt now.
  💻 echo "1" > /tmp/note_1.txt   (0.1s)
  I've created /tmp/note_1.txt with the content "1". I'll continue with the remaining files on the next turn as you specified.

  ↻ Continuing toward goal (1/20): Only 1 of 4 files has been created; 3 files remain.

Hermes: [Continuing toward your standing goal]
  💻 echo "2" > /tmp/note_2.txt   (0.1s)
  Created /tmp/note_2.txt. Two more to go.

  ↻ Continuing toward goal (2/20): 2 of 4 files created; 2 remain.

Hermes: [Continuing toward your standing goal]
  💻 echo "3" > /tmp/note_3.txt   (0.1s)
  Created /tmp/note_3.txt.

  ↻ Continuing toward goal (3/20): 3 of 4 files created; 1 remains.

Hermes: [Continuing toward your standing goal]
  💻 echo "4" > /tmp/note_4.txt   (0.1s)
  All four files have been created: /tmp/note_1.txt through /tmp/note_4.txt, each containing its number.

  ✓ Goal achieved: All four files were created with the specified content, completing the goal.

You: _
```

Four turns, one `/goal` invocation, zero "keep going" prompts from you.

## Correcting an outcome

The primary model has the full task context, but you remain in control. If it keeps recording `continue` after the work is done, the turn budget pauses the loop and you can `/goal clear` or send a new message. If it records `complete` too early, send a follow-up or set a more precise completion contract. The reason shown in `↻ Continuing toward goal`, `✓ Goal achieved`, or `⏸ Goal blocked` is the model-authored reason attached to the structured outcome.

## Attribution

`/goal` is Hermes' take on the **Ralph loop** pattern. The user-facing design — keep a goal alive across turns, don't stop until it's achieved, with create/pause/resume/clear controls — was popularised and shipped in [Codex CLI 0.128.0](https://github.com/openai/codex) by Eric Traut on OpenAI's Codex team. Our implementation is independent (central `CommandDef` registry, `SessionDB.state_meta` persistence, exact-turn model-authored outcomes, and adapter-FIFO continuation on the gateway side) but the idea is theirs. Credit where credit's due.
