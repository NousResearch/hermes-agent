---
sidebar_position: 24
title: "Session Handoff Commands"
description: "Developer notes, smoke tests, and follow-up design for /handoff, /handoff-save, and /handoff-new"
---

# Session Handoff Commands

Hermes provides session handoff slash commands for moving long-running or heavily compressed conversations into a fresh session without losing the important working context.

## Commands

- `/handoff` — queues a structured `SESSION HANDOFF` prompt as the next agent turn.
- `/handoff-save` — queues the same handoff prompt and embeds guarded runtime markers so the final handoff can be saved under `$HERMES_HOME/handoffs/`.
- `/handoff-new` — generates a fresh-session transition package and a ready-to-paste prompt. It does not automatically run `/new`.

## Implementation paths

- Classic CLI: generated prompts are inserted at the front of the pending input queue so handoff runs before later queued user messages.
- Platform gateway, idle session: `/handoff*` is rewritten into the shared handoff prompt and handled as an agent turn.
- Platform gateway, active session: `/handoff*` is queued without interrupting the running agent.
- TUI gateway: `slash.exec` rejects these as pending-input commands; `command.dispatch` returns `{type: "send", message: prompt}`.

## `/handoff-save` runtime fallback

The generated prompt contains machine-readable markers:

```text
HERMES_HANDOFF_SAVE_PATH: <path>
HERMES_HANDOFF_SAVE_TOKEN: <token>
```

The runtime fallback saves the final response only when all constraints pass:

- response starts with `SESSION HANDOFF` after leading whitespace is stripped
- generated token marker is present
- target path resolves under `$HERMES_HOME/handoffs/`
- filename matches `handoff_*.md`
- existing files are not overwritten
- response size is at most 512 KiB

Gateway streaming behavior:

- If the body was not already sent, the save notice is appended to the response.
- If the body was already streamed/sent, the runtime sends a separate trailing save notice instead of resending the body.
- If queued follow-up recursion is about to run, the current handoff response is saved/notified before processing the queued follow-up.

## Safety guard

Handoff prompts must explicitly avoid quoting or revealing:

- hidden system or developer messages
- internal policy text
- secrets
- credentials
- unrelated private context

The handoff should carry only user-visible preferences, workflow constraints, decisions, file paths, commands, evidence, blockers, and next actions.

## Acceptance line for the current implementation pass

Stop implementation once all of the following are true:

1. Classic CLI registers `/handoff`, `/handoff-save`, and `/handoff-new` and queues generated prompts as next-turn input.
2. Gateway idle mode rewrites `/handoff*` into an agent-facing handoff prompt.
3. Gateway active-agent mode queues `/handoff*` without interrupting the running agent.
4. TUI gateway dispatch returns `{type: "send", message: prompt}` for `/handoff*` and keeps `slash.exec` rejection behavior.
5. `/handoff-save` deterministic fallback saves only valid `SESSION HANDOFF` responses to `$HERMES_HOME/handoffs/handoff_*.md`.
6. `/handoff-save` fallback rejects marker/path injection, requires token marker, refuses overwrites, and caps saved response size.
7. Gateway streaming/already_sent mode does not resend streamed body and still sends a trailing save-path notice.
8. Gateway queued follow-up recursion saves/notifies the current `/handoff-save` response before processing the queued follow-up.
9. Handoff prompt safety guard excludes hidden system/developer messages, internal policy text, secrets, credentials, and unrelated private context.
10. The focused regression suite passes.
11. At least one adversarial review after the final code change reports no blocker.
12. Any profile-local skill or operator note that documents this workflow reflects the shipped behavior and known limitations.

Explicitly out of scope for this implementation pass:

- automatically executing `/new` after `/handoff-new`
- full repository test suite
- production gateway restart or cutover
- perfect coverage of every platform adapter's thread metadata behavior
- filesystem race/symlink hardening beyond current guarded path/token/non-overwrite checks

## Focused test command

```bash
venv/bin/python -m pytest \
  tests/test_cli_handoff_commands.py \
  tests/gateway/test_unknown_command.py \
  tests/hermes_cli/test_commands.py \
  tests/test_cli_manual_compress.py \
  tests/test_tui_gateway_server.py \
  -q -o 'addopts='
```

## Restart smoke test checklist

Run this after restarting the relevant Hermes process so the slash registry and Python code are reloaded.

### CLI

1. Start a fresh Hermes CLI process from this checkout/profile.
2. Run `/help` or command autocomplete and verify `/handoff`, `/handoff-save`, and `/handoff-new` are visible.
3. Run `/handoff`.
   - Expected: the next agent turn receives a handoff-generation prompt.
   - Expected: response starts with `SESSION HANDOFF`.
4. Run `/handoff-save`.
   - Expected: response starts with `SESSION HANDOFF`.
   - Expected: output includes `✓ Handoff saved: <path>`.
   - Verify the file exists under `$HERMES_HOME/handoffs/handoff_*.md` and starts with `SESSION HANDOFF`.
5. Run `/handoff-new`.
   - Expected: response includes a ready-to-paste fresh-session prompt and tells the user to run `/new`.
   - Expected: it does not automatically reset the session.

### Gateway foreground, non-production

1. Start gateway in foreground for a test profile/channel only:

```bash
hermes --profile <profile> gateway run
```

2. In a permitted DM/test channel, send `/handoff`.
   - Expected: bot replies with a `SESSION HANDOFF` response.
3. Send `/handoff-save`.
   - Expected: bot replies with handoff content plus `✓ Handoff saved: ...`, or in streaming mode sends the save notice as a trailing message.
   - Verify the saved file under the profile's `handoffs/` directory.
4. While the agent is busy, send `/handoff-save`, then immediately send a normal follow-up message.
   - Expected: `/handoff-save` is queued rather than interrupting.
   - Expected: the handoff response is saved/notified before the queued follow-up result is delivered.
5. Send `/handoff-new`.
   - Expected: bot produces handoff/new-session instructions but does not reset the gateway session automatically.
6. Stop the foreground gateway after the smoke test.

### TUI gateway

1. Start a fresh TUI/CLI session after code reload.
2. Trigger `/handoff`, `/handoff-save`, and `/handoff-new` through the TUI command path.
3. Expected:
   - slash execution does not treat these as immediate local-only commands
   - command dispatch returns a sendable prompt
   - `/handoff-save` saves under `$HERMES_HOME/handoffs/` after the agent response

## `/handoff-new --apply` follow-up design

Status: design only. Keep the current `/handoff-new` manual behavior unchanged in this implementation pass.

Desired future behavior:

1. Generate and save a `SESSION HANDOFF`.
2. Prepare a fresh-session prompt from that handoff.
3. Let the user explicitly choose whether to start the new session now.
4. Preserve rollback: the old session remains searchable and the handoff file exists.
5. Avoid hidden automatic `/new` while the agent is still responding.

Recommended first target: CLI-only `--apply`.

Proposed CLI flow:

```text
/handoff-new
```

Responds with the handoff, saved path, fresh-session prompt, and an instruction such as:

```text
Run /new and paste the prompt above, or run /handoff-new --apply to start a fresh session automatically.
```

Optional future command:

```text
/handoff-new --apply
```

Expected behavior:

1. Generate and save handoff.
2. Store the fresh-session prompt in a one-shot post-reset queue.
3. Reset the session using the existing `/new` mechanism.
4. Inject the fresh-session prompt as the first pending input, or place it in the input buffer if direct injection risks role alternation.

Safety gates for implementation:

- Never auto-apply in group/channel sessions unless explicitly configured.
- Never discard queued messages silently.
- Always save handoff before reset.
- If saving fails, do not reset automatically.
- If the response does not start with `SESSION HANDOFF`, do not reset automatically.

Acceptance criteria for a future implementation phase:

1. `/handoff-new` manual behavior remains unchanged.
2. `/handoff-new --apply` works in classic CLI first.
3. Handoff file is saved before reset.
4. New session starts with the generated fresh-session prompt or the prompt is queued as first input.
5. Save failure prevents reset.
6. Non-handoff response prevents reset.
7. Tests cover the happy path, save failure, non-handoff response, and queued user message behavior.
8. Gateway apply mode remains deferred unless separately scoped.
