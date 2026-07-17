---
sidebar_position: 9
sidebar_label: "History Timeline"
title: "History Timeline and conversation branching"
description: "Use /history in the TUI to jump to a previous prompt or answer and continue in a non-destructive conversation branch."
---

# History Timeline and conversation branching

The TUI `/history` command opens a **History Timeline** overlay for the current conversation. Use it when you want to revisit an earlier prompt or answer, then continue from that point without overwriting the original session.

This is a **conversation branch** feature. It changes which chat transcript Hermes continues from; it does **not** automatically roll back files in your working directory.

## Open the timeline

In the TUI, type:

```text
/history
```

You can also press `Ctrl+H` in the TUI to open the same History Timeline overlay without typing the slash command.

The overlay lists the current session's messages in order. User and assistant messages are actionable; system and tool rows may appear as context-only rows and are marked `(view)`.

The right side shows the full text for the selected row, including readable placeholders for empty assistant messages that only contain reasoning or tool calls.

## Keyboard controls

| Key | Action |
|-----|--------|
| `↑` / `k` | Move to the previous timeline row |
| `↓` / `j` | Move to the next timeline row |
| `PgUp` / `PgDn` | Page up or down through long histories |
| `g` / `G` | Jump to the first / last row |
| `/` | Filter rows by role plus message text in the current session only |
| `n` / `N` | While filtering, move to the next / previous match |
| `Tab` / `Shift+Tab` | While filtering, move to the next / previous match |
| `Enter` | Run the default branch action for the selected persisted user/assistant message |
| `e` or `b` | Branch from the selected persisted user/assistant message |
| `r` | Retry from the selected persisted user message in a new branch |
| `c` | Copy the selected actionable message text |
| `Esc` or `q` | Clear the active filter; press again to close the overlay |

Filtering is intentionally scoped to the **current TUI session timeline**. It is not a cross-session search. Use session browsing or resume commands when you need to find another saved conversation.

## Branch actions and semantics

The selected message role controls what branching means.

### User prompt: Edit & branch

Select an earlier user prompt and press `Enter`, `e`, or `b` to open a new branch from that prompt's point in the transcript.

For a user prompt, the branch starts from the conversation context before that prompt and uses the selected prompt as the draft continuation. This lets you edit the prompt and explore a different path without deleting the original session.

### User prompt: Retry in branch

Select a user prompt and press `r` to retry that prompt in a new branch.

Hermes keeps the conversation up to and including the selected user message, drops the later assistant/tool results in the new branch, switches to that branch, and submits the same prompt again. Use this when you want a fresh answer to the same prompt while preserving the original answer in the parent session.

### Assistant answer: Branch after this answer

Select an assistant answer and press `Enter`, `e`, or `b` to continue after that answer in a new branch.

Hermes keeps the conversation up through the selected assistant answer, switches to the new branch, and waits for your next prompt. It does not regenerate the selected answer automatically.

## What branches preserve

Branches are non-destructive:

- The parent conversation remains saved and resumable.
- The new branch records parent lineage so you can tell where it came from.
- Branching requires a persisted user or assistant message. If a row only exists in the transient live transcript, Hermes will ask you to use a persisted row.
- Tool and system rows are visible for context, but they are not direct branch targets.

## Filesystem boundary: not a coding-world rollback

History Timeline branching only changes the **conversation history** that Hermes replays into the next agent turn. It does not automatically restore files, undo shell commands, reset git state, or recreate a previous working directory.

If you need the codebase or filesystem to match the historical conversation point, combine branching with one of these workflows:

- [`/rollback`](checkpoints-and-rollback) or `hermes chat --checkpoints` for Hermes filesystem checkpoints.
- A git commit, branch, or worktree for explicit source-control isolation.
- `hermes --worktree` for parallel coding sessions that should not share one working tree.

Think of `/history` as "continue the chat from here," not "restore my project to here."

## Quick workflow

1. Run `/history` in the TUI.
2. Move to the old prompt or answer with `↑` / `↓`, `j` / `k`, or `PgUp` / `PgDn`.
3. Optionally press `/` and type a phrase to filter the current session's timeline.
4. Press:
   - `r` on a user prompt for **Retry in branch**.
   - `e`, `b`, or `Enter` on a user prompt for **Edit & branch**.
   - `Enter`, `e`, or `b` on an assistant answer for **Branch after this answer**.
5. Continue in the new branch. Resume the parent later from the session list if needed.
