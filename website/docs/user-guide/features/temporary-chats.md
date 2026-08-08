---
title: "Temporary Chats"
description: "Private conversations that leave nothing on disk — no transcript, no memory, no trace"
---

# Temporary Chats

A temporary chat runs like any other conversation, but persists nothing: no
session record, no message history, no AI-generated title, no memory
extraction, and no way to resume it later. When it ends, it is gone.

Temporary is not the same as encrypted. Messages still travel to your model
provider like every other chat — a temporary chat is simply **not written
down** on your machine.

## Starting one

| Surface | Start | End |
|---|---|---|
| Desktop | **New temporary session** in the sidebar, or `mod+alt+n` | Start any normal session |
| CLI (interactive) | `/temp` — the prompt becomes an incognito `temp >` | `/temp off` |
| CLI (one-shot) | `hermes --no-session -q "..."` | ends with the run |
| Chat platforms | `/temp` | `/temp off` (`/temp status` to check) |

Entering and leaving temporary mode always starts a fresh conversation: the
preceding chat is closed normally, and the temporary transcript is discarded
rather than carried over.

On chat platforms — which have no persistent badge — the agent restates that
nothing is being saved every 10th reply, and again on its first reply after a
gateway restart. Temporary mode survives restarts: a chat you marked
temporary can never silently downgrade to a saved one.

## What is not saved

- **No session or message rows** in the session store — including the empty
  billing row other features would create. Token accounting for a temporary
  chat is not persisted.
- **No title.** An AI-generated title is a durable summary of a conversation
  you asked us not to keep.
- **No memory.** End-of-session extraction, per-turn syncing to memory
  providers, and conversation trajectories are all skipped.
- **No self-improvement.** The background review that normally distills
  conversations into memory and skill updates (including `/refine`) does
  not run on a temporary chat — its entire output would be durable state
  derived from the conversation.
- **No resumability.** `/resume` has nothing to find.
- **Minimal logs.** The agent's operational log records that a turn happened,
  but the message preview is redacted.

## What is blocked

Anything that would let the conversation outlive itself through a durable
side door is refused at the tool level, with an explanation:

- **Memory writes** — adding, replacing, or removing memory entries,
  including batched operations.
- **Skill changes** — creating, editing, or deleting skills and their files.
- **Cron jobs** — creating, updating, pausing, resuming, or removing jobs.
  Triggering an existing job (`run`) is also blocked, because the run
  executes as a normal saved session and would carry this conversation's
  text into its transcript.
- **External memory-provider writes** — provider tools that store to
  services like Hindsight, Supermemory, Mem0, OpenViking, RetainDB,
  ByteRover, or Honcho are refused. Providers that don't declare which of
  their tools are read-only have all their tools blocked, on purpose.
- **Delegated subagents inherit all of this.** A subagent spawned from a
  temporary chat is itself temporary: its session is not persisted and the
  same write blocks apply.

## What still works

Reading is never blocked — a temporary chat that couldn't use your memory or
skills would be a degraded assistant, not a private one:

- Memory recall and provider context injection
- Listing and viewing skills, listing cron jobs, searching provider memory
- Every other tool: terminal, file editing, web search, browsing, code
  execution, image generation

## Honest boundaries

A temporary chat controls what **Hermes** writes down, not what you ask it
to do:

- **File edits and terminal commands are real.** If you ask a temporary chat
  to edit a file or run a command, that change persists — that is the point
  of tools. The conversation about it is what disappears.
- **Background processes you start keep running** after the chat ends.
- **Provider read queries still leave your machine** — recalling memory from
  an external provider sends the query to that service, like in any chat.
