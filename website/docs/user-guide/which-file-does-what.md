---
sidebar_position: 4
title: "Which File Does What?"
description: "SOUL.md vs USER.md vs MEMORY.md vs AGENTS.md — a one-page map of the agent's files, who writes each one, and when the agent actually sees them"
---

# Which File Does What?

"I told my agent something and it forgot." "Which file is my agent's brain?" "I edited SOUL.md — why doesn't it know my name?" These questions all come down to the same thing: Hermes Agent is shaped by several markdown files, and each one has a different job. This page maps them all in one place. For depth on any of them, follow the links to [Persistent Memory](/user-guide/features/memory), [Personality & SOUL.md](/user-guide/features/personality), and [Context Files](/user-guide/features/context-files).

## The Master Table

| File | What it holds | Who writes it | When the agent sees it | Where it lives |
|------|---------------|---------------|------------------------|----------------|
| **SOUL.md** | The agent's primary identity — personality, tone, communication style, what to avoid stylistically | You. Hermes seeds a starter file automatically if one doesn't exist; existing files are never overwritten | Slot #1 of the system prompt, at session start | `~/.hermes/SOUL.md` (or `$HERMES_HOME/SOUL.md` with a custom home) — never the working directory |
| **USER.md** | User profile — your name, role, preferences, communication style, expectations | The agent, via the `memory` tool (you can gate saves with `write_approval`, or edit entries via `hermes journey edit`) | Injected into the system prompt as a frozen snapshot at session start | `~/.hermes/memories/` |
| **MEMORY.md** | Agent's personal notes — environment facts, project conventions, tool quirks, things learned | The agent, via the `memory` tool (same gating and editing options as USER.md) | Injected into the system prompt as a frozen snapshot at session start | `~/.hermes/memories/` |
| **AGENTS.md** | Project instructions, conventions, architecture — commands, ports, paths, repo-specific workflows | You (or whoever authors the project) | Loaded into the system prompt at startup from your working directory; nested copies are discovered progressively as the agent navigates subdirectories | Project working directory + subdirectories |
| **.hermes.md** / **HERMES.md** | Project instructions, like AGENTS.md but Hermes-specific and highest priority | You | Loaded into the system prompt at startup (first match wins over AGENTS.md) | Your project — discovery walks up to the git root |

:::info One project context file per session
Only **one** project context type is loaded per session, first match wins: `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`. `SOUL.md` is always loaded independently as the agent identity — it is not part of that priority chain. See [Context Files](/user-guide/features/context-files) for the full list, including `CLAUDE.md` and `.cursorrules` compatibility.
:::

A useful shorthand:

- **SOUL.md** is who the agent *is* — if it should follow you everywhere, it belongs here.
- **USER.md** is who *you* are — the agent maintains it for you.
- **MEMORY.md** is what the agent has *learned* — it maintains this itself too.
- **AGENTS.md** (or `.hermes.md`) is what the *project* needs — if it belongs to a project, it belongs here.

## "Why did it forget what I just said?"

Memory (MEMORY.md and USER.md) is injected into the system prompt as a **frozen snapshot** captured once at session start — when the agent saves something mid-session, the change is persisted to disk immediately but won't appear in the system prompt until the next session starts. This is intentional: it preserves the LLM's prefix cache for performance, and tool responses always show the live state, so nothing is lost — start a new session and the updated memory is there. Full details in [How Memory Appears in the System Prompt](/user-guide/features/memory#how-memory-appears-in-the-system-prompt).

## Common Mix-Ups

### "I put facts about myself in SOUL.md, but USER.md stayed empty"

`SOUL.md` and `USER.md` are separate systems that never feed each other. `SOUL.md` is a personality file **you** edit directly — it shapes tone and identity, and its content is injected verbatim as slot #1 of the prompt. `USER.md` is part of persistent memory and is written by **the agent** through the `memory` tool. If you want facts about yourself in USER.md, tell the agent ("remember that I prefer concise answers") and it saves them — editing SOUL.md won't populate memory, and memory entries won't change the persona. Use SOUL.md for durable voice and personality guidance; leave preferences and profile facts to memory. See [What should go in SOUL.md?](/user-guide/features/personality#what-should-go-in-soulmd) and [Two Targets Explained](/user-guide/features/memory#two-targets-explained).

### "I told it my name mid-session and it acted like it never heard it"

If the agent saved your name to memory, the save worked — check with the `memory` tool's responses or `hermes journey list`. What you're seeing is the frozen-snapshot rule above: the system prompt doesn't refresh mid-session, so the *injected* memory block still shows the session-start state. The agent can still use what you told it within the current conversation (it's in the context), and the saved entry will be in the system prompt from the next session onward. The same applies to edits you make to `SOUL.md` or `AGENTS.md` while a session is running: context is assembled at session start, so restart the session to pick up changes.

:::tip Quick decision guide
- Want to change how the agent **talks**? Edit `~/.hermes/SOUL.md` — [Personality & SOUL.md](/user-guide/features/personality).
- Want the agent to **remember a fact**? Just tell it — it saves to memory itself. [Persistent Memory](/user-guide/features/memory).
- Want to set **project rules**? Put an `AGENTS.md` (or `.hermes.md`) in the project — [Context Files](/user-guide/features/context-files).
- Need a **temporary** personality change? Use `/personality` — it's a session-level overlay, no file edits needed.
:::

## Related Docs

- [Persistent Memory](/user-guide/features/memory) — MEMORY.md, USER.md, the `memory` tool, capacity limits, `write_approval`
- [Personality & SOUL.md](/user-guide/features/personality) — SOUL.md content guidance, `/personality` presets, the prompt stack
- [Context Files](/user-guide/features/context-files) — AGENTS.md, `.hermes.md`, progressive discovery, security scanning
