---
sidebar_position: 27
title: "Troubleshooting: \"My Agent Feels Dumber\""
description: "A diagnostic checklist for when Hermes seems less capable than before or forgets things mid-session — model switches, context pressure, wrong context detection, and the frozen memory snapshot"
---

# Troubleshooting: "My Agent Feels Dumber"

Sometimes Hermes seems less sharp than it was yesterday, or forgets something you told it twenty minutes ago. This is almost never mysterious — there's usually one specific, checkable cause. Work through this checklist in order: the steps are sorted by how often each one turns out to be the answer.

## 1. Check which model the session is actually using

**Symptom:** Answers are shallower, code quality dropped, reasoning feels off — across the board.

**Check:** Run `/model` with no arguments to show the current model, or `/status` to see the session's model, provider, and profile in one view.

**What it means:** A model switch is a capability change, and it's easy to end up on a different model than you think:

- A plain `/model <name>` switch is **session-only** by default (unless `model.persist_switch_by_default: true` is set), so the model you're on may not match what's in `config.yaml`.
- Changing the main model from the dashboard's Models page applies to **new sessions only** — an already-open chat keeps running whatever model it started with.
- If you switched to a faster model for a simple task (a pattern [Tips & Best Practices](/guides/tips#choose-the-right-model) recommends), remember to switch back for complex reasoning work.

If the model is wrong, `/model <name>` fixes it for this session; add `--global` to persist the change to `config.yaml`. Note that a mid-session switch resets the prompt cache, so the next turn re-reads the conversation at full input price — on a long session it can be cheaper to start fresh on the right model.

## 2. Check context usage

**Symptom:** The session started strong but responses are slowing down, getting truncated, or losing track of earlier details.

**Check:** Run `/usage` to see token usage and context window state, or `/context` for a visual breakdown of what's occupying the window (system prompt, tool definitions, skills, memory, conversation) versus free space.

**What it means:** Extended conversations accumulate messages and tool outputs, approaching context limits. When you notice degradation in a long session:

```bash
# Compress the conversation (summarizes history, preserves key context)
/compress

# Or start a fresh session
/new
```

`/compress` summarizes the conversation history, dramatically reducing token count while preserving key context. `/compress here [N]` keeps the most recent N exchanges verbatim and summarizes the rest, and a focus topic (`/compress focus <topic>`) narrows what a full summary preserves.

:::tip
Use `/compress` regularly during long sessions rather than waiting for problems, and `/usage` periodically to see where you stand.
:::

## 3. Verify the detected context length

**Symptom:** Context problems appear surprisingly early — the first long conversation already hits limits, or compression fires far sooner than the model's advertised window should allow.

**Check:** Look at the CLI startup line — it shows the detected context length (e.g., `📊 Context limit: 128000 tokens`). You can also check with `/usage` during a session.

**What it means:** Hermes may have auto-detected the wrong context length for your model. Set it explicitly:

```yaml
# In ~/.hermes/config.yaml
model:
  default: your-model-name
  context_length: 131072  # your model's actual context window
```

Or for custom endpoints, per-model on the provider entry:

```yaml
providers:
  my-server:
    api: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 64000
```

Ollama users: if you set a custom `num_ctx`, set the matching context length in Hermes — Ollama's `/api/show` reports the model's *maximum* context, not the effective `num_ctx` you configured. On a running gateway, edits to `model.context_length` or any `compression.*` key take effect on the next message — no restart needed.

See [Context Length Detection](/integrations/providers#context-length-detection) for how auto-detection works and all override options.

## 4. "I told it something and it forgot" — the frozen memory snapshot

**Symptom:** You asked Hermes to remember something during this session, it confirmed the save, but later in the *same* session it doesn't seem to know it.

**Check:** Nothing is broken — check the timing. Memory saved mid-session is written to disk immediately, but the system prompt won't reflect it until the next session.

**What it means:** This is documented, intentional behavior. Memory is injected into the system prompt as a **frozen snapshot at session start**, and that injection never changes mid-session — it preserves the LLM's prefix cache for performance. When the agent adds or removes memory entries during a session, the changes persist to disk right away but appear in the system prompt only when the next session starts. Tool responses always show the live state, so the save itself is confirmed and real.

:::info
Frozen snapshot in practice: "remember X" during a session means X is guaranteed available **next** session. Within the current session, the fact still exists in the conversation history itself — the agent forgets it only if that part of the conversation has since been compressed away (see step 7).
:::

See [Persistent Memory](/user-guide/features/memory#how-memory-appears-in-the-system-prompt) for the full mechanics.

## 5. Memory is bounded and curated — not a transcript

**Symptom:** Hermes doesn't recall a detail from a session last week, even though you discussed it at length.

**Check:** Memory capacity and contents. The system prompt memory header shows usage (e.g., `[67% — 1,474/2,200 chars]`), and `hermes journey list` shows every saved memory entry and skill.

**What it means:** Persistent memory is intentionally bounded — 2,200 chars (~800 tokens) for MEMORY.md and 1,375 chars (~500 tokens) for USER.md. It holds curated key facts, not conversation transcripts. Things worth saving are preferences, environment facts, conventions, and corrections; raw discussion detail is not stored there by design.

For "did we discuss X last week?" recall, the agent has a separate mechanism: `session_search` queries all past sessions (stored in SQLite with full-text search) and can find things discussed weeks ago even when they're not in active memory. Just ask — "search our past sessions for the deploy discussion."

You can also help directly: say "remember this for next time" after a productive session, or "clean up your memory" when it's near capacity so the agent consolidates entries. See [Memory & Skills tips](/guides/tips#memory--skills) and [Capacity Management](/user-guide/features/memory#capacity-management).

## 6. Check that skills and tools are loaded

**Symptom:** Hermes used to handle a specific workflow expertly and now approaches it naively, or says it can't do something it did before.

**Check:**

- `/skills` — browse installed skills (a skill the agent relied on may have been removed).
- `/reload-skills` — re-scan `~/.hermes/skills/` for newly installed or removed skills.
- `/tools list` — see available tools; a tool disabled earlier with `/tools disable` stays out of the agent's toolset for the session.
- `/context all` — per-skill and per-toolset cost listing, which doubles as an inventory of what's actually loaded.

**What it means:** Skills are the agent's procedural knowledge — multi-step workflows and tool-specific instructions. If a skill is missing or a toolset was trimmed (e.g., a session started with `hermes chat -t "terminal"` to reduce prompt weight), the agent genuinely has less to work with in that session. Re-enable tools with `/tools enable`, or invoke the skill explicitly by name (`/github-pr-workflow`) to confirm it loads.

## 7. Compression side-effects

**Symptom:** After a long session (or right after running `/compress`), Hermes remembers the broad strokes but has lost fine detail from earlier in the conversation.

**Check:** Whether compression has fired — `/usage` and `/context` show compression stats and context state on messaging platforms, and manual `/compress` always reports its result.

**What it means:** Compression replaces older conversation history with a summary — that's the point, and it necessarily trades detail for headroom. Know the shape of it:

- Recent messages are protected: by default the last 20 messages stay uncompressed (`protect_last_n`) and the opening exchange is pinned (`protect_first_n: 3`) so the original goal stays visible.
- Compaction is non-destructive: with the default `compression.in_place: true`, the session keeps one durable id and pre-compaction turns are soft-archived — still searchable via `session_search` and recoverable, not deleted.
- With `in_place: false` (legacy behavior), each compaction rotates to a **new session linked to the old one** — a titled session becomes `"my project" → "my project #2" → "my project #3"`. If you resume by title, `hermes -c "my project"` automatically picks the most recent variant.
- A focus topic narrows what a full summary preserves: `/compress focus auth-refactor` keeps that thread's detail at the expense of the rest.

If a compressed-away detail matters, ask the agent to search for it (`session_search` reaches the archived turns), or re-paste the key facts into the conversation.

See [Context Compression](/user-guide/configuration#context-compression) for the full settings reference and [Auto-Lineage on Compression](/user-guide/sessions#auto-lineage-on-compression) for how titled sessions chain.

---

## Quick reference

| Symptom | First command | Likely cause |
|---------|--------------|--------------|
| Everything feels less capable | `/model` | Session is on a different model than you think |
| Long session degrading | `/usage` | Context pressure — compress or start fresh |
| Limits hit surprisingly early | CLI startup line / `/usage` | Wrong auto-detected context length |
| Forgot what I said this session | — (by design) | Frozen memory snapshot — appears next session |
| Forgot last week's discussion | ask it to `session_search` | Memory is bounded, curated facts only |
| Lost a specific ability | `/skills`, `/tools list` | Skill or toolset not loaded this session |
| Lost old detail after long session | `/usage`, `/context` | Compression summarized older history |
