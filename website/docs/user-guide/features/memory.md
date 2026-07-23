---
sidebar_position: 3
title: "Persistent Memory"
description: "How Hermes Agent remembers across sessions — MEMORY.md, USER.md, and session search"
---

# Persistent Memory

Hermes Agent has bounded, curated memory that persists across sessions. This lets it remember your preferences, your projects, your environment, and things it has learned.

## How It Works

Two files make up the agent's memory:

| File | Purpose | Char Limit |
|------|---------|------------|
| **MEMORY.md** | Agent's personal notes — environment facts, conventions, things learned | 2,200 chars (~800 tokens) |
| **USER.md** | User profile — your preferences, communication style, expectations | 1,375 chars (~500 tokens) |

Both are stored in `~/.hermes/memories/` and are injected into the system prompt as a frozen snapshot at session start. The agent manages its own memory via the `memory` tool — it can add, replace, or remove entries.

:::info
Character limits keep memory focused. Memory does **not** auto-compact: when a
write would exceed the limit, the `memory` tool returns an error instead of
silently dropping entries. The agent then makes room itself — consolidating or
removing entries in the same turn before retrying (see [What Happens When Memory
is Full](#what-happens-when-memory-is-full)). Note that `replace` is also bound
by the limit: swapping an entry for a longer one can still overflow, so the new
content must be shortened (or another entry removed) to fit.
:::

## How Memory Appears in the System Prompt

At the start of every session, memory entries are loaded from disk and rendered into the system prompt as a frozen block:

```
══════════════════════════════════════════════
MEMORY (your personal notes) [67% — 1,474/2,200 chars]
══════════════════════════════════════════════
User's project is a Rust web service at ~/code/myapi using Axum + SQLx
§
This machine runs Ubuntu 22.04, has Docker and Podman installed
§
User prefers concise responses, dislikes verbose explanations
```

The format includes:
- A header showing which store (MEMORY or USER PROFILE)
- Usage percentage and character counts so the agent knows capacity
- Individual entries separated by `§` (section sign) delimiters
- Entries can be multiline

**Frozen snapshot pattern:** The system prompt injection is captured once at session start and never changes mid-session. This is intentional — it preserves the LLM's prefix cache for performance. When the agent adds/removes memory entries during a session, the changes are persisted to disk immediately but won't appear in the system prompt until the next session starts. Tool responses always show the live state.

## Memory Tool Actions

The agent uses the `memory` tool with these actions:

- **add** — Add a new memory entry
- **replace** — Replace an existing entry with updated content (uses substring matching via `old_text`)
- **remove** — Remove an entry that's no longer relevant (uses substring matching via `old_text`)

There is no `read` action — memory content is automatically injected into the system prompt at session start. The agent sees its memories as part of its conversation context.

### Substring Matching

The `replace` and `remove` actions use short unique substring matching — you don't need the full entry text. The `old_text` parameter just needs to be a unique substring that identifies exactly one entry:

```python
# If memory contains "User prefers dark mode in all editors"
memory(action="replace", target="memory",
       old_text="dark mode",
       content="User prefers light mode in VS Code, dark mode in terminal")
```

If the substring matches multiple entries, an error is returned asking for a more specific match.

## Two Targets Explained

### `memory` — Agent's Personal Notes

For information the agent needs to remember about the environment, workflows, and lessons learned:

- Environment facts (OS, tools, project structure)
- Project conventions and configuration
- Tool quirks and workarounds discovered
- Completed task diary entries
- Skills and techniques that worked

### `user` — User Profile

For information about the user's identity, preferences, and communication style:

- Name, role, timezone
- Communication preferences (concise vs detailed, format preferences)
- Pet peeves and things to avoid
- Workflow habits
- Technical skill level

## What to Save vs Skip

### Save These (Proactively)

The agent saves automatically — you don't need to ask. It saves when it learns:

- **User preferences:** "I prefer TypeScript over JavaScript" → save to `user`
- **Environment facts:** "This server runs Debian 12 with PostgreSQL 16" → save to `memory`
- **Corrections:** "Don't use `sudo` for Docker commands, user is in docker group" → save to `memory`
- **Conventions:** "Project uses tabs, 120-char line width, Google-style docstrings" → save to `memory`
- **Completed work:** "Migrated database from MySQL to PostgreSQL on 2026-01-15" → save to `memory`
- **Explicit requests:** "Remember that my API key rotation happens monthly" → save to `memory`

### Skip These

- **Trivial/obvious info:** "User asked about Python" — too vague to be useful
- **Easily re-discovered facts:** "Python 3.12 supports f-string nesting" — can web search this
- **Raw data dumps:** Large code blocks, log files, data tables — too big for memory
- **Session-specific ephemera:** Temporary file paths, one-off debugging context
- **Information already in context files:** SOUL.md and AGENTS.md content

## Capacity Management

Memory has strict character limits to keep system prompts bounded:

| Store | Limit | Typical entries |
|-------|-------|----------------|
| memory | 2,200 chars | 8-15 entries |
| user | 1,375 chars | 5-10 entries |

### What Happens When Memory is Full

When you try to add an entry that would exceed the limit, the tool returns an error:

```json
{
  "success": false,
  "error": "Memory at 2,100/2,200 chars. Adding this entry (250 chars) would exceed the limit. Consolidate now: use 'replace' to merge overlapping entries into shorter ones or 'remove' stale or less important entries (see current_entries below), then retry this add — all in this turn.",
  "current_entries": ["..."],
  "usage": "2,100/2,200"
}
```

The agent should then:
1. Read the current entries (shown in the error response)
2. Identify entries that can be removed or consolidated
3. Use `replace` to merge related entries into shorter versions
4. Then `add` the new entry

**Best practice:** When memory is above 80% capacity (visible in the system prompt header), consolidate entries before adding new ones. For example, merge three separate "project uses X" entries into one comprehensive project description entry.

### Practical Examples of Good Memory Entries

**Compact, information-dense entries work best:**

```
# Good: Packs multiple related facts
User runs macOS 14 Sonoma, uses Homebrew, has Docker Desktop and Podman. Shell: zsh with oh-my-zsh. Editor: VS Code with Vim keybindings.

# Good: Specific, actionable convention
Project ~/code/api uses Go 1.22, sqlc for DB queries, chi router. Run tests with 'make test'. CI via GitHub Actions.

# Good: Lesson learned with context
The staging server (10.0.1.50) needs SSH port 2222, not 22. Key is at ~/.ssh/staging_ed25519.

# Bad: Too vague
User has a project.

# Bad: Too verbose
On January 5th, 2026, the user asked me to look at their project which is
located at ~/code/api. I discovered it uses Go version 1.22 and...
```

## Duplicate Prevention

The memory system automatically rejects exact duplicate entries. If you try to add content that already exists, it returns success with a "no duplicate added" message.

## Security Scanning

Memory entries are scanned for injection and exfiltration patterns before being accepted, since they're injected into the system prompt. Content matching threat patterns (prompt injection, credential exfiltration, SSH backdoors) or containing invisible Unicode characters is blocked.

## Session Search

Beyond MEMORY.md and USER.md, the agent can search its past conversations using the `session_search` tool:

- All CLI and messaging sessions are stored in SQLite (`~/.hermes/state.db`) with FTS5 full-text search
- Search queries return actual messages from the DB — no LLM summarization, no truncation
- The agent can find things it discussed weeks ago, even if they're not in its active memory
- The agent can also scroll forward/backward inside any session it finds

```bash
hermes sessions list    # Browse past sessions
```

See [Session Search Tool](/user-guide/sessions#session-search-tool) for the three calling shapes (discovery / scroll / browse) and the response format.

### session_search vs memory

| Feature | Persistent Memory | Session Search |
|---------|------------------|----------------|
| **Capacity** | ~1,300 tokens total | Unlimited (all sessions) |
| **Speed** | Instant (in system prompt) | ~20ms FTS5 query, ~1ms scroll |
| **Cost** | Token cost in every prompt | Free — no LLM calls |
| **Use case** | Key facts always available | Finding specific past conversations |
| **Management** | Manually curated by agent | Automatic — all sessions stored |
| **Token cost** | Fixed per session (~1,300 tokens) | On-demand (searched when needed) |

**Memory** is for critical facts that should always be in context. **Session search** is for "did we discuss X last week?" queries where the agent needs to recall specifics from past conversations.

## Configuration

```yaml
# In ~/.hermes/config.yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
  write_approval: false     # false = write freely (default) | true = require approval
```

## Controlling memory writes (`write_approval`)

By default the agent saves memory freely — including from the background
self-improvement review that runs after a turn. If you'd rather approve saves
first, set `memory.write_approval: true`. It's a simple on/off gate applied to
**both** foreground turns and the background review:

| `write_approval` | Behaviour |
|------------------|-----------|
| `false` (default) | Write freely — the gate is off (the pre-gate behaviour). |
| `true` | Require approval before anything is saved. In the interactive CLI, foreground writes prompt you inline (entries are small enough to read in full). Everywhere else — messaging platforms, scripts, and the background self-improvement review — writes are **staged** for review with `/memory pending`. |

> To turn memory off entirely (not just gate it), set `memory_enabled: false`.

Review staged writes from the CLI or any messaging platform:

```
/memory pending             # list staged memory writes (auto ones tagged [auto])
/memory approve <id>        # apply one (or 'all')
/memory reject <id>         # drop one (or 'all')
/memory approval on         # turn the gate on (or 'off') and persist it
```

This is the answer to "the agent saved a wrong assumption about me": set
`write_approval: true`, and every save — especially the unprompted background
ones — waits for your yes/no before it ever enters your profile.

## Background review notifications (`display.memory_notifications`)

After a turn, the background self-improvement review may quietly save a memory
or update a skill. This is Hermes' consent-aware learning loop: repeated
corrections and durable workflow lessons become compact memory entries or
procedural skills, while `write_approval` can stage those writes for review
before they affect future sessions. By default it surfaces a short
`💾 Memory updated` line in chat so you know it happened. Control how chatty
that is:

```yaml
display:
  memory_notifications: on    # off | on (default) | verbose
```

| Value | Behaviour |
|-------|-----------|
| `off` | No chat notification. The review still runs and still writes — you just don't see a line for it. |
| `on` (default) | Generic line, e.g. `💾 Memory updated`, `💾 Skill 'foo' patched`. |
| `verbose` | Includes a compact preview of what changed, e.g. `💾 Memory ➕ User prefers terse replies` or a `"old" → "new"` skill diff snippet. |

> This only governs the **gateway** chat notification. The review itself, and
> writes to your memory/skill stores, are unaffected by this setting. Set it
> per-platform via `display.platforms.<platform>.memory_notifications`.

## Running the review on a cheaper model (`auxiliary.background_review`)

The review runs on your **main chat model** by default, replaying the
conversation — which is already warm in the prompt cache, so it's cheap cache
reads. On an expensive main model you can run the review on a cheaper model
instead:

```yaml
auxiliary:
  background_review:
    provider: openrouter
    model: google/gemini-3-flash-preview   # auto (default) = main chat model
```

When you point it at a model **different** from your main one, the review runs
there for substantially lower cost (~3–5× in benchmarks). Because a different
model can't reuse your main model's prompt cache anyway, the fork automatically
replays a compact **digest** of the conversation (recent turns verbatim + a
summary of older ones) rather than the full transcript — minimizing what it
writes to the new cache. Capture holds: in testing, memory capture was
identical and skill capture near-identical to the main-model review.

Leave it at `auto` (or set it to your main model) and nothing changes — the
review keeps running on the main model with the full warm-cache replay.

## Controlling skill writes (`skills.write_approval`)

Skills use the same on/off gate, but the review UX differs because a
`SKILL.md` is far too large to read in a chat bubble:

```yaml
skills:
  write_approval: false     # false = write freely (default) | true = require approval
```

When `write_approval: true`, skill writes (create / edit / patch / write_file /
delete) always **stage** regardless of origin. You review the one-line gist
inline, but the full diff stays out-of-band:

```
/skills pending             # list staged skill writes + a one-line gist each
/skills diff <id>           # full unified diff (best viewed in CLI or dashboard)
/skills approve <id>        # apply it (or 'all')
/skills reject <id>         # drop it (or 'all')
/skills approval on         # turn the gate on (or 'off') and persist it
```

On a messaging platform, approve a skill from its gist + metadata, or open
`/skills diff` on the CLI / dashboard / the staged file under
`~/.hermes/pending/skills/<id>.json` when you want to read the whole change.
Full details in [Gating agent skill writes](/user-guide/features/skills#gating-agent-skill-writes-skillswrite_approval).


## External Memory Providers

For deeper, persistent memory that goes beyond MEMORY.md and USER.md, Hermes ships with 8 external memory provider plugins — including Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, and Supermemory.

External providers run **alongside** built-in memory (never replacing it) and add capabilities like knowledge graphs, semantic search, automatic fact extraction, and cross-session user modeling.

```bash
hermes memory setup      # pick a provider and configure it
hermes memory status     # check what's active
```

See the [Memory Providers](./memory-providers.md) guide for full details on each provider, setup instructions, and comparison.
