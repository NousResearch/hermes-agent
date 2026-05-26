---
sidebar_position: 5
title: "Prompt Assembly"
description: "How Hermes builds the system prompt, preserves cache stability, and injects ephemeral layers"
---

# Prompt Assembly

Hermes deliberately separates:

- **cached system prompt state**
- **ephemeral API-call-time additions**

This is one of the most important design choices in the project because it affects:

- token usage
- prompt caching effectiveness
- session continuity
- memory correctness

Primary files:

- `run_agent.py`
- `agent/prompt_builder.py`
- `tools/memory_tool.py`

## Cached system prompt layers

The cached system prompt is assembled in roughly this order:

1. Base identity — `SOUL.md` from `HERMES_HOME` when available, otherwise the Built-in fallback `DEFAULT_AGENT_IDENTITY` in `prompt_builder.py`
2. tool-aware behavior guidance
3. Honcho static block (when active)
4. optional system message
5. frozen MEMORY snapshot
6. frozen USER profile snapshot
7. skills index
8. Project context files (`.hermes.md` / `HERMES.md`, `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `.cursor/rules/*.mdc`) — `SOUL.md` is **not** duplicated here when it was already loaded as Base identity in step 1
9. timestamp / optional session ID
10. platform hint

When `skip_context_files` is set (e.g., subagent delegation), Project context is skipped. Base identity is also skipped unless the caller explicitly sets `load_soul_identity=True`; otherwise the Built-in fallback `DEFAULT_AGENT_IDENTITY` is used.

### Concrete example: assembled system prompt

Here is a simplified view of what the final system prompt looks like when all layers are present (comments show the source of each section):

```
# Layer 1: Base Identity (from ~/.hermes/SOUL.md)
You are Hermes, an AI assistant created by Nous Research.
You are an expert software engineer and researcher.
You value correctness, clarity, and efficiency.
...

# Layer 2: Tool-aware behavior guidance
You have persistent memory across sessions. Save durable facts using
the memory tool: user preferences, environment details, tool quirks,
and stable conventions. Memory is injected into every turn, so keep
it compact and focused on facts that will still matter later.
...
When the user references something from a past conversation or you
suspect relevant cross-session context exists, use session_search
to recall it before asking them to repeat themselves.

# Tool-use enforcement (for GPT/Codex models only)
You MUST use your tools to take action — do not describe what you
would do or plan to do without actually doing it.
...

# Layer 3: Honcho static block (when active)
[Honcho personality/context data]

# Layer 4: Optional system message (from config or API)
[User-configured system message override]

# Layer 5: Frozen MEMORY snapshot
## Persistent Memory
- User prefers Python 3.12, uses pyproject.toml
- Default editor is nvim
- Working on project "atlas" in ~/code/atlas
- Timezone: US/Pacific

# Layer 6: Frozen USER profile snapshot
## User Profile
- Name: Alice
- GitHub: alice-dev

# Layer 7: Skills index
## Skills (mandatory)
Before replying, scan the skills below. If one clearly matches
your task, load it with skill_view(name) and follow its instructions.
...
<available_skills>
  software-development:
    - code-review: Structured code review workflow
    - test-driven-development: TDD methodology
  research:
    - arxiv: Search and summarize arXiv papers
</available_skills>

# Layer 8: Project context files (from project directory)
# Project Context
The following project context files have been loaded and should be followed:

## AGENTS.md
This is the atlas project. Use pytest for testing. The main
entry point is src/atlas/main.py. Always run `make lint` before
committing.

# Layer 9: Timestamp + session
Current time: 2026-03-30T14:30:00-07:00
Session: abc123

# Layer 10: Platform hint
You are a CLI AI Agent. Try not to use markdown but simple text
renderable inside a terminal.
```

## How SOUL.md appears in the prompt

`SOUL.md` lives at `~/.hermes/SOUL.md` and serves as Base identity — the very first section of the system prompt. The loading logic in `prompt_builder.py` works as follows:

```python
# From agent/prompt_builder.py (simplified)
def load_soul_md() -> Optional[str]:
    soul_path = get_hermes_home() / "SOUL.md"
    if not soul_path.exists():
        return None
    content = soul_path.read_text(encoding="utf-8").strip()
    content = _scan_context_content(content, "SOUL.md")  # Security scan
    content = _truncate_content(content, "SOUL.md")       # Cap at 20k chars
    return content
```

When `load_soul_md()` returns content, it replaces the hardcoded Built-in fallback `DEFAULT_AGENT_IDENTITY`. The `build_context_files_prompt()` function is then called with `skip_soul=True` to prevent `SOUL.md` from appearing twice (once as Base identity, once as a context helper output).

If `SOUL.md` doesn't exist, the system falls back to:

```
You are Hermes Agent, an intelligent AI assistant created by Nous Research.
You are helpful, knowledgeable, and direct. You assist users with a wide
range of tasks including answering questions, writing and editing code,
analyzing information, creative work, and executing actions via your tools.
You communicate clearly, admit uncertainty when appropriate, and prioritize
being genuinely useful over being verbose unless otherwise directed below.
Be targeted and efficient in your exploration and investigations.
```

## How context files are injected

`build_context_files_prompt()` uses a **priority system** — only one project context type is loaded (first match wins):

```python
# From agent/prompt_builder.py (simplified)
def build_context_files_prompt(cwd=None, skip_soul=False):
    cwd_path = Path(cwd).resolve()

    # Priority: first match wins — only ONE project context loaded
    project_context = (
        _load_hermes_md(cwd_path)       # 1. .hermes.md / HERMES.md (walks to git root)
        or _load_agents_md(cwd_path)    # 2. AGENTS.md (cwd only)
        or _load_claude_md(cwd_path)    # 3. CLAUDE.md (cwd only)
        or _load_cursorrules(cwd_path)  # 4. .cursorrules / .cursor/rules/*.mdc
    )

    identity_sections = []
    context_sections = []
    if project_context:
        context_sections.append(project_context)

    # Base identity from HERMES_HOME (independent of Project context)
    if not skip_soul:
        soul_content = load_soul_md()
        if soul_content:
            identity_sections.append(soul_content)

    # Returned sections are labeled separately so Base identity is not
    # described as Project context when this helper is called directly.
    ...
```

### Context file discovery details

| Priority | Files | Search scope | Notes |
|----------|-------|-------------|-------|
| 1 | `.hermes.md`, `HERMES.md` | CWD up to git root | Hermes-native project config |
| 2 | `AGENTS.md` | CWD only | Common agent instruction file |
| 3 | `CLAUDE.md` | CWD only | Claude Code compatibility |
| 4 | `.cursorrules`, `.cursor/rules/*.mdc` | CWD only | Cursor compatibility |

All context files are:
- **Security scanned** — checked for prompt injection patterns (invisible unicode, "ignore previous instructions", credential exfiltration attempts)
- **Truncated** — capped at 20,000 characters using 70/20 head/tail ratio with a truncation marker
- **YAML frontmatter stripped** — `.hermes.md` frontmatter is removed (reserved for future config overrides)

## API-call-time-only layers

These are intentionally *not* persisted as part of the cached system prompt:

- `ephemeral_system_prompt`
- prefill messages
- gateway-derived session context overlays
- later-turn Honcho recall injected into the current-turn user message

This separation keeps the stable prefix stable for caching.

## Memory snapshots

Local memory and user profile data are injected as frozen snapshots at session start. Mid-session writes update disk state but do not mutate the already-built system prompt until a new session or forced rebuild occurs.

## Context files

`agent/prompt_builder.py` scans and sanitizes project context files using a **priority system** — only one type is loaded (first match wins):

1. `.hermes.md` / `HERMES.md` (walks to git root)
2. `AGENTS.md` (CWD at startup; subdirectories discovered progressively during the session via `agent/subdirectory_hints.py`)
3. `CLAUDE.md` (CWD only)
4. `.cursorrules` / `.cursor/rules/*.mdc` (CWD only)

`SOUL.md` is loaded separately via `load_soul_md()` for the Base identity slot. When it loads successfully, `build_context_files_prompt(skip_soul=True)` prevents it from appearing twice.

Long files are truncated before injection.

## Skills index

The skills system contributes a compact skills index to the prompt when skills tooling is available.

## Supported prompt customization surfaces

Most users should treat `agent/prompt_builder.py` as implementation code, not a configuration surface. The supported customization path is to change the prompt inputs Hermes already loads, rather than editing Python templates in place.

### Use these surfaces first

- `~/.hermes/SOUL.md` — replace the Built-in fallback identity block with your own Base identity and standing behavior.
- `~/.hermes/MEMORY.md` and `~/.hermes/USER.md` — provide durable cross-session facts and user profile data that should be snapshotted into new sessions.
- Project context files such as `.hermes.md`, `HERMES.md`, `AGENTS.md`, `CLAUDE.md`, or `.cursorrules` — inject repo-specific working rules.
- Skills — package reusable workflows and references without editing core prompt code.
- Optional system prompt config / API overrides — add deployment-specific instruction text without forking Hermes.
- Ephemeral overlays such as `HERMES_EPHEMERAL_SYSTEM_PROMPT` or prefill messages — add turn-scoped guidance that should not become part of the cached prompt prefix.

### When to edit code instead

Edit `agent/prompt_builder.py` only if you are intentionally maintaining a fork or contributing upstream behavior changes. That file assembles the prompt plumbing, cache boundaries, and injection order for every session. Direct edits there are global product changes, not per-user prompt customization.

In other words:

- if you want a different assistant identity, edit `SOUL.md`
- if you want different repo rules, edit project context files
- if you want reusable operating procedures, add or modify skills
- if you want to change how Hermes assembles prompts for everyone, change Python and treat it as a code contribution

## Why prompt assembly is split this way

The architecture is intentionally optimized to:

- preserve provider-side prompt caching
- avoid mutating history unnecessarily
- keep memory semantics understandable
- let gateway/ACP/CLI add context without poisoning persistent prompt state

## Grandmaster runtime cleanup implementation-readiness checklist

Use this checklist before proposing any future code or live-runtime cleanup. It is documentation-only guidance unless a separate implementation approval names a scoped code/doc batch. It does not authorize config changes, service starts/restarts, provider calls, activating a candidate `SOUL.md`, or clearing overlays.

### Prompt-surface terminology alignment

Docs should use descriptive names for roadmap items rather than opaque `R#` labels. Use the same names for the same prompt surfaces:

- **Base identity:** `HERMES_HOME/SOUL.md`, loaded into the stable identity slot when available.
- **Built-in fallback:** `DEFAULT_AGENT_IDENTITY`, used when `SOUL.md` is missing, empty, unreadable, or intentionally skipped. If the prompt-injection scanner blocks content, the identity slot may contain a blocked-content marker instead of the original file text.
- **Project context:** `.hermes.md`/`HERMES.md`, `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, and `.cursor/rules/*.mdc`; these are separate from `SOUL.md` and should carry repo/workflow rules.
- **Persisted personality/config overlays:** `/personality`, `agent.system_prompt`, and `display.personality`; these can affect behavior after startup and may need separate rollback.
- **Ephemeral API-call-time overlays:** `ephemeral_system_prompt`, `HERMES_EPHEMERAL_SYSTEM_PROMPT`, gateway/channel/session overlays, API instructions, and prefill messages; these preserve the cached prefix but can mask the base identity.
- **Prompt cache boundary:** base prompt assembly belongs in the cached system prompt; ephemeral overlays are appended immediately before model calls.

### Runtime prompt assembly code-change readiness

Before a code-change batch, prepare an implementation spec that names exact edits and tests for:

- `agent/prompt_builder.py`: `load_soul_md()` path, scanning, truncation, empty/missing fallback, and context-file separation.
- `agent/system_prompt.py`: prompt assembly order, `load_soul_identity`, `skip_context_files`, and `skip_soul=True` behavior.
- `agent/chat_completion_helpers.py` and `agent/conversation_loop.py`: API-call-time append behavior for `ephemeral_system_prompt`.
- `cli.py`, `gateway/run.py`, and `tui_gateway/server.py`: `/personality none`, persisted config fields, and in-memory overlay clearing.
- `cron/scheduler.py`: `load_soul_identity=True`, job `workdir`, and project-context loading boundaries.
- `tools/delegate_tool.py`: delegated subagent `skip_context_files=True`, `skip_memory=True`, and subagent-specific prompt behavior.
- `hermes_cli/profiles.py` and `hermes_cli/profile_distribution.py`: clone/copy/reapply behavior for `SOUL.md` and distribution-owned paths.

#### Scoped follow-up: Discord channel/thread toolset limits

Scope this separately from identity/runtime cleanup. The goal is to reduce gateway token overhead by allowing Discord channels or threads to expose only the tool schemas needed for that surface, without changing global platform defaults for every Discord session.

Status: implementation was deferred after local work; parked in git stash as the Discord channel toolset-limits work. Do not treat this feature as activated until the stash is reapplied, reviewed, committed, configured, and the gateway is restarted/fresh sessions are created as needed.

Proposed config surface:

```yaml
discord:
  channel_toolsets:
    "1508848229419323563": ["x_search", "web", "discord", "clarify"]
    "coding-channel-id": ["terminal", "file", "code_execution", "web", "discord"]
```

Resolution contract:

1. Prefer an exact current Discord channel/thread ID match.
2. If the message is inside a thread and no exact entry exists, fall back to the parent channel/forum ID.
3. If no channel-scoped entry matches, use existing `platform_toolsets.discord` behavior unchanged.
4. Do not mutate global config, persisted transcript history, memory, or provider/model settings.
5. Treat values as toolset names, not individual tool names; invalid entries should be warned about and ignored, with safe fallback if no valid scoped toolsets remain.

Implementation spec should name edits and tests for:

- `gateway/platforms/base.py`: add `MessageEvent.channel_toolsets` and a `resolve_channel_toolsets(config_extra, channel_id, parent_id)` helper parallel to `resolve_channel_prompt()` / `resolve_channel_skills()`.
- `gateway/platforms/discord.py`: resolve scoped toolsets for messages, interactions, and threads using exact ID then parent fallback.
- `gateway/run.py`: pass `event.channel_toolsets` into `AIAgent(enabled_toolsets=...)` only when present; otherwise keep platform-wide Discord tool resolution.
- `hermes_cli/tools_config.py` / `toolsets.py`: verify names resolve consistently and preserve existing default-off / MCP passthrough semantics.
- `website/docs/user-guide/messaging/discord.md`: document `discord.channel_toolsets`, examples, fallback rules, and restart/fresh-session caveats.

Candidate tests for approved offline/runtime cleanup batches:

- `load_soul_identity=True` still loads base identity when `skip_context_files=True` while keeping project context disabled.
- `skip_context_files=True` without forced identity falls back predictably.
- `/personality none` clears persisted and in-memory overlays for CLI, gateway, and TUI as documented.
- Gateway/TUI updates do not pretend to rebuild the cached base prompt when they only change an ephemeral overlay.
- Cron loads `SOUL.md` from the scheduler/profile `HERMES_HOME` and only injects project context when `workdir` is configured.
- Delegated subagents remain isolated from parent project context and memory unless a future approved design changes that.
- Distribution-owned `SOUL.md` reapplication is tested for install/update and rollback documentation.
  - Status: offline hardening implemented locally. `hermes_cli/profile_distribution.py` now copies only manifest-owned paths, supports nested `distribution_owned` paths, preserves protected user-owned paths, and removes distribution-owned target paths when the source no longer ships them. Tests added in `tests/hermes_cli/test_profile_distribution.py`.
  - Verification: `python -m pytest tests/hermes_cli/test_profile_distribution.py -q -o 'addopts='` → `66 passed`; broader CLI/profile suite → `201 passed`; prompt/runtime-adjacent suite → `116 passed`.
- `resolve_channel_toolsets()` exact-ID match wins over parent fallback; parent fallback works for Discord threads/forums; malformed config returns `None`.
- Discord events carry scoped `channel_toolsets` when configured and omit them when not configured.
- Gateway agent creation passes scoped `enabled_toolsets` only when the event supplies them; existing platform-wide `platform_toolsets.discord` remains unchanged otherwise.
- Existing `channel_prompts` and `channel_skill_bindings` behavior remains unchanged.

Stop and request review if the proposed code change would alter provider/model/tool settings outside the scoped Discord channel, read secrets or session bodies, change memory/Honcho behavior, weaken prompt-cache stability, merge base identity with ephemeral overlays, or make rollback depend on hidden chat state.

## Related docs

- [Context Compression & Prompt Caching](./context-compression-and-caching.md)
- [Session Storage](./session-storage.md)
- [Gateway Internals](./gateway-internals.md)
