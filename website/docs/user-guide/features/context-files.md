---
sidebar_position: 8
title: "Context Files"
description: "Project context files — .hermes.md, AGENTS.md, CLAUDE.md, global SOUL.md, and .cursorrules — automatically injected into every conversation"
---

# Context Files

Hermes Agent automatically discovers and loads context files that shape how it behaves. Some are project-local and discovered from your working directory. `SOUL.md` is now global to the Hermes instance and is loaded from `HERMES_HOME` only.

## Supported Context Files

| File | Purpose | Discovery |
|------|---------|-----------| 
| **.hermes.md** / **HERMES.md** | Project instructions (highest priority) | Walks to git root |
| **AGENTS.md** (cwd) | Project instructions, conventions, architecture | CWD at startup + subdirectories progressively |
| **AGENTS.md** (HERMES_HOME) | Global operational policy — workflow rules, coding policy, safety guardrails — applied across all sessions/cwds | `HERMES_HOME/AGENTS.md` only |
| **CLAUDE.md** | Claude Code context files (also detected) | CWD at startup + subdirectories progressively |
| **SOUL.md** | Global personality and tone customization for this Hermes instance | `HERMES_HOME/SOUL.md` only |
| **.cursorrules** | Cursor IDE coding conventions | CWD only |
| **.cursor/rules/*.mdc** | Cursor IDE rule modules | CWD only |

:::info Priority system
Only **one** project context type is loaded per session from the working directory (first match wins): `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`. **SOUL.md** and the **HERMES_HOME `AGENTS.md`** load independently and always (when present), regardless of cwd. Home-level `AGENTS.md` is appended *before* the cwd project context, so project files can override or augment baseline policy.
:::

## AGENTS.md

`AGENTS.md` is the primary project context file. It tells the agent how your project is structured, what conventions to follow, and any special instructions.

### Progressive Subdirectory Discovery

At session start, Hermes loads the `AGENTS.md` from your working directory into the system prompt. As the agent navigates into subdirectories during the session (via `read_file`, `terminal`, `search_files`, etc.), it **progressively discovers** context files in those directories and injects them into the conversation at the moment they become relevant.

```
my-project/
├── AGENTS.md              ← Loaded at startup (system prompt)
├── frontend/
│   └── AGENTS.md          ← Discovered when agent reads frontend/ files
├── backend/
│   └── AGENTS.md          ← Discovered when agent reads backend/ files
└── shared/
    └── AGENTS.md          ← Discovered when agent reads shared/ files
```

This approach has two advantages over loading everything at startup:
- **No system prompt bloat** — subdirectory hints only appear when needed
- **Prompt cache preservation** — the system prompt stays stable across turns

Each subdirectory is checked at most once per session. The discovery also walks up parent directories, so reading `backend/src/main.py` will discover `backend/AGENTS.md` even if `backend/src/` has no context file of its own.

:::info
Subdirectory context files go through the same [security scan](#security-prompt-injection-protection) as startup context files. Malicious files are blocked.
:::

### Example AGENTS.md

```markdown
# Project Context

This is a Next.js 14 web application with a Python FastAPI backend.

## Architecture
- Frontend: Next.js 14 with App Router in `/frontend`
- Backend: FastAPI in `/backend`, uses SQLAlchemy ORM
- Database: PostgreSQL 16
- Deployment: Docker Compose on a Hetzner VPS

## Conventions
- Use TypeScript strict mode for all frontend code
- Python code follows PEP 8, use type hints everywhere
- All API endpoints return JSON with `{data, error, meta}` shape
- Tests go in `__tests__/` directories (frontend) or `tests/` (backend)

## Important Notes
- Never modify migration files directly — use Alembic commands
- The `.env.local` file has real API keys, don't commit it
- Frontend port is 3000, backend is 8000, DB is 5432
```

## SOUL.md

`SOUL.md` controls the agent's personality, tone, and communication style. See the [Personality](/user-guide/features/personality) page for full details.

**Location:**

- `~/.hermes/SOUL.md`
- or `$HERMES_HOME/SOUL.md` if you run Hermes with a custom home directory

Important details:

- Hermes seeds a default `SOUL.md` automatically if one does not exist yet
- Hermes loads `SOUL.md` only from `HERMES_HOME`
- Hermes does not probe the working directory for `SOUL.md`
- If the file is empty, nothing from `SOUL.md` is added to the prompt
- If the file has content, the content is injected verbatim after scanning and truncation

## AGENTS.md (HERMES_HOME)

In addition to the cwd-based `AGENTS.md` (project context), Hermes also loads an `AGENTS.md` from your `HERMES_HOME` if present. This is the right home for **operational policy** — workflow rules, coding/test conventions, safety guardrails, formatting preferences — that should apply across every session, regardless of working directory.

**Location:**

- `~/.hermes/AGENTS.md`
- or `$HERMES_HOME/AGENTS.md` if you run Hermes with a custom home directory

**Why both?**

- **`HERMES_HOME/AGENTS.md`** = baseline policy that follows you everywhere (CLI sessions, Telegram/Discord gateway, cron jobs, subagents, any cwd).
- **cwd `AGENTS.md`** = project-specific context (architecture, conventions, repo-local quirks). Loaded when you run Hermes inside that project.

When both exist they coexist: home AGENTS.md is appended **first**, then the cwd project context is appended after, so the project file can override or extend home rules. The cwd-based progressive subdirectory discovery still works for the project tree.

**When to use this vs. SOUL.md:**

- **SOUL.md** → identity, personality, voice, tone (*who* the agent is)
- **AGENTS.md** → workflow rules, policies, procedures (*how* the agent operates)

If you find yourself writing imperative procedural rules in SOUL.md, move them to `~/.hermes/AGENTS.md` to keep identity and policy cleanly separated.

## .cursorrules

Hermes is compatible with Cursor IDE's `.cursorrules` file and `.cursor/rules/*.mdc` rule modules. If these files exist in your project root and no higher-priority context file (`.hermes.md`, `AGENTS.md`, or `CLAUDE.md`) is found, they're loaded as the project context.

This means your existing Cursor conventions automatically apply when using Hermes.

## How Context Files Are Loaded

### At startup (system prompt)

Context files are loaded by `build_context_files_prompt()` in `agent/prompt_builder.py`:

1. **Scan working directory** — checks for `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules` (first match wins)
2. **Content is read** — each file is read as UTF-8 text
3. **Security scan** — content is checked for prompt injection patterns
4. **Truncation** — files exceeding `context_file_max_chars` characters (default 20,000) are head/tail truncated (70% head, 20% tail, with a marker in the middle)
5. **Assembly** — all sections are combined under a `# Project Context` header
6. **Injection** — the assembled content is added to the system prompt

### During the session (progressive discovery)

`SubdirectoryHintTracker` in `agent/subdirectory_hints.py` watches tool call arguments for file paths:

1. **Path extraction** — after each tool call, file paths are extracted from arguments (`path`, `workdir`, shell commands)
2. **Ancestor walk** — the directory and up to 5 parent directories are checked (stopping at already-visited directories)
3. **Hint loading** — if an `AGENTS.md`, `CLAUDE.md`, or `.cursorrules` is found, it's loaded (first match per directory)
4. **Security scan** — same prompt injection scan as startup files
5. **Truncation** — capped at 8,000 characters per file
6. **Injection** — appended to the tool result, so the model sees it in context naturally

The final prompt section looks roughly like:

```text
# Project Context

The following project context files have been loaded and should be followed:

## AGENTS.md

[Your AGENTS.md content here]

## .cursorrules

[Your .cursorrules content here]

[Your SOUL.md content here]
```

Notice that SOUL content is inserted directly, without extra wrapper text.

## Security: Prompt Injection Protection

All context files are scanned for potential prompt injection before being included. The scanner checks for:

- **Instruction override attempts**: "ignore previous instructions", "disregard your rules"
- **Deception patterns**: "do not tell the user"
- **System prompt overrides**: "system prompt override"
- **Hidden HTML comments**: `<!-- ignore instructions -->`
- **Hidden div elements**: `<div style="display:none">`
- **Credential exfiltration**: `curl ... $API_KEY`
- **Secret file access**: `cat .env`, `cat credentials`
- **Invisible characters**: zero-width spaces, bidirectional overrides, word joiners

If any threat pattern is detected, the file is blocked:

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

:::warning
This scanner protects against common injection patterns, but it's not a substitute for reviewing context files in shared repositories. Always validate AGENTS.md content in projects you didn't author.
:::

## Size Limits

| Limit | Value |
|-------|-------|
| Max chars per file | `context_file_max_chars` (default 20,000, ~7,000 tokens) |
| Head truncation ratio | 70% |
| Tail truncation ratio | 20% |
| Truncation marker | 10% (shows char counts and suggests using file tools) |

When a file exceeds the configured limit, the truncation message reads:

```
[...truncated AGENTS.md: kept 14000+4000 of 25000 chars. Use file tools to read the full file.]
```

## Tips for Effective Context Files

:::tip Best practices for AGENTS.md
1. **Keep it concise** — stay under your configured `context_file_max_chars`; the agent reads it every turn
2. **Structure with headers** — use `##` sections for architecture, conventions, important notes
3. **Include concrete examples** — show preferred code patterns, API shapes, naming conventions
4. **Mention what NOT to do** — "never modify migration files directly"
5. **List key paths and ports** — the agent uses these for terminal commands
6. **Update as the project evolves** — stale context is worse than no context
:::

### Per-Subdirectory Context

For monorepos, put subdirectory-specific instructions in nested AGENTS.md files:

```markdown
<!-- frontend/AGENTS.md -->
# Frontend Context

- Use `pnpm` not `npm` for package management
- Components go in `src/components/`, pages in `src/app/`
- Use Tailwind CSS, never inline styles
- Run tests with `pnpm test`
```

```markdown
<!-- backend/AGENTS.md -->
# Backend Context

- Use `poetry` for dependency management
- Run the dev server with `poetry run uvicorn main:app --reload`
- All endpoints need OpenAPI docstrings
- Database models are in `models/`, schemas in `schemas/`
```
