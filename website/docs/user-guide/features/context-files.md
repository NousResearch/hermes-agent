---
sidebar_position: 8
title: "Context Files"
description: "Project context files — AGENTS.md, SOUL.md, and .cursorrules — automatically injected into every conversation"
---

# Context Files

Hermes Agent automatically discovers and loads project context files from your working directory. These files are injected into the system prompt at the start of every session, giving the agent persistent knowledge about your project's conventions, architecture, and preferences.

## Supported Context Files

| File | Purpose | Discovery |
|------|---------|-----------|
| **AGENTS.md** | Project instructions, conventions, architecture | Recursive (walks subdirectories) |
| **SOUL.md** | Personality and tone customization | CWD → `~/.hermes/SOUL.md` fallback |
| **.cursorrules** | Cursor IDE coding conventions | CWD only |
| **.cursor/rules/*.mdc** | Cursor IDE rule modules | CWD only |

## AGENTS.md

`AGENTS.md` is the primary project context file. It tells the agent how your project is structured, what conventions to follow, and any special instructions.

### Hierarchical Discovery

Hermes walks the directory tree starting from the working directory and loads **all** `AGENTS.md` files found, sorted by depth. This supports monorepo-style setups:

```
my-project/
├── AGENTS.md              ← Top-level project context
├── frontend/
│   └── AGENTS.md          ← Frontend-specific instructions
├── backend/
│   └── AGENTS.md          ← Backend-specific instructions
└── shared/
    └── AGENTS.md          ← Shared library conventions
```

All four files are concatenated into a single context block with relative path headers.

:::info
Directories that are skipped during the walk: `.`-prefixed dirs, `node_modules`, `__pycache__`, `venv`, `.venv`.
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

`SOUL.md` controls the agent's personality, tone, and communication style. See the [Personality](/docs/user-guide/features/personality) page for full details.

**Discovery order:**

1. `SOUL.md` or `soul.md` in the current working directory
2. `~/.hermes/SOUL.md` (global fallback)

When a SOUL.md is found, the agent is instructed:

> *"If SOUL.md is present, embody its persona and tone. Avoid stiff, generic replies; follow its guidance unless higher-priority instructions override it."*

## .cursorrules

Hermes is compatible with Cursor IDE's `.cursorrules` file and `.cursor/rules/*.mdc` rule modules. If these files exist in your project root, they're loaded alongside AGENTS.md.

This means your existing Cursor conventions automatically apply when using Hermes.

## How Context Files Are Loaded

Context files are loaded by `build_context_files_prompt()` in `agent/prompt_builder.py`:

1. **At session start** — the function scans the working directory
2. **Content is read** — each file is read as UTF-8 text
3. **Security scan** — content is checked for prompt injection patterns
4. **Truncation** — files exceeding 20,000 characters are head/tail truncated (70% head, 20% tail, with a marker in the middle)
5. **Assembly** — all sections are combined under a `# Project Context` header
6. **Injection** — the assembled content is added to the system prompt

The final prompt section looks like:

```
# Project Context

The following project context files have been loaded and should be followed:

## AGENTS.md

[Your AGENTS.md content here]

## .cursorrules

[Your .cursorrules content here]

## SOUL.md

If SOUL.md is present, embody its persona and tone...

[Your SOUL.md content here]
```

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
| Max chars per file | 20,000 (~7,000 tokens) |
| Head truncation ratio | 70% |
| Tail truncation ratio | 20% |
| Truncation marker | 10% (shows char counts and suggests using file tools) |

When a file exceeds 20,000 characters, the truncation message reads:

```
[...truncated AGENTS.md: kept 14000+4000 of 25000 chars. Use file tools to read the full file.]
```

## Tips for Effective Context Files

:::tip Best practices for AGENTS.md
1. **Keep it concise** — stay well under 20K chars; the agent reads it every turn
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
