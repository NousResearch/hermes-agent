---
sidebar_position: 12
---

# Codebase Index

[mcp-codebase-index](https://github.com/Mibayy/mcp-codebase-index) is a community MCP server that builds a structural index
of your codebase — functions, classes, imports, and dependency graphs — so
Hermes can answer code questions without reading entire files.

Instead of `cat run_agent.py` (3 000 lines into context), the agent calls
`find_symbol("_honcho_prefetch")` and gets a 20-line preview plus the file
and line number. The savings compound quickly on large projects.

**Measured across 92 real sessions on production codebases:**

| Project | Sessions | Queries | Chars used | Chars (naive) | Chars saved | Saving |
|---------|----------|---------|------------|---------------|-------------|--------|
| project-alpha | 35 | 360 | 4,801,108 | 639,560,872 | 634,759,764 | 99% |
| project-beta | 26 | 189 | 766,508 | 20,936,204 | 20,169,696 | 96% |
| project-gamma | 30 | 232 | 410,816 | 3,679,868 | 3,269,052 | 89% |
| project-delta | 1 | 1 | 3,036 | 52,148 | 49,112 | 94% |
| **TOTAL** | **92** | **782** | **5,981,476** | **664,229,092** | **658,247,616** | **99%** |

These savings apply to any model that supports tool use — Anthropic, OpenRouter,
Ollama, Gemini, etc. The index reduces what enters the context window regardless
of which model processes it.

## Installation

mcp-codebase-index runs as a separate MCP server process. Install it in a
dedicated virtual environment so its dependencies don't conflict with Hermes:

```bash
python -m venv ~/.local/mcp-codebase-index-venv
~/.local/mcp-codebase-index-venv/bin/pip install mcp-codebase-index
```

## Configuration

Add the server to `~/.hermes/cli-config.yaml`:

```yaml
mcp_servers:
  codebase-index:
    command: ~/.local/mcp-codebase-index-venv/bin/mcp-codebase-index
    env:
      WORKSPACE_ROOTS: /path/to/project1:/path/to/project2
    timeout: 120
    connect_timeout: 30
```

`WORKSPACE_ROOTS` is a colon-separated list of absolute paths. Each project
gets its own isolated index, loaded lazily on first use.

Restart Hermes after editing the config.

## Verifying the setup

In a Hermes session:

```
hermes> list all indexed projects
```

The agent will call `list_projects` and show each root with its index status.

To trigger a first index on a project, ask anything about its code:

```
hermes> find the function that handles Honcho prefetch
```

The first index takes a few seconds (cold start). Subsequent queries are
instant — the index is incremental and git-aware: only changed files are
re-indexed between sessions.

## Available tools

Once configured, Hermes has access to:

| Tool | What it does |
|------|--------------|
| `search_codebase` | Regex search across all indexed files |
| `find_symbol` | Locate a function/class by name — returns file, line, preview |
| `get_function_source` | Full source of a function without reading the whole file |
| `get_class_source` | Full source of a class |
| `get_dependencies` | What does this symbol call? |
| `get_dependents` | What calls this symbol? |
| `get_change_impact` | Transitive impact analysis of a change |
| `get_project_summary` | High-level overview: file count, top classes/functions |
| `list_projects` | All indexed projects and their status |
| `switch_project` | Change the active project |
| `reindex` | Force a full re-index |
| `get_usage_stats` | Token savings per project across sessions |

## Excluding build directories

By default the index excludes `node_modules`, `__pycache__`, `.git`, and
common build outputs (`.next`, `dist`, `build`). To add extra exclusions:

```yaml
mcp_servers:
  codebase-index:
    command: ~/.local/mcp-codebase-index-venv/bin/mcp-codebase-index
    env:
      WORKSPACE_ROOTS: /path/to/project
      EXCLUDE_EXTRA: "**/vendor/**:**/generated/**"
```

## Teaching Hermes to use it

Having the tools available doesn't guarantee the agent will prefer them over
`terminal("grep ...")` or `read_file`. Load the `codebase-index` skill at the
start of a coding session to guide the agent's behavior:

```
hermes> load the codebase-index skill
```

See [Skills](../skills/) for how bundled skills work.

## Pairs well with structured memory

codebase-index works well alongside the native structured memory toolset
(PR #3093), which provides a typed, searchable fact store built directly into
Hermes — no external process, no configuration beyond enabling the toolset.

- **codebase-index** keeps code navigation cheap — the agent reads symbols,
  not files, so the context stays small during exploration.
- **structured memory** keeps decisions durable — architectural constraints,
  design choices, and open questions are stored as typed facts (`C[]`, `D[]`,
  `V[]`) that survive context compression and are searchable across turns.

Together they address the two main ways long sessions degrade: context bloat
from reading too much code, and decision drift from forgetting what was agreed.
