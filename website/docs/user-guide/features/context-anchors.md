---
sidebar_position: 9
title: "Context Anchors"
description: "Persistent project memory files that survive context compression"
---

# Context Anchors

Context anchors are markdown files that persist your project state across context compressions. When a long conversation triggers compression, critical project details normally get lost in the summary. Context anchors solve this by:

1. **Auto-injecting** anchor files into the conversation after every compression
2. **Auto-updating** anchor files when you work on the associated project

This means your agent always knows what's running, what's fixed, and what NOT to touch, even after hours of conversation.

## Quick Start

Add to your `~/.hermes/config.yaml`:

```yaml
context_anchors:
  - path: ~/.hermes/context/my-project.md
    keywords: [myproject, /var/www/myproject, myproject.com]
```

Create the file:

```markdown
# My Project Context

## Current State
- Backend running on port 8080
- Database migrated to v3
- Auth bug FIXED on 2026-03-15, do NOT re-debug

## Services
- nginx -> port 8080 (backend)
- redis on port 6379

## Rules
- Never modify migration files directly
- Always restart nginx after config changes
```

That's it. After compression, this file is automatically re-injected so the agent remembers everything.

## Configuration

### Full anchor config

```yaml
context_anchors:
  - path: ~/.hermes/context/my-project.md
    keywords: [myproject, /var/www/myproject]
    max_chars: 5000

  - path: ~/.hermes/context/trading-bot.md
    keywords: [sirius, polymarket, trading]
    max_chars: 5000

# Global settings
context_anchors_max_total_chars: 20000  # max total injection size
context_anchors_auto_save: true         # auto-update anchors (default: true)
```

### Simple form

If you just need the path, keywords are auto-derived from the filename:

```yaml
context_anchors:
  - ~/.hermes/context/eclatauto.md
  - ~/.hermes/context/sirius.md
```

### Configuration options

| Option | Default | Description |
|--------|---------|-------------|
| `path` | required | Path to the anchor markdown file |
| `keywords` | filename stem | Keywords that trigger project detection |
| `max_chars` | 5000 | Max characters per anchor file |
| `context_anchors_max_total_chars` | 20000 | Max total characters for all anchors |
| `context_anchors_auto_save` | true | Auto-update anchors when working on project |

## How It Works

### Read Path (Post-Compression Injection)

When context compression fires:

1. The agent saves memories (`flush_memories`) as usual
2. **NEW:** The agent auto-saves to relevant anchor files (if `auto_save` is enabled)
3. Middle conversation turns are summarized
4. **NEW:** All anchor files are loaded from disk and injected as a message
5. The agent continues with full project context intact

The injected message looks like:

```
[ANCHORED PROJECT CONTEXT - These files contain persistent project state
that survives compression. They reflect the current ground truth. Do NOT
repeat work described here. Do NOT re-debug issues marked as fixed.]

## ~/.hermes/context/my-project.md

[your file contents here]
```

### Write Path (Auto-Save)

When the agent detects it's working on a project with a configured anchor:

1. **Detection:** Keywords are matched against recent messages, tool calls, and file paths
2. **Trigger:** Before compression, the agent is asked to update the anchor file
3. **Method:** Uses `read_file` + `patch` (never overwrites the whole file)
4. **Result:** The anchor file reflects the latest project state

### Summary Enhancement

The compression summarizer is told which anchor files exist, so it:
- Does NOT duplicate anchor content in the summary (saves tokens)
- References anchor files instead: "See ~/.hermes/context/project.md for current state"
- Focuses on actions, decisions, and preferences not captured in anchors

## Best Practices

:::tip Writing effective anchor files
1. **Start with current state** - what's running, what ports, what versions
2. **Mark completed work** - "Auth bug FIXED on date, do NOT re-debug"
3. **List rules and constraints** - "Never restart service X without asking"
4. **Keep it under 3000 chars** - the agent reads this every compression
5. **Use clear headers** - `## Current State`, `## Rules`, `## TODO`
:::

:::tip Choosing keywords
- Include **paths** the agent touches: `/var/www/myproject`, `/root/mybot/`
- Include **URLs**: `myproject.com`, `api.myproject.com`
- Include **identifiers**: project name, service names, database names
- Keywords are case-insensitive
:::

## Example: Multi-Project Setup

```yaml
context_anchors:
  - path: ~/.hermes/context/website.md
    keywords: [eclatauto, eclatauto13.fr, /var/www/eclatauto]

  - path: ~/.hermes/context/trading.md
    keywords: [sirius, polymarket, /root/sirius-5min, paper_results]

  - path: ~/.hermes/context/dashboard.md
    keywords: [hermes-workspace, dashboard, port 3001, port 8642]
```

Each project gets its own persistent memory that the agent maintains automatically.

## Limitations

- Anchor files are plain markdown; no structured data or queries
- Auto-save depends on the auxiliary LLM being available
- Large anchor files consume context tokens after injection
- Keywords must be configured manually (no auto-discovery)
