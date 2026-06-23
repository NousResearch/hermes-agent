---
title: Graphnosis
description: Connect Hermes Agent to Graphnosis local encrypted memory
---

# Graphnosis + Hermes Agent

Graphnosis gives Hermes persistent, encrypted memory on your machine — engrams, semantic recall, and optional skills.

## Prerequisites

- [Graphnosis](https://graphnosis.com/download) installed
- Cortex unlocked (app running)

## Recommended setup (both paths)

```bash
hermes memory setup          # select graphnosis — auto-prefetch + memory tools
hermes mcp install graphnosis  # full MCP tools (edit, cross_search, skills)
hermes skills install https://graphnosis.com/skills/graphnosis/SKILL.md
```

Start a new Hermes chat session after setup.

## Memory provider

Sets `memory.provider: graphnosis` and enables:

- Automatic prefetch before each turn
- `graphnosis_recall`, `graphnosis_remember`, `graphnosis_stats`

## MCP catalog

Registers `mcp_graphnosis_*` tools for mutations and advanced operations.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Socket not found | Unlock Graphnosis; check `~/.graphnosis/mcp.sock` |
| Recall returns locked | Open Graphnosis app and unlock cortex |
| Provider inactive on Desktop | Set `memory.provider: graphnosis` in `~/.hermes/config.yaml` |

Full docs: [graphnosis.com/getting-started/connect-ai](https://graphnosis.com/getting-started/connect-ai)
