---
sidebar_position: 3
title: Visual workflows (Hermes Studio import)
---

# Visual workflows — importing from Hermes Studio

Hermes Agent ships a visual workflow builder in its web dashboard (drag-and-drop
agent graphs that execute through subagent delegation). Workflows built in the
dashboard are stored as blueprints under `~/.hermes/workflows/<name>.json`.

Hermes Studio — the community desktop/web console for Hermes Agent — ships its
own visual workflow canvas and can **export** workflows in a portable envelope
format (`hermes-studio.workflow`, version 1). This command imports those exports
so a workflow designed in Studio's canvas runs in Hermes Agent.

```bash
hermes workflows import-studio ./my-workflow-export.json
```

The importer:

- validates the export envelope (format, version, node/edge limits, credential
  fields, path-safe names, acyclic graph);
- converts each Studio agent node into a Hermes Agent `agent` node (goal text,
  provider/model when present, selected skills mapped into the node context);
- turns `approvalRequired: true` into a `gate` node and reroutes the agent's
  outgoing edges through it;
- adds the synthetic `start`/`end` nodes the dashboard builder expects;
- writes the blueprint to `~/.hermes/workflows/<name>.json` in the exact shape
  the dashboard visual workflow builder reads.

## Usage

```bash
# Import an export file (name comes from the export, slugified)
hermes workflows import-studio ./export.json

# Override the blueprint name
hermes workflows import-studio ./export.json --name "My Pipeline"

# Write to a custom directory instead of ~/.hermes/workflows/
hermes workflows import-studio ./export.json --out ./blueprints

# Print the converted blueprint without writing a file
hermes workflows import-studio ./export.json --print

# List and inspect imported blueprints
hermes workflows list
hermes workflows show my-pipeline
```

## What is preserved

| Hermes Studio concept | Hermes Agent blueprint |
| --- | --- |
| Agent node (`title`, `input`) | `agent` node (`label`, `prompt`) |
| `provider` + `model` (legacy exports) | `model` as `provider/model` |
| `skills` | `context` ("Selected skills: …") |
| `approvalRequired: true` | `gate` node, edges rerouted through it |
| Edge dependencies | Plain `source` → `target` edges |
| (implicit graph starts/ends) | Synthetic `start` / `end` nodes |

## What is downgraded (with a warning)

- **Conditions** on edges are not yet supported by the Agent executor; the edge
  is kept as a plain dependency.
- **Feedback/loop** edges are dropped — the Agent executor rejects cycles.
- **`join: "any"`** is downgraded to an all-join (the executor waits for every
  upstream).
- **Coding-agent nodes** (`claude-code`, `codex`) import as Hermes `agent`
  nodes: the visual-workflow executor delegates through Hermes subagents and
  does not spawn external coding agents.

## Exporting from Hermes Studio

In Hermes Studio, open a workflow and use its **export** action to download the
portable definition (a JSON envelope). That file is the input to
`hermes workflows import-studio`.
