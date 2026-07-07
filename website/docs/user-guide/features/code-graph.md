# Code Graph

Hermes can optionally index a repository into a local read-only code graph and expose compact graph queries to the agent.

## Enable

Run `hermes tools` and enable the `code_graph` toolset for the surfaces where you want it available. Start a new session after changing toolsets.

## Tools

- `code_graph_index` - index or refresh the current repository into the Hermes profile cache.
- `code_graph_status` - check whether the index is missing, fresh, or stale.
- `code_graph_search` - search indexed symbols.
- `code_graph_symbol` - inspect one symbol's definition.
- `code_graph_neighbors` - inspect imports and textual references around a symbol.
- `code_graph_impact` - estimate affected symbols, imports, references, and likely tests for changed paths.
- `code_graph_context` - build a compact implementation context for a goal.

## Storage

Indexes are stored in the Hermes profile cache under `$HERMES_HOME/cache/code_graph/` and are safe to delete. Hermes will rebuild them when `code_graph_index` runs again.

## Limitations

The MVP focuses on Python symbols/imports plus language-agnostic file chunks and textual references. It does not perform type-aware resolution, semantic embeddings, LSP resolution, or automatic refactoring.

