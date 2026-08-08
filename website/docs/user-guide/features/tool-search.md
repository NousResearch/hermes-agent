---
title: Tool Search
sidebar_position: 95
---

# Tool Search

When you have many MCP servers or non-core plugin tools attached to a
session, their JSON schemas can consume a substantial fraction of the
context window on every turn — even when only a few of them are relevant
to what the user actually asked for.

**Tool Search** is Hermes' progressive-disclosure layer for that problem.
MCP and plugin schemas stay out of the model-visible tools array; three small,
byte-stable bridge tools discover the current catalog and load specific schemas
on demand.

:::info Built-in Hermes tools never defer
The tools that make up Hermes' core capability set (`terminal`,
`read_file`, `write_file`, `patch`, `search_files`, `todo`, `memory`,
`browser_*`, `web_search`, `web_extract`, `clarify`, `execute_code`,
`delegate_task`, `session_search`, and the rest of
`_HERMES_CORE_TOOLS`) are *always* loaded directly. Only MCP tools and
non-core plugin tools are eligible for deferral.
:::

## How it works

While Tool Search is enabled, the model always sees the same three bridge tools
in place of deferred schemas—even when no MCP server is currently connected:

```
tool_search(query, limit?)     — search the deferred-tool catalog
tool_describe(name)            — load the full schema for one tool
tool_call(name, arguments)     — invoke a deferred tool
```

A typical interaction looks like:

```
Model: tool_search("create a github issue")
  → { matches: [{ name: "mcp_github_create_issue", ... }, ...] }
Model: tool_describe("mcp_github_create_issue")
  → { parameters: { type: "object", properties: { ... } } }
Model: tool_call("mcp_github_create_issue", { title: "...", body: "..." })
  → { ok: true, issue_number: 42 }
```

When the model invokes `tool_call`, Hermes **unwraps the bridge** and
dispatches the underlying tool exactly as if the model had called it
directly. Pre-tool-call hooks, guardrails, approval prompts, and
post-tool-call hooks all run against the real tool name — not against
`tool_call`. The activity feed in the CLI and gateway also unwraps so you
see the underlying tool, not the bridge.

## Stable bridge and live discovery

The bridge descriptions do not contain a tool count, server summary, or catalog
listing. Every `tool_search`, `tool_describe`, and `tool_call` invocation resolves
the current live registry, filtered to the session's enabled/disabled toolsets.

Keeping the bridge present from the start makes the model-facing `tools=` prefix
identical across MCP additions, removals, equal-count swaps, and schema edits.

The model still needs to know which deferred capabilities exist. Hermes attaches
a compact, skills-style catalog snapshot to the API copy of the first real user
turn. The stored user text stays clean, and the exact API copy is retained for
byte-identical cache replay. Building the snapshot does **not** wait for every
MCP server: it lists the tools currently ready plus the names of in-scope
servers still initializing. On surfaces that start discovery in the background,
those pending names keep unavailable-yet domains discoverable without putting
MCP startup latency back on the chat critical path.

A full-schema fingerprint identifies the snapshot. When membership,
descriptions, parameter schemas, or pending-server state changes, Hermes
attaches a new snapshot to the next real user turn and explicitly supersedes
older ones. The rendered manifest form is part of the identifier too, so a
budget change that exposes a richer or more compact listing is also announced.
Hermes never inserts a standalone synthetic user message, so strict role
alternation is preserved.

The snapshot uses the same size fallbacks as the earlier embedded listing:
names plus short descriptions, names only, then per-server summaries. The live
registry remains authoritative for search, schema loading, and invocation.

## Configuration

```yaml
tools:
  tool_search:
    enabled: auto       # auto (default), on, or off
    search_default_limit: 5
    max_search_limit: 20
    listing: auto       # catalog snapshot on a real user turn
    threshold_pct: 5    # snapshot budget as a percentage of context
    listing_max_tokens: 4000
```

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `auto` | `auto`/`on` keep the stable bridge enabled; `off` exposes eligible schemas directly. |
| `search_default_limit` | `5` | Hits returned when the model calls `tool_search` without a `limit`. |
| `max_search_limit` | `20` | Hard upper bound the model can request via `limit`. Range 1–50. |
| `listing` | `auto` | `auto`/`on` attach a catalog snapshot to the first real user turn and again when its full-schema fingerprint changes; `off` uses a bare stable bridge. The listing is never embedded in a tool schema. |
| `threshold_pct` | `5` | Snapshot budget as a percentage of the active model's context length. Range 0–100. |
| `listing_max_tokens` | `4000` | Absolute cap on the snapshot manifest. Range 200–60000. Large catalogs degrade to names-only or per-server summaries, keeping full schemas available through search. |

### Why the listing exists

Without it, deferred capabilities are *invisible* — live benchmarking showed
models substituting visible core tools (running `gh` in the terminal instead
of searching for the deferred GitHub tool) or declaring a capability
nonexistent instead of calling `tool_search`. The listing applies the skills
pattern to tools: every capability stays discoverable by name at all times,
while full parameter schemas remain deferred. If the model sees the exact
tool name in the listing, it can skip `tool_search` and go straight to
`tool_describe`, saving a round trip.

You can also flip the legacy boolean shape:

```yaml
tools:
  tool_search: true   # equivalent to {enabled: auto}
```

## When NOT to use it

Tool Search trades a fixed per-turn token cost (the three small bridge schemas),
an append-only catalog snapshot on first use/change, and at least one extra round
trip on cold tools (describe → call when the name is listed; otherwise search →
describe → call) for the savings and cache stability of deferred schemas.

If you want the old always-eager behavior for a small toolset, set
`enabled: off`.

## Trade-offs that don't go away

These come from the prompt-cache integrity invariant — they are inherent
to any progressive-disclosure design, not specific to this implementation:

- **One extra round trip on cold tools.** The first time the model needs
  a deferred tool, it spends one or two extra model calls to find and
  load the schema. The token savings on the static side are real, but a
  portion is paid back at runtime.
- **No cache benefit on deferred schemas.** A loaded `tool_describe`
  result enters the conversation history (so it does get cached on
  subsequent turns) but it never benefits from the system-prompt cache
  prefix.
- **Model-quality dependence.** Tool Search assumes the model can write a
  reasonable search query for the tool it wants. Smaller models do this
  less well; the published Anthropic numbers (49% → 74% on Opus 4 with
  vs. without tool search) show the upside but also that ~26 points of
  accuracy is still retrieval failure.
- **Direct mode still invalidates cache.** With `enabled: off`, MCP/plugin
  schemas are exposed directly. Adding, removing, or editing one changes the
  model-facing `tools=` prefix and invalidates the provider prompt cache.

## Implementation details

- **Retrieval:** BM25 over tokenized tool name + description + parameter
  names. Falls back to a literal substring match on the tool name when
  BM25 returns no positive-score hits, which protects against
  zero-IDF degenerate cases (e.g. searching `"github"` against a
  catalog where every tool name contains "github").
- **Catalog is live, not embedded.** Each bridge invocation rebuilds its
  session-scoped view from the current registry. Tool changes therefore take
  effect immediately without replacing the model-facing bridge schemas.
- **Catalog hints are append-only.** Snapshot metadata rides on a real user
  turn and is fingerprinted from the full scoped schemas, rendered manifest,
  and pending MCP server names. A resumed agent reads the prior `api_content`
  sidecar and does not repeat an unchanged snapshot. If the supplied history no
  longer contains the current snapshot, Hermes attaches it again at the
  append-only edge.
- **Compression re-anchors the current snapshot.** A turn can compress after
  its prologue has already decided that an older catalog snapshot is present.
  Before the compressed transcript is committed or the model is called again,
  Hermes attaches the current snapshot to the newest surviving user message's
  API sidecar (or a text part for multimodal content). This introduces no
  synthetic turn and prevents same-turn compaction from erasing tool discovery
  context.
- **Slow MCP startup does not gate the snapshot.** The first snapshot reflects
  the catalog ready when the user turn is assembled and marks slower in-scope
  servers as initializing. When they finish, the next real user turn carries a
  superseding snapshot; bridge calls always consult the live registry in the
  meantime.
- **The catalog is scoped to the session's toolsets.** `tool_search`,
  `tool_describe`, and `tool_call` only ever see and invoke tools the
  session was actually granted. A subagent, kanban worker, or gateway
  session restricted to a subset of toolsets cannot use the bridge to
  discover or call a tool outside that subset — the deferred catalog is
  the deferrable slice of the session's own enabled/disabled toolsets,
  not the whole process registry.
- **No JS sandbox.** Hermes uses the simpler "structured tools" mode
  (search / describe / call as plain functions). The JS-sandbox "code
  mode" some other implementations offer is a large surface area; we
  skip it.

## See also

- `tools/tool_search.py` — the implementation
- `tests/tools/test_tool_search.py` — the regression suite
