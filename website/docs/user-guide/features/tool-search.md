---
title: Tool Search
sidebar_position: 95
---

# Tool Search

When you have many MCP servers or non-core plugin tools attached to a
session, their JSON schemas can consume a substantial fraction of the
context window on every turn — even when only a few of them are relevant
to what the user actually asked for.

**Tool Search** is Hermes' opt-in progressive-disclosure layer for that
problem. When activated, MCP and plugin tools are replaced in the
model-visible tools array by three bridge tools, and the model loads each
specific tool's schema on demand.

:::info Built-in Hermes tools never defer
The tools that make up Hermes' core capability set (`terminal`,
`read_file`, `write_file`, `patch`, `search_files`, `todo`, `memory`,
`browser_*`, `web_search`, `web_extract`, `clarify`, `execute_code`,
`delegate_task`, `session_search`, and the rest of
`_HERMES_CORE_TOOLS`) are *always* loaded directly. Only MCP tools and
non-core plugin tools are eligible for deferral.
:::

## How it works

When Tool Search activates for a turn, the model sees three new tools in
place of the deferred ones:

```
tool_search(queries, limit?)   — search the deferred-tool catalog (one or more queries)
tool_describe(names)           — load the full schemas for one or more tools
tool_call(name, arguments)     — invoke a deferred tool
```

A typical interaction looks like:

```
Model: tool_search(["create a github issue", "send a slack message"])
  → { results: [ { query: "create a github issue",
                   matches: ["mcp_github_create_issue", ...] },
                 { query: "send a slack message",
                   matches: ["mcp_slack_post_message", ...] } ],
      tools: { mcp_github_create_issue: { description: "...",
                                          required: ["title"], ... },
               mcp_slack_post_message: { ... } } }
Model: tool_describe(["mcp_github_create_issue", "mcp_slack_post_message"])
  → { tools: { mcp_github_create_issue: { parameters: { ... } },
               mcp_slack_post_message: { parameters: { ... } } } }
Model: tool_call("mcp_github_create_issue", { title: "...", body: "..." })
  → { ok: true, issue_number: 42 }
```

Each query in a `tool_search` call is searched independently against the
same catalog (`limit` applies per query); the per-query groups carry tool
names only, while the shared `tools` map holds each matched tool's
description and required parameter names once. Queries are stemmed, so
"issues" finds `create_issue`. Each query group that returns no matches
includes an `available_sources` summary of the connected servers so a lexical
miss is not mistaken for a missing capability.
`tool_describe` resolves every requested name in one call; unknown names
are reported in `not_found` without failing the rest of the batch.

When the model invokes `tool_call`, Hermes **unwraps the bridge** and
dispatches the underlying tool exactly as if the model had called it
directly. Pre-tool-call hooks, guardrails, approval prompts, and
post-tool-call hooks all run against the real tool name — not against
`tool_call`. The activity feed in the CLI and gateway also unwraps so you
see the underlying tool, not the bridge.

## When does it activate?

Tool Search uses **tiered disclosure**: the presence of *any* deferrable
(MCP/plugin) tool activates the bridge; what scales with catalog size is
how much of the catalog stays visible, not whether schemas defer.

| Tier | Condition | What the model sees |
| --- | --- | --- |
| **0** | No MCP/plugin tools | Every tool eager, no bridge. Pass-through. |
| **1** | Deferred catalog's listing fits the budget | Bridge + a skills-style manifest of every deferred tool (name + short description, degrading to names-only when over budget). Degradation is **per server**: when one oversized server (Cloudflare) is attached alongside small ones (Linear), the small servers keep their per-tool listings and only the oversized server collapses to a summary line. |
| **2** | Per-tool listing exceeds the budget even names-only for every server (e.g. Cloudflare's flat API surface alone: ~3,300 tools whose names are ~32K tokens) | Bare bridge + a one-line-per-server summary (server name + tool count), so the model knows which domains are reachable; individual tools are discoverable only through `tool_search`. |

The listing budget is `min(threshold_pct% of context, listing_max_tokens)`.
The decision is re-evaluated every time the tools array is built, so
adding or removing MCP servers mid-session moves the session between
tiers on the next assembly.

## Configuration

```yaml
tools:
  tool_search:
    enabled: auto       # auto (default), on, or off
    threshold_pct: 5    # listing budget as a percentage of context
    search_default_limit: 5
    max_search_limit: 25
    listing: auto       # embed a grouped name+description catalog manifest
    listing_max_tokens: 4000
```

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `auto` | `auto`/`on` activate whenever at least one deferrable tool exists; `off` disables entirely (everything stays eager). `auto` is currently an alias of `on` — it is reserved for a future mode that inlines schemas when they fit the context and defers only when they don't. Pin `on` or `off` if you want today's behavior guaranteed across upgrades. |
| `threshold_pct` | `5` | Listing budget as a percentage of the active model's context length. Range 0–100. |
| `search_default_limit` | `5` | Hits returned per query when the model calls `tool_search` without a `limit`. |
| `max_search_limit` | `25` | Hard upper bound the model can request via `limit` (per query). Range 1–50. |
| `listing` | `auto` | Embed a skills-style manifest of every deferred tool (name + first sentence of its description, ≤60 chars, grouped by MCP server) in the `tool_search` bridge description. `auto` includes it when it fits the budget (falling back to names-only, then to the tier-2 server summary); `on`/`off` force either way. |
| `listing_max_tokens` | `4000` | Absolute cap on the embedded listing, regardless of context size. Range 200–60000. Large catalogs degrade to names-only or per-server summaries, keeping full schemas available through search. |

Per-call array caps are internal safety bounds, not configuration. Over-cap
calls return an error so the model can retry with a smaller batch.

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

Tool Search trades a fixed per-turn token cost (the three bridge tool
schemas plus the catalog listing) and at least one extra round trip on
cold tools (describe → call) for the savings on the deferred schemas.
At tier 1 the listing keeps every capability visible, so the discovery
round trip usually disappears — the model goes straight to
`tool_describe`. Live benchmarking showed the listing mode matching
eager loading's task success while costing less than the bare bridge.

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
- **Toolset edits invalidate cache.** Adding or removing a tool mid-
  session changes the bridge tools' descriptions (which include the
  count of deferred tools) and the catalog, so the prompt cache is
  invalidated. This is the same trade-off as any toolset edit.

## Implementation details

- **Retrieval:** BM25 over tokenized tool name, source name (the MCP
  server or plugin toolset the tool belongs to, so searching `"linear"`
  finds that server's tools even when a tool's own name doesn't carry
  the service), description, and parameter names, with Snowball
  stemming (English) applied to both the index and the query so
  morphological variants match ("issues" finds `create_issue`). Falls
  back to a literal substring match on the tool name when no query
  token matches any document (e.g. searching `"hub"` where the token is
  `github`).
- **Parallel execution unwraps the bridge.** The batch planner decides
  concurrency on the *underlying* tool of a `tool_call`, not on the
  literal bridge name — so an MCP server opted in via
  `supports_parallel_tool_calls: true` keeps its concurrency when its
  tools are called through the bridge, and `tool_search` /
  `tool_describe` lookups batch concurrently like any read-only tool.
- **Catalog is stateless across turns.** It rebuilds from the current
  tool-defs list every assembly — no session-keyed `Map`. This avoids
  the class of bug where a stored catalog drifts out of sync with the
  live tool registry.
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
- The `openclaw-tool-search-report` PDF in the original implementation
  PR for the research that shaped the design
