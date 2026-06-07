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
`delegate_task`, `session_search`, `send_message`, and the rest of
`_HERMES_CORE_TOOLS`) are *always* loaded directly. Only MCP tools and
non-core plugin tools are eligible for deferral.
:::

## How it works

When Tool Search activates for a turn, the model sees three new tools in
place of the deferred ones:

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

## When does it activate?

By default Tool Search runs in `auto` mode: it activates only when the
deferrable tool schemas would consume at least 10% of the active model's
context window. Below that, the tools-array assembly is a pure
pass-through and you pay no overhead.

This decision is re-evaluated every time the tools array is built, so:

- A session with just a few MCP tools and a long context model never
  activates Tool Search.
- A session with many MCP servers attached (15+ tools typically) starts
  activating it.
- Removing MCP servers mid-session correctly returns to direct exposure
  on the next assembly.

## Configuration

```yaml
tools:
  tool_search:
    enabled: auto       # auto (default), on, or off
    threshold_pct: 10   # percentage of context — only used in auto mode
    search_default_limit: 5
    max_search_limit: 20
```

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `auto` | `auto` activates above threshold; `on` always activates if there's at least one deferrable tool; `off` disables entirely. |
| `threshold_pct` | `10` | Percentage of context length at which `auto` mode kicks in. Range 0–100. |
| `search_default_limit` | `5` | Hits returned when the model calls `tool_search` without a `limit`. |
| `max_search_limit` | `20` | Hard upper bound the model can request via `limit`. Range 1–50. |

You can also flip the legacy boolean shape:

```yaml
tools:
  tool_search: true   # equivalent to {enabled: auto}
```

## When NOT to use it

Tool Search trades a fixed per-turn token cost (the three bridge tool
schemas, ~300 tokens) and at least one extra round trip (search →
describe → call) for the savings on the deferred schemas. It's a clear
win when you have many tools and use few per turn; it's overhead when
you have few tools total.

The `auto` default handles this for you. If you set `enabled: on`
unconditionally, expect a slight per-turn cost on small toolsets.

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

- **Retrieval:** BM25 over tokenized tool name + description + parameter
  names. Falls back to a literal substring match on the tool name when
  BM25 returns no positive-score hits, which protects against
  zero-IDF degenerate cases (e.g. searching `"github"` against a
  catalog where every tool name contains "github").
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

## Embedding Reranker (issue #13332)

The optional embedding reranker improves Recall@5 significantly beyond
BM25 for semantic queries. It is **default OFF** — enabling it requires
an OpenAI-compatible embeddings endpoint.

**Benchmark results** (194 tools / 98 labeled queries,
nomic-embed-text-v2-moe, fulfils NousResearch/hermes-agent#13332):

| Mode | R@5 | vs BM25 baseline | Notes |
|------|-----|-----------------|-------|
| BM25 only (default) | **0.634** | — | Stock default, no new deps |
| `rerank` (pure cosine, full catalog) | **0.810** | +0.176 | Highest accuracy |
| `rrf` k=10 (RRF fusion) | **0.770** | +0.136 | Zero regressions, safe default |

**Critical finding: nomic-embed-text-v2-moe REQUIRES task prefixes.**
Without `search_query:` and `search_document:` prefixes, R@5 drops by
~0.194. The config defaults include these prefixes; set to `""` for
models that do not want them.

:::caution Prefix mismatch silently degrades accuracy
The `query_prefix` and `doc_prefix` default to `"search_query: "` and
`"search_document: "` (nomic-embed-text-v2-moe task prefixes). Models that
do **not** use task prefixes — such as `all-MiniLM-L6-v2`, `e5-small-v2`,
and OpenAI's `text-embedding-3-*` family — **must** set **both** to `""`
or accuracy silently degrades by approximately −0.19 R@5:

```yaml
tools:
  tool_search:
    reranker:
      enabled: true
      endpoint: http://localhost:11434/v1/embeddings
      model: all-MiniLM-L6-v2
      query_prefix: ""    # ← required for non-nomic models
      doc_prefix: ""      # ← required for non-nomic models
```

The mismatch is silent because the model still produces valid-shaped vectors;
cosine scores are just much lower (nomic treats an unprefixed query as a
document, not a query), and the fallback to BM25 does not trigger. Always
verify prefix requirements in your embedding model's documentation.
:::

**Full-catalog retrieve:** tool embeddings are cached — only one query embed
call recurs per search. Narrow retrieval (N=10) leaves +0.119 R@5 on the
table compared to full-catalog (N=194). The implementation always retrieves
the full catalog.

**Endpoint compatibility:** any OpenAI-compatible `/v1/embeddings` endpoint.
The benchmark used a GPU-hosted nomic model, but CPU models such as
`all-MiniLM-L6-v2` at ~200 ms latency are equally valid. No GPU required.
Pi-friendly.

**Graceful fallback:** any endpoint failure (timeout, non-200, bad JSON,
shape mismatch) → the module logs a debug line and returns the BM25 result
unchanged. Tool discovery is never blocked by an unavailable reranker.

```yaml
tools:
  tool_search:
    reranker:
      enabled: true                                   # default false
      endpoint: http://localhost:11434/v1/embeddings  # required when enabled
      model: nomic-embed-text-v2-moe
      mode: rerank       # "rerank" (pure cosine) or "rrf" (RRF fusion)
      rrf_k: 10          # RRF k parameter — k=10 beat k=60 in benchmarks
      top_k: 5           # results to return (should match search_default_limit)
      query_prefix: "search_query: "    # nomic task prefix for queries
      doc_prefix:   "search_document: " # nomic task prefix for tool docs
      api_key: ""        # optional bearer token
      timeout: 5.0       # seconds before fallback to BM25
```

| Key | Default | Meaning |
| --- | --- | --- |
| `reranker.enabled` | `false` | Enable embedding reranker. |
| `reranker.endpoint` | `""` | OpenAI-compatible `/v1/embeddings` URL. Required when enabled. |
| `reranker.model` | `nomic-embed-text-v2-moe` | Embedding model name. |
| `reranker.mode` | `rerank` | `rerank`: sort by cosine similarity. `rrf`: Reciprocal Rank Fusion of BM25 + embedding ranks. |
| `reranker.rrf_k` | `10` | RRF k parameter. Lower values boost top-ranked items more. k=10 outperformed the textbook k=60 in benchmarks. |
| `reranker.top_k` | `5` | Number of results to return. Should match `search_default_limit`. |
| `reranker.query_prefix` | `"search_query: "` | Task prefix prepended to the query embed text. Set to `""` for models that do not use prefixes. |
| `reranker.doc_prefix` | `"search_document: "` | Task prefix prepended to each tool's embed text. Set to `""` for models that do not use prefixes. |
| `reranker.api_key` | `""` | Bearer token for authenticated endpoints. |
| `reranker.timeout` | `5.0` | HTTP timeout in seconds. After expiry, falls back to BM25. |

### Three-tier architecture

| Tier | What | Dependencies | R@5 (194 tools) |
|------|------|-------------|-----------------|
| 0 (default) | BM25 only | none (stdlib) | ~0.634 |
| 1 rerank | BM25 + embedding cosine rerank | embeddings endpoint | ~0.810 |
| 1 rrf | BM25 + RRF fusion | embeddings endpoint | ~0.770 |
| 2 (future) | cross-encoder rerank | reranker endpoint | est. ~0.85–0.89 |

Tier 0 is the stock default with no external dependencies. Tier 1 adds
the embedding reranker (opt-in). Tier 2 (cross-encoder) is the documented
next step but not yet implemented; it requires a `/rerank`-style endpoint
(e.g. `bge-reranker-v2-m3` in Ollama, or Cohere's rerank API). Literature
estimates suggest +3–8% Recall@5 over Tier 1 on well-formed tool-selection
corpora.

## See also

- `tools/tool_search.py` — the implementation
- `tests/tools/test_tool_search.py` — the regression suite
- `/tmp/tool-rerank-poc/BENCHMARK_WRITEUP.md` — benchmark evidence for the
  hybrid pre-selection design (issue #13332)
- The `openclaw-tool-search-report` PDF in the original implementation
  PR for the research that shaped the base design
