---
sidebar_position: 5
title: "Community Memory Providers"
description: "External memory provider plugins maintained by the community — Memex Zero RAG"
---

# Community Memory Providers

Beyond the 8 built-in providers, the following **standalone memory provider plugins** are available as community-maintained packages. They integrate with Hermes via the same `MemoryProvider` ABC and plugin discovery system — install them into `~/.hermes/plugins/`, and Hermes picks them up automatically.

## Available Community Providers

### Memex Zero RAG

[Memex Zero RAG](https://github.com/JPeetz/MeMex-Zero-RAG) is a citation-first knowledge system inspired by Karpathy's LLM Wiki. The [memex-hermes-plugin](https://github.com/JPeetz/memex-hermes-plugin) exposes it to Hermes as a memory provider that stores individual, cited **Facts** rather than free-form vector chunks.

:::info v0.2.0 status
This plugin is at **v0.2.0** — file-backed local storage, plus remote [MeMex Zero RAG](https://github.com/JPeetz/MeMex-Zero-RAG) backends over MCP (`mcp+stdio://` / `mcp+sse://`). Still no PyPI package — install by git clone. Time-based confidence decay stays backend-side and unshipped. See the [CHANGELOG](https://github.com/JPeetz/memex-hermes-plugin/blob/main/CHANGELOG.md) for the honest status delta.
:::

| | |
|---|---|
| **Best for** | Multi-session project knowledge, citation-critical workflows, teams that want a single-writer store of hand-curated Facts rather than a vector recall soup |
| **Requires** | `git clone` into `~/.hermes/plugins/memex` + `MEMEX_ENDPOINT` set to a `file://`, `mcp+stdio://`, or `mcp+sse://` URL (or `hermes memory setup memex`) |
| **Data storage** | Local filesystem (`file://`, mode `0o700`, JSON facts) or a remote MeMex Zero RAG backend over MCP (`mcp+stdio://` / `mcp+sse://`) |
| **Cost** | Free (open-source, see plugin repo for license) |
| **Tools** | `memex_search`, `memex_read`, `memex_list`, `memex_write`, `memex_flag`, `memex_revalidate` (six total; the last three are write-gated) |

**Key differentiators:**

- **Citations required for `source` / `entity` / `concept` facts** — `memex_write` raises `CitationRequiredError` if a citation field is missing on those fact types.
- **Write-gate on the primary agent** — mutating tools (`memex_write`, `memex_flag`, `memex_revalidate`) are enabled **only** when the plugin is loaded by the primary agent. Subagents and cron jobs get read-only access, preserving a single-writer invariant on the shared store.
- **Bounded prefetch** — on every turn Hermes calls `prefetch(query)`; the plugin runs a bounded 2-second `memex_search()` on a daemon worker thread and injects the top-5 FactRefs as a `<memex-context>` block into the system prompt. Timeouts, empty results, and client failures all yield an empty block silently — no errors surface to the model.
- **Immutable + flag/revalidate workflow** — facts can be flagged for revalidation, then confirmed, updated, or retired via `memex_revalidate`. Attempting to mutate an immutable fact raises `ImmutableFactError`.
- **File-first, MCP for shared/remote use** — `file://` needs no API key at all. `mcp+stdio://` and `mcp+sse://` connect to a real MeMex Zero RAG server; `MEMEX_API_KEY` is sent as an env var to the stdio subprocess or as an `Authorization: Bearer` header over SSE.
- **Atomic + mode-0600 config writes** — the fallback `~/.hermes/memex.json` writer uses `tempfile.mkstemp` + `fchmod 0o600` + atomic rename; no mode-0644 window.

**Setup:**

```bash
# Clone the plugin into the Hermes plugin directory
git clone https://github.com/JPeetz/memex-hermes-plugin.git ~/.hermes/plugins/memex

# Option A — env var (activates on next Hermes start)
export MEMEX_ENDPOINT=file:///path/to/your/memex-store
# or, for a remote MeMex Zero RAG backend:
# export MEMEX_ENDPOINT=mcp+stdio:///path/to/MeMex-Zero-RAG/mcp/server.py
# export MEMEX_ENDPOINT=mcp+sse://your-memex-host:8000

# Option B — interactive setup (writes ~/.hermes/memex.json)
hermes memory setup memex

# Verify
hermes memory status   # should show: memex — Status: available ✓
```

The plugin activates when `MEMEX_ENDPOINT` is set in the environment **or** when `~/.hermes/memex.json` contains an `endpoint` value (whichever is present; env var wins).

**Configuration (env vars):**

| Env var | Required | Default | Purpose |
|---|---|---|---|
| `MEMEX_ENDPOINT` | yes | — | Where the fact store lives: `file://...` for local, `mcp+stdio://...` (spawns a subprocess) or `mcp+sse://host:port` (remote server) for a MeMex Zero RAG backend. |
| `MEMEX_API_KEY` | no | — | Sent to remote MCP endpoints only: env var for `mcp+stdio://` subprocess env, `Authorization: Bearer` header for `mcp+sse://`. Unused for `file://`. |
| `MEMEX_SESSION_SCOPE` | no | `profile` | `session` / `profile` / `global` (stored, not yet enforced — reserved for search filtering). |
| `MEMEX_PREFETCH_TIMEOUT` | no | `2.0` | Prefetch deadline in seconds. Must be a positive finite float. |

**Config file:** `~/.hermes/memex.json` (fallback when env vars are unset)

```json
{
  "endpoint": "file:///path/to/your/memex-store",
  "api_key": null
}
```

Endpoint resolution precedence: `MEMEX_ENDPOINT` env var → `~/.hermes/memex.json` → fallback of `~/.hermes/memex/`.

**Known limitations in v0.2.0:**

- **No BM25 for `file://`** — local search is a substring match over title/body/tags. Remote backends (`mcp+stdio://` / `mcp+sse://`) get the MeMex Zero RAG server's own hybrid FTS5 search for free.
- **No in-plugin decay maths** — confidence is stored/returned as provided; time-based decay is a backend concern.
- **`memex_list` capped at 500 rows** at the tool boundary. Larger stores must paginate via `tags` / `types` / `statuses` filters.
- **`statuses=["retired"]` returns empty** — the client hides retired facts from `list()` before the tool-boundary post-filter runs.
- **`file://` is single-writer** — locking is advisory; concurrent multi-process writes will race. Remote backends inherit whatever write coordination the backend itself implements.
- **Remote `revalidate()` can't restore original confidence** on `confirm`/`update` the way the local `file://` client does — the MeMex Zero RAG backend has no `initial_confidence` field to restore from. A genuine backend schema gap, documented, not a plugin bug.

**One-external-provider invariant:** per Hermes PLUGIN-ABI §Q1, only one external memory provider may be active per Hermes session. If another external provider is already registered, Hermes will refuse to enable this one. Run `hermes memory list` to see which is active.

**Plugin source:** [github.com/JPeetz/memex-hermes-plugin](https://github.com/JPeetz/memex-hermes-plugin) · [DESIGN.md](https://github.com/JPeetz/memex-hermes-plugin/blob/main/DESIGN.md) · [CHANGELOG.md](https://github.com/JPeetz/memex-hermes-plugin/blob/main/CHANGELOG.md)

---

To add your own community memory provider, publish a standalone plugin following the [Memory Provider Plugin guide](/developer-guide/memory-provider-plugin) and open a PR to add it to this section.
