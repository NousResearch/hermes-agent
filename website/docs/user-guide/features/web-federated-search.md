---
sidebar_position: 5
title: "Federated Search"
description: "Aggregate results from multiple search backends with LLM-based relevance ranking"
---

# Federated Search

The **federated** web search provider fans out each search request to multiple
configured sub-backends (Tavily, SearXNG, custom HTTP endpoints, etc.) and ranks
the combined results by relevance. It plugs into the existing
`web.search_backend` config key — no changes to the tool chain required.

## Why Federated?

- **Multi-backend resilience.** If one backend is down or rate-limited, results
  from others still come through. Configure Tavily for English search and a
  custom backend (e.g., MiniMax) for Chinese search — both contribute to every
  query.
- **LLM-based ranking.** An auxiliary LLM reranks merged results for relevance.
  Falls back to keyword-based scoring automatically if the LLM is unavailable.
- **Zero tool changes.** The `web_search` tool passes results through the same
  pipeline; the provider boundary is transparent to the agent.

## Quick Start

Enable federated search with two backends:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: federated
  federated:
    backends:
      - name: tavily
      - name: minimax
        type: custom
        base_url: "https://api.minimaxi.com"
        api_key_env: MINIMAX_CN_API_KEY
```

With this configuration, every `web_search` call queries both Tavily and MiniMax
in parallel, merges results, ranks by keyword scoring, and returns up to 8
results.

## Configuration Reference

All federated configuration lives under `web.federated` in `config.yaml`.

### Top-level Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `timeout` | int | `10` | Maximum seconds to wait for all backends. If the deadline expires, partial results from completed backends are returned. |
| `max_results` | int | `8` | Maximum number of results returned after ranking (ceiling). The caller's `limit` parameter is still respected — the actual output count is `min(limit, max_results)`. |
| `ranker` | dict | — | Optional LLM ranking configuration (see below). |
| `backends` | list | — | **Required.** List of sub-backend configurations. |

### Ranking (`ranker`)

The `ranker` section controls how merged results are ordered by relevance.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ranker.provider` | str | — | LLM provider for ranking. Set to `"none"` (or omit) for keyword-based scoring (fast, zero cost). Use a real provider (e.g., `"opencode-go"`) for LLM ranking. |
| `ranker.model` | str | — | Model name for the ranking LLM (e.g., `"deepseek-v4-flash"`). |

**Ranking modes:**
- **Keyword** (`provider: "none"` or unset): Scores results by query term
  frequency in title + description. Fast, no API cost.
- **LLM** (real provider): Sends results to an auxiliary LLM for relevance
  ranking. Falls back to keyword mode automatically on timeout, rate limit, or
  parse failure.

**Recommendation:** Use a non-main-model provider for ranking to avoid
concurrency contention. When ≤ 3 results, ranking is skipped entirely.

### Backend Configurations (`backends`)

Each entry in the `backends` list represents one search source.

#### Registered Providers

Use an existing Hermes web search provider (e.g., Tavily, Firecrawl, SearXNG):

```yaml
backends:
  - name: tavily
  - name: firecrawl
```

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `name` | str | ✅ | Registered provider name. Must match an available provider (tavily, firecrawl, searxng, exa, parallel, ddgs, brave_free, xai). |

Registered providers use their existing credentials (`.env` API keys). No
additional configuration needed.

#### Custom HTTP Backends

Connect any search API that accepts a query parameter and returns JSON:

```yaml
backends:
  - name: minimax
    type: custom
    base_url: "https://api.minimaxi.com"
    api_key_env: MINIMAX_CN_API_KEY
    search_path: /v1/coding_plan/search
    query_param: "q"
    auth_style: bearer
```

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `name` | str | ✅ | — | Display name for logging and error messages. |
| `type` | str | ✅ | — | Must be `"custom"` to enable custom HTTP mode. |
| `base_url` | str | ✅ | — | Base URL of the search API (e.g., `https://api.minimaxi.com`). |
| `api_key_env` | str | ✅ | — | Environment variable that holds the API key (e.g., `MINIMAX_CN_API_KEY`). Set in `.env`. |
| `search_path` | str | — | `"/v1/coding_plan/search"` | API path appended to `base_url`. |
| `query_param` | str | — | `"q"` | Query string parameter name. |
| `auth_style` | str | — | `"bearer"` | Authentication style: `"bearer"` (Authorization header) or `"x-api-key"` (x-api-key header). |

**Supported response shapes:** The custom backend parser handles three common
API response formats:

- `{"organic": [{"title", "link", "snippet"}, ...]}`
- `{"data": {"web": [{"title", "url", "description"}, ...]}}`
- `{"results": [{"title", "url", "content"}, ...]}`

## Execution Flow

For each search request:

1. **Parallel fan-out.** All backends are queried concurrently via
   `ThreadPoolExecutor`. Each backend runs in its own thread.
2. **Timeout enforcement.** `concurrent.futures.wait()` with the configured
   timeout bounds the total wait time. Backends that haven't finished are
   cancelled; completed results proceed.
3. **Merge & rank.** Results from all successful backends are combined. If a
   `ranker` is configured, the merged set is ranked by relevance (LLM or keyword).
4. **Top N truncation.** Output count is capped at `min(limit, max_results)` —
   the caller's requested limit is respected, with `max_results` serving as a
   ceiling.

## Full Configuration Example

```yaml
web:
  search_backend: federated
  federated:
    # Wait up to 15 seconds for all backends
    timeout: 15

    # Return at most 10 results after ranking
    max_results: 10

    # Use keyword-based ranking (fast, zero LLM cost)
    ranker:
      provider: none

    backends:
      # Registered providers
      - name: tavily
      - name: searxng

      # Custom HTTP backend — MiniMax Chinese search
      - name: minimax
        type: custom
        base_url: "https://api.minimaxi.com"
        api_key_env: MINIMAX_CN_API_KEY
        search_path: /v1/coding_plan/search
```

## Troubleshooting

| Symptom | Likely Cause |
|---------|-------------|
| `federated search not configured` | `web.federated` section missing from `config.yaml` or `web.search_backend` not set to `"federated"` |
| `no search backends configured` | `web.federated.backends` is empty or not a list |
| `backend '<name>' not registered` | `name` references an unavailable provider. Check `hermes tools` to see registered providers. |
| `All backends failed` | Every backend returned errors. Check credentials and network. The gateway log shows per-backend details. |
| `Custom backend timed out` | The configured API didn't respond within 15s. Verify the `base_url` and `search_path` are correct. |
| `LLM ranking failed, falling back to keyword ranking` | Normal fallback — the ranking LLM was unavailable. Results use keyword scoring instead. |
| `load_config() strips web.federated` | This is expected. The provider reads raw YAML directly; config changes take effect on next search call without restart. |

## Known Limitations

- **No content extraction.** `supports_extract()` returns `False`. Use another
  provider (e.g., Tavily or Firecrawl) for `web_extract` by setting
  `web.extract_backend`.
- **No nested federation.** Sub-backends are leaf nodes; you cannot chain
  federated providers.
- **Thread-pool overhead.** Each backend runs in a thread. With many backends
  (>10), consider using fewer, higher-quality providers instead.
