---
title: Web Search & Extract
description: Search the web, extract page content, and crawl websites with multiple backend providers — including free self-hosted SearXNG.
sidebar_label: Web Search
sidebar_position: 6
---

# Web Search & Extract

Hermes Agent includes two model-callable web tools backed by multiple providers:

- **`web_search`** — search the web and return ranked results
- **`web_extract`** — fetch and extract readable content from one or more URLs (with built-in deep-crawl support when the backend provides it)

Both are configured through a single backend selection. Providers are chosen via `hermes tools` or set directly in `config.yaml`. Recursive crawling capabilities (Firecrawl/Tavily) are exposed through `web_extract` rather than as a separate `web_crawl` tool.

## Backends

| Provider | Env Var | Search | Extract | Crawl | Free tier |
|----------|---------|--------|---------|-------|-----------|
| **Firecrawl** (default) | `FIRECRAWL_API_KEY` | ✔ | ✔ | ✔ | 500 credits/mo |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | — | ✔ Free (self-hosted) |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | ✔ | 1 000 searches/mo |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | — | 1 000 searches/mo |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | — | Paid |

**Per-capability split:** you can use different providers for search and extract independently — for example SearXNG (free) for search and Firecrawl for extract. See [Per-capability configuration](#per-capability-configuration) below.

:::tip Nous Subscribers
If you have a paid [Nous Portal](https://portal.nousresearch.com) subscription, web search and extract are available through the **[Tool Gateway](tool-gateway.md)** via managed Firecrawl — no API key needed. Run `hermes tools` to enable it.
:::

---

## How `web_extract` handles long pages

Backends return raw page markdown, which can be huge (forum threads, docs sites, news articles with embedded comments). To keep your context window usable and your costs down, `web_extract` runs returned content through the **`web_extract` auxiliary model** before handing it to the agent. Behavior is purely size-driven:

| Page size (characters) | What happens |
|------------------------|--------------|
| Under 5 000 | Returned as-is — no LLM call, full markdown reaches the agent |
| 5 000 – 500 000 | Single-pass summary via the `web_extract` auxiliary model, capped at ~5 000 chars of output |
| 500 000 – 2 000 000 | Chunked: split into 100 k-char chunks, summarize each in parallel, then synthesize a final summary (~5 000 chars) |
| Over 2 000 000 | Refused with a hint to use `web_crawl` with focused extraction instructions or a more specific source |

The summary keeps quotes, code blocks, and key facts in their original formatting — it's a content compressor, not a paraphraser. If summarization fails or times out, Hermes falls back to the first ~5 000 chars of raw content rather than a useless error.

### Which model does the summarizing?

The `web_extract` auxiliary task. By default (`auxiliary.web_extract.provider: "auto"`), this is your **main chat model** — same provider, same model as `hermes model`. That's fine for most setups, but on expensive reasoning models (Opus, MiniMax M2.7, etc.) every long-page extract adds meaningful cost.

To route extraction summaries to a cheap, fast model regardless of your main:

```yaml
# ~/.hermes/config.yaml
auxiliary:
  web_extract:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 360       # seconds; raise if you hit summarization timeouts
```

Or pick interactively: `hermes model` → **Configure auxiliary models** → `web_extract`.

See [Auxiliary Models](/docs/user-guide/configuration#auxiliary-models) for the full reference and per-task override patterns.

### When summarization gets in the way

If you specifically need raw, unsummarized page content — for example, you're scraping a structured page where the LLM summary would drop important fields — use `browser_navigate` + `browser_snapshot` instead. The browser tool returns the live accessibility tree without auxiliary-model rewriting (subject to its own 8 000-char snapshot cap on huge pages).

---

## Setup

### Quick setup via `hermes tools`

Run `hermes tools`, navigate to **Web Search & Extract**, and pick a provider. The wizard prompts for the required URL or API key and writes it to your config.

```bash
hermes tools
```

---

### Firecrawl (default)

Full-featured search, extract, and crawl. Recommended for most users.

```bash
# ~/.hermes/.env
FIRECRAWL_API_KEY=fc-your-key-here
```

Get a key at [firecrawl.dev](https://firecrawl.dev). The free tier includes 500 credits/month.

**Self-hosted Firecrawl:** Point at your own instance instead of the cloud API:

```bash
# ~/.hermes/.env
FIRECRAWL_API_URL=http://localhost:3002
```

When `FIRECRAWL_API_URL` is set, the API key is optional (disable server auth with `USE_DB_AUTHENTICATION=false`).

---

### SearXNG (free, self-hosted)

SearXNG is a privacy-respecting, open-source metasearch engine that aggregates results from 70+ search engines. **No API key required** — just point Hermes at a running SearXNG instance.

SearXNG is **search-only** — `web_extract` (including its crawl modes) requires a separate extract provider.

#### Option A — Self-host with Docker (recommended)

This gives you a private instance with no rate limits.

**1. Create a working directory:**

```bash
mkdir -p ~/searxng/searxng
cd ~/searxng
```

**2. Write a `docker-compose.yml`:**

```yaml
# ~/searxng/docker-compose.yml
services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8888:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://localhost:8888/
    restart: unless-stopped
```

**3. Start the container:**

```bash
docker compose up -d
```

**4. Enable the JSON API format:**

SearXNG ships with JSON output disabled by default. Copy the generated config and enable it:

```bash
# Copy the auto-generated config out of the container
docker cp searxng:/etc/searxng/settings.yml ~/searxng/searxng/settings.yml
```

Open `~/searxng/searxng/settings.yml` and find the `formats` block (around line 84):

```yaml
# Before (default — JSON disabled):
formats:
  - html

# After (enable JSON for Hermes):
formats:
  - html
  - json
```

**5. Restart to apply:**

```bash
docker cp ~/searxng/searxng/settings.yml searxng:/etc/searxng/settings.yml
docker restart searxng
```

**6. Verify it works:**

```bash
curl -s "http://localhost:8888/search?q=test&format=json" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(f'{len(d[\"results\"])} results')"
```

You should see something like `10 results`. If you get a `403 Forbidden`, JSON format is still disabled — recheck step 4.

**7. Configure Hermes:**

```bash
# ~/.hermes/.env
SEARXNG_URL=http://localhost:8888
```

Then select SearXNG as the search backend in `~/.hermes/config.yaml`:

```yaml
web:
  search_backend: "searxng"
```

Or set via `hermes tools` → Web Search & Extract → SearXNG.

---

#### Option B — Use a public instance

Public SearXNG instances are listed at [searx.space](https://searx.space/). Filter by instances that have **JSON format enabled** (shown in the table).

```bash
# ~/.hermes/.env
SEARXNG_URL=https://searx.example.com
```

:::caution Public instances
Public instances have rate limits, variable uptime, and may disable JSON format at any time. For production use, self-hosting is strongly recommended.
:::

---

#### Pair SearXNG with an extract provider

SearXNG handles search; you need a separate provider for `web_extract` (including any deep-crawl modes). Use the per-capability keys:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"   # or tavily, exa, parallel
```

With this config, Hermes uses SearXNG for all search queries and Firecrawl for URL extraction — combining free search with high-quality extraction.

---

### Tavily

AI-optimised search, extract, and crawl with a generous free tier.

```bash
# ~/.hermes/.env
TAVILY_API_KEY=tvly-your-key-here
```

Get a key at [app.tavily.com](https://app.tavily.com/home). The free tier includes 1 000 searches/month.

---

### Exa

Neural search with semantic understanding. Good for research and finding conceptually related content.

```bash
# ~/.hermes/.env
EXA_API_KEY=your-exa-key-here
```

Get a key at [exa.ai](https://exa.ai). The free tier includes 1 000 searches/month.

---

### Parallel

AI-native search and extraction with deep research capabilities.

```bash
# ~/.hermes/.env
PARALLEL_API_KEY=your-parallel-key-here
```

Get access at [parallel.ai](https://parallel.ai).

---

## Configuration

### Single backend

Set one provider for all web capabilities:

```yaml
# ~/.hermes/config.yaml
web:
  backend: "searxng"   # firecrawl | searxng | tavily | exa | parallel
```

### Per-capability configuration

Use different providers for search vs extract. This lets you combine free search (SearXNG) with a paid extract provider, or vice versa:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"     # used by web_search
  extract_backend: "firecrawl"  # used by web_extract (and its deep-crawl modes)
```

When per-capability keys are empty, both fall through to `web.backend`. When `web.backend` is also empty, the backend is auto-detected from whichever API key/URL is present.

**Priority order (per capability):**
1. `web.search_backend` / `web.extract_backend` (explicit per-capability)
2. `web.backend` (shared fallback)
3. Auto-detect from environment variables

### Auto-detection

If no backend is explicitly configured, Hermes picks the first available one based on which credentials are set:

| Credential present | Auto-selected backend |
|--------------------|-----------------------|
| `FIRECRAWL_API_KEY` or `FIRECRAWL_API_URL` | firecrawl |
| `PARALLEL_API_KEY` | parallel |
| `TAVILY_API_KEY` | tavily |
| `EXA_API_KEY` | exa |
| `SEARXNG_URL` | searxng |

---

## Verify your setup

Run `hermes setup` to see which web backend is detected:

```
✅ Web Search & Extract (searxng)
```

Or check via the CLI:

```bash
# Activate the venv and run the web tools module directly
source ~/.hermes/hermes-agent/.venv/bin/activate
python -m tools.web_tools
```

This prints the active backend and its status:

```
✅ Web backend: searxng
   Using SearXNG (search only): http://localhost:8888
```

---

## Troubleshooting

### `web_search` returns `{"success": false}`

- Check `SEARXNG_URL` is reachable: `curl -s "http://localhost:8888/search?q=test&format=json"`
- If you get HTTP 403, JSON format is disabled — add `json` to the `formats` list in `settings.yml` and restart
- If you get a connection error, the container may not be running: `docker ps | grep searxng`

### `web_extract` says "search-only backend"

SearXNG cannot extract URL content. Set `web.extract_backend` to a provider that supports extraction:

```yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"  # or tavily / exa / parallel
```

### SearXNG returns 0 results

Some public instances disable certain search engines or categories. Try:
- A different query
- A different public instance from [searx.space](https://searx.space/)
- Self-hosting your own instance for reliable results

### Rate limited on a public instance

Switch to a self-hosted instance (see [Option A](#option-a--self-host-with-docker-recommended) above). With Docker, your own instance has no rate limits.

### `web_extract` returns truncated content with a "summarization timed out" note

The auxiliary model didn't finish summarizing within the configured timeout. Either:

- Raise `auxiliary.web_extract.timeout` in `config.yaml` (default 360s on fresh installs, 30s if the key is missing)
- Switch the `web_extract` auxiliary task to a faster model (e.g. `google/gemini-3-flash-preview`) — see [How `web_extract` handles long pages](#how-web_extract-handles-long-pages)
- For pages where summarization is the wrong tool, use `browser_navigate` instead

---

## Optional skill: `searxng-search`

For agents that need to use SearXNG via `curl` directly (e.g. as a fallback when the web toolset isn't available), install the `searxng-search` optional skill:

```bash
hermes skills install official/research/searxng-search
```

This adds a skill that teaches the agent how to:
- Call the SearXNG JSON API via `curl` or Python
- Filter by category (`general`, `news`, `science`, etc.)
- Handle pagination and error cases
- Fall back gracefully when SearXNG is unreachable
