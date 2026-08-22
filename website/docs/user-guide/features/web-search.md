---
title: Web Search & Extract
description: Search the web and extract page content with multiple backend providers — including free self-hosted SearXNG.
sidebar_label: Web Search
sidebar_position: 6
---

# Web Search & Extract

Hermes Agent includes two model-callable web tools backed by multiple providers:

- **`web_search`** — search the web and return ranked results
- **`web_extract`** — fetch and extract readable content from one or more URLs

Both are configured through a single backend selection. Providers are chosen via `hermes tools` or set directly in `config.yaml`.

## Backends

| Provider | Env Var | Search | Extract | Free tier |
|----------|---------|--------|---------|-----------|
| **Firecrawl** (default) | `FIRECRAWL_API_KEY` (optional — keyless when selected) | ✔ | ✔ | 500 credits/mo · keyless cloud when selected |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | ✔ Free (self-hosted) |
| **Brave Search (free tier)** | `BRAVE_SEARCH_API_KEY` | ✔ | — | 2 000 queries/mo |
| **DDGS (DuckDuckGo)** | — (no key) | ✔ | — | ✔ Free |
| **Tavily** | `TAVILY_API_KEY` (optional) | ✔ | ✔ | ✔ Keyless ring member · 1 000 searches/mo with a free key |
| **Exa** | `EXA_API_KEY` (optional) | ✔ | ✔ | ✔ Keyless ring member · 1 000 searches/mo with key |
| **Parallel** | `PARALLEL_API_KEY` (optional) | ✔ | ✔ | ✔ Keyless ring member · paid with key |
| **Keenable** | `KEENABLE_API_KEY` (optional) | ✔ | ✔ | ✔ Keyless ring member · paid with key |
| **Anthropic (native)** | `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` (the model's own credential — Anthropic-served models only) | ✔ | ✔ | Paid (per-search charge + tokens) |
| **xAI (Grok)** | `XAI_API_KEY` or `hermes auth add xai-oauth` | ✔ | — | Paid (SuperGrok or per-token) |

Brave Search, DDGS, and xAI are **search-only** — pair any of them with Firecrawl/Tavily/Exa/Parallel when you also need `web_extract`. DDGS uses the [`ddgs` Python package](https://pypi.org/project/ddgs/) under the hood; if it isn't already installed, run `pip install ddgs` (or let Hermes lazy-install it on first use). xAI runs Grok's server-side `web_search` tool on the Responses API — results are LLM-generated rather than index-backed, so titles, descriptions, and URL choice are all model output (see the [trust-model caveat](#xai-grok) below).

**Per-capability split:** you can use different providers for search and extract independently — for example SearXNG (free) for search and Firecrawl for extract. See [Per-capability configuration](#per-capability-configuration) below.

:::info Works out of the box — keyless free-tier rotation
A fresh install with **no web credentials at all** gets working `web_search` and `web_extract` out of the box: requests rotate round-robin across five vendors' public free tiers — **Exa, Parallel, Tavily, Firecrawl, and Keenable** — spreading load evenly, and a rate-limited request automatically retries on the next vendor in the ring (multi-hop, until one serves or all are throttled). No signup, no key. This tier is strictly last-resort — any configured backend or present API key always wins — and requests carry no user identifiers (only a random per-process session id, rotated on restart). For guaranteed, unthrottled service, set up a keyed provider. Disable the keyless tier entirely with `web.keyless_fallback: false`.
:::

**Choosing free vs paid explicitly:** in `hermes tools`, Exa, Parallel, and Keenable each appear as two rows — **Free (keyless)** and **Paid (API key)**. Picking Free pins that vendor's anonymous endpoint (even if you later add a key); picking Paid pins the keyed path (a missing key then errors instead of silently downgrading to the free tier). The selection is stored as `web.provider_tier.<name>: free|paid`; leave it unset for auto (key present → paid, otherwise the keyless ring).

**Anthropic (native) runs inside the model request:** selecting `anthropic` maps `web_search` and `web_extract` onto Anthropic's server-side `web_search` and `web_fetch` tools. They execute inside the Messages API call, reuse the same Anthropic credential as the model, and return source citations in Claude's response — and they only ever execute there. There is no client-side fallback, no other transport carries them, and compatible third-party Anthropic endpoints are not assumed to host them. Hermes therefore counts `anthropic` as usable only while the model is served by Anthropic itself: when every web capability routes to `anthropic` on a model served elsewhere, the `web` toolset reports as unconfigured rather than advertising tools no request can carry. `hermes tools` keeps listing Anthropic as your stored selection — the picker never overrides a choice you made — but marks the row **not usable on the current model**. The keyless free tier does not cover that gap — an explicit `web.backend` outranks it — so web stays off until you select a backend the active model can reach. If another backend keeps `web` enabled while `anthropic` stays selected for just one capability, that tool remains callable and reports that it cannot run locally (nothing is dispatched, so the one-shot keyless rescue has nothing to rescue). See [Anthropic (native)](#anthropic-native) for setup.

:::tip Nous Subscribers
If you have a paid [Nous Portal](https://portal.nousresearch.com) subscription, web search and extract are available through the **[Tool Gateway](tool-gateway.md)** via managed Firecrawl — no API key needed. New installs can run `hermes setup --portal` to log in and turn on all gateway tools at once; existing installs can flip just web via `hermes tools`.
:::

---

## How `web_extract` handles long pages

Backends return raw page markdown, which can be huge (forum threads, docs sites, news articles with embedded comments). To keep your context window usable, `web_extract` applies a **deterministic character budget** — no LLM summarization is involved:

| Page size (characters) | What happens |
|------------------------|--------------|
| At or under the budget (default 15 000) | Returned whole — full markdown reaches the agent |
| Over the budget | Head+tail window (~75% head / ~25% tail, cut on markdown line boundaries) plus an explicit `[TRUNCATED]` footer. The full clean text is stored to disk and the footer tells the agent the file path and the exact `read_file` call to page through the omitted middle |
| Over 2 000 000 | Stored text is capped at 2 MB |

The per-page budget is configurable via `web.extract_char_limit` in `config.yaml` (default `15000`, clamped to 2 000–500 000), and the agent can raise it per-call with the tool's `char_limit` argument.

### When truncation gets in the way

If you specifically need the live DOM rather than extracted markdown — for example, a JS-heavy page where extraction returns little content — use `browser_navigate` + `browser_snapshot` instead. The browser tool returns the live accessibility tree (subject to its own snapshot cap on huge pages).

---

## Setup

### Quick setup via `hermes tools`

Run `hermes tools`, navigate to **Web Search & Extract**, and pick a provider. The wizard prompts for the required URL or API key and writes it to your config.

```bash
hermes tools
```

---

### Firecrawl (default)

Full-featured search and extract. Recommended for most users.

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

SearXNG is **search-only** — `web_extract` requires a separate extract provider.

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

Open `~/searxng/searxng/settings.yml`.
If `use_default_settings: true` is present, the file only contains your overrides. All other settings are inherited from the built-in defaults.
To enable JSON responses for Hermes, add the following override:

```yaml
search:
  formats:
    - html
    - json
```

Your `settings.yml` should look similar to:

```yaml
# Read the documentation before extending the defaults:
# https://docs.searxng.org/admin/settings/

use_default_settings: true

server:
  secret_key: "abcdef12345678"
  image_proxy: true

search:
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

SearXNG handles search; you need a separate provider for `web_extract`. Use the per-capability keys:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"   # or tavily, exa, parallel
```

With this config, Hermes uses SearXNG for all search queries and Firecrawl for URL extraction — combining free search with high-quality extraction.

---

### Tavily

AI-optimised search and extract. Select Tavily in `hermes tools` (or set `web.backend: tavily`) to use it **keyless** with no account (rate-limited). Set an API key when you want higher limits.

```bash
# optional — skip this for keyless access after selecting Tavily
# ~/.hermes/.env
TAVILY_API_KEY=tvly-your-key-here
```

Get a key at [app.tavily.com](https://app.tavily.com/home). See [Tavily keyless](https://docs.tavily.com/documentation/keyless).

Empty installs keep Firecrawl as the named default. Keyless Tavily is not auto-selected.

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

### Anthropic (native) {#anthropic-native}

Choose **Anthropic Web Search & Fetch** in `hermes tools`, or configure it directly:

```yaml
web:
  backend: anthropic
```

No additional credential is required when `ANTHROPIC_API_KEY` (or `CLAUDE_CODE_OAUTH_TOKEN`) was already configured for the model, but the model itself must be served by Anthropic's own API — the key alone does not enable this backend elsewhere. Search uses `web_search_20250305`; fetch uses `web_fetch_20250910` with citations enabled. Both are capped at five uses per model request, and fetched page text is capped at roughly 25 000 tokens so a single large page cannot fill the context window — note that Anthropic applies this limit to text only, not to binary content such as PDFs. Anthropic web search has a per-search charge in addition to normal token usage.

**No `[TRUNCATED]` footers here:** `web.extract_char_limit` and the head+tail window described [above](#how-web_extract-handles-long-pages) bound results that pass through Hermes, and the native fetch never does — it runs inside the Messages API request, so the ~25 000-token server-side cap is what bounds it instead.

---

### xAI (Grok) {#xai-grok}

Routes `web_search` through Grok's server-side [web_search tool](https://docs.x.ai/developers/tools/web-search) on the Responses API. Grok runs the actual searching and returns the top results as structured JSON.

Works with either credential path — no new env vars, no new setup wizard:

```bash
# ~/.hermes/.env (env-var path)
XAI_API_KEY=sk-xai-your-key-here
```

or for SuperGrok subscribers:

```bash
hermes auth add xai-oauth
```

Then select xAI as the search backend:

```yaml
# ~/.hermes/config.yaml
web:
  backend: "xai"
```

**Optional knobs:**

```yaml
web:
  backend: "xai"
  xai:
    model: grok-build-0.1        # reasoning model required by web_search (default)
    allowed_domains:             # optional, max 5 — mutex with excluded_domains
      - arxiv.org
    excluded_domains:            # optional, max 5
      - example-spam.com
    timeout: 90                  # seconds (default)
```

**Search-only** — pair with Firecrawl / Tavily / Exa / Parallel if you also need `web_extract`. On 401 the provider performs a single forced OAuth-token refresh and retries (covers mid-window revocation and opaque tokens the proactive expiry check can't decode); env-var credentials skip the retry.

:::caution Trust model
Unlike index-backed providers (Brave, Tavily, Exa) which return verbatim search-engine results, xAI is an LLM choosing which URLs to surface and writing the titles and descriptions itself. The *content* of the query influences the output, so a maliciously crafted query (e.g. injected via untrusted upstream input the agent picked up) can in principle steer Grok into emitting attacker-chosen URLs. Treat returned URLs the same way you'd treat any model-generated link — validate before fetching, especially if the query came from untrusted input.
:::

---

## Configuration

### Single backend

Set one provider for all web capabilities:

```yaml
# ~/.hermes/config.yaml
web:
  backend: "searxng"   # firecrawl | searxng | brave-free | ddgs | tavily | exa | parallel | anthropic | xai
```

### Per-capability configuration

Use different providers for search vs extract. This lets you combine free search (SearXNG) with a paid extract provider, or vice versa:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"     # used by web_search
  extract_backend: "firecrawl"  # used by web_extract
```

When per-capability keys are empty, both fall through to `web.backend`. Only when no web selection has ever been written is the backend auto-detected from whichever API key/URL is present — once a selection exists, the runtime always uses it, and adding a key to `.env` does not reroute web traffic.

**Priority order (per capability):**
1. `web.search_backend` / `web.extract_backend` (explicit per-capability)
2. `web.backend` (shared fallback; `nous` = managed Tool Gateway)
3. Auto-detect from environment variables (never-configured setups only)

### Auto-detection

If no backend has **ever** been selected (no `web.backend` / per-capability key written by you or `hermes tools`), Hermes picks the first available one based on which credentials are set:

| Credential present | Auto-selected backend |
|--------------------|-----------------------|
| `TAVILY_API_KEY` | tavily |
| `EXA_API_KEY` | exa |
| `PARALLEL_API_KEY` | parallel |
| `FIRECRAWL_API_KEY` or `FIRECRAWL_API_URL` (or the Nous Tool Gateway is ready) | firecrawl |
| `SEARXNG_URL` | searxng |
| `BRAVE_SEARCH_API_KEY` | brave-free |
| `ddgs` package importable | ddgs |
| *(nothing set at all)* | keyless ring: exa / parallel / tavily / firecrawl / keenable (round-robin) |

**Keyless free-tier ring:** when *no* credential above is present, requests rotate across five vendors' public free tiers (Exa, Parallel, Tavily, Firecrawl, Keenable) so web tools work on a fresh install with zero setup — and a rate-limited request fails over to the next vendor in the ring automatically. Pin one vendor in `hermes tools` to stop the rotation (the ring is then only used as failover succession on throttles). All free tiers are vendor-rate-limited under burst load; sustained normal usage goes through fine. Set `web.keyless_fallback: false` to turn the tier off — with it off and no credentials, web tools are unavailable until a provider is configured.

**One-shot keyless rescue for keyed backends:** when your chosen/keyed backend fails a call (bad key, outage, upstream 5xx), that single call automatically retries on the keyless free-tier ring instead of erroring — the result notes which vendor served it and why (`rescued_from` / `backend_error`). The failover is never sticky: the very next `web_search`/`web_extract` call attempts your chosen backend again. Disable with `web.keyless_rescue: false` (also off whenever `keyless_fallback` is off).

xAI Web Search is **not** in the auto-detection chain — having `XAI_API_KEY` set (or being signed in via xAI Grok OAuth) does not automatically route web traffic through xAI, since those credentials are also used for inference / TTS / image gen and the user may want a different backend for web. Opt in explicitly with `web.backend: "xai"`.

Anthropic's native web tools are **not** in the auto-detection chain either, for the same reason plus a stronger one: `ANTHROPIC_API_KEY` / `CLAUDE_CODE_OAUTH_TOKEN` is a model credential that says nothing about which endpoint serves the model, and the tools only run on Anthropic's own endpoint. An Anthropic-only install with no web selection is therefore served by the keyless ring like any other credential-free install. Opt in explicitly with `web.backend: "anthropic"`.

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

### `web_extract` returns truncated content with a `[TRUNCATED]` footer

That's expected for pages over the character budget. The footer names the on-disk file holding the full clean text and the exact `read_file` call to page through the omitted middle. To see more inline, raise `web.extract_char_limit` in `config.yaml` or pass a larger `char_limit` on the call.

Neither knob has any effect on the [Anthropic (native)](#anthropic-native) backend — its fetch is bounded server-side and never produces a `[TRUNCATED]` footer.

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
