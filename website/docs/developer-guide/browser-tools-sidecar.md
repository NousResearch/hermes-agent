---
title: Browser-tools Sidecar
---

# Browser-tools sidecar

The browser-tools sidecar is an optional shared HTTP service for agent fleets
that need browser rendering, stealth fetches, structured scraping, and webpage
enrichment without installing heavy browser dependencies in every Hermes agent
container.

It is designed to combine complementary tools:

- **CloakBrowser** for stealth/humanized browser rendering, screenshots, and
  visual artifact capture.
- **Scrapling** for fast HTTP scraping, CSS/XPath extraction, adaptive selectors,
  and crawler/spider workflows.
- **Plain HTTP fallback** for simple public pages when optional runtimes are not
  installed.

This should be used as a shared capability layer, not as a replacement for
platform-specific ingestion routes such as YouTube transcript extraction,
Cobalt media capture, `yt-dlp`, Instagram discovery, or other social APIs.

## Architecture

```text
Hermes agent / worker / SocialScraperAgent
        │
        │ BROWSER_TOOLS_URL=http://browser-tools:8790
        ▼
browser-tools sidecar
        ├── CloakBrowser runtime: stealth render, screenshot, humanized browsing
        ├── Scrapling: HTTP fetch, selectors, extraction, future crawls
        └── plain HTTP fallback
```

Run it on the same internal Docker network as the agents. Expose the port to
`127.0.0.1` only for operator smoke tests.

## Compose

A minimal compose file is included at the repository root:

```bash
docker compose -f docker-compose.browser-tools.yml up --build browser-tools
```

For dependent agent containers, inject:

```yaml
environment:
  BROWSER_TOOLS_URL: http://browser-tools:8790
```

## API

### `GET /health`

Returns service liveness and installed runtime capabilities:

```json
{
  "ok": true,
  "capabilities": {
    "cloak_fetch_cli": true,
    "scrapling": true,
    "playwright": true,
    "patchright": true
  }
}
```

### `POST /fetch`

Fetch a page with automatic or explicit routing.

```json
{
  "url": "https://example.com",
  "mode": "auto",
  "timeout_ms": 30000,
  "text_limit": 12000,
  "screenshot": false,
  "humanize": false,
  "wait_until": "domcontentloaded"
}
```

Modes:

- `auto` — choose the best available runtime.
- `scrapling_http` — fast HTTP fetch and text extraction.
- `cloak` — CloakBrowser-backed render/fetch, useful for screenshots and
  fingerprint-sensitive pages.
- `plain_http` — low-dependency fallback.
- `scrapling_dynamic` / `scrapling_stealth` — reserved API modes for future
  browser-backed Scrapling routes once browser installation is pinned.

### `POST /extract`

Run structured extraction with Scrapling selectors.

```json
{
  "url": "https://quotes.toscrape.com/",
  "mode": "scrapling_http",
  "selectors": {
    "quotes": ".quote .text::text",
    "authors": ".quote .author::text",
    "links": "a::attr(href)"
  },
  "include_links": true
}
```

Selector values default to CSS. Prefix with `xpath:` for XPath selectors.

## Hermes tools

When `BROWSER_TOOLS_URL` is configured, Hermes exposes two browser-toolset
functions:

- `browser_tools_fetch` — fetch through the sidecar.
- `browser_tools_extract` — structured selector extraction through the sidecar.

These tools are intentionally gated by the environment variable so ordinary
Hermes installs do not see sidecar-specific tools unless the service is wired.

## Routing guidance

Use the sidecar this way:

```text
Need ordinary webpage text?          -> browser_tools_fetch mode=scrapling_http
Need structured CSS/XPath fields?    -> browser_tools_extract
Need screenshot / visual artifact?   -> browser_tools_fetch mode=cloak screenshot=true
Need fingerprint-sensitive render?   -> browser_tools_fetch mode=cloak
Need social video/transcript?        -> keep platform-specific ingestion routes
Need multi-page docs/product crawl?  -> future Scrapling spider endpoint
```

## Smoke tests

From the host, if the compose file maps `127.0.0.1:8790`:

```bash
curl -fsS http://127.0.0.1:8790/health

curl -fsS -X POST http://127.0.0.1:8790/fetch \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com","text_limit":120}'

curl -fsS -X POST http://127.0.0.1:8790/extract \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://quotes.toscrape.com/","selectors":{"quotes":".quote .text::text"}}'
```

Expected results:

- `/health` returns `ok: true`.
- `/fetch` returns title `Example Domain` and status `200`.
- `/extract` returns quote text in `fields.quotes`.

## Security and operations

- Keep the service on an internal Docker network by default.
- Map the port to `127.0.0.1` only for local smoke tests.
- Treat browser runtimes and downloaded Chromium builds as privileged third-party
  binaries; pin versions for production images.
- Use stealth/fingerprint-sensitive modes only for owned, permitted, or
  contracted automation. The sidecar removes a technical blocker; it does not
  grant legal or Terms-of-Service permission.
- Do not install every browser runtime into every agent. Keep this service as the
  shared dependency boundary.
