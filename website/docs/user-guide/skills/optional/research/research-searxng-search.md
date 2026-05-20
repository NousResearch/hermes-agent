---
title: "Searxng Search — Free meta-search via SearXNG — aggregates results from 70+ search engines"
sidebar_label: "Searxng Search"
description: "Free meta-search via SearXNG — aggregates results from 70+ search engines"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Searxng Search

Free meta-search via SearXNG — aggregates results from 70+ search engines. Self-hosted or use a public instance. No API key needed. Falls back automatically when the web search toolset is unavailable.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/searxng-search` |
| Path | `optional-skills/research/searxng-search` |
| Version | `1.0.0` |
| Author | hermes-agent |
| License | MIT |
| Platforms | linux, macos |
| Tags | `search`, `searxng`, `meta-search`, `self-hosted`, `free`, `fallback` |
| Related skills | [`duckduckgo-search`](/docs/user-guide/skills/optional/research/research-duckduckgo-search), [`domain-intel`](/docs/user-guide/skills/optional/research/research-domain-intel) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# SearXNG Search

Free meta-search using [SearXNG](https://searxng.org/) — a privacy-respecting, self-hosted search aggregator that queries 70+ search engines simultaneously.

**No API key required** when using a public instance. Can also be self-hosted for full control. Automatically appears as a fallback when the main web search toolset (`FIRECRAWL_API_KEY`) is not configured.

## Configuration

SearXNG requires a `SEARXNG_URL` environment variable pointing to your SearXNG instance:

```bash
# Public instances (no setup required)
SEARXNG_URL=https://searxng.example.com

# Self-hosted SearXNG
SEARXNG_URL=http://localhost:8888
```

If no instance is configured, this skill is unavailable and the agent falls back to other search options.

## Detection Flow

Check what is actually available before choosing an approach:

```bash
# Check if SEARXNG_URL is set and the instance is reachable
curl -s --max-time 5 "${SEARXNG_URL}/search?q=test&format=json" | head -c 200
```

Decision tree:
1. If `SEARXNG_URL` is set and the instance responds, use SearXNG
2. If `SEARXNG_URL` is unset or unreachable, fall back to other available search tools
3. If the user wants SearXNG specifically, help them set up an instance or find a public one

## Method 1: CLI via curl (Preferred)

Use `curl` via `terminal` to call the SearXNG JSON API. This avoids assuming any particular Python package is installed.

```bash
# Text search (JSON output)
curl -s --max-time 10 \
  "${SEARXNG_URL}/search?q=python+async+programming&format=json&engines=google,bing&limit=10"

# With Safesearch off
curl -s --max-time 10 \
  "${SEARXNG_URL}/search?q=example&format=json&safesearch=0"

# Specific categories (general, news, science, etc.)
curl -s --max-time 10 \
  "${SEARXNG_URL}/search?q=AI+news&format=json&categories=news"
```

### Common CLI Flags

| Flag | Description | Example |
|------|-------------|---------|
| `q` | Query string (URL-encoded) | `q=python+async` |
| `format` | Output format: `json`, `csv`, `rss` | `format=json` |
| `engines` | Comma-separated engine names | `engines=google,bing,ddg` |
| `limit` | Max results per engine (default 10) | `limit=5` |
| `categories` | Filter by category | `categories=news,science` |
| `safesearch` | 0=none, 1=moderate, 2=strict | `safesearch=0` |
| `time_range` | Filter: `day`, `week`, `month`, `year` | `time_range=week` |

### Parsing JSON Results

```bash
# Extract titles and URLs from JSON
curl -s --max-time 10 "${SEARXNG_URL}/search?q=fastapi&format=json&limit=5" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data.get('results', []):
    print(r.get('title',''))
    print(r.get('url',''))
    print(r.get('content','')[:200])
    print()
"
```

Returns per result: `title`, `url`, `content` (snippet), `engine`, `parsed_url`, `img_src`, `thumbnail`, `author`, `published_date`

## Method 2: Python API via `requests`

Use the SearXNG REST API directly from Python with the `requests` library:

```python
import os, requests, urllib.parse

base_url = os.environ.get("SEARXNG_URL", "")
if not base_url:
    raise RuntimeError("SEARXNG_URL is not set")

query = "fastapi deployment guide"
params = {
    "q": query,
    "format": "json",
    "limit": 5,
    "engines": "google,bing",
}

resp = requests.get(f"{base_url}/search", params=params, timeout=10)
resp.raise_for_status()
data = resp.json()

for r in data.get("results", []):
    print(r["title"])
    print(r["url"])
    print(r.get("content", "")[:200])
    print()
```

## Method 3: searxng-data Python Package

For more structured access, install the `searxng-data` package:

```bash
pip install searxng-data
```

```python
from searxng_data import engines

# List available engines
print(engines.list_engines())
```

Note: This package only provides engine metadata, not the search API itself.

## Self-Hosting SearXNG

To run your own SearXNG instance:

```bash
# Using Docker
docker run -d -p 8888:8080 \
  -v $(pwd)/searxng:/etc/searxng \
  searxng/searxng:latest

# Then set
SEARXNG_URL=http://localhost:8888
```

Or install via pip:
```bash
pip install searxng
# Edit /etc/searxng/settings.yml
searxng-run
```

Public SearXNG instances are available at:
- `https://searxng.example.com` (replace with any public instance)

## Workflow: Search then Extract

SearXNG returns titles, URLs, and snippets — not full page content. To get full page content, search first and then extract the most relevant URL with `web_extract`, browser tools, or `curl`.

```bash
# Search for relevant pages
curl -s "${SEARXNG_URL}/search?q=fastapi+deployment&format=json&limit=3"
# Output: list of results with titles and URLs

# Then extract the best URL with web_extract
```

## Limitations

- **Instance availability**: If the SearXNG instance is down or unreachable, search fails. Always check `SEARXNG_URL` is set and the instance is reachable.
- **No content extraction**: SearXNG returns snippets, not full page content. Use `web_extract`, browser tools, or `curl` for full articles.
- **Rate limiting**: Some public instances limit requests. Self-hosting avoids this.
- **Engine coverage**: Available engines depend on the SearXNG instance configuration. Some engines may be disabled.
- **Results freshness**: Meta-search aggregates external engines — result freshness depends on those engines.

## Troubleshooting

| Problem | Likely Cause | What To Do |
|---------|--------------|------------|
| `SEARXNG_URL` not set | No instance configured | Use a public SearXNG instance or set up your own |
| Connection refused | Instance not running or wrong URL | Check the URL is correct and the instance is running |
| Empty results | Instance blocks the query | Try a different instance or self-host |
| Slow responses | Public instance under load | Self-host or use a less-loaded public instance |
| `json` format not supported | Old SearXNG version | Try `format=rss` or upgrade SearXNG |

## Pitfalls

- **Always set `SEARXNG_URL`**: Without it, the skill cannot function.
- **URL-encode queries**: Spaces and special characters must be URL-encoded in curl, or use `urllib.parse.quote()` in Python.
- **Use `format=json`**: The default format may not be machine-readable. Always request JSON explicitly.
- **Set a timeout**: Always use `--max-time` or `timeout=` to avoid hanging on unreachable instances.
- **Self-hosting is best**: Public instances may go down, rate-limit, or block. A self-hosted instance is reliable.

## Instance Discovery

If `SEARXNG_URL` is not set and the user asks about SearXNG, help them either:
1. Find a public SearXNG instance (search for "public searxng instance")
2. Set up their own with Docker or pip

Public instances are listed at: https://searxng.org/
