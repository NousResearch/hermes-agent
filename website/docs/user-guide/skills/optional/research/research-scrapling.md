---
title: "Scrapling"
sidebar_label: "Scrapling"
description: "Difficult web extraction fallback with Scrapling — selector-driven static/dynamic extraction for cases where web_extract is empty, imprecise, or too coarse."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Scrapling

Difficult web extraction fallback with Scrapling — selector-driven static/dynamic extraction for cases where web_extract is empty, imprecise, or too coarse.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/scrapling` |
| Path | `optional-skills/research/scrapling` |
| Version | `1.2.0` |
| Author | FEUAZUR + Hermes Agent |
| License | MIT |
| Tags | `Web Extraction`, `Selector Extraction`, `Fallback`, `Research`, `Batch Extraction` |
| Related skills | [`duckduckgo-search`](/docs/user-guide/skills/optional/research/research-duckduckgo-search), [`domain-intel`](/docs/user-guide/skills/optional/research/research-domain-intel) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Scrapling

[Scrapling](https://github.com/D4Vinci/Scrapling) is an optional fallback for difficult web extraction: selector-driven static, dynamic, and session-aware extraction when Hermes' normal `web_extract` output is empty, wrong, or too coarse.

Use this skill narrowly. Scrapling is **not** a default web backend, not a replacement for `web_search`, not a replacement for `web_extract`, not a browser-agent replacement, and not a general anti-bot promise.

**Compliance first:** only fetch public pages that are allowed for the task. Respect robots.txt, website Terms of Service, rate limits, privacy boundaries, and local law. Do not use this skill for paywall bypass, private/authenticated data, credentialed scraping, or adversarial anti-bot circumvention.

## When to Use

Consider Scrapling only after the normal Hermes web path is insufficient:

- `web_extract` returns empty, boilerplate, or the wrong main content.
- The task needs CSS, XPath, text, or regex selection from a known page structure.
- The task has many homogeneous pages such as announcements, products, listings, jobs, or tables.
- A lightweight session/cookie flow is needed but full browser automation would be too heavy.
- A page is lightly blocked and a controlled fallback is worth trying, without treating stealth mode as guaranteed bypass.

## When NOT to Use

Do not use Scrapling for:

- ordinary search or broad web reconnaissance — use `web_search`.
- ordinary article/page extraction where `web_extract` works.
- visual inspection, screenshots, forms, clicking, or full page interaction — use browser tools.
- authenticated/private data, paywalled content, explicit no-scrape contexts, or clear ToS conflict.
- strong CAPTCHA, enterprise WAF, or adversarial anti-bot systems.
- default Hermes tool exposure or global MCP-style scraping.

## Routing Position

Default acquisition ladder:

1. `web_search` for discovery.
2. `web_extract` for ordinary URL-to-markdown extraction.
3. Browser tools for interaction, visual inspection, screenshots, login flows, or complex JavaScript workflows.
4. Scrapling only as a difficult extraction fallback when the caller can provide a narrow target, selector, or homogeneous batch pattern.

Do not add Scrapling to `_HERMES_CORE_TOOLS`, do not change the public `web_extract` schema, and do not register a new top-level tool. The only accepted route exposure is the task-named `difficult_web_extract` `/bg` fallback.

## Runtime Setup

Do **not** install Scrapling into the Hermes main virtualenv.

Recommended isolated runtime:

```text
~/.hermes/runtimes/scrapling/
```

Recommended Python on Hank's current machine:

```text
/Users/zhaopufan/.local/bin/python3.11
```

Use the bundled pilot setup script instead of installing by hand:

```bash
python optional-skills/research/scrapling/scripts/setup_runtime.py --dry-run
python optional-skills/research/scrapling/scripts/setup_runtime.py
```

Only install browser assets when dynamic/stealth extraction is explicitly needed:

```bash
python optional-skills/research/scrapling/scripts/setup_runtime.py --install-browsers
```

The script installs `optional-skills/research/scrapling/requirements.txt` into the isolated runtime and emits a JSON setup receipt. Browser assets remain opt-in so Playwright/Patchright downloads do not happen during ordinary static pilot checks.

## Fetcher Selection

| Mode | Class | Use When | Default |
| --- | --- | --- | --- |
| Static | `Fetcher` / `FetcherSession` | Static HTML, selector extraction, fast homogeneous pages | Preferred first attempt |
| Dynamic | `DynamicFetcher` / dynamic sessions | JavaScript-rendered content, `wait_selector`, `network_idle` | Explicit only |
| Stealth | `StealthyFetcher` / stealth sessions | Light blocking after static/dynamic fail | Explicit fallback only |

Stealth mode must stay opt-in. It can help with some light bot checks, but it is slower, heavier, and not a universal bypass mechanism.

## CLI Examples

### Static selector extraction

```bash
~/.hermes/runtimes/scrapling/bin/python \
  optional-skills/research/scrapling/scripts/scrapling_extract.py \
  --url 'https://example.com/article' \
  --selector '.article' \
  --selector-type css \
  --mode static \
  --fallback-reason selector_required
```

### Dynamic extraction with a wait condition

```bash
~/.hermes/runtimes/scrapling/bin/python \
  optional-skills/research/scrapling/scripts/scrapling_extract.py \
  --url 'https://example.com/results' \
  --selector '.result-card' \
  --selector-type css \
  --mode dynamic \
  --wait-selector '.result-card' \
  --network-idle \
  --fallback-reason web_extract_empty
```

### Stealth fallback, explicit only

```bash
~/.hermes/runtimes/scrapling/bin/python \
  optional-skills/research/scrapling/scripts/scrapling_extract.py \
  --url 'https://example.com/protected-public-page' \
  --selector '.content' \
  --selector-type css \
  --mode stealth \
  --fallback-reason light_block
```

The pilot runner emits the JSON receipt schema below for both success and failure. Do not enable stealth mode casually. Record why static/dynamic failed first.

## Python Examples

### Static CSS selection

```python
from scrapling.fetchers import Fetcher

page = Fetcher.get("https://quotes.toscrape.com/", timeout=20)
quotes = page.css(".quote .text::text").getall()
print(quotes)
```

### XPath selection

```python
from scrapling.fetchers import Fetcher

page = Fetcher.get("https://example.com/report", timeout=20)
rows = page.xpath('//table[contains(@class, "data")]//tr').getall()
print(rows)
```

### Batch homogeneous pages with a session

```python
from scrapling.fetchers import FetcherSession

urls = [
    "https://example.com/jobs/1",
    "https://example.com/jobs/2",
]

with FetcherSession(impersonate="chrome") as session:
    for url in urls:
        page = session.get(url, timeout=20)
        title = page.css("h1::text").get()
        location = page.css(".location::text").get()
        print({"url": url, "title": title, "location": location})
```

### Dynamic wait selector

```python
from scrapling.fetchers import DynamicFetcher

page = DynamicFetcher.fetch(
    "https://example.com/results",
    wait_selector=(".result-card", "visible"),
    network_idle=True,
    headless=True,
)
items = page.css(".result-card::text").getall()
print(items)
```

## Element Selection Quick Reference

```python
page.css("h1::text").get()
page.css("a::attr(href)").getall()
page.xpath('//div[@class="content"]/text()').getall()
page.find_all("div", class_="listing")
page.find_by_text("Read more", tag="a")
page.find_by_regex(r"\$\d+\.\d{2}")
```

For product/listing pages, `find_similar()` can help discover repeated structures after one representative element is found.

## Output Receipt Schema

Any reusable script or future adapter should normalize output to an auditable JSON receipt:

```json
{
  "backend": "scrapling",
  "mode": "static|dynamic|stealth",
  "url": "https://example.com/page",
  "selector": ".article",
  "content": "...",
  "elapsed_ms": 1234,
  "fallback_reason": "web_extract_empty|selector_required|batch_homogeneous|light_block",
  "errors": []
}
```

Receipts should preserve URL, selector, mode, elapsed time, fallback reason, and errors. Do not return unbounded raw HTML unless explicitly needed.

## Failure Fallback Rules

1. If static extraction fails because the target content is not in HTML, try dynamic extraction with a specific `wait_selector`.
2. If dynamic extraction times out, lower resource load, narrow the selector, or fall back to browser tools for inspection.
3. If the page is blocked, record the block reason before trying stealth mode.
4. If stealth fails or raises compliance ambiguity, stop. Do not escalate into adversarial bypass.
5. If the task requires interaction, screenshots, forms, login, or visual judgment, switch to browser tools.

## Pitfalls

- **Runtime isolation:** install under `~/.hermes/runtimes/scrapling/`, not the Hermes main venv.
- **Python version:** Scrapling currently supports Python 3.10+ and official classifiers through Python 3.13; avoid relying on Hermes' Python 3.14 venv.
- **Browser install:** `DynamicFetcher` / `StealthyFetcher` may require `scrapling install` inside the isolated runtime.
- **Timeout units:** check Scrapling fetcher docs; timeout units can differ between static and browser-backed fetchers.
- **Stealth cost:** stealth fetchers run browser/fingerprint tooling; limit concurrency and record why they were used.
- **Compliance:** never promise Cloudflare/CAPTCHA/paywall bypass. Stop when the site policy or risk boundary is unclear.

## Related Design Note

See `docs/specs/web-acquisition-routing.md` in the Hermes repo for the Phase 0 routing boundary and upgrade/kill gates.
