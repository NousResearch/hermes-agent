---
title: "Olostep — Cloud web search, scraping and cited answers"
sidebar_label: "Olostep"
description: "Cloud web search, scraping and cited answers"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Olostep

Cloud web search, scraping and cited answers.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/olostep` |
| Path | `optional-skills/research/olostep` |
| Version | `1.1.0` |
| Author | Umer K. (umerkay), Zeeshan Adil (ZeeshanAdilButt) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Web Search`, `Scraping`, `Research`, `AI Answers`, `Cloud API` |
| Related skills | [`duckduckgo-search`](../../optional/research/research-duckduckgo-search.md), [`scrapling`](../../optional/research/research-scrapling.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Olostep Skill

[Olostep](https://www.olostep.com) is a cloud web data API. This skill covers three calls through the Python SDK: web search, scraping one URL to Markdown or HTML, and web-sourced answers with source URLs. It does not cover crawls, batches, maps or monitors.

## When to Use

- The user asks for Olostep by name, or `web_search` / `web_extract` are not configured.
- You need page content from a JavaScript-heavy or bot-protected page without a local browser.
- You need an answer with source URLs, optionally in a fixed JSON shape.
- Prefer `web_search` and `web_extract` for ordinary one-off lookups.

## Prerequisites

- `OLOSTEP_API_KEY` set (Hermes prompts for it on load).
- The SDK: `pip install -U olostep` (checked against 1.2.2; older releases may lack `searches`).

## How to Run

Run the helper through the `terminal` tool. `SKILL_DIR` is the directory containing this SKILL.md.

```bash
python SKILL_DIR/scripts/olostep_cli.py search "best python web frameworks" --limit 5
python SKILL_DIR/scripts/olostep_cli.py scrape https://example.com --format markdown
python SKILL_DIR/scripts/olostep_cli.py answer "Who wrote the latest Dune novel?"
```

`search` and `answer` print JSON. `scrape` prints the page content.

## Quick Reference

| Task | SDK call | Read from the result |
|------|----------|----------------------|
| Search | `client.searches.create(query, limit=5)` | `result.links`, each with `title`, `url`, `description` |
| Scrape | `client.scrapes.create(url_to_scrape=url, formats=["markdown"])` | `result.markdown_content` or `result.html_content` |
| Answer | `client.answers.create(task=task, json_format={...})` | `result.json_content` and `result.sources` |

Search also takes `include_domains`, `exclude_domains` and `scrape_options={"formats": ["markdown"]}` to embed page content in each link.

## Procedure

1. Pick the call from the table above. Search to find pages, scrape to read one known URL, answer for a question that needs sources.
2. Use the helper script for a single call. Use the SDK directly when you need options the script does not expose.
3. Scrape in Python:

```python
import os
from olostep import Olostep

client = Olostep(api_key=os.environ["OLOSTEP_API_KEY"])
result = client.scrapes.create(url_to_scrape="https://example.com", formats=["markdown"])
print(getattr(result, "markdown_content", None))
```

4. Answers in Python. `json_content` is a JSON string, so parse it, and report `sources` with the answer:

```python
import json

answer = client.answers.create(
    task="What is the latest book by J.K. Rowling?",
    json_format={"book_title": "", "author": "", "release_date": ""},
)
data = json.loads(answer.json_content)
print(data, answer.sources)
```

5. Async variant, for several calls at once:

```python
import asyncio
from olostep import AsyncOlostep

async def main():
    async with AsyncOlostep(api_key=os.environ["OLOSTEP_API_KEY"]) as client:
        pages = await asyncio.gather(
            client.scrapes.create(url_to_scrape="https://example.com", formats=["markdown"]),
            client.scrapes.create(url_to_scrape="https://example.org", formats=["markdown"]),
        )
        return [getattr(p, "markdown_content", None) for p in pages]

asyncio.run(main())
```

6. Catch `Olostep_BaseError` (from `olostep`) for API failures.

## Pitfalls

- Scrape takes `url_to_scrape=`, as in the Olostep docs. The SDK also accepts `url=`, but passing both raises `ValueError`.
- `formats` is a list: `["markdown"]`, not `"markdown"`.
- The result only has the content attributes that came back, so `result.markdown_content` can raise `AttributeError`. Read it with `getattr(result, "markdown_content", None)`.
- `answers.create` takes `task=` (not `query=`) and `json_format=` for structured output. Without `json_format`, `json_content` came back as `{"result": "<text>"}` in a live API check.
- `json_content` is a string. `result.answer` is only an SDK alias for it and carries no citations, so use `result.sources`.
- Fields the answer agent cannot find come back as `NOT_FOUND`. Do not present them as facts.
- `sources` can be an empty list for simple factual questions. Say so instead of inventing citations.
- `limit` on search must be between 1 and 25.

## Verification

```bash
python SKILL_DIR/scripts/olostep_cli.py answer "What is the capital of France?"
```

Expect JSON with an `answer` containing "Paris" and a `sources` list. An `OLOSTEP_API_KEY is not set` message means the key is missing.
