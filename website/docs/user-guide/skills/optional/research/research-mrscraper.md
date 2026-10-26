---
title: "Mrscraper — Search and retrieve web pages with MrScraper"
sidebar_label: "Mrscraper"
description: "Search and retrieve web pages with MrScraper"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Mrscraper

Search and retrieve web pages with MrScraper.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/mrscraper` |
| Path | `optional-skills/research/mrscraper` |
| Version | `1.0.0` |
| Author | Riandra Diva (riandradiva18), Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `MrScraper`, `Google SERP`, `Web Scraping`, `Rendered Pages`, `Account Usage` |
| Related skills | [`scrapling`](../../optional/research/research-scrapling.md), [`grounded-citations`](../../bundled/research/research-grounded-citations.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# MrScraper Skill

Use this skill to connect the cataloged standalone MrScraper plugin or [official hosted MCP server](https://docs.mrscraper.com/docs/getting-started/mcp-server) to Hermes and choose the right registered tools. The native plugin calls MrScraper's HTTP API directly through three tools: `mrscraper_scraper_api`, `mrscraper_web_search`, and `mrscraper_account`; the hosted MCP connection is a separate alternative with different tool names and authentication. Neither connection provides an interactive browser session for clicking or typing.

## When to Use

- Find relevant pages in Google results before retrieving their contents.
- Retrieve HTML, Markdown, or a screenshot from a known public URL.
- Derive named fields or repeated listings from returned page content.
- Check account usage or remaining quota before expanding a task.

## Prerequisites

- A MrScraper account. Calls to the service can incur charges; target URLs, search queries, and request options sent through either connection leave Hermes.
- **Native plugin:** In `terminal`, run `hermes plugins install mrscraper`, then `hermes plugins enable mrscraper`. The catalog install uses a reviewed commit pin and prompts for `MRSCRAPER_API_TOKEN`. Start a new Hermes session after enabling. To route standard `web_search` and `web_extract` through it, select MrScraper under Web Search & Extract in `hermes tools`; the three `mrscraper_*` tools do not require that provider selection.
- **Hosted MCP alternative:** In `terminal`, run `hermes mcp add mrscraper --url https://mcp.mrscraper.com/mcp --auth oauth`. Complete the browser sign-in, choose the tools to enable, and start a new session. If already configured, use `hermes mcp test mrscraper`. The server currently documents `scraperAPI`, `webSearch`, and `status`; inspect the live MCP schemas before calling them. The MCP connection does not configure the native web provider.

Install this optional skill with `hermes skills install official/research/mrscraper`. Installing the skill does not install either MrScraper connection.

## How to Run

Choose the connection already available in this session. Use `web_search` and `web_extract` when the native plugin is selected as the web provider; use `mrscraper_scraper_api`, `mrscraper_web_search`, or `mrscraper_account` for direct access to their API controls. With hosted MCP, Hermes registers server tools as `mcp__mrscraper__<tool>`; discover their live argument schemas before use. If neither connection is available, guide setup under Prerequisites and continue after the connection appears in a new session.

## Quick Reference

| Task | Native plugin | Hosted MCP |
| --- | --- | --- |
| Google results | `web_search` or `mrscraper_web_search` | `mcp__mrscraper__webSearch` |
| Retrieve one page | `web_extract` or `mrscraper_scraper_api` | `mcp__mrscraper__scraperAPI` |
| Request HTML, Markdown, or a screenshot | `mrscraper_scraper_api` with the corresponding options | Inspect `mcp__mrscraper__scraperAPI` options |
| Extract fields or listings | Derive fields from returned page content | Derive fields from returned page content |
| Check account usage and quota | `mrscraper_account` | `mcp__mrscraper__status` |

MCP names in the right column assume the server was configured as `mrscraper`. Use the names and schemas registered in the current session if the server name differs. Native plugin arguments are not interchangeable with MCP arguments.

## Procedure

1. Identify the user's target URL or search query, desired fields, geography, and scope. Keep the number of searches and page requests as small as the task allows. Use only public HTTP or HTTPS page targets.
2. For discovery, call `web_search` with MrScraper selected, `mrscraper_web_search` with query text or a Google search URL, or `mcp__mrscraper__webSearch`. Set region, language, and result page only from the user's context. Treat search snippets as leads, then fetch relevant URLs before claiming page facts.
3. For a known page, inspect the source through `web_extract`, `mrscraper_scraper_api`, or `mcp__mrscraper__scraperAPI`. Request HTML or Markdown as needed. With the native scraper tool, request browser rendering or a wait selector when the content needs JavaScript, and a screenshot when visual output is required. Derive requested fields or repeated listings from the returned content; these three native tools do not create managed extraction jobs.
4. For a multi-page task, select relevant URLs from search results or links in returned page content. Retrieve each page within the agreed scope and deduplicate records by a stable source URL or ID.
5. Return requested fields with source URLs. Separate missing fields from empty values and identify pages that could not be retrieved. Verify uncertain fields against the returned page content or a second source.

## Pitfalls

- `mrscraper_scraper_api` makes one request per URL, not a persistent CDP session. Use Hermes `browser_navigate` and browser action tools for click/type/scroll workflows.
- The plugin token and hosted MCP OAuth connection are independent. A working MCP connection does not make `web_search` use MrScraper.
- Native `mrscraper_web_search` defaults to structured JSON results. Use raw HTML only when needed; raw mode overrides the selected output format.
- Avoid repeated broad requests and blind retries: a rate-limit or quota error needs a scope or account check. Use `mrscraper_account` or `mcp__mrscraper__status` when account state matters. The native account tool returns a filtered usage summary; do not request or reproduce raw account credentials.
- Never put an API token in a prompt, URL, result, or `terminal` command. Let the plugin install prompt or MCP OAuth flow store credentials.
- A request that fails or returns empty content is not completed extraction. Report the failure and adjust only the relevant page options before retrying.

## Verification

Confirm the connection exposes the chosen tool. Run one small query or page request, inspect the returned source URL and content, and verify requested fields before expanding the scope. Report inaccessible pages, missing fields, and any account limit that prevents completion.
