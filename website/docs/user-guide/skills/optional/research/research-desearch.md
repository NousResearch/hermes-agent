---
title: "Desearch — Search the web and X, or get sourced AI answers"
sidebar_label: "Desearch"
description: "Search the web and X, or get sourced AI answers"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Desearch

Search the web and X, or get sourced AI answers.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/desearch` |
| Path | `optional-skills/research/desearch` |
| Version | `1.0.0` |
| Author | Desearch |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Research`, `Web`, `Search`, `X`, `MCP` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Desearch Skill

Use the Desearch MCP server to search the web, search X posts and users, and get AI answers with links to their sources. It also extracts the text of a public page. It does not run without a Desearch API key, and it is a paid service billed by usage.

## When to Use

- The user asks for Desearch by name, or has `DESEARCH_API_KEY` configured and wants it used.
- The task needs X (Twitter) posts, replies, user timelines or trends.
- The task needs one summarized answer drawn from several sources (web, Reddit, Hacker News, arXiv, Wikipedia, X).

For a quick one-off lookup when Desearch is not requested, prefer native `web_search` and `web_extract`.

## Prerequisites

1. Create an API key at https://console.desearch.ai/api-keys. Hermes asks for `DESEARCH_API_KEY` the first time the skill loads in the CLI and stores it in the Hermes env file. On messaging gateways, run `hermes setup` locally instead.

2. Pick one access path. Both use the same hosted server and the same key.

   **Native MCP (recommended).** Add the server to `~/.hermes/config.yaml`. Hermes resolves `${DESEARCH_API_KEY}` at connect time:

   ```yaml
   mcp_servers:
     desearch:
       url: "https://mcp.desearch.ai/mcp"
       headers:
         Authorization: "Bearer ${DESEARCH_API_KEY}"
   ```

   To run the server locally over stdio instead (needs Node.js):

   ```yaml
   mcp_servers:
     desearch:
       command: npx
       args: ["-y", "desearch-mcp-server"]
       env:
         DESEARCH_API_KEY: "${DESEARCH_API_KEY}"
   ```

   The tools then appear as `mcp__desearch__web_search`, `mcp__desearch__ai_search`, `mcp__desearch__x_search`, `mcp__desearch__extract` and the rest of the table below.

   **Helper script.** Without MCP config, run `scripts/desearch.py` through the `terminal` tool. It is stdlib Python 3 and sends the key only as a Bearer header to the hosted server.

## How to Run

```bash
python3 "${HERMES_SKILL_DIR}/scripts/desearch.py" web "bittensor subnet 22"
```

Every subcommand prints the tool's JSON result to stdout. Errors go to stderr with exit code 1.

## Quick Reference

| Script subcommand | MCP tool | Use for |
|---|---|---|
| `web QUERY [--start N]` | `web-search` | Ranked titles, links and snippets |
| `ai PROMPT [--sources web,reddit] [--date-filter PAST_WEEK] [--links-only]` | `ai-search` | Summary with sources; default sources `web,twitter` |
| `x QUERY [--count N] [--user NAME] [--lang en] [--since DATE] [--until DATE]` | `x-search` | X posts matching a query |
| `links PROMPT [--count N]` | `web-links-search` | Web links for a prompt (10 to 200) |
| `extract URL [--format text\|html]` | `extract` | Page content of a public URL |
| `call TOOL '{"key": "value"}'` | any | Tools without a shortcut, see below |
| `tools` | `tools/list` | Names of the tools the server exposes |

Tools reached through `call` or native MCP: `x-links-search` (`prompt`, `count`), `x-posts-by-urls` (`urls`), `x-post-by-id` (`id`), `x-posts-by-user` (`user`, `query`, `count`), `x-user-posts` (`username`, `cursor`), `x-user-replies` (`user`, `query`, `count`), `x-post-replies` (`post_id`, `query`, `count`), `x-post-retweeters` (`id`, `cursor`), `x-trends` (`woeid`, `count`), `web-crawl` (legacy, prefer `extract`).

AI search sources: `web`, `twitter`, `arxiv`, `wikipedia`, `hackernews`, `reddit`. Date filters: `PAST_24_HOURS`, `PAST_2_DAYS`, `PAST_WEEK`, `PAST_2_WEEKS`, `PAST_MONTH`, `PAST_2_MONTHS`, `PAST_YEAR`, `PAST_2_YEARS`.

## Procedure

1. Check access. If `mcp__desearch__*` tools are present, use them. Otherwise run the `tools` subcommand through `terminal`; it should list `web-search`. Done when one path answers.
2. Choose the tool from the Quick Reference. Use `web` for links, `ai` for a summarized answer, `x` for posts, `extract` to read a page. Done when the tool and arguments are fixed.
3. Run the call. Example AI search over Reddit and Hacker News for the past week:

   ```bash
   python3 "${HERMES_SKILL_DIR}/scripts/desearch.py" ai "open source agent frameworks" --sources reddit,hackernews --date-filter PAST_WEEK
   ```

   Done when the command exits 0 and prints JSON.
4. Save large results with `write_file` if follow-up questions are likely, then read them back with `read_file` instead of repeating paid calls. Done when the file exists.
5. Answer from the returned data only. Cite only URLs present in the result. Done when every cited link appears in the JSON.

## Pitfalls

- Every call is billed to the key's Desearch account. Do not loop or retry blindly; reuse saved results.
- HTTP 401 or 403 means the key is missing or invalid. Check the stored `DESEARCH_API_KEY` and the key's status in the console.
- AI search and link searches can return a body with billing fields and no links. Treat that as no results, not as an error, and try a broader prompt or `web`.
- `web-links-search` accepts only the `web` source. Use `ai --sources twitter` or `x` for X content.
- `extract` reads public pages only. Pages behind a login or heavy scripts may return little text; try `call extract '{"url": "...", "js": true}'`.
- Shell variables do not persist between `terminal` calls. Repeat the full script path each time.

## Verification

```bash
python3 "${HERMES_SKILL_DIR}/scripts/desearch.py" web "hermes agent" --start 0
```

Success is exit code 0 and JSON with result titles and links.
