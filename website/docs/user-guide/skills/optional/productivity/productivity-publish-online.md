---
title: "Publish Online — Publish a site at a ShipStatic link, no account needed"
sidebar_label: "Publish Online"
description: "Publish a site at a ShipStatic link, no account needed"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Publish Online

Publish a site at a ShipStatic link, no account needed.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/productivity/publish-online` |
| Path | `optional-skills/productivity/publish-online` |
| Version | `3.1.6` |
| Author | ShipStatic (shipstatic-com) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `shipstatic`, `publish`, `deploy`, `hosting`, `static-site`, `website`, `share`, `URL`, `password` |
| Related skills | [`publish-site`](../../optional/web-development/web-development-publish-site.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Publish online (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [shipstatic/plugin](https://github.com/shipstatic/plugin): the repo ships
> the skill directory (`skills/publish-online/`), the same file ShipStatic
> publishes to every agent host. `hermes skills install
> official/productivity/publish-online` pulls the current skill live from
> that repo (quarantined and scanned like any hub install), so this directory
> holds only the catalog metadata and the vendored copy can never go stale.

The skill puts a site, page, report, slide deck, dashboard or small web app
online at a ShipStatic link anyone can open, with no account needed. Without
an account the site stays up for 3 days and comes with a claim link that
keeps it for good. It also offers built-in password protection.

## Prerequisites

- A folder on disk goes through the ShipStatic CLI, run with the `terminal`
  tool: `npx -y @shipstatic/ship@3 <folder> --json`. That needs Node.js 20.19
  or newer. Nothing else to install, and no key.
- Without Node, the skill uses the hosted ShipStatic MCP server's
  `deployments_upload` tool instead. Add it to `~/.hermes/config.yaml`:

  ```yaml
  mcp_servers:
    shipstatic:
      url: https://mcp.shipstatic.com
  ```

- Optional: `SHIP_TOKEN` (an API key from https://my.shipstatic.com/api-key)
  makes CLI deploys land in the user's account and never expire.

Full documentation: https://docs.shipstatic.com/skill
