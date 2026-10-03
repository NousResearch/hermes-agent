---
id: robinhood
title: "robinhood"
sidebar_label: "robinhood"
description: "Robinhood agentic trading: portfolio, balances, and orders."
---

<!-- This page is auto-generated from optional-mcps/robinhood/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# robinhood

Robinhood agentic trading: portfolio, balances, and orders.

## Overview

**Source:** [https://robinhood.com/us/en/support/articles/agentic-trading-overview/](https://robinhood.com/us/en/support/articles/agentic-trading-overview/)

Install this catalog entry with:

```bash
hermes mcp install robinhood
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall robinhood` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://agent.robinhood.com/mcp/trading`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Robinhood (or run `hermes mcp login robinhood`). Approve access,
then restart the session so tools load.

CAUTION: this server can place REAL trades (equities, options, crypto)
in a dedicated Robinhood agentic account. Hermes's normal tool-approval
flow applies, but review orders carefully before approving.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/robinhood/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: robinhood
description: 'Robinhood agentic trading: portfolio, balances, and orders.'
source: https://robinhood.com/us/en/support/articles/agentic-trading-overview/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://agent.robinhood.com/mcp/trading

auth:
  type: oauth

# Excluded: options-tier upsell link and social watchlist engagement
# features. Trading tools stay enabled by design — see the post_install
# caution.
tools:
  default_excluded:
    - get_option_level_upgrade_info
    - get_popular_watchlists
    - follow_watchlist
    - unfollow_watchlist

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - robinhood
    - stock trade
  hosts:
    - robinhood.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Robinhood (or run `hermes mcp login robinhood`). Approve access,
  then restart the session so tools load.

  CAUTION: this server can place REAL trades (equities, options, crypto)
  in a dedicated Robinhood agentic account. Hermes's normal tool-approval
  flow applies, but review orders carefully before approving.
```
