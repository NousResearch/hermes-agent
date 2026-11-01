---
id: twelve-data
title: "twelve-data"
sidebar_label: "twelve-data"
description: "Stocks, forex, and crypto market data from Twelve Data."
---

<!-- This page is auto-generated from optional-mcps/twelve-data/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# twelve-data

Stocks, forex, and crypto market data from Twelve Data.

## Overview

**Source:** [https://twelvedata.com/docs](https://twelvedata.com/docs)

Install this catalog entry with:

```bash
hermes mcp install twelve-data
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall twelve-data` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.twelvedata.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Twelve Data (or run `hermes mcp login twelve-data`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/twelve-data/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: twelve-data
description: Stocks, forex, and crypto market data from Twelve Data.
source: https://twelvedata.com/docs

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.twelvedata.com/mcp

auth:
  type: oauth

# Excluded: OAuth plumbing exposed as tools on the cloud server, plus the
# account-quota probe. All remaining tools are read-only market data.
tools:
  default_excluded:
    - oauth_login
    - auth_status
    - oauth_configure
    - get_api_usage

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - twelve data
    - stock price
  hosts:
    - twelvedata.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Twelve Data (or run `hermes mcp login twelve-data`). Approve access,
  then restart the session so tools load.
```
