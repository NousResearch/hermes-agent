---
id: notion
title: "notion"
sidebar_label: "notion"
description: "Pages and databases from your Notion workspace."
---

<!-- This page is auto-generated from optional-mcps/notion/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# notion

Pages and databases from your Notion workspace.

## Overview

**Source:** [https://developers.notion.com/docs/mcp](https://developers.notion.com/docs/mcp)

Install this catalog entry with:

```bash
hermes mcp install notion
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall notion` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.notion.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Notion (or run `hermes mcp login notion`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/notion/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: notion
connector_slug: notion
description: >-
  Pages and databases from your Notion workspace.
source: https://developers.notion.com/docs/mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration;
# Hermes's MCP client + mcp_oauth_manager handle discovery, PKCE, token
# exchange, and refresh.
transport:
  type: http
  url: https://mcp.notion.com/mcp

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - notion
  hosts:
    - notion.so
    - notion.site

post_install: |
  On first connection Hermes opens a browser to authorize with
  Notion (or run `hermes mcp login notion`). Approve access,
  then restart the session so tools load.
```
