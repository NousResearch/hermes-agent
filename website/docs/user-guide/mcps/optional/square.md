---
id: square
title: "square"
sidebar_label: "square"
description: "Catalog, orders, and payments via Square's hosted MCP."
---

<!-- This page is auto-generated from optional-mcps/square/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# square

Catalog, orders, and payments via Square's hosted MCP.

## Overview

**Source:** [https://developer.squareup.com/docs/mcp](https://developer.squareup.com/docs/mcp)

Install this catalog entry with:

```bash
hermes mcp install square
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall square` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.squareup.com/sse`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Square (or run `hermes mcp login square`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/square/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: square
description: >-
  Catalog, orders, and payments via Square's hosted MCP.
source: https://developer.squareup.com/docs/mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration;
# Hermes's MCP client + mcp_oauth_manager handle discovery, PKCE, token
# exchange, and refresh.
# "square" the English word is everywhere ("square brackets") — only the
# unambiguous brand form triggers a suggestion.
transport:
  type: http
  url: https://mcp.squareup.com/sse

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - squareup
  hosts:
    - squareup.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Square (or run `hermes mcp login square`). Approve access,
  then restart the session so tools load.
```
