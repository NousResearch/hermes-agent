---
id: trivago
title: "trivago"
sidebar_label: "trivago"
description: "trivago hotel search: compare prices by city and dates."
---

<!-- This page is auto-generated from optional-mcps/trivago/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# trivago

trivago hotel search: compare prices by city and dates.

## Overview

**Source:** [https://mcp.trivago.com/mcp](https://mcp.trivago.com/mcp)

Install this catalog entry with:

```bash
hermes mcp install trivago
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall trivago` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.trivago.com/mcp`

## Auth

**Type:** `none`

No credentials required.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

No account or credentials needed — tools are available as soon as the
session restarts.

Search only — booking happens on the linked booking sites, never
in-conversation.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/trivago/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: trivago
description: 'trivago hotel search: compare prices by city and dates.'
source: https://mcp.trivago.com/mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). No authentication required (verified live:
# initialize succeeds anonymously).

transport:
  type: http
  url: https://mcp.trivago.com/mcp

auth:
  type: none

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - hotel search
    - trivago
  hosts:
    - trivago.com

post_install: |
  No account or credentials needed — tools are available as soon as the
  session restarts.

  Search only — booking happens on the linked booking sites, never
  in-conversation.
```
