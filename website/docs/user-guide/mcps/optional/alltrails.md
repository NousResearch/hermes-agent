---
id: alltrails
title: "alltrails"
sidebar_label: "alltrails"
description: "AllTrails: find hikes and trails with reviews and ratings."
---

<!-- This page is auto-generated from optional-mcps/alltrails/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# alltrails

AllTrails: find hikes and trails with reviews and ratings.

## Overview

**Source:** [https://www.alltrails.com/mcp](https://www.alltrails.com/mcp)

Install this catalog entry with:

```bash
hermes mcp install alltrails
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall alltrails` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://www.alltrails.com/mcp`

## Auth

**Type:** `none`

No credentials required.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

No account or credentials needed — tools are available as soon as the
session restarts.

Heads-up: only 5 tools, but their schemas are unusually verbose (~24K
tokens total). If you only browse trails occasionally, consider leaving
this server disabled and enabling it on demand, or prune tools with:
  hermes mcp configure alltrails

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/alltrails/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: alltrails
description: 'AllTrails: find hikes and trails with reviews and ratings.'
source: https://www.alltrails.com/mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). No authentication required (verified live:
# initialize succeeds anonymously).

transport:
  type: http
  url: https://www.alltrails.com/mcp

auth:
  type: none

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - hiking
    - trail
    - alltrails
  hosts:
    - alltrails.com

post_install: |
  No account or credentials needed — tools are available as soon as the
  session restarts.

  Heads-up: only 5 tools, but their schemas are unusually verbose (~24K
  tokens total). If you only browse trails occasionally, consider leaving
  this server disabled and enabling it on demand, or prune tools with:
    hermes mcp configure alltrails
```
