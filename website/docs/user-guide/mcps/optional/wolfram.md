---
id: wolfram
title: "wolfram"
sidebar_label: "wolfram"
description: "Wolfram|Alpha computation, math, and curated knowledge."
---

<!-- This page is auto-generated from optional-mcps/wolfram/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# wolfram

Wolfram|Alpha computation, math, and curated knowledge.

## Overview

**Source:** [https://www.wolfram.com/agent-tools/](https://www.wolfram.com/agent-tools/)

Install this catalog entry with:

```bash
hermes mcp install wolfram
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall wolfram` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://agenttools.wolfram.com/mcp`

## Auth

**Type:** `none`

No credentials required.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

No account or credentials needed — tools are available as soon as the
session restarts.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/wolfram/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: wolfram
description: Wolfram|Alpha computation, math, and curated knowledge.
source: https://www.wolfram.com/agent-tools/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). No authentication required (verified live:
# initialize succeeds anonymously).

transport:
  type: http
  url: https://agenttools.wolfram.com/mcp

auth:
  type: none

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - wolfram
    - wolfram alpha
    - computation
  hosts:
    - wolfram.com
    - wolframalpha.com

post_install: |
  No account or credentials needed — tools are available as soon as the
  session restarts.
```
