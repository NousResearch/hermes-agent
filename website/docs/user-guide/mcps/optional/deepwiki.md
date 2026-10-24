---
id: deepwiki
title: "deepwiki"
sidebar_label: "deepwiki"
description: "Ask questions about any public GitHub repo (Devin's DeepWiki)."
---

<!-- This page is auto-generated from optional-mcps/deepwiki/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# deepwiki

Ask questions about any public GitHub repo (Devin's DeepWiki).

## Overview

**Source:** [https://docs.devin.ai/work-with-devin/deepwiki-mcp](https://docs.devin.ai/work-with-devin/deepwiki-mcp)

Install this catalog entry with:

```bash
hermes mcp install deepwiki
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall deepwiki` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.deepwiki.com/mcp`

## Auth

**Type:** `none`

No credentials required.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

No account or credentials needed — tools are available as soon as the
session restarts.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/deepwiki/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: deepwiki
description: Ask questions about any public GitHub repo (Devin's DeepWiki).
source: https://docs.devin.ai/work-with-devin/deepwiki-mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). No authentication required (verified live:
# initialize succeeds anonymously).

transport:
  type: http
  url: https://mcp.deepwiki.com/mcp

auth:
  type: none

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - deepwiki
    - repo wiki
  hosts:
    - deepwiki.com

post_install: |
  No account or credentials needed — tools are available as soon as the
  session restarts.
```
