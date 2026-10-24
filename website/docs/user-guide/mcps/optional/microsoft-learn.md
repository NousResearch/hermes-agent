---
id: microsoft-learn
title: "microsoft-learn"
sidebar_label: "microsoft-learn"
description: "Official Microsoft, Azure, and .NET docs and code samples."
---

<!-- This page is auto-generated from optional-mcps/microsoft-learn/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# microsoft-learn

Official Microsoft, Azure, and .NET docs and code samples.

## Overview

**Source:** [https://learn.microsoft.com/en-us/training/support/mcp-get-started](https://learn.microsoft.com/en-us/training/support/mcp-get-started)

Install this catalog entry with:

```bash
hermes mcp install microsoft-learn
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall microsoft-learn` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://learn.microsoft.com/api/mcp`

## Auth

**Type:** `none`

No credentials required.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

No account or credentials needed — tools are available as soon as the
session restarts.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/microsoft-learn/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: microsoft-learn
description: Official Microsoft, Azure, and .NET docs and code samples.
source: https://learn.microsoft.com/en-us/training/support/mcp-get-started

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). No authentication required (verified live:
# initialize succeeds anonymously).

transport:
  type: http
  url: https://learn.microsoft.com/api/mcp

auth:
  type: none

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - microsoft docs
    - azure docs
    - msdn
  hosts:
    - learn.microsoft.com

post_install: |
  No account or credentials needed — tools are available as soon as the
  session restarts.
```
