---
id: context7
title: "context7"
sidebar_label: "context7"
description: "Up-to-date, version-specific library docs and code examples."
---

<!-- This page is auto-generated from optional-mcps/context7/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# context7

Up-to-date, version-specific library docs and code examples.

## Overview

**Source:** [https://context7.com/docs/resources/all-clients](https://context7.com/docs/resources/all-clients)

Install this catalog entry with:

```bash
hermes mcp install context7
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall context7` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.context7.com/mcp`

## Auth

**Type:** `none`

No credentials required.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

No account or credentials needed — tools are available as soon as the
session restarts.

Works anonymously with rate limits; for higher limits create an API key
at context7.com/dashboard and add it as a Bearer header via
mcp_servers.context7.headers in config.yaml.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/context7/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: context7
description: Up-to-date, version-specific library docs and code examples.
source: https://context7.com/docs/resources/all-clients

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). No authentication required (verified live:
# initialize succeeds anonymously).

transport:
  type: http
  url: https://mcp.context7.com/mcp

auth:
  type: none

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - context7
    - library docs
  hosts:
    - context7.com

post_install: |
  No account or credentials needed — tools are available as soon as the
  session restarts.

  Works anonymously with rate limits; for higher limits create an API key
  at context7.com/dashboard and add it as a Bearer header via
  mcp_servers.context7.headers in config.yaml.
```
