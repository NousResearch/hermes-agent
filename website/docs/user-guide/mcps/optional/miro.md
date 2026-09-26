---
id: miro
title: "miro"
sidebar_label: "miro"
description: "Read and edit Miro boards, diagrams, and frames."
---

<!-- This page is auto-generated from optional-mcps/miro/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# miro

Read and edit Miro boards, diagrams, and frames.

## Overview

**Source:** [https://developers.miro.com/docs/connecting-to-miro-mcp](https://developers.miro.com/docs/connecting-to-miro-mcp)

Install this catalog entry with:

```bash
hermes mcp install miro
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall miro` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.miro.com/`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Miro (or run `hermes mcp login miro`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/miro/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: miro
description: Read and edit Miro boards, diagrams, and frames.
source: https://developers.miro.com/docs/connecting-to-miro-mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.miro.com/

auth:
  type: oauth

# Excluded: diagram_create/diagram_get_dsl are vendor-deprecated; the four
# layout_* tools are 'deprecating soon', duplicated by the canvas_* set.
tools:
  default_excluded:
    - diagram_create
    - diagram_get_dsl
    - layout_create
    - layout_get_dsl
    - layout_read
    - layout_update

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - miro
    - whiteboard
  hosts:
    - miro.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Miro (or run `hermes mcp login miro`). Approve access,
  then restart the session so tools load.
```
