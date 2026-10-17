---
id: vercel
title: "vercel"
sidebar_label: "vercel"
description: "Deployments, logs, and projects via Vercel's hosted MCP."
---

<!-- This page is auto-generated from optional-mcps/vercel/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# vercel

Deployments, logs, and projects via Vercel's hosted MCP.

## Overview

**Source:** [https://vercel.com/docs/mcp](https://vercel.com/docs/mcp)

Install this catalog entry with:

```bash
hermes mcp install vercel
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall vercel` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.vercel.com`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Vercel (or run `hermes mcp login vercel`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/vercel/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: vercel
connector_slug: vercel_mcp
description: >-
  Deployments, logs, and projects via Vercel's hosted MCP.
source: https://vercel.com/docs/mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration;
# Hermes's MCP client + mcp_oauth_manager handle discovery, PKCE, token
# exchange, and refresh.
# No `vercel.app` suggest host on purpose: pasted deploy-preview links are
# about the site being previewed, not about managing Vercel.
transport:
  type: http
  url: https://mcp.vercel.com

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - vercel
  hosts:
    - vercel.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Vercel (or run `hermes mcp login vercel`). Approve access,
  then restart the session so tools load.
```
