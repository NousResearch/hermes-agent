---
id: railway
title: "railway"
sidebar_label: "railway"
description: "Railway: projects, services, deployments, and environments."
---

<!-- This page is auto-generated from optional-mcps/railway/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# railway

Railway: projects, services, deployments, and environments.

## Overview

**Source:** [https://docs.railway.com/guides/mcp-server](https://docs.railway.com/guides/mcp-server)

Install this catalog entry with:

```bash
hermes mcp install railway
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall railway` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.railway.com`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Railway (or run `hermes mcp login railway`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/railway/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: railway
connector_slug: railway
description: 'Railway: projects, services, deployments, and environments.'
source: https://docs.railway.com/guides/mcp-server

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.
transport:
  type: http
  url: https://mcp.railway.com

auth:
  type: oauth

# Excluded: railway-agent hands the request to Railway's server-side AI
# agent for multi-step infra operations — an opaque delegation meta-layer
# that acts outside Hermes's per-tool approval loop. The remaining tools
# are direct (and destructive ones carry vendor destructive-hints).
tools:
  default_excluded:
    - railway-agent

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - railway
    - railway deploy
  hosts:
    - railway.com
    - railway.app

post_install: |
  On first connection Hermes opens a browser to authorize with
  Railway (or run `hermes mcp login railway`). Approve access,
  then restart the session so tools load.
```
