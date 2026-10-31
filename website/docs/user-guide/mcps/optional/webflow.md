---
id: webflow
title: "webflow"
sidebar_label: "webflow"
description: "Sites, CMS collections, and pages via Webflow's hosted MCP."
---

<!-- This page is auto-generated from optional-mcps/webflow/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# webflow

Sites, CMS collections, and pages via Webflow's hosted MCP.

## Overview

**Source:** [https://developers.webflow.com/mcp/reference/getting-started](https://developers.webflow.com/mcp/reference/getting-started)

Install this catalog entry with:

```bash
hermes mcp install webflow
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall webflow` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.webflow.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Webflow (or run `hermes mcp login webflow`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/webflow/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: webflow
description: >-
  Sites, CMS collections, and pages via Webflow's hosted MCP.
source: https://developers.webflow.com/mcp/reference/getting-started

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration;
# Hermes's MCP client + mcp_oauth_manager handle discovery, PKCE, token
# exchange, and refresh.
# No `webflow.io` suggest host — that's published staging sites, not Webflow intent.
transport:
  type: http
  url: https://mcp.webflow.com/mcp

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - webflow
  hosts:
    - webflow.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Webflow (or run `hermes mcp login webflow`). Approve access,
  then restart the session so tools load.
```
