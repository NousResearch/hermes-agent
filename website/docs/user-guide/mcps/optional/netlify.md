---
id: netlify
title: "netlify"
sidebar_label: "netlify"
description: "Sites, deploys, and env vars via Netlify's hosted MCP."
---

<!-- This page is auto-generated from optional-mcps/netlify/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# netlify

Sites, deploys, and env vars via Netlify's hosted MCP.

## Overview

**Source:** [https://docs.netlify.com/build/build-with-ai/agent-setup-guides/agent-setup-overview/](https://docs.netlify.com/build/build-with-ai/agent-setup-guides/agent-setup-overview/)

Install this catalog entry with:

```bash
hermes mcp install netlify
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall netlify` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://netlify-mcp.netlify.app/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Netlify (or run `hermes mcp login netlify`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/netlify/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: netlify
description: >-
  Sites, deploys, and env vars via Netlify's hosted MCP.
source: https://docs.netlify.com/build/build-with-ai/agent-setup-guides/agent-setup-overview/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration;
# Hermes's MCP client + mcp_oauth_manager handle discovery, PKCE, token
# exchange, and refresh.
# No `netlify.app` suggest host for the same deploy-preview reason as vercel.app.
transport:
  type: http
  url: https://netlify-mcp.netlify.app/mcp

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - netlify
  hosts:
    - netlify.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Netlify (or run `hermes mcp login netlify`). Approve access,
  then restart the session so tools load.
```
