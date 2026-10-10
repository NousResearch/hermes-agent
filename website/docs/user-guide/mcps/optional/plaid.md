---
id: plaid
title: "plaid"
sidebar_label: "plaid"
description: "Plaid dashboard: integrations, Items, and usage debugging."
---

<!-- This page is auto-generated from optional-mcps/plaid/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# plaid

Plaid dashboard: integrations, Items, and usage debugging.

## Overview

**Source:** [https://plaid.com/docs/resources/mcp/](https://plaid.com/docs/resources/mcp/)

Install this catalog entry with:

```bash
hermes mcp install plaid
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall plaid` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://api.dashboard.plaid.com/mcp/`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Plaid (or run `hermes mcp login plaid`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/plaid/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: plaid
description: 'Plaid dashboard: integrations, Items, and usage debugging.'
source: https://plaid.com/docs/resources/mcp/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://api.dashboard.plaid.com/mcp/

auth:
  type: oauth

# Excluded: meta-intro tool that spends a call explaining the other four.
tools:
  default_excluded:
    - plaid_get_tools_introduction

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - plaid
  hosts:
    - plaid.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Plaid (or run `hermes mcp login plaid`). Approve access,
  then restart the session so tools load.
```
