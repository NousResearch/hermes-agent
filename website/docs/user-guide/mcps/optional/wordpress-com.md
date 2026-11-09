---
id: wordpress-com
title: "wordpress-com"
sidebar_label: "wordpress-com"
description: "WordPress.com: posts, pages, drafts, stats, and comments."
---

<!-- This page is auto-generated from optional-mcps/wordpress-com/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# wordpress-com

WordPress.com: posts, pages, drafts, stats, and comments.

## Overview

**Source:** [https://developer.wordpress.com/docs/mcp/](https://developer.wordpress.com/docs/mcp/)

Install this catalog entry with:

```bash
hermes mcp install wordpress-com
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall wordpress-com` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://public-api.wordpress.com/wpcom/v2/mcp/v1`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
WordPress.com (or run `hermes mcp login wordpress-com`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/wordpress-com/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: wordpress-com
description: 'WordPress.com: posts, pages, drafts, stats, and comments.'
source: https://developer.wordpress.com/docs/mcp/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://public-api.wordpress.com/wpcom/v2/mcp/v1

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - wordpress
  hosts:
    - wordpress.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  WordPress.com (or run `hermes mcp login wordpress-com`). Approve access,
  then restart the session so tools load.
```
