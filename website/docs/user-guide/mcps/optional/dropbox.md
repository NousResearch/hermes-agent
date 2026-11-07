---
id: dropbox
title: "dropbox"
sidebar_label: "dropbox"
description: "Search, read, and manage files in Dropbox."
---

<!-- This page is auto-generated from optional-mcps/dropbox/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# dropbox

Search, read, and manage files in Dropbox.

## Overview

**Source:** [https://help.dropbox.com/integrations/connect-dropbox-mcp-server](https://help.dropbox.com/integrations/connect-dropbox-mcp-server)

Install this catalog entry with:

```bash
hermes mcp install dropbox
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall dropbox` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.dropbox.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Dropbox (or run `hermes mcp login dropbox`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/dropbox/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: dropbox
connector_slug: dropbox
description: Search, read, and manage files in Dropbox.
source: https://help.dropbox.com/integrations/connect-dropbox-mcp-server

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.dropbox.com/mcp

auth:
  type: oauth

# Excluded: account quota probe and the niche file-request feature trio.
tools:
  default_excluded:
    - GetUsageAndQuota
    - CreateFileRequest
    - GetFileRequest
    - ListFileRequests

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - dropbox
  hosts:
    - dropbox.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Dropbox (or run `hermes mcp login dropbox`). Approve access,
  then restart the session so tools load.
```
