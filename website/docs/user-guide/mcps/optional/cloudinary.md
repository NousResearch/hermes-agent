---
id: cloudinary
title: "cloudinary"
sidebar_label: "cloudinary"
description: "Upload, search, and transform media assets in Cloudinary."
---

<!-- This page is auto-generated from optional-mcps/cloudinary/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# cloudinary

Upload, search, and transform media assets in Cloudinary.

## Overview

**Source:** [https://cloudinary.com/documentation/cloudinary_llm_mcp](https://cloudinary.com/documentation/cloudinary_llm_mcp)

Install this catalog entry with:

```bash
hermes mcp install cloudinary
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall cloudinary` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://asset-management.mcp.cloudinary.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Cloudinary (or run `hermes mcp login cloudinary`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/cloudinary/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: cloudinary
description: Upload, search, and transform media assets in Cloudinary.
source: https://cloudinary.com/documentation/cloudinary_llm_mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://asset-management.mcp.cloudinary.com/mcp

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - cloudinary
    - media assets
  hosts:
    - cloudinary.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Cloudinary (or run `hermes mcp login cloudinary`). Approve access,
  then restart the session so tools load.
```
