---
id: supabase
title: "supabase"
sidebar_label: "supabase"
description: "Database, auth, and storage from your Supabase projects."
---

<!-- This page is auto-generated from optional-mcps/supabase/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# supabase

Database, auth, and storage from your Supabase projects.

## Overview

**Source:** [https://supabase.com/docs/guides/ai-tools/mcp](https://supabase.com/docs/guides/ai-tools/mcp)

Install this catalog entry with:

```bash
hermes mcp install supabase
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall supabase` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.supabase.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Supabase (or run `hermes mcp login supabase`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/supabase/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: supabase
connector_slug: supabase
description: >-
  Database, auth, and storage from your Supabase projects.
source: https://supabase.com/docs/guides/ai-tools/mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration;
# Hermes's MCP client + mcp_oauth_manager handle discovery, PKCE, token
# exchange, and refresh.
transport:
  type: http
  url: https://mcp.supabase.com/mcp

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - supabase
  hosts:
    - supabase.com
    - supabase.co

post_install: |
  On first connection Hermes opens a browser to authorize with
  Supabase (or run `hermes mcp login supabase`). Approve access,
  then restart the session so tools load.
```
