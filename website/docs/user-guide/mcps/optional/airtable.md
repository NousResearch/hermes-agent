---
id: airtable
title: "airtable"
sidebar_label: "airtable"
description: "Bases, tables, and records from your Airtable workspace."
---

<!-- This page is auto-generated from optional-mcps/airtable/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# airtable

Bases, tables, and records from your Airtable workspace.

## Overview

**Source:** [https://support.airtable.com/articles/9897799762-using-the-airtable-mcp-server](https://support.airtable.com/articles/9897799762-using-the-airtable-mcp-server)

Install this catalog entry with:

```bash
hermes mcp install airtable
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall airtable` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.airtable.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Airtable (or run `hermes mcp login airtable`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/airtable/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: airtable
connector_slug: airtable
description: >-
  Bases, tables, and records from your Airtable workspace.
source: https://support.airtable.com/articles/9897799762-using-the-airtable-mcp-server

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration;
# Hermes's MCP client + mcp_oauth_manager handle discovery, PKCE, token
# exchange, and refresh.
transport:
  type: http
  url: https://mcp.airtable.com/mcp

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - airtable
  hosts:
    - airtable.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Airtable (or run `hermes mcp login airtable`). Approve access,
  then restart the session so tools load.
```
