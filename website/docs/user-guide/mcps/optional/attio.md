---
id: attio
title: "attio"
sidebar_label: "attio"
description: "CRM records, lists, and notes in Attio."
---

<!-- This page is auto-generated from optional-mcps/attio/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# attio

CRM records, lists, and notes in Attio.

## Overview

**Source:** [https://attio.com/help/apps/other-apps/using-the-attio-mcp-server](https://attio.com/help/apps/other-apps/using-the-attio-mcp-server)

Install this catalog entry with:

```bash
hermes mcp install attio
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall attio` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.attio.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Attio (or run `hermes mcp login attio`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/attio/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: attio
connector_slug: attio
description: CRM records, lists, and notes in Attio.
source: https://attio.com/help/apps/other-apps/using-the-attio-mcp-server

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.attio.com/mcp

auth:
  type: oauth

# Excluded: trivial identity probe; query-particle-sql is a plan-gated
# generic SQL escape hatch.
tools:
  default_excluded:
    - whoami
    - query-particle-sql

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - attio
    - crm
  hosts:
    - attio.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Attio (or run `hermes mcp login attio`). Approve access,
  then restart the session so tools load.
```
