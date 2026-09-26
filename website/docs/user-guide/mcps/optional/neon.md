---
id: neon
title: "neon"
sidebar_label: "neon"
description: "Neon serverless Postgres: projects, branches, and SQL."
---

<!-- This page is auto-generated from optional-mcps/neon/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# neon

Neon serverless Postgres: projects, branches, and SQL.

## Overview

**Source:** [https://neon.com/docs/ai/neon-mcp-server](https://neon.com/docs/ai/neon-mcp-server)

Install this catalog entry with:

```bash
hermes mcp install neon
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall neon` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.neon.tech/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Neon (or run `hermes mcp login neon`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/neon/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: neon
connector_slug: neon_mcp
description: 'Neon serverless Postgres: projects, branches, and SQL.'
source: https://neon.com/docs/ai/neon-mcp-server

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.neon.tech/mcp

auth:
  type: oauth

# Excluded: search/fetch nav indirection (redundant with list/describe
# tools); docs-lookup pair; observability beta (single-region, dead weight
# for most); Neon Auth product provisioning trio.
tools:
  default_excluded:
    - search
    - fetch
    - list_docs_resources
    - get_doc_resource
    - query_logs
    - list_log_fields
    - list_log_field_values
    - provision_neon_auth
    - configure_neon_auth
    - get_neon_auth_config

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - neon
    - postgres
  hosts:
    - neon.tech
    - neon.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Neon (or run `hermes mcp login neon`). Approve access,
  then restart the session so tools load.
```
