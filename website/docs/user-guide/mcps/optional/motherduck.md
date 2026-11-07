---
id: motherduck
title: "motherduck"
sidebar_label: "motherduck"
description: "MotherDuck: query DuckDB cloud warehouses with SQL."
---

<!-- This page is auto-generated from optional-mcps/motherduck/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# motherduck

MotherDuck: query DuckDB cloud warehouses with SQL.

## Overview

**Source:** [https://motherduck.com/docs/key-tasks/ai-and-motherduck/mcp-setup/](https://motherduck.com/docs/key-tasks/ai-and-motherduck/mcp-setup/)

Install this catalog entry with:

```bash
hermes mcp install motherduck
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall motherduck` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://api.motherduck.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

By default, only these tools are enabled at install time (others are hidden until the user opts in via the install-time checklist):

- `list_columns`
- `list_databases`
- `list_macros`
- `list_shares`
- `list_tables`
- `list_views`
- `query`
- `query_rw`
- `search_catalog`

## Post-install notes

On first connection Hermes opens a browser to authorize with
MotherDuck (or run `hermes mcp login motherduck`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/motherduck/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: motherduck
description: 'MotherDuck: query DuckDB cloud warehouses with SQL.'
source: https://motherduck.com/docs/key-tasks/ai-and-motherduck/mcp-setup/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://api.motherduck.com/mcp

auth:
  type: oauth

# Curated default: the 9 core catalog/SQL tools. The other 30 (Dive, Flight
# scheduled-jobs, Guide products + ask_docs_question) stay available via
# `hermes mcp configure motherduck`.
tools:
  default_enabled:
    - list_columns
    - list_databases
    - list_macros
    - list_shares
    - list_tables
    - list_views
    - query
    - query_rw
    - search_catalog

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - motherduck
    - duckdb
  hosts:
    - motherduck.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  MotherDuck (or run `hermes mcp login motherduck`). Approve access,
  then restart the session so tools load.
```
