---
id: prisma-postgres
title: "prisma-postgres"
sidebar_label: "prisma-postgres"
description: "Create and manage Prisma Postgres databases."
---

<!-- This page is auto-generated from optional-mcps/prisma-postgres/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# prisma-postgres

Create and manage Prisma Postgres databases.

## Overview

**Source:** [https://www.prisma.io/docs/postgres/integrations/mcp-server](https://www.prisma.io/docs/postgres/integrations/mcp-server)

Install this catalog entry with:

```bash
hermes mcp install prisma-postgres
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall prisma-postgres` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.prisma.io/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Prisma Postgres (or run `hermes mcp login prisma-postgres`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/prisma-postgres/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: prisma-postgres
connector_slug: prisma
description: Create and manage Prisma Postgres databases.
source: https://www.prisma.io/docs/postgres/integrations/mcp-server

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.prisma.io/mcp

auth:
  type: oauth

# Excluded: docs Q&A tool bundled into a DB-management server.
tools:
  default_excluded:
    - search_prisma_documentation

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - prisma
  hosts:
    - prisma.io

post_install: |
  On first connection Hermes opens a browser to authorize with
  Prisma Postgres (or run `hermes mcp login prisma-postgres`). Approve access,
  then restart the session so tools load.
```
