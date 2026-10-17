---
id: monday
title: "monday"
sidebar_label: "monday"
description: "Boards, items, docs, and workflows in monday.com."
---

<!-- This page is auto-generated from optional-mcps/monday/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# monday

Boards, items, docs, and workflows in monday.com.

## Overview

**Source:** [https://developer.monday.com/apps/docs/mondaycom-mcp-integration](https://developer.monday.com/apps/docs/mondaycom-mcp-integration)

Install this catalog entry with:

```bash
hermes mcp install monday
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall monday` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.monday.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
monday.com (or run `hermes mcp login monday`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/monday/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: monday
description: Boards, items, docs, and workflows in monday.com.
source: https://developer.monday.com/apps/docs/mondaycom-mcp-integration

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.monday.com/mcp

auth:
  type: oauth

# Excluded: the 'Advanced API access' trio is a generic execute-any-GraphQL
# escape hatch (meta-layer — Hermes policy: no server-side API-execute
# indirection); get_sprint_summary is an AI-product feature;
# create_notification pings other users' bell/email.
tools:
  default_excluded:
    - all_monday_api
    - get_graphql_schema
    - get_type_details
    - get_sprint_summary
    - create_notification

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - monday.com
    - monday board
  hosts:
    - monday.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  monday.com (or run `hermes mcp login monday`). Approve access,
  then restart the session so tools load.
```
