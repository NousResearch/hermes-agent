---
id: sentry
title: "sentry"
sidebar_label: "sentry"
description: "Issues, stack traces, and error context from Sentry."
---

<!-- This page is auto-generated from optional-mcps/sentry/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# sentry

Issues, stack traces, and error context from Sentry.

## Overview

**Source:** [https://docs.sentry.io/product/sentry-mcp/](https://docs.sentry.io/product/sentry-mcp/)

Install this catalog entry with:

```bash
hermes mcp install sentry
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall sentry` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.sentry.dev/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Sentry (or run `hermes mcp login sentry`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/sentry/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: sentry
connector_slug: sentry_mcp
description: >-
  Issues, stack traces, and error context from Sentry.
source: https://docs.sentry.io/product/sentry-mcp/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration;
# Hermes's MCP client + mcp_oauth_manager handle discovery, PKCE, token
# exchange, and refresh.
transport:
  type: http
  url: https://mcp.sentry.dev/mcp

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - sentry
    - stack trace
    - crash report
  hosts:
    - sentry.io

post_install: |
  On first connection Hermes opens a browser to authorize with
  Sentry (or run `hermes mcp login sentry`). Approve access,
  then restart the session so tools load.
```
