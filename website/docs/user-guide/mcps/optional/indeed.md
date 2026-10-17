---
id: indeed
title: "indeed"
sidebar_label: "indeed"
description: "Search jobs and listings on Indeed."
---

<!-- This page is auto-generated from optional-mcps/indeed/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# indeed

Search jobs and listings on Indeed.

## Overview

**Source:** [https://docs.indeed.com/indeed-mcp](https://docs.indeed.com/indeed-mcp)

Install this catalog entry with:

```bash
hermes mcp install indeed
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall indeed` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.indeed.com/claude/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Indeed (or run `hermes mcp login indeed`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/indeed/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: indeed
description: Search jobs and listings on Indeed.
source: https://docs.indeed.com/indeed-mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.indeed.com/claude/mcp

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - indeed
    - job search
  hosts:
    - indeed.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Indeed (or run `hermes mcp login indeed`). Approve access,
  then restart the session so tools load.
```
