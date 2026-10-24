---
id: globalping
title: "globalping"
sidebar_label: "globalping"
description: "Ping, traceroute, DNS, and HTTP tests from global probes."
---

<!-- This page is auto-generated from optional-mcps/globalping/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# globalping

Ping, traceroute, DNS, and HTTP tests from global probes.

## Overview

**Source:** [https://github.com/jsdelivr/globalping-mcp-server](https://github.com/jsdelivr/globalping-mcp-server)

Install this catalog entry with:

```bash
hermes mcp install globalping
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall globalping` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.globalping.dev/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Globalping (or run `hermes mcp login globalping`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/globalping/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: globalping
description: Ping, traceroute, DNS, and HTTP tests from global probes.
source: https://github.com/jsdelivr/globalping-mcp-server

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.globalping.dev/mcp

auth:
  type: oauth

# Excluded: in-band documentation/usage-guide/rate-limit probes.
tools:
  default_excluded:
    - help
    - compareLocations
    - limits

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - globalping
    - traceroute
    - ping
  hosts:
    - globalping.io

post_install: |
  On first connection Hermes opens a browser to authorize with
  Globalping (or run `hermes mcp login globalping`). Approve access,
  then restart the session so tools load.
```
