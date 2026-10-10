---
id: datadog
title: "datadog"
sidebar_label: "datadog"
description: "Logs, monitors, dashboards, and incidents from Datadog."
---

<!-- This page is auto-generated from optional-mcps/datadog/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# datadog

Logs, monitors, dashboards, and incidents from Datadog.

## Overview

**Source:** [https://docs.datadoghq.com/bits_ai/mcp_server/](https://docs.datadoghq.com/bits_ai/mcp_server/)

Install this catalog entry with:

```bash
hermes mcp install datadog
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall datadog` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.datadoghq.com/api/unstable/mcp-server/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Datadog (or run `hermes mcp login datadog`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/datadog/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: datadog
connector_slug: datadog_mcp
description: >-
  Logs, monitors, dashboards, and incidents from Datadog.
source: https://docs.datadoghq.com/bits_ai/mcp_server/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration;
# Hermes's MCP client + mcp_oauth_manager handle discovery, PKCE, token
# exchange, and refresh.
transport:
  type: http
  url: https://mcp.datadoghq.com/api/unstable/mcp-server/mcp

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - datadog
    - apm
  hosts:
    - datadoghq.com
    - datadoghq.eu

post_install: |
  On first connection Hermes opens a browser to authorize with
  Datadog (or run `hermes mcp login datadog`). Approve access,
  then restart the session so tools load.
```
