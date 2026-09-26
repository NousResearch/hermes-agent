---
id: strava
title: "strava"
sidebar_label: "strava"
description: "Strava: activities, fitness trends, training load (read-only)."
---

<!-- This page is auto-generated from optional-mcps/strava/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# strava

Strava: activities, fitness trends, training load (read-only).

## Overview

**Source:** [https://support.strava.com/en-us/articles/15401531-strava-mcp-connector](https://support.strava.com/en-us/articles/15401531-strava-mcp-connector)

Install this catalog entry with:

```bash
hermes mcp install strava
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall strava` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.strava.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Strava (or run `hermes mcp login strava`). Approve access,
then restart the session so tools load.

Requires a Strava subscription.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/strava/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: strava
description: 'Strava: activities, fitness trends, training load (read-only).'
source: https://support.strava.com/en-us/articles/15401531-strava-mcp-connector

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.strava.com/mcp

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - strava
    - running
    - cycling
  hosts:
    - strava.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Strava (or run `hermes mcp login strava`). Approve access,
  then restart the session so tools load.

  Requires a Strava subscription.
```
