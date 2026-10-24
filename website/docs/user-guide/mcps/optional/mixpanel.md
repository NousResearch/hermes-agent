---
id: mixpanel
title: "mixpanel"
sidebar_label: "mixpanel"
description: "Mixpanel analytics: events, funnels, retention, dashboards."
---

<!-- This page is auto-generated from optional-mcps/mixpanel/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# mixpanel

Mixpanel analytics: events, funnels, retention, dashboards.

## Overview

**Source:** [https://docs.mixpanel.com/docs/mcp](https://docs.mixpanel.com/docs/mcp)

Install this catalog entry with:

```bash
hermes mcp install mixpanel
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall mixpanel` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.mixpanel.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Mixpanel (or run `hermes mcp login mixpanel`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/mixpanel/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: mixpanel
description: 'Mixpanel analytics: events, funnels, retention, dashboards.'
source: https://docs.mixpanel.com/docs/mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.mixpanel.com/mcp

auth:
  type: oauth

# Excluded: six guidance pseudo-tools returning static best-practice text;
# chart-widget renderer; deep-link generator; bulk variants of
# Edit-Event/Edit-Property.
tools:
  default_excluded:
    - Get-Experiment-Setup-Guidance
    - Get-Experiment-Results-Interpretation-Guidance
    - Explain-Experiment-Health-Check
    - Run-Experiment-Pre-Launch-Checks
    - Get-Feature-Flag-Setup-Guidance
    - Get-Feature-Flag-Lifecycle-Guidance
    - Display-Query
    - Get-Lexicon-URL
    - Bulk-Edit-Events
    - Bulk-Edit-Properties

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - mixpanel
  hosts:
    - mixpanel.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Mixpanel (or run `hermes mcp login mixpanel`). Approve access,
  then restart the session so tools load.
```
