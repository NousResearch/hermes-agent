---
id: amplitude
title: "amplitude"
sidebar_label: "amplitude"
description: "Amplitude analytics: charts, dashboards, experiments, flags."
---

<!-- This page is auto-generated from optional-mcps/amplitude/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# amplitude

Amplitude analytics: charts, dashboards, experiments, flags.

## Overview

**Source:** [https://amplitude.com/docs/amplitude-ai/amplitude-mcp](https://amplitude.com/docs/amplitude-ai/amplitude-mcp)

Install this catalog entry with:

```bash
hermes mcp install amplitude
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall amplitude` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.amplitude.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Amplitude (or run `hermes mcp login amplitude`). Approve access,
then restart the session so tools load.

EU-resident orgs: change mcp_servers.amplitude.url to
https://mcp.eu.amplitude.com/mcp in config.yaml.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/amplitude/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: amplitude
description: 'Amplitude analytics: charts, dashboards, experiments, flags.'
source: https://amplitude.com/docs/amplitude-ai/amplitude-mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.amplitude.com/mcp

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - amplitude
  hosts:
    - amplitude.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Amplitude (or run `hermes mcp login amplitude`). Approve access,
  then restart the session so tools load.

  EU-resident orgs: change mcp_servers.amplitude.url to
  https://mcp.eu.amplitude.com/mcp in config.yaml.
```
