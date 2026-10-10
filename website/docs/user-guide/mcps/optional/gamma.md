---
id: gamma
title: "gamma"
sidebar_label: "gamma"
description: "Gamma: generate and edit AI presentations, docs, and sites."
---

<!-- This page is auto-generated from optional-mcps/gamma/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# gamma

Gamma: generate and edit AI presentations, docs, and sites.

## Overview

**Source:** [https://developers.gamma.app/docs/gamma-mcp-server](https://developers.gamma.app/docs/gamma-mcp-server)

Install this catalog entry with:

```bash
hermes mcp install gamma
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall gamma` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.gamma.app/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Gamma (or run `hermes mcp login gamma`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/gamma/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: gamma
description: 'Gamma: generate and edit AI presentations, docs, and sites.'
source: https://developers.gamma.app/docs/gamma-mcp-server

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.gamma.app/mcp

auth:
  type: oauth

# Excluded: per-person viewer tracking (incl. emails) — privacy-sensitive
# telemetry-grade analytics; get_gamma_analytics covers the useful case.
tools:
  default_excluded:
    - get_gamma_viewer_analytics
    - get_gamma_viewer_detail_analytics
    - get_gamma_card_analytics

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - gamma
    - presentation
  hosts:
    - gamma.app

post_install: |
  On first connection Hermes opens a browser to authorize with
  Gamma (or run `hermes mcp login gamma`). Approve access,
  then restart the session so tools load.
```
