---
id: fireflies
title: "fireflies"
sidebar_label: "fireflies"
description: "Meeting transcripts, summaries, and action items."
---

<!-- This page is auto-generated from optional-mcps/fireflies/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# fireflies

Meeting transcripts, summaries, and action items.

## Overview

**Source:** [https://docs.fireflies.ai/getting-started/mcp-configuration](https://docs.fireflies.ai/getting-started/mcp-configuration)

Install this catalog entry with:

```bash
hermes mcp install fireflies
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall fireflies` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://api.fireflies.ai/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Fireflies (or run `hermes mcp login fireflies`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/fireflies/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: fireflies
description: Meeting transcripts, summaries, and action items.
source: https://docs.fireflies.ai/getting-started/mcp-configuration

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://api.fireflies.ai/mcp

auth:
  type: oauth

# Excluded: experimental ChatGPT-connector search/fetch shims duplicating
# fireflies_get_transcripts / fireflies_get_transcript + get_summary.
tools:
  default_excluded:
    - fireflies_search
    - fireflies_fetch

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - fireflies
    - meeting notes
    - transcript
  hosts:
    - fireflies.ai

post_install: |
  On first connection Hermes opens a browser to authorize with
  Fireflies (or run `hermes mcp login fireflies`). Approve access,
  then restart the session so tools load.
```
