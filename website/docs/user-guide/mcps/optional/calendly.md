---
id: calendly
title: "calendly"
sidebar_label: "calendly"
description: "Scheduling links, events, and invitees from Calendly."
---

<!-- This page is auto-generated from optional-mcps/calendly/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# calendly

Scheduling links, events, and invitees from Calendly.

## Overview

**Source:** [https://developer.calendly.com/calendly-mcp-server](https://developer.calendly.com/calendly-mcp-server)

Install this catalog entry with:

```bash
hermes mcp install calendly
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall calendly` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.calendly.com`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Calendly (or run `hermes mcp login calendly`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/calendly/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: calendly
connector_slug: calendly
description: Scheduling links, events, and invitees from Calendly.
source: https://developer.calendly.com/calendly-mcp-server

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.calendly.com

auth:
  type: oauth

# Excluded: vendor skill-discovery pair (instruction-loading indirection —
# Hermes skills/tool_search cover this).
tools:
  default_excluded:
    - list_calendly_skills
    - load_calendly_skill

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - calendly
    - scheduling
  hosts:
    - calendly.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Calendly (or run `hermes mcp login calendly`). Approve access,
  then restart the session so tools load.
```
