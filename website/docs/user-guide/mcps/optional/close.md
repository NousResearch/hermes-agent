---
id: close
title: "close"
sidebar_label: "close"
description: "Sales CRM: leads, opportunities, calls, and emails."
---

<!-- This page is auto-generated from optional-mcps/close/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# close

Sales CRM: leads, opportunities, calls, and emails.

## Overview

**Source:** [https://help.close.com/docs/mcp-server](https://help.close.com/docs/mcp-server)

Install this catalog entry with:

```bash
hermes mcp install close
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall close` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.close.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Close (or run `hermes mcp login close`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/close/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: close
description: 'Sales CRM: leads, opportunities, calls, and emails.'
source: https://help.close.com/docs/mcp-server

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.close.com/mcp

auth:
  type: oauth

# Excluded (107-tool surface): product-help search; generic search/fetch/
# paginate layer duplicating lead_search/activity_search; AI field
# enrichment; and the entire voice-agent cluster — schedule_voice_agent_call
# places REAL outbound AI phone calls to contacts. Re-enable any with
# `hermes mcp configure close`.
tools:
  default_excluded:
    - close_product_knowledge_search
    - customized_builtin_labels
    - search
    - fetch
    - paginate_search
    - enrich_field
    - schedule_voice_agent_call
    - apply_voice_agent_update
    - propose_voice_agent_update
    - find_voice_agents
    - find_agent_configs
    - get_voice_agents
    - get_voice_agent_overview_report
    - get_voice_agent_performance_report

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - close crm
  hosts:
    - close.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Close (or run `hermes mcp login close`). Approve access,
  then restart the session so tools load.
```
