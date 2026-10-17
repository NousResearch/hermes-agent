---
id: atlassian
title: "atlassian"
sidebar_label: "atlassian"
description: "Jira issues and Confluence pages via Atlassian's hosted remote MCP."
---

<!-- This page is auto-generated from optional-mcps/atlassian/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# atlassian

Jira issues and Confluence pages via Atlassian's hosted remote MCP.

## Overview

**Source:** [https://support.atlassian.com/rovo/docs/getting-started-with-the-atlassian-remote-mcp-server/](https://support.atlassian.com/rovo/docs/getting-started-with-the-atlassian-remote-mcp-server/)

Install this catalog entry with:

```bash
hermes mcp install atlassian
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall atlassian` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.atlassian.com/v1/mcp/authv2`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Atlassian (or run `hermes mcp login atlassian`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/atlassian/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: atlassian
connector_slug: jira
description: >-
  Jira issues and Confluence pages via Atlassian's hosted remote MCP.
source: https://support.atlassian.com/rovo/docs/getting-started-with-the-atlassian-remote-mcp-server/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration;
# Hermes's MCP client + mcp_oauth_manager handle discovery, PKCE, token
# exchange, and refresh.
transport:
  type: http
  # /v1/sse was deprecated by Atlassian after June 30, 2026 (404s since the
  # cutover — OAuth succeeded but every handshake failed; issue #91538).
  # /v1/mcp/authv2 is the endpoint Atlassian's docs recommend for custom
  # clients (Streamable HTTP, native OAuth 2.1 + DCR — verified live).
  url: https://mcp.atlassian.com/v1/mcp/authv2

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - jira
    - confluence
    - atlassian
    - bitbucket
  hosts:
    - atlassian.net
    - atlassian.com
    - jira.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Atlassian (or run `hermes mcp login atlassian`). Approve access,
  then restart the session so tools load.
```
