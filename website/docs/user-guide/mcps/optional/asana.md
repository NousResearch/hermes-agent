---
id: asana
title: "asana"
sidebar_label: "asana"
description: "Tasks, projects, and goals from your Asana workspace."
---

<!-- This page is auto-generated from optional-mcps/asana/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# asana

Tasks, projects, and goals from your Asana workspace.

## Overview

**Source:** [https://developers.asana.com/docs/integrating-with-asanas-mcp-server](https://developers.asana.com/docs/integrating-with-asanas-mcp-server)

Install this catalog entry with:

```bash
hermes mcp install asana
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall asana` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.asana.com/v2/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

Asana V2 has no Dynamic Client Registration: you need your own Asana
MCP app. In the developer console (https://app.asana.com/0/my-apps)
create an app of type "MCP app", then:
  - OAuth → Redirect URL: http://localhost:27890/callback (exactly).
  - Manage distribution → allow the workspace(s) you will use.
Its Client ID / Client secret are the ASANA_CLIENT_ID / ASANA_CLIENT_SECRET
values prompted above. The secret is stored in the profile's .env; the
non-secret client id is inlined into config.yaml.

Then run `hermes mcp login asana` (or click Authorize in the dashboard /
Desktop: the callback still arrives at http://localhost:27890/callback, so
the browser must run on the same machine as Hermes), approve access, and
restart (or `/reload-mcp`) the Hermes session or gateway that should
expose the Asana tools.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/asana/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: asana
connector_slug: asana
description: >-
  Tasks, projects, and goals from your Asana workspace.
source: https://developers.asana.com/docs/integrating-with-asanas-mcp-server

# Asana's V2 remote MCP (Streamable HTTP). The V1 beta endpoint
# https://mcp.asana.com/sse is retired. V2 does not support Dynamic Client
# Registration: every client must be a user-owned, pre-registered Asana
# "MCP app" (developer console), so the OAuth block below pins that client
# and a fixed callback URL that the app must register verbatim. Hermes's MCP
# client + mcp_oauth_manager still handle discovery, PKCE, token exchange,
# and refresh with the configured credentials.
transport:
  type: http
  url: https://mcp.asana.com/v2/mcp

auth:
  type: oauth
  # Prompted at install (CLI, dashboard, desktop). Secrets are stored in the
  # profile's .env; non-secret values (client id below) are inlined into
  # config.yaml — config.yaml only ever carries ${VAR} references for secrets.
  env:
    - name: ASANA_CLIENT_ID
      prompt: "Asana MCP app Client ID (developer console → your MCP app → OAuth)"
      secret: false
    - name: ASANA_CLIENT_SECRET
      prompt: "Asana MCP app Client secret"
  oauth:
    client_id: "${ASANA_CLIENT_ID}"
    client_secret: "${ASANA_CLIENT_SECRET}"
    # Asana matches the redirect URL exactly; localhost and 127.0.0.1 are not
    # interchangeable. Register http://localhost:27890/callback on the app.
    redirect_host: localhost
    redirect_port: 27890

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - asana
  hosts:
    - asana.com

post_install: |
  Asana V2 has no Dynamic Client Registration: you need your own Asana
  MCP app. In the developer console (https://app.asana.com/0/my-apps)
  create an app of type "MCP app", then:
    - OAuth → Redirect URL: http://localhost:27890/callback (exactly).
    - Manage distribution → allow the workspace(s) you will use.
  Its Client ID / Client secret are the ASANA_CLIENT_ID / ASANA_CLIENT_SECRET
  values prompted above. The secret is stored in the profile's .env; the
  non-secret client id is inlined into config.yaml.

  Then run `hermes mcp login asana` (or click Authorize in the dashboard /
  Desktop: the callback still arrives at http://localhost:27890/callback, so
  the browser must run on the same machine as Hermes), approve access, and
  restart (or `/reload-mcp`) the Hermes session or gateway that should
  expose the Asana tools.
```
