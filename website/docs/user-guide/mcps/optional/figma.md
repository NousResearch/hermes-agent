---
id: figma
title: "figma"
sidebar_label: "figma"
description: "Official Figma remote MCP — design context, Code Connect, and write-to-canvas via https://mcp.figma.com/mcp (OAuth)."
---

<!-- This page is auto-generated from optional-mcps/figma/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# figma

Official Figma remote MCP — design context, Code Connect, and write-to-canvas via https://mcp.figma.com/mcp (OAuth).

## Overview

**Source:** [https://developers.figma.com/docs/figma-mcp-server/remote-server-installation/](https://developers.figma.com/docs/figma-mcp-server/remote-server-installation/)

Install this catalog entry with:

```bash
hermes mcp install figma
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall figma` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.figma.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with Figma
(or run `hermes mcp login figma`). Approve access, then restart the
session so tools load.

Figma walks OAuth DCR by exact client_name. Hermes registers as
"Claude Code" automatically so registration is not 403'd — you do not
need to paste a client_id.

Prompt with a frame/file link:
  implement this design: https://www.figma.com/design/&lt;fileKey>/...

Desktop alternative (no OAuth, local only): enable Dev Mode MCP in the
Figma desktop app (Shift+D → Inspect → Enable desktop MCP server) and
point url at http://127.0.0.1:3845/mcp instead.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/figma/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: figma
connector_slug: figma
description: >-
  Official Figma remote MCP — design context, Code Connect, and
  write-to-canvas via https://mcp.figma.com/mcp (OAuth).
source: https://developers.figma.com/docs/figma-mcp-server/remote-server-installation/

# Figma's hosted endpoint requires OAuth 2.1 + DCR. Their registration
# endpoint allowlists *exact* client_name strings (not arbitrary clients):
# "Claude Code" and "Codex" succeed; "Hermes Agent" 403s. Hermes's MCP
# OAuth layer auto-sets client_name to "Claude Code" for this host (see
# tools/mcp_oauth.apply_oauth_provider_defaults) so the browser flow can
# start. Users can override via oauth.client_name if they want Codex
# instead. Scope defaults to mcp:connect.
transport:
  type: http
  url: https://mcp.figma.com/mcp

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - figma
    - mockup
    - wireframe
  hosts:
    - figma.com

post_install: |
  On first connection Hermes opens a browser to authorize with Figma
  (or run `hermes mcp login figma`). Approve access, then restart the
  session so tools load.

  Figma walks OAuth DCR by exact client_name. Hermes registers as
  "Claude Code" automatically so registration is not 403'd — you do not
  need to paste a client_id.

  Prompt with a frame/file link:
    implement this design: https://www.figma.com/design/<fileKey>/...

  Desktop alternative (no OAuth, local only): enable Dev Mode MCP in the
  Figma desktop app (Shift+D → Inspect → Enable desktop MCP server) and
  point url at http://127.0.0.1:3845/mcp instead.
```
