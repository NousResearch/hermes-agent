---
id: linear
title: "linear"
sidebar_label: "linear"
description: "Find, create, and update Linear issues, projects, and comments."
---

<!-- This page is auto-generated from optional-mcps/linear/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# linear

Find, create, and update Linear issues, projects, and comments.

## Overview

**Source:** [https://linear.app/docs/mcp](https://linear.app/docs/mcp)

Install this catalog entry with:

```bash
hermes mcp install linear
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall linear` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.linear.app/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection, Hermes will open a browser to authenticate with Linear.
After auth, restart your Hermes session so the Linear tools are loaded.

You can re-run the tool checklist any time with:
  hermes mcp configure linear

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/linear/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: linear
description: Find, create, and update Linear issues, projects, and comments.
source: https://linear.app/docs/mcp

# Linear ships a remote MCP server with native OAuth 2.1 + Dynamic Client
# Registration over Streamable HTTP. Hermes's MCP client + mcp_oauth_manager
# handle discovery, PKCE, token exchange, and refresh — nothing to install
# locally.
transport:
  type: http
  url: https://mcp.linear.app/mcp

auth:
  type: oauth
  # No `provider:` — this is native MCP OAuth (case 1), not a third-party
  # provider like Google. The MCP client triggers the browser flow on the
  # first probe / first connect.

# Tool selection at install time:
# Linear's MCP server exposes a moderate-sized tool surface (find/get/list +
# create/update across issues/projects/comments). We leave `default_enabled`
# unset so the install-time checklist starts with everything pre-checked —
# users prune what they don't want.
#
# If you want to encode a curated subset here once it stabilizes, list the
# tool names under `tools.default_enabled`. Probe failure would then apply
# that list directly.

post_install: |
  On first connection, Hermes will open a browser to authenticate with Linear.
  After auth, restart your Hermes session so the Linear tools are loaded.

  You can re-run the tool checklist any time with:
    hermes mcp configure linear
```
