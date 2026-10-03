---
id: todoist
title: "todoist"
sidebar_label: "todoist"
description: "Manage Todoist tasks and projects."
---

<!-- This page is auto-generated from optional-mcps/todoist/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# todoist

Manage Todoist tasks and projects.

## Overview

**Source:** [https://www.todoist.com/help/articles/todoist-mcp-server](https://www.todoist.com/help/articles/todoist-mcp-server)

Install this catalog entry with:

```bash
hermes mcp install todoist
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall todoist` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://ai.todoist.net/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Todoist (or run `hermes mcp login todoist`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/todoist/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: todoist
connector_slug: todoist
description: Manage Todoist tasks and projects.
source: https://www.todoist.com/help/articles/todoist-mcp-server

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://ai.todoist.net/mcp

auth:
  type: oauth

# Excluded: search/fetch are the OpenAI-connector compatibility layer that
# duplicates the find-* tools; template import/export is niche.
tools:
  default_excluded:
    - search
    - fetch
    - export-project-template
    - import-project-template

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - todoist
    - todo
  hosts:
    - todoist.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Todoist (or run `hermes mcp login todoist`). Approve access,
  then restart the session so tools load.
```
