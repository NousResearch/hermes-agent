---
id: hugging_face
title: "hugging_face"
sidebar_label: "hugging_face"
description: "Models, datasets, Spaces, and papers from the Hugging Face Hub."
---

<!-- This page is auto-generated from optional-mcps/hugging_face/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# hugging_face

Models, datasets, Spaces, and papers from the Hugging Face Hub.

## Overview

**Source:** [https://huggingface.co/docs/hub/agents-mcp](https://huggingface.co/docs/hub/agents-mcp)

Install this catalog entry with:

```bash
hermes mcp install hugging_face
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall hugging_face` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://huggingface.co/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Hugging Face (or run `hermes mcp login hugging_face`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/hugging_face/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: hugging_face
description: >-
  Models, datasets, Spaces, and papers from the Hugging Face Hub.
source: https://huggingface.co/docs/hub/agents-mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration;
# Hermes's MCP client + mcp_oauth_manager handle discovery, PKCE, token
# exchange, and refresh.
# Underscored name so UIs render "Hugging Face", not "Huggingface".
transport:
  type: http
  url: https://huggingface.co/mcp

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - hugging face
    - huggingface
  hosts:
    - huggingface.co
    - hf.co

post_install: |
  On first connection Hermes opens a browser to authorize with
  Hugging Face (or run `hermes mcp login hugging_face`). Approve access,
  then restart the session so tools load.
```
