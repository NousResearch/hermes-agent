---
id: twilio-docs
title: "twilio-docs"
sidebar_label: "twilio-docs"
description: "Twilio developer docs search (public beta, read-only)."
---

<!-- This page is auto-generated from optional-mcps/twilio-docs/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# twilio-docs

Twilio developer docs search (public beta, read-only).

## Overview

**Source:** [https://www.twilio.com/docs/ai/mcp](https://www.twilio.com/docs/ai/mcp)

Install this catalog entry with:

```bash
hermes mcp install twilio-docs
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall twilio-docs` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.twilio.com/docs`

## Auth

**Type:** `none`

No credentials required.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

No account or credentials needed — tools are available as soon as the
session restarts.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/twilio-docs/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: twilio-docs
description: 'Twilio developer docs search (public beta, read-only).'
source: https://www.twilio.com/docs/ai/mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). No authentication required (verified live:
# initialize succeeds anonymously).

transport:
  type: http
  url: https://mcp.twilio.com/docs

auth:
  type: none

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - twilio docs
  hosts:
    - twilio.com

post_install: |
  No account or credentials needed — tools are available as soon as the
  session restarts.
```
