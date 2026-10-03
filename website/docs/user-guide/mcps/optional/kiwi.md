---
id: kiwi
title: "kiwi"
sidebar_label: "kiwi"
description: "Kiwi.com flight search: itineraries with direct booking links."
---

<!-- This page is auto-generated from optional-mcps/kiwi/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# kiwi

Kiwi.com flight search: itineraries with direct booking links.

## Overview

**Source:** [https://www.kiwi.com/stories/kiwi-mcp-connector/](https://www.kiwi.com/stories/kiwi-mcp-connector/)

Install this catalog entry with:

```bash
hermes mcp install kiwi
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall kiwi` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.kiwi.com`

## Auth

**Type:** `none`

No credentials required.

## Tools

By default, only these tools are enabled at install time (others are hidden until the user opts in via the install-time checklist):

- `search-flight`

## Post-install notes

No account or credentials needed — tools are available as soon as the
session restarts.

Search only — booking and payment happen on kiwi.com via the returned
links, never in-conversation.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/kiwi/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: kiwi
description: 'Kiwi.com flight search: itineraries with direct booking links.'
source: https://www.kiwi.com/stories/kiwi-mcp-connector/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). No authentication required (verified live:
# initialize succeeds anonymously).

transport:
  type: http
  url: https://mcp.kiwi.com

auth:
  type: none

# The server ships exactly two tools: search-flight (the product) and
# feedback-to-devs (outbound feedback channel — excluded per Hermes policy:
# no telemetry/feedback tools without explicit user opt-in).
tools:
  default_enabled:
    - search-flight

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - flight search
    - kiwi.com
  hosts:
    - kiwi.com

post_install: |
  No account or credentials needed — tools are available as soon as the
  session restarts.

  Search only — booking and payment happen on kiwi.com via the returned
  links, never in-conversation.
```
