---
id: stripe
title: "stripe"
sidebar_label: "stripe"
description: "Payments, customers, and invoices via Stripe's hosted MCP."
---

<!-- This page is auto-generated from optional-mcps/stripe/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# stripe

Payments, customers, and invoices via Stripe's hosted MCP.

## Overview

**Source:** [https://docs.stripe.com/mcp](https://docs.stripe.com/mcp)

Install this catalog entry with:

```bash
hermes mcp install stripe
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall stripe` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.stripe.com`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Stripe (or run `hermes mcp login stripe`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/stripe/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: stripe
connector_slug: stripe_mcp
description: >-
  Payments, customers, and invoices via Stripe's hosted MCP.
source: https://docs.stripe.com/mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration;
# Hermes's MCP client + mcp_oauth_manager handle discovery, PKCE, token
# exchange, and refresh.
transport:
  type: http
  url: https://mcp.stripe.com

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - stripe
  hosts:
    - dashboard.stripe.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Stripe (or run `hermes mcp login stripe`). Approve access,
  then restart the session so tools load.
```
