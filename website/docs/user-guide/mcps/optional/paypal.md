---
id: paypal
title: "paypal"
sidebar_label: "paypal"
description: "Payments, invoices, and subscriptions via PayPal's hosted MCP."
---

<!-- This page is auto-generated from optional-mcps/paypal/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# paypal

Payments, invoices, and subscriptions via PayPal's hosted MCP.

## Overview

**Source:** [https://developer.paypal.com/tools/mcp-server/](https://developer.paypal.com/tools/mcp-server/)

Install this catalog entry with:

```bash
hermes mcp install paypal
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall paypal` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.paypal.com/sse`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Paypal (or run `hermes mcp login paypal`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/paypal/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: paypal
description: >-
  Payments, invoices, and subscriptions via PayPal's hosted MCP.
source: https://developer.paypal.com/tools/mcp-server/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration;
# Hermes's MCP client + mcp_oauth_manager handle discovery, PKCE, token
# exchange, and refresh.
transport:
  type: http
  url: https://mcp.paypal.com/sse

auth:
  type: oauth

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - paypal
  hosts:
    - developer.paypal.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Paypal (or run `hermes mcp login paypal`). Approve access,
  then restart the session so tools load.
```
