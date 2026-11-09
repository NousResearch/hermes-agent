---
id: circleci
title: "circleci"
sidebar_label: "circleci"
description: "CircleCI: diagnose build failures, read logs, rerun workflows."
---

<!-- This page is auto-generated from optional-mcps/circleci/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# circleci

CircleCI: diagnose build failures, read logs, rerun workflows.

## Overview

**Source:** [https://circleci.com/docs/guides/toolkit/circleci-mcp-overview/](https://circleci.com/docs/guides/toolkit/circleci-mcp-overview/)

Install this catalog entry with:

```bash
hermes mcp install circleci
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall circleci` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.circleci.com/v1/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
CircleCI (or run `hermes mcp login circleci`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/circleci/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: circleci
description: 'CircleCI: diagnose build failures, read logs, rerun workflows.'
source: https://circleci.com/docs/guides/toolkit/circleci-mcp-overview/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.circleci.com/v1/mcp

auth:
  type: oauth

# Excluded: connectivity ping and billing-CSV power-user probe.
tools:
  default_excluded:
    - hello
    - download_usage_data

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - circleci
  hosts:
    - circleci.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  CircleCI (or run `hermes mcp login circleci`). Approve access,
  then restart the session so tools load.
```
