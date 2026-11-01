---
id: buildkite
title: "buildkite"
sidebar_label: "buildkite"
description: "CI/CD pipelines, builds, and test results from Buildkite."
---

<!-- This page is auto-generated from optional-mcps/buildkite/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# buildkite

CI/CD pipelines, builds, and test results from Buildkite.

## Overview

**Source:** [https://buildkite.com/docs/apis/mcp-server](https://buildkite.com/docs/apis/mcp-server)

Install this catalog entry with:

```bash
hermes mcp install buildkite
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall buildkite` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.buildkite.com/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Buildkite (or run `hermes mcp login buildkite`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/buildkite/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: buildkite
description: CI/CD pipelines, builds, and test results from Buildkite.
source: https://buildkite.com/docs/apis/mcp-server

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.buildkite.com/mcp

auth:
  type: oauth

# Excluded: access_token (token self-probe) and get_job_env (can expose
# secrets from CI env vars into model context).
tools:
  default_excluded:
    - access_token
    - get_job_env

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - buildkite
    - ci
  hosts:
    - buildkite.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  Buildkite (or run `hermes mcp login buildkite`). Approve access,
  then restart the session so tools load.
```
