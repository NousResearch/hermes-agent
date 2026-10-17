---
id: semgrep
title: "semgrep"
sidebar_label: "semgrep"
description: "Scan code for security vulnerabilities with Semgrep."
---

<!-- This page is auto-generated from optional-mcps/semgrep/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# semgrep

Scan code for security vulnerabilities with Semgrep.

## Overview

**Source:** [https://semgrep.dev/docs/mcp](https://semgrep.dev/docs/mcp)

Install this catalog entry with:

```bash
hermes mcp install semgrep
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall semgrep` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://mcp.semgrep.ai/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
Semgrep (or run `hermes mcp login semgrep`). Approve access,
then restart the session so tools load.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/semgrep/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: semgrep
description: Scan code for security vulnerabilities with Semgrep.
source: https://semgrep.dev/docs/mcp

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://mcp.semgrep.ai/mcp

auth:
  type: oauth

# Excluded: security_check duplicates semgrep_scan; static metadata and
# schema probes. (Vendor archived the standalone repo — live surface may
# drift; excludes no-op harmlessly if names change.)
tools:
  default_excluded:
    - security_check
    - supported_languages
    - semgrep_rule_schema

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - semgrep
    - static analysis
  hosts:
    - semgrep.dev

post_install: |
  On first connection Hermes opens a browser to authorize with
  Semgrep (or run `hermes mcp login semgrep`). Approve access,
  then restart the session so tools load.
```
