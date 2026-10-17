---
id: gitlab
title: "gitlab"
sidebar_label: "gitlab"
description: "GitLab: issues, merge requests, pipelines, and repo context."
---

<!-- This page is auto-generated from optional-mcps/gitlab/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# gitlab

GitLab: issues, merge requests, pipelines, and repo context.

## Overview

**Source:** [https://docs.gitlab.com/user/model_context_protocol/mcp_server/](https://docs.gitlab.com/user/model_context_protocol/mcp_server/)

Install this catalog entry with:

```bash
hermes mcp install gitlab
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall gitlab` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://gitlab.com/api/v4/mcp`

## Auth

**Type:** `oauth`

OAuth is handled at first connection. For native MCP OAuth, Hermes's MCP client triggers the browser flow on the first probe.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

On first connection Hermes opens a browser to authorize with
GitLab (or run `hermes mcp login gitlab`). Approve access,
then restart the session so tools load.

Covers gitlab.com. Self-managed GitLab (18.6+): point
mcp_servers.gitlab.url at https://&lt;your-host>/api/v4/mcp.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/gitlab/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: gitlab
description: 'GitLab: issues, merge requests, pipelines, and repo context.'
source: https://docs.gitlab.com/user/model_context_protocol/mcp_server/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). Native OAuth 2.1 + Dynamic Client Registration
# (verified live: RFC 9728 protected-resource metadata -> AS metadata with
# registration_endpoint); Hermes's MCP client + mcp_oauth_manager handle
# discovery, PKCE, token exchange, and refresh.

transport:
  type: http
  url: https://gitlab.com/api/v4/mcp

auth:
  type: oauth

# Excluded: version probe and Duo-product session listing.
tools:
  default_excluded:
    - get_mcp_server_version
    - list_duo_sessions

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - gitlab
    - merge request
  hosts:
    - gitlab.com

post_install: |
  On first connection Hermes opens a browser to authorize with
  GitLab (or run `hermes mcp login gitlab`). Approve access,
  then restart the session so tools load.

  Covers gitlab.com. Self-managed GitLab (18.6+): point
  mcp_servers.gitlab.url at https://<your-host>/api/v4/mcp.
```
