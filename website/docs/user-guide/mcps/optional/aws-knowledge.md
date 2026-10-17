---
id: aws-knowledge
title: "aws-knowledge"
sidebar_label: "aws-knowledge"
description: "Authoritative AWS docs, API references, and best practices."
---

<!-- This page is auto-generated from optional-mcps/aws-knowledge/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# aws-knowledge

Authoritative AWS docs, API references, and best practices.

## Overview

**Source:** [https://awslabs.github.io/mcp/servers/aws-knowledge-mcp-server/](https://awslabs.github.io/mcp/servers/aws-knowledge-mcp-server/)

Install this catalog entry with:

```bash
hermes mcp install aws-knowledge
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall aws-knowledge` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `https://knowledge-mcp.global.api.aws`

## Auth

**Type:** `none`

No credentials required.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

No account or credentials needed — tools are available as soon as the
session restarts.

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/aws-knowledge/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: aws-knowledge
description: Authoritative AWS docs, API references, and best practices.
source: https://awslabs.github.io/mcp/servers/aws-knowledge-mcp-server/

# Official vendor-hosted remote MCP (URL-only — Hermes never spawns a local
# process for this entry). No authentication required (verified live:
# initialize succeeds anonymously).

transport:
  type: http
  url: https://knowledge-mcp.global.api.aws

auth:
  type: none

# aws___retrieve_skill fetches vendor-authored SKILL.md workflow files — a
# vendor skill/discovery layer. Hermes's own skills and tool_search are the
# only instruction/deferral layers we ship (verified live via tools/list).
tools:
  default_excluded:
    - aws___retrieve_skill

# Composer-suggestion triggers (desktop brand pills).
suggest:
  keywords:
    - aws docs
    - aws knowledge
  hosts:
    - aws.amazon.com
    - docs.aws.amazon.com

post_install: |
  No account or credentials needed — tools are available as soon as the
  session restarts.
```
