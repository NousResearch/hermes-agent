---
id: unreal-engine
title: "unreal-engine"
sidebar_label: "unreal-engine"
description: "Drive the Unreal Engine 5.8 editor over its local MCP server."
---

<!-- This page is auto-generated from optional-mcps/unreal-engine/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifest, not this page. -->

# unreal-engine

Drive the Unreal Engine 5.8 editor over its local MCP server.

## Overview

**Source:** [https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor](https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor)

Install this catalog entry with:

```bash
hermes mcp install unreal-engine
```

or pick it interactively with `hermes mcp`. Uninstall with `hermes mcp uninstall unreal-engine` (the server's config block is removed; any credentials in `~/.hermes/.env` are preserved).

## Transport

**Type:** `http`

**URL:** `http://127.0.0.1:8000/mcp`

## Auth

**Type:** `none`

No credentials required.

## Tools

No default tool filter is declared. The install-time checklist starts with every probed tool pre-checked — users prune what they don't want.

## Post-install notes

This entry connects to Epic's official Unreal MCP plugin, which runs INSIDE
the Unreal Editor. Before Hermes can connect:

  1. Open your project in Unreal Editor 5.8+.
  2. Edit > Plugins, search "Unreal MCP", enable it, restart the editor
     (the Toolset Registry dependency enables automatically).
  3. Edit > Editor Preferences > General > Model Context Protocol, turn on
     "Auto Start Server" (or run `ModelContextProtocol.StartServer` in the
     editor console). It binds to http://127.0.0.1:8000/mcp by default.

Start Hermes AFTER the editor's server is running so the tools are probed.
If you changed the port or URL path in Editor Preferences, update the url in
mcp_servers.unreal-engine to match.

Status: Epic ships this as EXPERIMENTAL. The server runs Tool calls serially
on the engine game thread — avoid issuing overlapping calls.

Re-run the tool checklist any time with:
  hermes mcp configure unreal-engine

## Manifest

The manifest below is the source of truth. It lives at `optional-mcps/unreal-engine/manifest.yaml` in the hermes-agent repo.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: unreal-engine
description: Drive the Unreal Engine 5.8 editor over its local MCP server.
source: https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor

# Epic's official "Unreal MCP" plugin (internal id ModelContextProtocol)
# embeds an MCP server inside the running Unreal Editor process and serves it
# over local HTTP. There is nothing to install on the Hermes side — the user
# enables the plugin in-editor and the server binds to 127.0.0.1. Hermes's
# MCP client just connects to the URL.
#
# Default bind is http://127.0.0.1:8000/mcp (port + path are configurable in
# Editor Preferences > General > Model Context Protocol). If you change the
# port/path in-editor, edit the url in mcp_servers.unreal-engine afterward.
transport:
  type: http
  url: http://127.0.0.1:8000/mcp

# The editor-embedded server accepts connections only from the same machine
# and has no authentication of its own (Epic's experimental design — not for
# remote use). Nothing to prompt for.
auth:
  type: none

# Tool selection at install time:
# The plugin advertises engine tools (spawn actors, configure lighting, create
# material instances, inspect Slate widgets, run automation tests) and is
# user-extensible, so the exact surface depends on the project's enabled
# toolsets. Leave default_enabled unset — the install-time probe lists whatever
# the live editor exposes and pre-checks all of it; users prune from there.

post_install: |
  This entry connects to Epic's official Unreal MCP plugin, which runs INSIDE
  the Unreal Editor. Before Hermes can connect:

    1. Open your project in Unreal Editor 5.8+.
    2. Edit > Plugins, search "Unreal MCP", enable it, restart the editor
       (the Toolset Registry dependency enables automatically).
    3. Edit > Editor Preferences > General > Model Context Protocol, turn on
       "Auto Start Server" (or run `ModelContextProtocol.StartServer` in the
       editor console). It binds to http://127.0.0.1:8000/mcp by default.

  Start Hermes AFTER the editor's server is running so the tools are probed.
  If you changed the port or URL path in Editor Preferences, update the url in
  mcp_servers.unreal-engine to match.

  Status: Epic ships this as EXPERIMENTAL. The server runs Tool calls serially
  on the engine game thread — avoid issuing overlapping calls.

  Re-run the tool checklist any time with:
    hermes mcp configure unreal-engine
```
