---
sidebar_position: 10
title: "Optional MCPs Catalog"
description: "Nous-approved optional MCP servers shipped with hermes-agent — install via hermes mcp install <name>"
---

# Optional MCPs Catalog

Optional MCP servers ship with hermes-agent under `optional-mcps/` but are **not active by default**. They are discovered through `hermes mcp catalog` and activated explicitly with `hermes mcp install <name>`.

Presence in `optional-mcps/` is the trust signal: an entry is in the catalog only because a maintainer merged a PR adding it. There is no community tier and no automatic refresh — the manifest you see is the manifest you get until you re-run `hermes mcp install` after a repo update.

## CLI usage

```bash
hermes mcp                  # interactive picker (TUI) — toggle entries on/off
hermes mcp catalog          # plain-text list of Nous-approved entries (scriptable)
hermes mcp install <name>   # install a catalog entry by name (prompts for env/OAuth)
hermes mcp remove <name>    # remove the server's config block (.env credentials are preserved)
```

`hermes mcp install` writes a `mcp_servers.<name>` block into `~/.hermes/config.yaml` using the manifest's `transport:` keys, runs any `install:` bootstrap (e.g. `git clone` + `pip install`), and prompts for any `auth:` env vars defined by the manifest. Secrets go to `~/.hermes/.env`; non-secret env vars also land in `.env` to keep one credential store.

For the general MCP config shape (independent of the catalog), see the [MCP Config Reference](/reference/mcp-config-reference). For the conceptual overview, see [MCP (Model Context Protocol)](/user-guide/features/mcp).

## Catalog entries

| Name | Transport | Auth | Source | Description |
|------|-----------|------|--------|-------------|
| [**linear**](https://linear.app/docs/mcp) | http (remote) | oauth (native MCP) | [linear.app/docs/mcp](https://linear.app/docs/mcp) | Find, create, and update Linear issues, projects, and comments. |
| [**n8n**](https://github.com/CyberSamuraiX/hermes-n8n-mcp) | stdio | api_key | [CyberSamuraiX/hermes-n8n-mcp](https://github.com/CyberSamuraiX/hermes-n8n-mcp) | Manage and inspect n8n workflows from Hermes (stdio bridge, no public port). |
| [**unreal-engine**](https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor) | http (local) | none | [dev.epicgames.com](https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor) | Drive the Unreal Engine 5.8 editor over its local MCP server. |

## Trust model

The catalog policy is intentionally narrow, and is enforced at the directory level rather than via metadata:

- **Approval is a merged PR.** Entries are added only by merging a PR into hermes-agent. Presence in the `optional-mcps/` directory equals Nous approval. There is no community tier and no trust signals beyond "it's in the catalog".
- **Manifests pin transport details.** Each manifest fixes the command, args, install URL, and git ref. MCPs are never auto-updated — users re-run `hermes mcp install <name>` explicitly to pull a new manifest version after a repo update.
- **Secrets live in `~/.hermes/.env`.** Env vars prompted at install time go to `~/.hermes/.env` (the .env-is-for-secrets rule). Non-secret env vars also go to `.env` so there is one credential store.
- **Default tool surface is conservative.** When an entry specifies `tools.default_enabled`, the install-time checklist pre-prunes mutating or rarely-useful tools — users opt in to the full surface per their threat model.

## How to add a new entry

To propose a new optional MCP:

1. Add a directory under `optional-mcps/<name>/` containing a `manifest.yaml`.
2. Use the existing entries (`optional-mcps/linear/manifest.yaml`, `optional-mcps/n8n/manifest.yaml`) as templates. They cover the two supported transports (HTTP with native MCP OAuth; stdio with git-clone install).
3. Set `manifest_version: 1` — the current schema version constant in `hermes_cli/mcp_catalog.py`. Manifests with a higher version than the running CLI are skipped, so bumping the version is a coordinated change with the catalog loader.
4. Submit a PR. Maintainers review transport, auth, default tool surface, and source provenance. Once merged, the entry appears in `hermes mcp catalog` and in this reference page.

See [Contributing](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md) for the general PR workflow.

## Per-entry details

The manifests below are the source of truth. They are read verbatim at install time by `hermes_cli/mcp_catalog.py`.

### linear

Remote MCP server with native OAuth 2.1 + Dynamic Client Registration over Streamable HTTP. Hermes's MCP client and `mcp_oauth_manager` handle discovery, PKCE, token exchange, and refresh — nothing is installed locally.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
manifest_version: 1

name: linear
description: Find, create, and update Linear issues, projects, and comments.
source: https://linear.app/docs/mcp

# Linear ships a remote MCP server with native OAuth 2.1 + Dynamic Client
# Registration over Streamable HTTP. Hermes's MCP client + mcp_oauth_manager
# handle discovery, PKCE, token exchange, and refresh — nothing to install
# locally.
transport:
  type: http
  url: https://mcp.linear.app/mcp

auth:
  type: oauth
  # No `provider:` — this is native MCP OAuth (case 1), not a third-party
  # provider like Google. The MCP client triggers the browser flow on the
  # first probe / first connect.

# Tool selection at install time:
# Linear's MCP server exposes a moderate-sized tool surface (find/get/list +
# create/update across issues/projects/comments). We leave `default_enabled`
# unset so the install-time checklist starts with everything pre-checked —
# users prune what they don't want.
#
# If you want to encode a curated subset here once it stabilizes, list the
# tool names under `tools.default_enabled`. Probe failure would then apply
# that list directly.

post_install: |
  On first connection, Hermes will open a browser to authenticate with Linear.
  After auth, restart your Hermes session so the Linear tools are loaded.

  You can re-run the tool checklist any time with:
    hermes mcp configure linear
```

### n8n

stdio bridge to a running n8n instance. The catalog clones the bridge repo into a hermes-managed install directory, sets up a venv, and prompts for the n8n base URL and API key. The default tool surface is read-mostly — mutating tools (`activate_workflow`, `deactivate_workflow`, docker `container_logs`) are pruned from the install-time checklist but users can opt in.

```yaml
# Nous-approved MCP catalog entry.
# Presence in this directory = approval. Merged via PR review.
#
# Schema version 1.
manifest_version: 1

name: n8n
description: Manage and inspect n8n workflows from Hermes (stdio bridge, no public port).
source: https://github.com/CyberSamuraiX/hermes-n8n-mcp

# How to launch the server once installed. The keys here map 1:1 to the
# `mcp_servers.<name>` block written into ~/.hermes/config.yaml by the
# existing `_save_mcp_server()` helper in hermes_cli/mcp_config.py.
transport:
  type: stdio
  # For git-installed servers, ${INSTALL_DIR} is substituted at install time
  # with the path the catalog cloned the repo into. The catalog never
  # auto-updates: the user re-runs `hermes mcp install official/n8n` to
  # refresh.
  command: "${INSTALL_DIR}/.venv/bin/python"
  args:
    - "${INSTALL_DIR}/server.py"

# Optional install step. Omit for npm/uvx servers where transport.command
# is the install (`npx -y package`). Use for repos that need a local clone
# + dependency install.
install:
  type: git
  url: https://github.com/CyberSamuraiX/hermes-n8n-mcp.git
  # Pin to a commit/tag. Required — manifests do not float HEAD.
  ref: main
  # Bootstrap commands run inside the cloned directory after clone.
  bootstrap:
    - "python3 -m venv .venv"
    - ".venv/bin/pip install -r requirements.txt"

# Authentication. Three shapes:
#   type: api_key  — prompt for env vars, write to ~/.hermes/.env
#   type: oauth    — provider-mediated or remote MCP native OAuth (case 1/2)
#   type: none     — no credentials needed
auth:
  type: api_key
  env:
    - name: N8N_BASE_URL
      prompt: "n8n instance URL"
      default: "http://127.0.0.1:5678"
      required: true
      secret: false
    - name: N8N_API_KEY
      prompt: "n8n API key (generate under Settings → API)"
      required: true
      secret: true

# Tool selection at install time:
# n8n's bridge exposes 11 tools. Mutating ones (activate/deactivate, docker
# container_logs) are pruned from the default so a user who installs casually
# gets a read-mostly safe surface. Users see the full list in the install-time
# checklist and can opt into the mutating tools per their threat model.
tools:
  default_enabled:
    - health
    - list_workflows
    - get_workflow
    - find_workflows
    - list_executions
    - get_execution
    - recent_failures
    - export_workflow

post_install: |
  The n8n bridge expects to talk to a running n8n instance over the URL you
  provided. Generate an API key in n8n under Settings → API.

  Workflow activate/deactivate calls are real mutations against your live n8n.
  Treat them carefully.

  Start a new Hermes session to load the n8n tools.
```

### unreal-engine

Local HTTP connection to Epic's official "Unreal MCP" plugin, which embeds an MCP server inside the running Unreal Editor 5.8+ process and binds to `127.0.0.1`. There is nothing to install on the Hermes side — the user enables the plugin in-editor and Hermes's MCP client connects to the URL. The server has no authentication of its own (Epic's experimental design accepts same-machine connections only — not for remote use), so start Hermes *after* the editor's server is running so the live tool surface is probed.

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

## See also

- [MCP feature page](/user-guide/features/mcp) — conceptual overview, config patterns, gotchas
- [MCP Config Reference](/reference/mcp-config-reference) — compact reference for `mcp_servers.*` keys
- [Optional Skills Catalog](/reference/optional-skills-catalog) — the parallel catalog for skills
- [Use MCP with Hermes](/guides/use-mcp-with-hermes) — tutorial-style walkthrough
