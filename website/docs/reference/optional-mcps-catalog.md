---
sidebar_position: 10
title: "Optional MCPs Catalog"
description: "Nous-approved optional MCP servers shipped with hermes-agent — install via hermes mcp install <name>"
---

{/* This page is auto-generated from optional-mcps/*/manifest.yaml by website/scripts/generate-mcp-catalog-docs.py. Edit the manifests, not this page. */}

# Optional MCPs Catalog

Optional MCP servers ship with hermes-agent under `optional-mcps/` but are **not active by default**. They are discovered through `hermes mcp catalog` and activated explicitly with `hermes mcp install <name>`.

Presence in `optional-mcps/` is the trust signal: an entry is in the catalog only because a maintainer merged a PR adding it. There is no community tier and no automatic refresh — the manifest you see is the manifest you get until you re-run `hermes mcp install` after a repo update.

## CLI usage

```bash
hermes mcp                  # interactive picker (TUI) — toggle entries on/off
hermes mcp catalog          # plain-text list of Nous-approved entries (scriptable)
hermes mcp install <name>   # install a catalog entry by name (prompts for env/OAuth)
hermes mcp uninstall <name> # remove the server's config block (.env credentials are preserved)
```

`hermes mcp install` writes a `mcp_servers.<name>` block into `~/.hermes/config.yaml` using the manifest's `transport:` keys, runs any `install:` bootstrap (e.g. `git clone` + `pip install`), and prompts for any `auth:` env vars defined by the manifest. Secrets go to `~/.hermes/.env`; non-secret env vars also land in `.env` to keep one credential store.

For the general MCP config shape (independent of the catalog), see the [MCP Config Reference](/reference/mcp-config-reference). For the conceptual overview, see [MCP (Model Context Protocol)](/user-guide/features/mcp).

## Catalog entries

| Name | Transport | Auth | Source | Description |
|------|-----------|------|--------|-------------|
| [**linear**](/docs/user-guide/mcps/optional/linear) | http | oauth | [https://linear.app/docs/mcp](https://linear.app/docs/mcp) | Find, create, and update Linear issues, projects, and comments. |
| [**n8n**](/docs/user-guide/mcps/optional/n8n) | stdio | api_key | [https://github.com/CyberSamuraiX/hermes-n8n-mcp](https://github.com/CyberSamuraiX/hermes-n8n-mcp) | Manage and inspect n8n workflows from Hermes (stdio bridge, no public port). |

## Trust model

The catalog policy is intentionally narrow, and is enforced at the directory level rather than via metadata:

- **Approval is a merged PR.** Entries are added only by merging a PR into hermes-agent. Presence in the `optional-mcps/` directory equals Nous approval. There is no community tier and no trust signals beyond "it's in the catalog".
- **Manifests pin transport details.** Each manifest fixes the command, args, install URL, and git ref. MCPs are never auto-updated — users re-run `hermes mcp install <name>` explicitly to pull a new manifest version after a repo update.
- **Secrets live in `~/.hermes/.env`.** Env vars prompted at install time go to `~/.hermes/.env` (the .env-is-for-secrets rule). Non-secret env vars also go to `.env` so there is one credential store.
- **Default tool surface is conservative.** When an entry specifies `tools.default_enabled`, the install-time checklist pre-prunes mutating or rarely-useful tools — users opt in to the full surface per their threat model.

## How to contribute an entry

To propose a new optional MCP:

1. Add a directory under `optional-mcps/<name>/` containing a `manifest.yaml`.
2. Use the existing entries (`optional-mcps/linear/manifest.yaml`, `optional-mcps/n8n/manifest.yaml`) as templates. They cover the two supported transports (HTTP with native MCP OAuth; stdio with git-clone install).
3. Set `manifest_version: 1` — the current schema version constant in `hermes_cli/mcp_catalog.py`. Manifests with a higher version than the running CLI are skipped, so bumping the version is a coordinated change with the catalog loader.
4. Submit a PR. Maintainers review transport, auth, default tool surface, and source provenance. Once merged, the entry appears in `hermes mcp catalog` and gets its own page in this catalog.

See [Contributing](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md) for the general PR workflow.

## See also

- [MCP Config Reference](/reference/mcp-config-reference)
- [MCP (Model Context Protocol)](/user-guide/features/mcp)
