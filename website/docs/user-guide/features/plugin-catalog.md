---
sidebar_position: 13
sidebar_label: "Plugin Catalog"
title: "Plugin Catalog"
description: "Browse and install reviewed, SHA-pinned Hermes plugins from the curated catalog"
---

# Plugin Catalog

The plugin catalog is a curated, human-reviewed directory of Hermes plugins you
can install by name with a single command:

```bash
hermes plugins install <name>
```

Browse it visually at **[/docs/plugins](/plugins)** — search, tier filters
(Official / Community), capability chips, and copyable install commands for
every entry.

The catalog complements — it does not replace — the existing
[plugin system](plugins.md). Anything you can install from the catalog is a
normal plugin under the hood; the catalog just adds discovery and a review
layer on top.

## What's in an entry

Each catalog entry is a small YAML file in the
[`plugin-catalog/`](https://github.com/NousResearch/hermes-agent/tree/main/plugin-catalog)
directory of the hermes-agent repository, declaring:

| Field | Meaning |
|---|---|
| `name` | The catalog key you pass to `hermes plugins install` |
| `repo` | The plugin's public git repository |
| `sha` | The **exact 40-hex commit** that was reviewed — installs check out this pin, not a branch tip |
| `tier` | `official` (maintained by NousResearch) or `community` |
| `maintainer` | Who owns the plugin |
| `capabilities` | Declared tools, hooks, middleware, and required env vars |
| `requires_hermes` | Minimum Hermes version, e.g. `>=0.19` (optional) |
| `platforms` | OS restrictions, empty = all (optional) |
| `docs_url` | External documentation link (optional) |

## Trust model

The catalog is designed so you know exactly what you're installing:

- **Human-merged admission.** Every entry (and every pin update) lands via a
  pull request reviewed by a maintainer. Nothing enters the catalog
  automatically.
- **Exact SHA pins.** Entries pin a specific commit, not a branch. A plugin
  author pushing new code to their repo does **not** change what the catalog
  installs — updating the pin requires another reviewed PR.
- **Capability declarations.** Entries state up front which tools, hooks, and
  middleware the plugin provides and which environment variables (API keys
  etc.) it needs, so you can judge its blast radius before installing.
- **Removed list.** Plugins pulled from the catalog (for example after a
  security incident) go on `plugin-catalog/removed.yaml` with a reason and
  date. The installer refuses to install anything on the removed list.
- **Installed ≠ enabled.** Installing a catalog plugin puts it on disk; like
  any plugin it must still be enabled before it loads. See
  [Plugins → Enabling and disabling](plugins.md).

:::warning Catalog review is a point-in-time review
A catalog entry means the pinned commit was looked at by a human, capability
declarations were checked, and the repo met the submission bar. It is not a
security audit, and it says nothing about other commits in the same
repository. Review the code of anything you give credentials to.
:::

## Installing from the catalog

```bash
# Install a reviewed catalog entry by name (checks out the pinned SHA)
hermes plugins install <name>

# Then enable it, as with any plugin
hermes plugins enable <name>
```

The install prompt shows the entry's capability summary — declared tools,
hooks, and required env vars — before anything is cloned.

### Custom git URLs are different

`hermes plugins install <git-url>` still works for any repository, but it
bypasses the catalog entirely:

- **No review** — you get whatever is at the branch tip, not a reviewed pin.
- **A warning banner** is shown to make clear the code is unvetted.
- The removed list is still consulted (a known-bad repo is refused by URL).

Use the git-URL path for your own plugins and repos you already trust; use the
catalog for discovery.

## Submitting a plugin to the catalog

Submissions are pull requests that add one `plugin-catalog/<name>.yaml` file.
The full checklist lives in the
[plugin-catalog README](https://github.com/NousResearch/hermes-agent/tree/main/plugin-catalog);
in short, an entry must be:

1. **Owner-submitted** — the PR author owns or maintains the plugin repo.
2. **A public repository** — the `repo` URL is publicly cloneable.
3. **Released** — the repo has real releases/tags, not just a default branch.
4. **Passing validation** — the catalog validation GitHub Action is green on
   the PR (schema, SHA format, reachability).
5. **Pinned to settled code** — the pinned SHA is at least **2 weeks old**, so
   the catalog never points at code pushed moments before review.

Pin updates (bumping `sha` to a newer commit) follow the same PR + review
process.

## See also

- [Plugins](plugins.md) — the plugin system itself: manifest format, enabling,
  configuration
- [Built-in Plugins](built-in-plugins.md) — plugins that ship with Hermes
- [Build a Hermes Plugin](/developer-guide/plugins) — write your own
- [Plugin Catalog page](/plugins) — the browsable catalog
