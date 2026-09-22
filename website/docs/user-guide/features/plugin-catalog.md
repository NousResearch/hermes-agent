---
sidebar_position: 13
sidebar_label: "Plugin Catalog"
title: "Plugin Catalog"
description: "Give Hermes new powers with reviewed plugins you can install in one click"
---

# Plugin Catalog

The plugin catalog is a curated, human-reviewed directory of Hermes plugins you
can install by name with a single command:

```bash
hermes plugins install <name>
```

Browse it visually at **[/docs/plugins](/plugins)** — entries are shelved by
category (Memory, Desktop, Platforms, Web & Browser, Tools, Voice, Automation,
Models), with search, tier filters (Official / Community), capability chips, and
copyable install commands for every entry.

Every entry also has its own page at `/docs/plugins/<name>` (click a card):
the full description and any disclosure, the pinned commit, tools, hooks and
environment variables, the Desktop install button and CLI command, optional
screenshots and the README from the reviewed commit, plus a **More by this
author** shelf. Authors have a page at `/docs/plugins/by/<maintainer>` listing
everything they maintain in the catalog. Both are generated at build time from
the same catalog files, so a merged PR is the only way a page changes.

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
| `category` | Browse shelf: `desktop` (default), `memory`, `platform`, `web`, `tools`, `voice`, `automation`, `models` or `general` |
| `maintainer` | Who owns the plugin |
| `capabilities` | Declared tools, hooks, middleware, and required env vars |
| `requires_hermes` | Minimum Hermes version, e.g. `>=0.19` (optional) |
| `platforms` | OS restrictions, empty = all (optional) |
| `docs_url` | External documentation link (optional) |
| `version` | Human-readable label for the pinned sha, e.g. `"1.4.0"`; shown as `1.4.0 @ abcd1234` in the CLI, on the catalog card and on the Desktop **Update to** button (optional, cosmetic) |
| `image` | Banner image for the catalog card and the plugin page hero, shown at 2:1 (1200×600 works; other shapes are centre-cropped); an `https` URL on `raw.githubusercontent.com`, `github.com` or `*.githubusercontent.com` (optional). Pin it to the entry's commit (`raw.githubusercontent.com/owner/repo/<sha>/...`) so it never changes under the review |
| `screenshots` | Up to 6 images shown as a gallery on the plugin page, same host rule as `image` (optional). Pin them to the entry's commit too |
| `readme` | The plugin page renders the repository README (the entry's `subdir` first, else the repo root) by default. It is fetched **from the pinned commit** at docs build time — never from a branch — so the page shows the README the reviewer read and changes only when the pin does. Set `false` to hide it. GitHub and GitLab repos (optional, default `true`) |

## Trust model

The catalog is designed so you know exactly what you're installing:

- **Human-merged admission.** Every entry (and every pin update) lands via a
  pull request reviewed by a maintainer. Nothing enters the catalog
  automatically.
- **Exact SHA pins.** Entries pin a specific commit, not a branch. A plugin
  author pushing new code to their repo does **not** change what the catalog
  installs — updating the pin requires another reviewed PR.
- **Scanned at admission, trusted at install.** Admission CI runs the same
  security scanner the installer runs (`hermes plugins validate` includes a
  `security scan` check): a `dangerous` verdict fails the entry, `caution`
  findings are listed for the reviewer. Because the reviewer saw them, a
  catalog install checked out at exactly the pinned SHA does not stop to ask
  about `caution` again; `dangerous` still blocks, and anything installed from
  a raw URL or at another revision gets the normal prompt.
- **Desktop plugins run with the app's authority — review is the boundary.**
  A plugin's `desktop/plugin.js` is evaluated inside the Desktop app itself,
  in the same realm as the app's own code: there is no sandbox, and it can
  do anything the app can (gateway RPC, the full `window.hermesDesktop`
  bridge, storage of other plugins). What protects you is the trust model
  above — a human read the exact pinned commit, and the install is that
  commit — plus two tripwires: admission's `desktop surface` lint refuses
  the obvious moves outside the plugin SDK (patching built-in prototypes,
  `eval`, importing anything other than `@hermes/plugin-sdk`/`react`,
  including remote scripts), and the app's loader refuses every non-SDK
  import again at load time. Treat the lint as a review aid, not a
  guarantee; give Desktop halves the same scrutiny you'd give a Python half.
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

The catalog name and the plugin's own manifest name can differ; `hermes
plugins install` prints the installed name, and `enable` takes that one. For
example the `touchdesigner` entry (a portable Agent Plugins v1 package that
bundles the twozero MCP server with the `touchdesigner-mcp` skill) installs as
`td`, kept short so its generated MCP tool names stay under provider
function-name limits:

```bash
hermes plugins install touchdesigner
hermes plugins enable td
```

Portable packages can also carry a stdio MCP server. The `snyk` entry pins the
Snyk CLI (`npx -y snyk@<version> mcp`) and bundles the `snyk-security-scan`
skill, so one install gives Hermes code, dependency, container and IaC scanning
plus the workflow for using it; the catalog name and manifest name match:

```bash
hermes plugins install snyk
hermes plugins enable snyk
```

### Updating a catalog install

`hermes plugins update <name>` never runs `git pull` for catalog installs —
it compares your installed pin against the current catalog pin and, when the
catalog moved (via a reviewed PR), force-reinstalls at the new SHA. Your
enabled/disabled state is preserved. `hermes plugins list` shows catalog
installs as `catalog:<tier>@<sha>` so you can see provenance at a glance.

### Names not in the catalog

A bare name that isn't a catalog entry is an error: there is no second,
unreviewed name index. Install such plugins by `owner/repo` or Git URL instead
(custom source, see below), or submit them to the catalog.

### Live refresh

The docs build publishes the catalog as one JSON document
(`https://hermes-agent.nousresearch.com/docs/api/plugin-catalog.json`).
`search`/`install`/`update` fetch it at most every six hours and cache it under
`~/.hermes/cache/`, so new entries and removals reach installed clients without
updating Hermes. Offline, the copy shipped with your checkout is used (a failed
fetch is remembered for a minute, so `plugins list` and the dashboard's Plugins
page pay at most one connection timeout, not one per installed plugin). Removals
from the in-tree list and the live list are always both enforced.

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
   Maintainers also add batches of community plugins from a reviewed sweep
   (each pin validated and scanned at the pinned commit); if yours was swept
   in and you want it changed or removed, open a PR on your entry.
2. **A public repository** — the `repo` URL is publicly cloneable.
3. **Released** — the repo has real releases/tags, not just a default branch.
4. **Passing validation** — the catalog validation GitHub Action is green on
   the PR (schema, SHA format, reachability).
5. **Not self-updating** — the catalog build must not download and replace
   its own files; the pinned SHA is the only update path (a SHA-bump PR plus
   `hermes plugins update <name>`).

Pin updates (bumping `sha` to a newer commit) follow the same PR + review
process; bump `version` in the same PR so the label users see matches the
code, and re-pin any `image` / `screenshots` URLs that embed the sha. Your
plugin page (`/docs/plugins/<name>`) is built from the same file: add
`screenshots:` there to fill it out (the README renders by default) — there is no separate
listing to maintain. Installed plugins compare their recorded sha against the live pin:
`hermes plugins list --json` reports `update_available`, the Desktop Plugins
tab shows an **Update to 1.4.0** button, and `hermes plugins update <name>`
checks out exactly the new pin.

## See also

- [Plugins](plugins.md) — the plugin system itself: manifest format, enabling,
  configuration
- [Built-in Plugins](built-in-plugins.md) — plugins that ship with Hermes
- [Build a Hermes Plugin](../../developer-guide/plugins/index.md) — write your own
- [Plugin Catalog page](/plugins) — the browsable catalog
