---
sidebar_label: "Catalog submission"
title: "Submitting to the Plugin Catalog"
description: "The full admission guidelines for the Hermes plugin catalog: what to check before you submit, the rules every entry follows, and what reviewers look at"
---

# Submitting to the Plugin Catalog

The [plugin catalog](../../user-guide/features/plugin-catalog.md) is a human-reviewed
directory of Hermes plugins. Being listed is the trust signal: users install a
catalog plugin by name, at the exact commit a maintainer read. This page holds
the complete guidelines for getting a plugin in and keeping it there.

The canonical copy of the admission rules is
[`plugin-catalog/README.md`](https://github.com/NousResearch/hermes-agent/blob/main/plugin-catalog/README.md)
in the repository. The [rules section](#admission-rules) below mirrors it word
for word, and a test fails the build if the two drift apart.

## Before you submit

- **A public repository.** The `repo` URL is an `https://` URL anyone can clone
  (GitHub or GitLab).
- **A loadable plugin at the commit you pin.** The tree has a `plugin.yaml`
  manifest plus at least one entrypoint Hermes loads: `__init__.py` (Python),
  `desktop/plugin.js` (Desktop), `plugin.json` (portable Agent Plugin) or
  `dashboard/manifest.json` (web dashboard). If the plugin lives in a monorepo,
  point `subdir` at its directory. The
  [plugin developer guide](./index.md) covers the layout.
- **Public surfaces only.** Extend Hermes through hooks, middleware, the
  `ctx.register_*` APIs, provider plugins and the
  [Desktop plugin SDK](../desktop-plugin-sdk.md). Never patch Hermes
  code or Desktop markup at runtime. If the hook you need is missing, see
  [Asking for a hook](#asking-for-a-hook).
- **Validation passes locally.** Run the same check catalog CI runs, against a
  checkout at the commit you are about to pin:

  ```bash
  hermes plugins validate /path/to/your-plugin --install-deps
  ```

  It checks the manifest and `requires_hermes`, that the plugin loads, that the
  `capabilities` you declare match what it registers, `config_schema` and
  `requires_env`, Python dependencies against Hermes's core constraints, the
  install security scan, and the `desktop surface` and `no core override`
  rules. Fix every failure before opening the PR, and read the warnings, since
  a reviewer will.

## Opening the PR

1. Add **one** file, `plugin-catalog/<name>.yaml`, to
   [`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent).
   The fields are documented in the README's
   [entry schema](https://github.com/NousResearch/hermes-agent/blob/main/plugin-catalog/README.md#entry-schema)
   and in [What's in an entry](../../user-guide/features/plugin-catalog.md#whats-in-an-entry).
   Pin `sha` to a full 40-character commit, and quote `version`.
2. In the PR description, say what the plugin does, which Hermes surfaces it
   uses, and everything rule 13 asks you to disclose. Add screenshots for
   anything with a UI.
3. Wait for the catalog CI job to go green. It clones your repo at the pinned
   commit and runs `hermes plugins validate`.
4. A maintainer reviews the pinned tree and merges, asks for changes, or
   declines. Rule 3 and rule 9 problems are declines rather than bug-fix
   requests: the plugin has to change its design before it can be listed.

Your plugin's page at `/docs/plugins/<name>` is built from the same file. The
README at the pinned commit renders there by default, and `screenshots:` fills
the gallery. There is no separate listing to maintain.

## Admission rules

<!-- admission-rules:start (mirrored in website/docs/developer-guide/plugins/catalog-submission.md; tests/plugin_catalog keeps them identical) -->
1. **Human-merged gate.** Entries are added *only* via a PR to the
   `hermes-agent` repository, reviewed and merged by a maintainer. There is
   no self-serve registry, no automated ingestion.
2. **Exact SHA pins are mandatory.** Every entry pins a full 40-character
   commit SHA. Branches, tags, and short SHAs are rejected by the loader.
   Installs clone the repository and check out exactly that commit.
3. **No self-updating code.** A listed plugin must not fetch and replace
   its own files (in-app "check for updates", signed release downloaders,
   remote `plugin.js` loaders). The exact SHA pin *is* the trust model; a
   self-updater lets an installed copy move to a commit nobody reviewed.
   Updates reach users only through a SHA-bump PR here plus
   `hermes plugins update <name>`. Keep the updater in the standalone
   distribution if you want one; strip it from the catalog build.
4. **SHA bumps are new PRs.** Updating an entry's pin is a new PR whose diff
   (old SHA → new SHA) is re-reviewed like any other change — reviewers are
   expected to look at the upstream commit range being adopted.
5. **Owner-or-major-contributor submissions, or a maintainer-curated sweep.**
   An entry may be submitted by the plugin repository's owner or a major
   contributor to it; drive-by submissions of third-party repos are declined.
   Hermes maintainers may also add entries in batches from a reviewed sweep
   of community plugins (every pin validated and scanned at the pinned
   commit, self-updater and credential-store checks run, English-first UI).
   Authors of swept-in entries keep control: a PR from the owner adjusting
   or removing their entry is accepted on request, and SHA bumps stay
   owner-or-maintainer PRs under rule 4.
6. **Declared capabilities must match reality.** The `capabilities:` block
   (tools, hooks, middleware, env vars) must match what the plugin actually
   registers at the pinned commit. Validation fails the entry otherwise —
   undeclared capability creep is treated as a security issue.
7. **The install scanner runs at admission.** `hermes plugins validate` includes
   the `security scan` check: `dangerous` fails the entry; `caution` findings
   appear as warnings in the CI log and the reviewer reads them before merging.
   In exchange, installs at the pinned SHA accept `caution` without a prompt
   (`dangerous` still blocks). Review the warnings; do not merge past them.
8. **Desktop plugins stay inside the SDK surface.** A `desktop/plugin.js` runs
   in the Desktop renderer with the app's full authority (the loader isolates
   errors, not capabilities), so a listed one may only use the plugin SDK:
   no prototype patching (`X.prototype.y =`, `Object.defineProperty(...prototype`),
   no `eval`/`new Function`, no `import()` of anything but `@hermes/plugin-sdk`
   / `react` (app bundle chunks, blob or http URLs included), no script-tag
   injection, no reaching into the app's internal stores or its own markup
   (querying `data-slot` / `data-tour` / `data-sidebar` / `data-testid`
   elements from `document`, or a `document.body` MutationObserver, to restyle,
   hide, click or rewrite core UI). `hermes plugins validate` refuses these at
   admission (`desktop surface` check); a plugin that needs a capability the
   SDK lacks asks for an SDK hook instead of patching around it.
9. **No runtime overrides of Hermes core.** A listed plugin extends Hermes only
   through public surfaces: hooks, middleware, provider profiles and the
   other `register_*` APIs, and Desktop SDK slots and routes. It must not
   replace, wrap or rebind core functions, methods, module attributes or
   private dicts in place (`AIAgent.<method> = ...`, `setattr(server, ...)`,
   `sys.modules[...]`, writes into a core module's tables). Two plugins
   patching the same seam silently break each other, and every core release
   can break both. `hermes plugins validate` refuses these at admission (`no
   core override` check). If the hook you need does not exist, open an issue
   describing it: we would rather add the seam than list a patch.
10. **Dependency security policy is the plugin's.** Hermes's 14-day
   `exclude-newer` quarantine covers Hermes's own dependencies only; a plugin's
   `python_dependencies` / `pyproject.toml` install under the plugin's policy
   (no quarantine, still inside Hermes's core constraints). Reviewers read the
   dependency list at the pinned SHA: bare floors (`>=X` with no upper bound)
   and floors on the newest release get a request for the oldest
   API-compatible floor plus an upper bound, and authors are strongly
   recommended to run their own release quarantine (`uv --exclude-newer` in
   their CI) — see the developer guide's *Dependency security policy*. A
   recent floor alone is not grounds to hold an entry.

11. **Credentials stay with their owner.** A plugin reads the credentials it is
   configured with: the env vars in `requires_env` and its own `config_schema`
   secrets. Reading another tool's login (a vendor CLI's token file, a browser
   profile) must be disclosed in the PR and is a trust-tier call for a
   maintainer. Refreshing, rotating or writing another client's OAuth tokens, or
   presenting itself as another vendor's client, is not admitted without an
   explicit maintainer ruling; a read-only build is the usual way through.
12. **Approvals and unattended runs are respected.** A plugin never routes around
   Hermes's approval system: no auto-approving, no disabling guards, and no
   spawning Hermes or shell children that inherit YOLO or non-interactive mode
   to run commands nobody approved. Anything that waits for a person (a prompt,
   an OAuth browser flow) fails cleanly or times out under cron, the messaging
   gateway and other unattended runs instead of hanging the agent.
13. **Risky behaviour is disclosed.** What a user would want to know before
   installing goes in the PR description and the plugin's README: network calls
   to third-party services, reads outside the plugin's own data, shell commands,
   long-running background processes, stored credentials. Telemetry and usage
   reporting are opt-in. Reviewers summarise these as disclosure lines on the
   entry PR; undisclosed behaviour found in review is a request for changes.
14. **Compatibility metadata is truthful.** `requires_hermes` is a SemVer floor
   (`">=0.21.5"`), never a CalVer date, and never newer than the current release
   (the loader skips the plugin otherwise). `version` matches the pinned code,
   and Python dependencies resolve under Hermes's core constraints
   (`hermes plugins validate --install-deps` is what CI runs).
15. **No skins or forks of bundled plugins.** A change to a bundled plugin is a
   PR against `hermes-agent`, not a competing listing, and vendor-lookalike skins
   are not listed under Nous branding.
<!-- admission-rules:end -->

## Updating your entry

Pin updates (bumping `sha` to a newer commit) go through the same PR and review
process. A reviewer reads the commit range you are adopting. Bump `version` in
the same PR so the label users see matches the code, and re-pin any `image` /
`screenshots` URLs that embed the old sha.

Installed plugins compare their recorded sha against the live pin:
`hermes plugins list --json` reports `update_available`, the Desktop Plugins tab
shows an **Update to 1.4.0** button, and `hermes plugins update <name>` checks
out exactly the new pin.

If a maintainer swept your plugin into the catalog and you want the entry
changed or removed, open a PR on its file. Owners keep control of their entries.

## Delisting and removal

- **Delisting** deletes an entry's file: for plugins that are unmaintained,
  superseded, squatting a name, or no longer meet a rule. Users who already
  installed the plugin keep it, and it is welcome back once the problem is
  fixed. When a new rule affects plugins that are already listed, their
  authors get an issue on their repository explaining the change and time to
  update before anything is removed.
- **Removal** adds the plugin to
  [`removed.yaml`](https://github.com/NousResearch/hermes-agent/blob/main/plugin-catalog/README.md#removedyaml--the-blocklist),
  which is reserved for security and policy incidents. Installers refuse
  anything on that list, and installed copies stop updating and cannot be
  enabled.

## Asking for a hook

If your plugin needs something the plugin surface doesn't offer, open an issue
on [hermes-agent](https://github.com/NousResearch/hermes-agent/issues)
describing what you need and why. We would much rather add a proper seam than
list a patch, and a plugin that will use the hook is exactly the concrete
consumer a new hook needs.
