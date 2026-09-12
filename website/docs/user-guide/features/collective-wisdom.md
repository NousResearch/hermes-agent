# Collective Wisdom

Collective Wisdom lets teammates in the same Nous organization share instruction-only skills with each other: browse what the team has published, install an exact version, pick up updates, and share your own skills after a review. It ships as the bundled `wisdom` plugin and talks to the Nous Gateway with your existing `hermes login` session.

## Requirements

- A Nous login (`hermes login`) whose team has Collective Wisdom enabled. The token carries `wisdom:*` scopes; without them the plugin's tools do not appear in the model's toolset and every command says so.
- Nothing else to configure. The Gateway URL is the shared `sync.base_url` (defaults to production).

## Using it

All three surfaces run the same actions:

| Surface | Example |
|---|---|
| Terminal | `hermes wisdom list`, `hermes wisdom install <skill-id>`, `hermes wisdom share my-skill --description "..."` |
| In a chat | `/wisdom status`, `/wisdom show <skill-id>`, `/wisdom update` |
| The agent | tools `wisdom_browse`, `wisdom_install`, `wisdom_share` (visible only when entitled) |

Commands: `list` (team catalog), `show <id>` (versions and Gateway checks), `status` (installed skills and pending updates), `install <id> [--version N]`, `update [id]`, `uninstall <id>`, `share <skill> --description "..."`.

Installed skills land under `~/.hermes/skills/_wisdom/<org>/<slug>/` and are indexed like any other skill. The plugin keeps a ledger of the exact version and content hash it installed, so `status` can tell you when the team has published something newer.

## Consent

Every action that changes your machine or your team's catalog asks first, and a "yes" in conversation is never enough:

- **Terminal**: a prompt showing the skill, exact version, content hash and the Gateway's security verdict.
- **Agent tools**: the same human-approval gate used for dangerous shell commands. In the CLI you get the usual once / session / always / deny prompt; on a messaging platform it becomes an approval button; with nobody present (cron, `-q`, subagents) the action is blocked.

Sharing asks twice: once to approve the exact package (file list, byte counts, content hash, your description) before it is uploaded as an owner-private draft, and once more after the Gateway has run its security and professionalism checks, before publication. Declining the second prompt withdraws the draft. Depending on your organization's policy the result is published immediately or held for an admin's review.

## What can be shared

Wisdom packages are instruction-only: a root `SKILL.md`, an optional `skill.manifest.json`, and plain-text files under `refs/` or `assets/` (`.md`, `.txt`, `.rst`, `.adoc`). Scripts, templates, executables, binaries, symlinks and package-manager manifests are refused, as is a `SKILL.md` that points at `scripts/` or `templates/`. If the manifest is missing, one is generated from the skill's frontmatter and the authoring OS/architecture. Everything downloaded is verified against the hashes the Gateway publishes before it is written to disk.
