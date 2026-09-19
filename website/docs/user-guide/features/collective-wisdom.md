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
| In a chat | `/wisdom status`, `/wisdom show <skill-id>`, `/wisdom update`; on Telegram and Slack the replies are cards with buttons (see below) |
| The agent | tools `wisdom_browse`, `wisdom_install`, `wisdom_share` (visible only when entitled) |
| Desktop | **Team Skills** in the sidebar: the catalog, installed versions, updates needing a decision (with the team's policy and conflicts), skills worth sharing, Install/Update/Remove/Share; a status-bar count of waiting items |

Commands: `list` (team catalog), `show <id>` (versions and Gateway checks), `status` (installed skills, pending updates, notices), `updates` (pending updates with their policy verdict), `install <id> [--version N]`, `update [id] [--keep]`, `uninstall <id>`, `share <skill> --description "..."`, `candidates` (local skills worth sharing), `not-now <skill>`, `mute [hours]` (silence team notices; `0` unmutes).

## Telegram and Slack

On Telegram and Slack, `/wisdom` answers with cards: `list` offers an Install button per skill, `status`/`updates` shows Update / Replace / Keep-mine buttons per pending update and a Mute button, `candidates` offers Share / Not now, `show` offers Install. Buttons carry an opaque token, never the skill id, version or hash; after a gateway restart a stale button simply reports that it expired. Every tap is checked against the same allowlist as your messages before anything runs, and every action that changes something posts an **Approve / Deny** card first showing exactly what the terminal would show (skill, exact version, content hash, Gateway verdict). Only an authorized tap on Approve proceeds; nobody answering within ten minutes cancels.

The gateway also delivers team notices proactively to the platform's home channel (`/sethome`), or to the last chat that used `/wisdom`: a card when a teammate publishes or updates a skill, when the team's update policy applied an update, when an update needs your decision, and when one of your local skills qualifies as a share candidate. Each item is sent once per platform; `mute` silences all of it.

## Update policy and local edits

Each installation carries the update mode your organization set on the Gateway: `MANUAL` (you review every version), `AUTO_WITH_NOTICE` (applied for you, you are told) or `REQUIRED` (mandated). The plugin re-checks at most every ten minutes (at session start, from the chat poller, from Desktop) and applies what policy allows, but a managed skill whose files no longer match the hash you installed carries **local edits** and is never overwritten silently:

| Mode | Unmodified | You edited your copy |
|---|---|---|
| `MANUAL` | listed; `update` asks | listed; `update` asks, your copy is kept aside |
| `AUTO_WITH_NOTICE` | applied, you are notified | **conflict**: Replace (edits kept aside) or Keep mine (`update <id> --keep`) |
| `REQUIRED` | applied, you are notified | applied; your edited copy is kept aside and reported |

An automatic update also requires the Gateway's security check to read `pass`; anything else is queued for your review. "Keep mine" pins that exact version — a newer one asks again.

## Share candidates

Nothing about your skills leaves the machine unless you share them, but the plugin does keep a small local record of the skill lifecycle events Hermes already emits (a skill loaded, patched or edited) and applies two deterministic rules to suggest what might be worth sharing: a skill used on seven consecutive business days, or one refined at least three times that then stayed stable for a week and is still in use. Bundled, hub-installed and Wisdom-installed skills are never candidates, neither are skills you already shared. At most three suggestions surface per week; `not-now <skill>` silences one for thirty days. A candidate is only a suggestion: sharing still goes through the two confirmations below. In chats and on Desktop a **Share** button prefills the description from the skill's frontmatter.

## Team notices

When a teammate publishes or updates a skill, the next new conversation gets a one-line heads-up in its system prompt ("your team published deploy-checklist v2") and the agent will mention it once if relevant. The feed is polled at most every ten minutes per profile and the note is frozen into the session when it starts, so it never changes mid-conversation and never costs you a prompt-cache miss. Nothing is downloaded or installed by a notice; installing still goes through the consent flow below. Notices clear when you install or remove the skill; `hermes wisdom mute 24` (or `/wisdom mute`) silences them.

Installed skills land under `~/.hermes/skills/_wisdom/<org>/<slug>/` and are indexed like any other skill. The plugin keeps a ledger of the exact version and content hash it installed, so `status` can tell you when the team has published something newer. Updating or reinstalling swaps the new version in atomically; if you had edited the installed files, your copy is kept under the plugin's data directory (the result reports `preserved_local_edits`) rather than overwritten. In shared chats `/wisdom status` omits local filesystem paths, and `hermes wisdom` exits non-zero when a command fails so scripts can react.

## Consent

Every action that changes your machine or your team's catalog asks first, and a "yes" in conversation is never enough:

- **Terminal**: a prompt showing the skill, exact version, content hash and the Gateway's security verdict.
- **Agent tools**: the same human-approval gate used for dangerous shell commands. In the CLI you get the usual once / session / always / deny prompt; on a messaging platform it becomes an approval button; with nobody present (cron, `-q`, subagents) the action is blocked.
- **Desktop**: Install opens a dialog with the exact version, content hash, update policy and Gateway security verdict; the backend applies only if the hash it computes at click time matches the one you saw. Share opens a dialog with the file list, byte counts, description and package hash; publication then happens only if the Gateway's security and professionalism checks both pass, otherwise the draft is withdrawn and the verdict is shown.
- **Telegram / Slack**: an Approve / Deny card with the same details, answered by an authorized tap.

Sharing asks twice: once to approve the exact package (file list, byte counts, content hash, your description) before it is uploaded as an owner-private draft, and once more after the Gateway has run its security and professionalism checks, before publication. Declining the second prompt withdraws the draft. Depending on your organization's policy the result is published immediately or held for an admin's review.

## What can be shared

Wisdom packages are instruction-only: a root `SKILL.md`, an optional `skill.manifest.json`, and plain-text files under `refs/` or `assets/` (`.md`, `.txt`, `.rst`, `.adoc`). Scripts, templates, executables, binaries, symlinks and package-manager manifests are refused, as is a `SKILL.md` that points at `scripts/` or `templates/`. If the manifest is missing, one is generated from the skill's frontmatter and the authoring OS/architecture. Everything downloaded is verified against the hashes the Gateway publishes before it is written to disk.
