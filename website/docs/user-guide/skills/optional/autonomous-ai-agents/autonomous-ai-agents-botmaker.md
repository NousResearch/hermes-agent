---
title: "Botmaker — Design, scaffold, and certify specialist Hermes bots"
sidebar_label: "Botmaker"
description: "Design, scaffold, and certify specialist Hermes bots"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Botmaker

Design, scaffold, and certify specialist Hermes bots.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/autonomous-ai-agents/botmaker` |
| Path | `optional-skills/autonomous-ai-agents/botmaker` |
| Version | `0.2.0` |
| Author | techjanitor (adapted by Nous Research) |
| License | MIT |
| Platforms | linux, macos |
| Tags | `hermes`, `bots`, `profiles`, `soul` |
| Related skills | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Botmaker

Design, scaffold, certify, and document specialist Hermes bots. Load this skill on every in-scope question. You draft; the child earns. Seeding files is not a bot.

Details: `references/soul-craft.md` (SOUL + human gate), `references/process.md` (mechanics), `references/vault.md` (vault conventions). Shared-skill linker: `scripts/link_skill_tree.py`. Drift tripwire: `scripts/drift_check.py`.

## When to Use

- Your human wants a new specialist bot, or a SOUL drafted for one — in chat, or as a task card assigned to `botmaker` (if your fleet dispatches work by kanban). A card carrying a human-signed SOUL spec satisfies the human gate.
- An existing specialist bot's identity/job/voice is wrong and needs a SOUL rewrite (human gate)
- `Bots/` in the vault needs a specialist note, roster row, or `making-bots.md` changelog
- A shared skill is about to be copied into a profile (stop; symlink files instead)

Don't use for: operating the inference server, ComfyUI, or any service; general coding; rewriting default `$hermes_root/SOUL.md`; spawning `botmaker-2`. `@hermes` is not the bot coordinator — you are.

## Prerequisites

- Hermes CLI (`hermes profile create --help`)
- Resolve the installation root as `hermes_root` before any write (`references/process.md` §Installation root). Canonical skills live under `$hermes_root/skills/`, **not** the active named profile's skills directory. Never substitute a different installation's default path.
- Vault: `<your-vault-path>` (see `references/vault.md`). Vault documentation is an optional convention — if the user has no vault, skip the vault steps.
- Human guide, fleet roster, changelog: vault `Bots/making-bots.md` (create it on first ship). Method lives here only; fleet state lives there only — patch a lesson's one owner and log one changelog line.

## How to Run

1. Interview. Refuse to scaffold until job / independence failure / not-list are sharp. See Procedure.
2. Draft a one-screen SOUL (`references/soul-craft.md`). Stop. Your human signs it.
3. Scaffold (`references/process.md`). **Alias preflight first.** `--no-skills`, pin the brain, file-level skill links, USER.md lockstep sentence if the skill is shared.
4. Kickoff in the **child's** Bot Chat. Do not babysit its tools. Do not write its `MEMORY.md`.
5. After certification: vault note + roster row (`references/vault.md`). Not before.

## Quick Reference

| Need | Do |
|---|---|
| New bot | Alias preflight (Procedure C) → Interview → SOUL draft → human sign-off → `hermes profile create NAME --no-skills --description "…"` |
| Pin a brain | `hermes -p NAME config set model.provider …` then `model.default …`. Then **unset** the copied `base_url` / `api_key` (create copies your default profile's model block, whatever it currently is) |
| Shared skill | Canonical `$hermes_root/skills/<cat>/<name>/`; profile files via `scripts/link_skill_tree.py`. Never a directory symlink. Load linked `references/` with `read_file` on the canonical path |
| Vault after cert | `Bots/<name>.md` (`type: bot-reference`), roster row on `making-bots.md`, one line on `Home.md` |
| Child memory | Child writes it. You do not. |

## Procedure

### A. Interview (before any `profile create`)

Ask, and do not proceed until 1–3 are sharp:

1. One-sentence job. If "and also," split into two bots. Job may be a CLI, an HTTP API, a GUI sock — ComfyUI counts. "CLI-only" is how we *create* the profile, not what the specialist is allowed to operate.
2. Failure the brain must survive → model pin (if the bot's job lives on a local box, that usually means a hosted provider — the bot must still think when the box is down).
3. What it is not.
4. Does default Hermes need the same skill? → canonical under `$hermes_root/skills/` + file-level symlinks.
5. Who is the client to ping (`@hermes`, another specialist, nobody).
6. Voice: inherit vs write. Inheritance is tone, not the model's stock identity — that is costume unless the bot *is* a persona bot (if your fleet has one).

### B. SOUL draft

Write one screen. Identity (job name first), Job, Hard constraints as their own heading, Voice, What you are not, Disagreement. No live pins, no IPs, no last-incident notes, no "helpful assistant," no pasted stock-model `# Style` block. Full rules in `references/soul-craft.md`.

**Human gate:** do not run `hermes profile create` until your human signs the draft — chat sign-off or a signed spec on the task card (`references/soul-craft.md` §Human gate). Iterate in the conversation or on the card, not on disk.

### C. Scaffold

Full mechanics and the command sequence: `references/process.md`. The stop points:

**Alias preflight — mandatory, before any `profile create NAME`.** The default alias wrapper can destroy a real CLI through a `~/.local/bin` symlink:

```text
zsh -lc 'which -a NAME'
# and, even if which is empty:
test -e "$HOME/.local/bin/NAME" || test -L "$HOME/.local/bin/NAME"
```

If **either** finds anything: `--no-alias` (rationale and the landmines: `references/process.md` §--no-skills).

- `--no-skills`; never `--clone` / desktop clone. Pin the brain, then unset the copied `base_url`/`api_key`; `config get model` must show only `default` + `provider`.
- **Peers + memory provider — check after every create** (site-specific; commands and verification: `references/process.md` §Independence). The invariant: every rostered profile has every peer it must reach. Do not print keys.
- Sibling-profile CLI: use the verified installation root from `references/process.md`, with an unset-variable guard. Stop if root discovery fails or points outside the intended installation.
- By default, `SOUL.md` writes prompt your human (`security.protected_instruction_files: true`) — that prompt is the human gate materialized. If your fleet disabled it for headless provisioning (see `profile/config.yaml` in this repo), the signature is the only gate: no human-signed draft or signed spec, no write. If a write is blocked anyway, stop — do **not** sneak it in via the shell.
- Never hand-edit `config.yaml` — `config set` only (your human may hand-edit their own; you do not). Never copy credentials. Never print `auth.api_key` / `secret_key`.

Then: write signed `SOUL.md`, thin `memories/USER.md` — if the skill is shared, include the **lockstep sentence**: one line stating that the canonical skill lives under `$hermes_root/skills/<category>/<name>/`, the profile files are symlinks, and process changes patch the canonical tree (so later sessions do not re-flag the drift). Then empty-or-tiny `memories/MEMORY.md` (the child fills this). Install the runbook skill; link shared skills file-level (`references/process.md` §Shared skills). Not every specialist needs a custom skill — `@hostadmin` is `hermes-agent` + earned MEMORY. Do not invent a runbook so the folder looks complete.

Stop. Do not fill the child's `MEMORY.md`. Do not add a vault roster row yet.

### D. Certification (child's Bot Chat)

Kickoff: first look at the real box/job → one failure path → client ping if there is a client. The child patches its own skill + `MEMORY.md`. You may `message_agent` one job ("first look") and then get out.

Certified when it has done the job against the live system, rewritten the stale bits of its runbook (if it has one), and pinged the client (or you agreed there isn't one). A first look that finds nothing broken is still a first look — do not invent a failure to practice. Your human saying "document it in the vault" is a ship signal. Uncertified profiles do not get a `Bots/<name>.md` or a roster row unless your human overrides.

### E. Vault

Only after D. See `references/vault.md`.

## Failure modes (symptom → owner)

One line each; mechanics live with the owner.

| Symptom | Rule | Where |
|---|---|---|
| Skill invisible / empty index | never directory-symlink a skill (rglob) | `references/process.md` §Shared skills |
| `skill_view`/`skill_manage` `file_path` rejected | file-symlink escape — `read_file` canonical | `references/process.md` §Shared skills |
| Second copy after a patch | `skill_manage` unlinks — patch canonical, `ls -l` | `references/process.md` §Shared skills |
| "Independent" bot on default's provider | unset the create-leftover `base_url`/`api_key` | `references/process.md` §Independence |
| DM acks then vanishes | a peer missing on the sender profile | `references/process.md` §Independence |
| Honcho peer `(not set)` | (Honcho users) `honcho sync` + `memory.provider honcho` | `references/process.md` §Independence |
| Real CLI destroyed by alias | preflight before `profile create` | `references/process.md` §--no-skills |
| `Unknown skill(s)` at worker init | link the injected skill file-level; don't always-load | `references/process.md` §--no-skills |
| `ui_meta` empty / `toolsets` says `hermes-cli` | title in `profile.yaml`; spec list is `platform_toolsets.cli` | `references/process.md` §Scaffold sequence |
| Worker files to the wrong board / sibling inherits kanban env | strip `HERMES_KANBAN_*`; one terminal call per worker | `references/process.md` §Kanban intake |
| Child sounds like a costume | identity = job name; inherit tone, not self | `references/soul-craft.md` |
| SOUL written unsigned | the signature is the gate | `references/soul-craft.md` §Human gate |
| Uncertified bot in the vault; persona in `Bots/` | write after certification only | `references/vault.md` |
| Filling child MEMORY · `botmaker-2` · default SOUL · operating services | constitution | `profile/SOUL.md` |

## Verification

A new bot is done when:

1. `hermes profile list` shows it on the pinned model, not silently riding your default profile's.
2. `hermes -p NAME config get model` is only provider + default.
3. `.no-bundled-skills` exists in the profile root.
4. (If your fleet has peers) `hermes -p NAME peer list` shows every peer it must reach — or DMs will ack then vanish.
5. (Honcho users) `hermes honcho peers` lists NAME with a real user peer (not `(not set)`). `memory.provider` is `honcho`.
6. `SOUL.md` matches the signed draft (one screen).
7. Shared skills: `ls -l` on profile `SKILL.md` shows `l` (symlink), canonical is a regular file. `rglob` from the profile skills dir finds `SKILL.md`.
8. After certification only: vault note + roster row + Home line + `making-bots.md` changelog.
9. `python3 "${hermes_root:?resolve installation root first}/skills/autonomous-ai-agents/botmaker/scripts/drift_check.py" --hermes-root "$hermes_root"` exits 0. The roster check validates listed specialists exist; personas, daily drivers, and uncertified profiles need no row.
