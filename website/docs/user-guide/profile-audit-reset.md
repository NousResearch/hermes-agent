---
sidebar_position: 6
title: "Audit and reset a Hermes profile"
description: "Safely clean up a long-running Hermes install by auditing state, backing it up, and migrating only the pieces you still want into fresh profiles."
---

# Audit and reset a Hermes profile

Hermes gets more useful as it accumulates sessions, skills, memories, cron jobs, tools, model settings, and gateway configuration. That same state can also get noisy after months of experiments.

This guide is a safe cleanup workflow for a long-running install. It is **not** a destructive reset button. The safe path is:

1. back up the current profile;
2. inventory what is actually in use;
3. create a clean profile;
4. migrate only the state you still want;
5. verify the new profile;
6. retire old state only after a rollback period.

:::warning
Do not delete profiles, cron jobs, memories, skills, plugins, or secrets until you have a backup and have verified the replacement profile. Profile cleanup can remove useful operational history if you rush it.
:::

:::danger Keep backups private
Profile backups can contain API keys, bot tokens, session transcripts, cron prompts, logs, delivery targets, and provider configuration. Store archives outside synced/shared folders, restrict filesystem permissions, and delete or encrypt old backups when you no longer need them.
:::

## When to use this workflow

Use this playbook when Hermes still works but responses feel worse or harder to steer because the install has accumulated too much state:

- experiments are mixed with daily-use configuration;
- old skills or memories conflict with current preferences;
- cron jobs exist but you no longer know what they do;
- provider or model settings have been changed many times;
- multiple gateways, MCP servers, or tools were tried and abandoned;
- you want separate coding, research, personal, or sandbox agents.

If Hermes is currently broken, run `hermes doctor` first and fix the immediate failure before doing a cleanup migration.

## What lives in a profile

A Hermes profile is a separate `HERMES_HOME`. Each profile has its own state directory. A typical profile includes:

```text
~/.hermes/
├── config.yaml
├── .env
├── SOUL.md
├── memories/
├── skills/
├── cron/
├── sessions/
├── plugins/
└── logs/
```

Actual installs may also contain runtime databases, auth state, caches, process files, generated media, source checkouts, or other files created by tools and plugins. Inventory the filesystem before deleting anything.

Named profiles live under `~/.hermes/profiles/<name>/` and can be targeted with either their generated alias or `hermes -p <name>`. Aliases may not exist if they were skipped with `--no-alias` or if the profile name collides with an existing command. `hermes -p <name>` is the reliable form.

Profiles separate Hermes state. They do **not** sandbox filesystem access. If you need a predictable working directory for a profile, set `terminal.cwd` in that profile's `config.yaml`.

## Phase 1: Back up before touching anything

Create at least one backup before you clean up.

### Option A: Export a portable profile archive

For a named profile:

```bash
hermes profile export coder
```

For the default profile:

```bash
hermes profile export default
```

`hermes profile export` creates a portable archive and intentionally filters credential, runtime, cache, local backup, and infrastructure files. It is useful for moving profile content, but it is **not** a complete secret-preserving rollback backup. If you need to recover provider keys, bot tokens, OAuth credentials, live runtime databases, local backups, or logs, back those up separately or use a full filesystem archive.

### Option B: Create a full filesystem archive

For a complete default-profile rollback archive, stop gateways/cron where practical, then archive `~/.hermes`:

```bash
cd ~
tar -czf hermes-default-backup-$(date +%Y%m%d-%H%M%S).tar.gz .hermes
chmod 600 hermes-default-backup-*.tar.gz
```

A full archive can be large and can include secrets, sessions, logs, source checkouts, virtual environments, caches, and active SQLite files. Keep it private and store it outside `~/.hermes` so it does not get swept into future cleanup.

### Option C: Clone everything into a rollback profile

If you want an immediately runnable rollback copy, be explicit about the source profile.

For the default profile:

```bash
hermes profile create rollback --clone-all --clone-from default
hermes -p rollback doctor
```

For a named profile:

```bash
hermes profile create coder-rollback --clone-all --clone-from coder
hermes -p coder-rollback doctor
```

`--clone-all` creates a fresh runnable profile, not a complete snapshot. For every source profile it excludes session history and recovery artifacts: `state.db` (including WAL/SHM files), `sessions`, `backups`, `state-snapshots`, and `checkpoints`. When the source is the default profile, it also excludes default-only infrastructure: sibling `profiles`, the `hermes-agent` checkout, `.worktrees`, `bin`, and `node_modules`. Ordinary profile state such as logs remains included, while runtime PID/process files are stripped from the clone.

:::warning
A rollback profile may contain cron jobs, gateway settings, provider credentials, plugins, logs, and delivery targets. It does not retain session history. Do not start its gateway or enable/run its cron jobs unless you intend to duplicate those side effects.
:::

## Phase 2: Inventory the current profile

Run the inventory from the profile you want to clean up. For the default profile, use plain `hermes`. For a named profile, use its alias or `hermes -p <name>`.

Default profile example:

```bash
hermes doctor
hermes profile list
hermes profile show default
hermes config
hermes skills list
hermes cron list
```

Named profile example:

```bash
hermes -p coder doctor
hermes profile show coder
hermes -p coder config
hermes -p coder skills list
hermes -p coder cron list
```

Then inspect the state directory directly. The read-only audit command gives you a safe size/count baseline without printing file contents or secret values:

```bash
hermes profile audit default
hermes profile audit default --json > old-profile-audit.json
```

For a named profile:

```bash
hermes profile audit coder
hermes profile audit coder --json > old-profile-audit.json
```

Default profile:

```bash
for d in skills memories cron logs plugins sessions; do
  [ -e "$HOME/.hermes/$d" ] && ls -la "$HOME/.hermes/$d"
done
```

Named profile:

```bash
for d in skills memories cron logs plugins sessions; do
  [ -e "$HOME/.hermes/profiles/coder/$d" ] && ls -la "$HOME/.hermes/profiles/coder/$d"
done
```

Look for these common sources of drift:

| Area | What to check | Cleanup decision |
| --- | --- | --- |
| `config.yaml` | old providers, model overrides, terminal backends, toolsets, MCP entries | keep only settings you understand and still use |
| `.env` / `auth.json` | old API keys, bot tokens, provider keys, OAuth state | rotate or remove stale secrets; do not copy unknown keys forward |
| `SOUL.md` | long, conflicting, or outdated identity instructions | shorten to stable identity and operating preferences |
| `memories/` | stale user facts, old project state, duplicated preferences | keep durable facts; remove temporary task progress |
| `skills/` | abandoned experiments, duplicate workflows, stale commands | keep reusable, tested skills; archive or remove the rest |
| `cron/` | forgotten recurring jobs or delivery targets | pause/remove jobs you cannot explain |
| `plugins/` | unused memory providers, platform adapters, MCP helpers | keep only actively used plugins |
| `sessions/` | large history you do not need in the cleaned profile | leave behind unless the new profile needs continuity |
| `logs/` | errors pointing to broken providers, tools, gateways | fix root causes before migration |

:::tip
Do not copy secrets or tokens into notes, PRs, screenshots, or issue comments. Report secret presence as "configured" or "missing" only.
:::

## Phase 3: Choose a target profile layout

A reset is a good time to split one overloaded agent into smaller purpose-built profiles.

Example layout:

| Profile | Purpose | Suggested contents |
| --- | --- | --- |
| `default` | light personal assistant and emergency fallback | minimal config, trusted provider, few durable memories |
| `coder` | software development | coding skills, GitHub tooling, project `terminal.cwd`, code-review preferences |
| `research` | web and document research | research skills, browser/search tools, citation preferences |
| `ops` | gateways, cron, and always-on automation | stable cron jobs, gateway tokens, conservative model routing |
| `sandbox` | experiments | throwaway tools, new plugins, risky skill drafts |

Keep production gateways and recurring cron jobs out of experimental profiles. Keep experimental skills out of the profile that answers daily user messages.

## Phase 4: Create a clean profile

Choose a profile name that does not already exist. If the name you want is already taken, back up the old profile first and then use a temporary name, such as `coder-clean`.

For a clean profile with no inherited clutter:

```bash
hermes profile create coder-clean --description "Focused software-development assistant."
hermes -p coder-clean setup
hermes -p coder-clean doctor
```

For a profile that reuses provider credentials, `SOUL.md`, installed skills, and curated memory files from the source profile:

```bash
hermes profile create coder-clean --clone --description "Focused software-development assistant."
hermes -p coder-clean doctor
```

Use `--clone` only when the current provider setup, secrets, `SOUL.md`, skills, and curated memories are known-good. If you want truly fresh memory or separated credentials, use a blank profile and migrate manually.

If the profile should start terminal commands in a specific project:

```bash
hermes -p coder-clean config set terminal.cwd /absolute/path/to/project
```

## Phase 5: Migrate only the pieces you still want

Move state forward deliberately. Do not bulk-copy the whole old profile unless you are making a rollback snapshot.

### Config

Use `hermes config` and `hermes config set` where possible instead of hand-editing YAML:

```bash
hermes -p coder-clean config
hermes -p coder-clean config set model.default anthropic/claude-sonnet-4
hermes -p coder-clean config set terminal.cwd /absolute/path/to/project
```

Replace model names with a provider/model already configured for that profile. Verify with `hermes -p coder-clean config` or `hermes -p coder-clean model` before assuming a model alias is available.

Copy only settings that still have a clear purpose. Prefer the profile's `.env` for secrets and `config.yaml` for non-secret settings.

### Personality

Keep `SOUL.md` short and stable. A good reset removes one-off instructions, old project state, and conflicting persona fragments.

Default profile:

```bash
$EDITOR ~/.hermes/SOUL.md
```

Named profile:

```bash
$EDITOR ~/.hermes/profiles/coder-clean/SOUL.md
```

Start a new session after changing `SOUL.md`; existing sessions may still carry old prompt context.

### Memories

Migrate only durable facts that will still be useful later. Avoid copying completed task logs, temporary TODOs, PR numbers, issue numbers, stale paths, or old decisions that belong in project documentation instead.

Good memory candidates:

- stable user preferences;
- stable environment facts;
- recurring project conventions;
- durable safety boundaries.

Poor memory candidates:

- "fixed bug X";
- "opened PR Y";
- old execution status;
- dated troubleshooting notes;
- copied chat transcripts.

### Skills

Keep skills that are reusable, tested, and broad enough to deserve being loaded by an agent. Remove or archive skills that are stale, duplicated, or only documented a one-off task.

You can migrate a known-good local skill by copying its skill directory into the new profile's `skills/` directory, then checking that Hermes sees it:

```bash
mkdir -p ~/.hermes/profiles/coder-clean/skills
cp -R ~/.hermes/skills/software-development ~/.hermes/profiles/coder-clean/skills/
hermes -p coder-clean skills list
```

Adjust the source path for the skill or category you actually want to migrate. Do not copy every skill by default; that recreates the clutter you are trying to remove.

### Cron jobs and gateways

Cron jobs and gateways have side effects. Re-create or migrate them last.

Before enabling a job in the new profile, confirm:

- schedule;
- prompt;
- attached skills;
- delivery target;
- working directory;
- profile pin;
- whether it can send messages or mutate external systems.

Use list/inspect commands first:

```bash
hermes -p coder-clean cron list
hermes -p coder-clean gateway status
```

Re-create recurring jobs manually with safe delivery first, such as local-only delivery or a one-shot dry run, before restoring production delivery targets. Do not copy old cron files blindly unless you have read their prompts and delivery settings.

## Phase 6: Verify the cleaned profile

Run a small acceptance test before retiring the old profile:

```bash
hermes -p coder-clean doctor
hermes -p coder-clean skills list
hermes -p coder-clean cron list
hermes -p coder-clean chat -q "In one sentence, tell me your profile name and primary role."
```

Then prove that the reset made something better. A profile that merely exists is not enough. Capture before/after evidence for the problem you were trying to fix:

```bash
hermes profile audit default --json > old-profile-audit.json
hermes profile audit coder-clean --json > new-profile-audit.json
hermes profile audit default --compare coder-clean
hermes prompt-size --json > old-prompt-size.json
hermes -p coder-clean prompt-size --json > new-prompt-size.json

du -sh ~/.hermes ~/.hermes/profiles/coder-clean
```

Good proof can include a smaller prompt budget, fewer enabled tools, fewer stale skills or memories, faster startup/inspection commands, a smaller profile footprint, cleaner `doctor` output, or successful answers on the workflows that were previously noisy. If the clean profile does not improve the measured problem, keep iterating before switching to it.

If the profile uses a gateway:

```bash
hermes -p coder-clean gateway status
```

If the profile has a project `terminal.cwd`, ask it to run a harmless command in that workspace and confirm the path.

Do not delete the old profile until the new one can:

- load the expected model/provider;
- access the expected tools;
- see the expected skills;
- keep secrets out of logs and docs;
- run any required gateway or cron workflows;
- answer with the intended personality and scope.

## Phase 7: Retire old state carefully

Once the cleaned profile works, keep the backup for a while. Then retire old state in the least-destructive order:

1. pause old cron jobs;
2. stop old gateways;
3. rename old named profiles to make accidental use obvious;
4. after a cooling-off period, delete named profiles you no longer need.

For a named profile, replace `old-coder` with the old profile name — **not** the new replacement profile:

```bash
hermes profile rename old-coder old-coder-archive
hermes profile delete old-coder-archive
```

Profile deletion stops the gateway, removes the service and alias, and deletes all profile data. It asks for confirmation unless you pass `--yes`.

:::note
The default profile cannot be deleted with `hermes profile delete default`. Keep a backup and clean it in place, or use the documented uninstall/reset path if you truly want to remove the whole default install.
:::

## Reset checklist

Use this checklist as a final pass.

- [ ] Current profile backed up or exported.
- [ ] Backup secrecy understood and protected.
- [ ] `hermes doctor` run on old profile.
- [ ] Current config inventoried.
- [ ] Skills reviewed for stale or duplicate workflows.
- [ ] Memories reviewed for stale task progress.
- [ ] Cron jobs reviewed before being copied or re-enabled.
- [ ] Gateway tokens and delivery targets checked.
- [ ] Clean profile created.
- [ ] Only necessary config copied forward.
- [ ] `SOUL.md` simplified.
- [ ] Required skills copied and verified.
- [ ] Required memories copied and verified.
- [ ] Cron jobs re-created only after side effects are understood.
- [ ] New profile passes `doctor`.
- [ ] New profile answers with the expected role.
- [ ] Old profile kept as rollback until the new profile is trusted.

## What not to do

Avoid these shortcuts:

- deleting `~/.hermes` before backing it up;
- sharing a backup archive that may contain credentials or transcripts;
- copying every old memory into the new profile;
- copying old `.env` keys you do not recognize;
- enabling old cron jobs before reading their prompts and delivery targets;
- treating profiles as a filesystem sandbox;
- deleting the replacement profile when you meant to retire the old one;
- doing cleanup from a live production gateway session without a rollback profile.

A good reset should make Hermes easier to reason about. If you cannot explain why a setting, skill, memory, or cron job belongs in the cleaned profile, leave it behind and keep it in the backup instead.
