---
title: Backup and disaster recovery
---

# Backup and disaster recovery

Hermes stores most user state under your Hermes home directory (`~/.hermes` by default, or `$HERMES_HOME` when set). A disaster-recovery plan should separate **portable configuration** from **sensitive full-state archives** and **machine-local runtime files**.

## Which command should I use?

| Command | Use it for | Includes secrets? | Includes session history? | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `hermes backup` | Full machine migration or disaster recovery | Yes | Yes | Creates a sensitive `.zip` of Hermes home, excluding the Hermes source checkout and regenerable caches. Store encrypted. |
| `hermes import <zip>` | Restore a full backup | Restores secrets from backup | Restores history from backup | Refuses unsafe archives and skips volatile runtime files such as PID/lock/state files. |
| `hermes backup --quick` | Local rollback before risky changes or updates | Yes | Critical DBs only | Writes `state-snapshots/` under the active Hermes home. Not a replacement for an off-machine backup. |
| `hermes profile export <name> -o profile.tar.gz` | Move/share one profile | Named profiles: no `.env`/`auth.json`; default profile export also strips root-level secrets and DBs | Named profiles: yes; default profile export is filtered | Best for portable profile snapshots, not full disaster recovery. |
| `hermes profile import <archive> --name <name>` | Restore a profile export as a named profile | Uses whatever the export contains | Uses whatever the export contains | Cannot import over the built-in `default` root profile. |

## What `hermes backup` captures

A full backup walks the default Hermes root and writes a zip archive. It is intended to be a complete, restorable copy of user state:

- `config.yaml`
- `.env`
- `auth.json`
- `state.db` and other Hermes SQLite databases, copied with SQLite's backup API so WAL-mode databases are consistent while Hermes is running
- `sessions/`
- `skills/`, including user-created skills and the curator archive under `skills/.archive/`
- `cron/` jobs and scripts
- `memories/`
- `profiles/` and each named profile's state
- platform pairing/channel files such as `channel_directory.json` when present
- active memory-provider external state declared via `backup_paths()` when that state lives under your home directory

A full backup deliberately excludes:

- the root-level `hermes-agent/` source checkout — reinstall or reclone Hermes instead
- `.git/`, `node_modules/`, Python virtualenvs, `site-packages/`, and common tool caches
- nested previous `backups/` directories, to avoid exponential backup growth
- `checkpoints/`, which are per-session trajectory caches and do not port cleanly
- SQLite sidecars (`*.db-wal`, `*.db-shm`, `*.db-journal`), because the `.db` itself is snapshotted consistently
- PID files such as `gateway.pid` and `cron.pid`

`hermes import` also skips volatile runtime names from older archives, including `gateway_state.json`, `gateway.pid`, `cron.pid`, `gateway.lock`, and `processes.json`. Those files describe the source machine's running processes and must not overwrite the target machine's runtime state.

## What `hermes profile export` captures

Profile export is for **portable profile snapshots**, not full machine backup.

For a named profile, `hermes profile export <name>` stages that profile directory and excludes only credential files named `.env` and `auth.json`. This means it usually carries profile config, skills, memories, sessions, cron jobs, logs, and databases, but the target machine still needs fresh credentials or its own credential files.

For the built-in `default` profile, the export is more aggressively filtered because `default` is the whole `~/.hermes` root. It excludes root-level infrastructure and secrets such as `hermes-agent/`, `.worktrees/`, sibling `profiles/`, `bin/`, `node_modules/`, root-level `state.db`, `auth.json`, `.env`, runtime process state, logs, and caches. Importing an archive as `default` is disallowed; import it as a named profile instead.

## Private git repo vs encrypted backup

A good recovery setup uses both layers:

### Put in a private git repo

Use git for portable, non-secret configuration and runbooks that benefit from diff review:

- sanitized `config.yaml` templates, with placeholders instead of tokens
- MCP server config templates and setup notes
- custom skills that do not contain credentials or private transcripts
- scripts that are not secret-bearing
- cron job templates or documented schedules, with secrets removed
- restore checklists and machine-specific notes

### Keep out of git

Do not commit secrets or high-sensitivity runtime archives:

- `.env`
- `auth.json`
- OAuth refresh tokens and credential pools
- full `hermes backup` zip files
- session history and `state.db` unless you have intentionally redacted/exported it
- voice/audio caches, downloaded documents, screenshots, or logs that may contain private data

Store full `hermes backup` archives in an encrypted backup system instead. Treat them as equivalent to API keys plus private conversation history.

## Recommended backup routine

1. Keep sanitized setup templates and this runbook in private git.
2. Before risky local changes, run:

   ```bash
   hermes backup --quick --label pre-change
   ```

3. Periodically create a full encrypted off-machine backup:

   ```bash
   hermes backup -o /path/to/encrypted/storage/
   ```

4. After changing credentials, providers, gateway pairing, profiles, skills, or cron jobs, create a fresh full backup.
5. Test restore on a disposable profile or spare machine before relying on the backup.

## Restore checklist

On the target machine:

1. Install Hermes and verify the CLI works:

   ```bash
   hermes doctor
   ```

2. Stop the gateway if it is running on the target.
3. Restore the full archive:

   ```bash
   hermes import /path/to/hermes-backup.zip
   ```

4. Re-run setup checks:

   ```bash
   hermes doctor
   hermes status --all
   hermes mcp list
   hermes tools list
   hermes skills list
   hermes cron list
   hermes sessions list
   ```

5. Reinstall or update the source checkout if needed:

   ```bash
   hermes update
   ```

6. Re-authenticate providers whose credentials were intentionally not restored or are machine-bound.
7. Verify messaging gateways:

   ```bash
   hermes gateway status
   hermes gateway start
   ```

8. Send a test message from each configured platform, then verify the gateway logs show the inbound message and a successful response.
9. Verify MCP servers that depend on local paths. Machine-local paths often need edits after restore.
10. Verify memory and session search by searching for a known recent session.
11. Verify cron by listing jobs and manually running one harmless job.
12. Verify voice features if used: send a short voice message and check transcription/TTS behavior.

## Common restore fixes

- **Gateway does not reconnect:** check platform tokens, pairing/home-channel files, and `hermes gateway status`; then restart the gateway.
- **MCP tools are missing:** run `hermes mcp list` and fix machine-local command paths or Node/Python environment paths.
- **Session search is empty:** confirm `state.db` restored and run `hermes sessions list`.
- **Cron jobs do not fire:** check `hermes cron list`, gateway scheduler status, and any per-job `workdir` paths.
- **Provider auth fails:** re-run `hermes auth` or restore credentials from the encrypted backup if appropriate.
