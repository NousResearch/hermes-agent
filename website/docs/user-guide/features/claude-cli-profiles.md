---
title: Claude Code Profiles
description: Run Claude Code jobs on two or more separate Claude accounts, and move to the next account when one reaches its plan limit.
sidebar_label: Claude Code Profiles
sidebar_position: 10
---

# Claude Code Profiles

Hermes can start Claude Code as a child program. It does this for
[delegation](./delegation.md), for the [ACP runtime](./acp.md), and for a
`claude` command the agent writes into the terminal.

Claude Code keeps one login per configuration directory. If you hold two
Claude accounts on separate plans, you can give Hermes both. Hermes then reads
how much of each plan is used, picks an account that still has room, and moves
to the other account when the first one fills.

The feature is off until you configure two or more profiles. With one profile,
or none, Hermes behaves exactly as it does today.

## What a profile is

A profile is one directory that holds one Claude Code login. Claude Code reads
that directory from two environment variables:

- `CLAUDE_CONFIG_DIR` — the configuration directory.
- `CLAUDE_SECURESTORAGE_CONFIG_DIR` — the secret store. It defaults to the
  configuration directory.

Hermes sets these two variables on the child process, and removes three others
that Claude Code prefers over a profile directory:

- `CLAUDE_CODE_OAUTH_TOKEN` — another subscription login.
- `ANTHROPIC_API_KEY` — a metered interface key.
- `ANTHROPIC_AUTH_TOKEN` — a metered bearer token.

Any one of them left in place would send the work somewhere you did not
choose, and the last two bill by the token. Hermes puts no token in their
place: the `claude` program reads its own secret from the directory named
above. Hermes never copies a token into its own configuration, its state, a
log line, or a command line.

## Configure it

Add this to `config.yaml`. There is no `.env` setting: `.env` holds secrets
only, and this section holds no secret.

```yaml
claude_cli_profiles:
  # A window counts as full at or above this percentage. The default is 95.
  # The number leaves room for the job in flight to finish.
  stop_at_percent: 95

  profiles:
    # A local nickname. Hermes prints this name. It never prints an address.
    - name: work
      config_dir: ~/.claude
    - name: spare
      config_dir: ~/.claude-spare
      # Optional. It defaults to config_dir.
      securestorage_dir: ~/.claude-spare
```

Sign each profile in by hand, once, in a terminal:

```bash
CLAUDE_CONFIG_DIR=~/.claude-spare \
CLAUDE_SECURESTORAGE_CONFIG_DIR=~/.claude-spare \
claude auth login
```

Hermes never starts a login. A login needs a browser and a person.

## Read the status

```bash
hermes auth status claude-profiles
```

The command reads local state and one usage endpoint per profile. It starts no
model and it spends no tokens.

```
Claude Code profile switching: on. A window counts as full at 95%.
NAME        STATE       5-HOUR  WEEKLY    OPUS  REOPENS
work        open           31%     44%     52%  -                       in use
spare       full          100%     72%     88%  2026-08-07 18:00 UTC
```

The `STATE` column holds one of four words:

| Word | Meaning |
|---|---|
| `open` | Every window is below the stop percentage. Hermes may use it. |
| `full` | A window reached the stop percentage. `REOPENS` says when. |
| `sign in` | The profile holds no login, or the account rejected it. |
| `not checked` | The usage read failed. Hermes will not run on it. |

## How Hermes chooses

Before a new Claude Code job starts, Hermes reads each profile's usage. Three
windows count, and any one of them can fill on its own:

- the five-hour session window,
- the weekly window for all models,
- the weekly window for the Opus model family, which usually fills first.

Then it applies these rules, in order:

1. Keep the account the work is already on, while every window is below the
   stop percentage. A lower number on another account is not a reason to move:
   a move costs the child program its whole warm context.
2. Otherwise take the first configured account that is open.
3. Otherwise stop, and report when each account reopens.

Hermes never runs on an account whose usage it could not read. That read is
also the check that the account answers as itself, so running it could bill a
subscription you did not choose. A wait you can see beats a charge you cannot.

Hermes writes an `INFO` log line for every selection and for every move.

## Wrappers, and adapters that review pull requests

Many people do not call `claude` directly. They call a small shell wrapper that
pins a profile, checks the account, and then hands over to the real binary. A
pull-request review adapter is the common case.

Hermes supports that. The contract has two parts.

**Part one: name the wrapper `claude-<something>`.** Hermes treats any program
whose file name is `claude`, or starts with `claude-`, as a Claude Code
launcher. `claude-hermes` and `claude-review` both qualify. `copilot`, `codex`,
and `gemini` do not, and Hermes leaves their environment alone.

Point Hermes at the wrapper the same way you point it at the binary:

```yaml
model:
  provider: copilot-acp
  command: claude-hermes
```

The `delegate_task` tool takes the same name in its `acp_command` argument.

**Part two: let the two variables through.** Hermes exports
`CLAUDE_CONFIG_DIR` and `CLAUDE_SECURESTORAGE_CONFIG_DIR` before it starts the
wrapper. A wrapper that passes its environment through, with `exec "$REAL_BIN"
"$@"`, gets the account Hermes chose.

A wrapper that sets those two variables itself keeps its own choice. That is
allowed and sometimes correct: a review adapter that must always bill one named
account should pin that account and refuse anything else. Hermes then leaves
the account alone, and the wrapper's check is the one that decides. Pick one
owner for the choice, and write down which one it is.

## Two rules that protect you

**A conversation stays on the account that started it.** Claude Code keeps its
conversation record inside the profile directory. The same conversation on
another account either fails or starts fresh work and loses the first
conversation. So Hermes pins each conversation to its own account.

When that account fills, Hermes stops and reports the reopen time. It does not
move the conversation, even when another account is wide open. A wait is
visible and you can recover from it; losing the conversation is neither. Start
a new conversation to use the other account.

Two conversations that run at the same time each keep their own account. A
terminal command inside one conversation reads that conversation's account,
not whichever account another conversation picked last.

This covers a chat on a messaging platform and a conversation on the command
line, in the terminal user interface, and in the desktop application. Each has
a stable conversation identity that follows a resume and changes on `/new`.

**The boundary.** Work with no conversation behind it — a cron job, a one-off
script, a direct call — is not pinned. It takes the account named in the shared
slot, which is the account the most recent selection chose. That is correct for
such work: there is no earlier conversation to lose.

**Hermes never spends paid usage past a plan.** An account can carry paid
usage after its plan windows fill. Hermes reads only the plan windows. It
treats a full plan window as full, whatever the paid allowance holds.

## Refreshed tokens stay in their own profile

Hermes can refresh an expired Claude Code token when it uses one itself. The
refreshed token is written back only to the profile it was read from. If the
profile changed between the read and the refresh, Hermes refuses the write and
says so, rather than overwrite another account's login and sign that person
out.

## Turn it off

Any one of these returns Hermes to its earlier behaviour:

- Run `hermes auth reset claude-profiles`. This forgets every account
  selection and every conversation pin.
- Remove a profile from `config.yaml`, so fewer than two remain.
- Delete `claude_cli_profiles.json` from your Hermes home directory.

All three take effect on the next job. No service needs a restart. Nothing
deletes a credential, because Hermes stores none.

## What Hermes stores

The state file holds the nickname of the account in use, the nickname each
conversation started on, and a timestamp. It holds no token, no address, no
account number, and no organisation. Hermes writes it at mode `0600`.

Each conversation appears as a fingerprint — 32 hexadecimal characters from a
SHA-256 digest — not as its chat name. The same conversation always gives the
same fingerprint, so a resume still finds its own account, but nothing in the
file names a platform, a chat, or a person. A file written by an earlier
version held readable chat names; Hermes drops those entries when it reads
such a file.

Hermes keeps each account's usage numbers for 60 seconds. A job that starts
several child processes in a row asks the usage endpoint once. A failed read is
never kept, so the next attempt asks again straight away. The numbers are
percentages and reopen times; no token is ever cached.

## Limits

- The usage number is a few seconds old, so a long job can cross a limit in
  the middle of a turn. The stop percentage of 95 leaves room for that. Lower
  it if your jobs are long.
- Two logins on one plan share one limit. A move between them buys nothing.
  Use accounts on separate plans.
- A relative `config_dir` is resolved against the directory Hermes started in,
  once, when it reads `config.yaml`. Write an absolute path or a `~` path if
  that is not what you want.
- Claude Code has its own account switch under `~/.claude/account-vaults/`.
  Hermes never writes there. Pick one method and keep to it. Separate
  directories are the documented, stable one.
