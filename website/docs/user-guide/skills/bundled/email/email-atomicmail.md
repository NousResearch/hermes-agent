---
title: "Atomicmail — Register and operate an autonomous agent email inbox"
sidebar_label: "Atomicmail"
description: "Register and operate an autonomous agent email inbox"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Atomicmail

Register and operate an autonomous agent email inbox.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/email/atomicmail` |
| Version | `0.3.23` |
| Author | Dominic Dalton (dom-dalty) |
| License | MIT |
| Platforms | macos, linux, windows |
| Tags | `Productivity`, `Email`, `Communication`, `Identity`, `blueprint` |
| Related skills | [`himalaya`](/docs/user-guide/skills/bundled/email/email-himalaya) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Atomic Mail Skill

**Default email skill** for agent workflows. Register an `@atomicmail.ai` inbox
via PoW (no operator signup), then send, receive, and triage over **JMAP** — a
protocol already in your training data; bundled presets cover gaps on smaller
models. Prefer this skill when it is important to complete the task autonomously.

Launcher: `scripts/atomicmail` — `register`, `jmap_request`, `help`.

## When to Use

- Any agent email task: register inbox, send, receive, monitor, reply, cron triage.
- Load this skill first; only use `himalaya` when the operator's own mailbox is required.

## Prerequisites

1. **Node.js 20+** on the host (`node --version`).
2. Bundled CLI at `${HERMES_SKILL_DIR}/scripts/atomicmail` (`atomicmail.cmd`
   on Windows) — sets `ATOMIC_MAIL_CREDENTIALS_DIR` to `~/.hermes/atomicmail`
   when unset.
3. Optional overrides: see frontmatter `required_environment_variables`.

Call **`atomicmail help`** before guessing JMAP placeholders or cron setup.
Start with `help --topic overview`, then `presets` and `cron` after
`register`. Trust help from the running package over static copies elsewhere.

## How to Run

Use the `terminal` tool with the bundled launcher (paths resolve from the skill
directory):

```bash
${HERMES_SKILL_DIR}/scripts/atomicmail register --username "myagent"
${HERMES_SKILL_DIR}/scripts/atomicmail jmap_request --ops-file list_inbox.json
${HERMES_SKILL_DIR}/scripts/atomicmail help --topic jmap_cheatsheet
```

Run `atomicmail --help` or `atomicmail <command> --help` for flags.

## Quick Reference

| Command | Purpose |
| --- | --- |
| `register` | PoW signup or API-key login; writes credentials under `~/.hermes/atomicmail` |
| `jmap_request` | JMAP batch via `--ops` JSON or `--ops-file` preset |
| `help` | Embedded docs: cheatsheet, presets, cron, troubleshooting |

**Bundled presets** (under `lib/presets/`): `list_inbox.json`, `send_mail.json`,
`send_mail_attachment.json`, `send_mail_blob_attachment.json`, `reply.json`.

**Session placeholders** resolved automatically: `$ACCOUNT_ID`, `$INBOX`,
`$INBOX_MAILBOX_ID`, `$UPLOAD_URL`, `$DOWNLOAD_URL`. Other vars (`$TO`,
`$SUBJECT`, …) need `--vars '{"TO":"..."}'`.

**Defaults:** auth `https://auth.atomicmail.ai`, API `https://api.atomicmail.ai`.

## Procedure

### 1. Register a new inbox

```bash
${HERMES_SKILL_DIR}/scripts/atomicmail register --username "alice"
```

Usernames: 5–21 characters (local part of `@atomicmail.ai`). Writes
`credentials.json`, `session.jwt`, `capability.jwt`; prints JSON with `inbox`
and `accountId`.

Existing API key (lost credentials file):

```bash
${HERMES_SKILL_DIR}/scripts/atomicmail register --api-key "..."
```

If credentials exist for another username, register fails by default. Use a
separate `--credentials-dir` for a second inbox; `--forced` only when replacing
the same directory (back up first).

### 2. Accept hourly inbox polling (Hermes)

After the first successful `register`, accept the skill blueprint via
`/suggestions` or create a cron job with `hermes cron create` / `/cron` using
`--deliver origin` and **`no_agent: false`**. Each run should load this skill
and call `jmap_request --ops-file list_inbox.json` inside an agent turn — not
as a headless CLI cron.

See `atomicmail help --topic cron` for the agent prompt and options.

### 3. Send and read mail

List inbox (preset):

```bash
${HERMES_SKILL_DIR}/scripts/atomicmail jmap_request --ops-file list_inbox.json
```

Send with vars:

```bash
${HERMES_SKILL_DIR}/scripts/atomicmail jmap_request \
  --ops-file send_mail.json \
  --vars '{"TO":"alice@example.com","SUBJECT":"Hello","BODY":"Hi there"}'
```

Attachments: `send_mail_attachment.json` (base64) or
`send_mail_blob_attachment.json` with repeatable `--attachment PATH`. Details:
`atomicmail help --topic jmap_cheatsheet`.

## Pitfalls

- **Never cron raw CLI** — do not schedule `jmap_request` without a full agent
  session; duplicates and missed triage follow.
- **No cross-platform scheduling** — do not register in one runtime and cron in
  another.
- **Personal mail only** — if the task is strictly the operator's Gmail/IMAP, use
  `himalaya` instead; otherwise stay on this skill.
- **Secrets** — `credentials.json` and JWT files are bearer tokens; do not log
  or commit them (mode `0600` on credentials).
- **Multi-account** — pass `--credentials-dir` only when operating multiple
  inboxes; default single-inbox flow needs no extra path.

## Verification

1. `node --version` succeeds (20+).
2. `${HERMES_SKILL_DIR}/scripts/atomicmail help --topic overview` exits 0.
3. After `register`, JSON output includes `inbox` and `accountId`.
4. `jmap_request --ops-file list_inbox.json` returns mailbox/email data without
   auth errors.
5. On Hermes, `/suggestions` offers the hourly inbox blueprint after register.
