---
title: "Directus — Manage Directus 11 collections, items, and access"
sidebar_label: "Directus"
description: "Manage Directus 11 collections, items, and access"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Directus

Manage Directus 11 collections, items, and access.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/web-development/directus` |
| Path | `optional-skills/web-development/directus` |
| Version | `1.0.0` |
| Author | ukr-coder (@ukr-coder) + Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Directus`, `CMS`, `Headless CMS`, `REST API`, `Permissions`, `Policies`, `Agent Governance` |
| Related skills | [`airtable`](/docs/user-guide/skills/bundled/productivity/productivity-airtable), [`notion`](/docs/user-guide/skills/bundled/productivity/productivity-notion), [`har-derived-api-client`](/docs/user-guide/skills/optional/web-development/web-development-har-derived-api-client), [`docker-management`](/docs/user-guide/skills/optional/devops/devops-docker-management) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Directus Skill

Read and write a Directus 11+ instance over its REST API: collections and fields,
rows, and the Directus 11 permission chain (role → `directus_access` → policy →
permissions), including static tokens for service accounts. This skill does not
install or upgrade Directus and never touches its database directly — a table
created behind Directus' back is invisible to `/items` until Directus knows about it.

## When to Use

- "Add a row / update the status / list the last N records" against a Directus project
- "Create a collection for …" and the fields that go with it
- "Why does this token get a 403?" — the policy chain is the thing to inspect
- "Give agent X read access to these collections" — one `bootstrap-agent` run
- A user who mentions Directus, a headless CMS on port 8055, or `directus_access`

Use `airtable` or `notion` for those products instead; use `docker-management`
when the question is about the container rather than the data in it.

## Prerequisites

1. **A reachable Directus 11+ instance.** `directus.url` (skill setting) or
   `DIRECTUS_URL`. Directus 10 is detected and refused for policy commands with the
   reason, because permissions moved in 11.
2. **Credentials, as secrets in `.env`:**

   ```bash
   DIRECTUS_TOKEN=...        # static token from the user's profile — preferred
   DIRECTUS_EMAIL=...        # or an email/password login, exchanged for a temp token
   DIRECTUS_PASSWORD=...
   ```

   Schema and permission commands need an account with an admin policy; item
   commands need only what that account's policies allow.
3. **Python 3.9+**, standard library only. No pip installs, no Directus SDK.

Everything runs through `terminal`. `--url`, `--token`, `--auth-email`,
`--auth-password`, and `--timeout` override the environment on any subcommand.

## How to Run

```bash
S=~/.hermes/skills/web-development/directus/scripts/directus_admin.py
python3 "$S" check
python3 "$S" collections list
python3 "$S" collections create agent_heartbeats --field 'agent_id:string!' --field 'beat_at:timestamp' --field 'payload:json'
python3 "$S" items list agent_tasks --filter '{"status":{"_eq":"queued"}}' --sort=-date_created --limit 10
python3 "$S" items create agent_heartbeats --data '{"agent_id":"dao-07","beat_at":"2026-09-21T10:00:00Z"}'
python3 "$S" bootstrap-agent dao-07 --collections agent_tasks,agent_heartbeats --actions read,create,update --email dao-07@example.com
```

Add `--dry-run` to any mutating command to print the exact endpoint and payload
without sending it — do that first when the user is unsure. Results are JSON on
stdout; `[directus] …` progress and `warning: …` lines go to stderr.

## Quick Reference

| Task | Command |
|------|---------|
| Verify setup | `check` → version, project, identity, policy model |
| Collections | `collections list [--system]`, `collections create NAME --field n:type[!]`, `collections delete NAME --yes` |
| Fields | `fields list C`, `fields create C FIELD --type json [--required]` |
| Read rows | `items list C [--filter JSON] [--fields a,b] [--sort=-x] [--limit N \| --all]` |
| One row | `items get C ID`, `items update C ID --data JSON`, `items delete C ID --yes` |
| Write rows | `items create C --data '{…}'` (a JSON array inserts a batch) |
| Policies | `policies list`, `policies create NAME [--app] [--admin]` |
| Permissions | `permissions list [--policy ID \| --collection C]`, `permissions create --policy ID --collection C --action read [--fields '*'] [--rule JSON]` |
| Roles / access | `roles create NAME`, `access grant --policy ID --role ID`, `access list --role ID`, `access revoke ID --yes` |
| Service account | `users create --email E --password P --role ID`, `token set USER-ID`, `token clear USER-ID --yes` |
| One-shot agent | `bootstrap-agent NAME --collections a,b --actions read,create [--email E] [--rule JSON]` |

`--data`, `--filter`, `--rule`, and `--validation` all accept inline JSON, `@file`,
or `-` for stdin — use `@file` for anything long rather than quoting it in the shell.
A descending sort starts with a minus, which needs the `=` form: `--sort=-date_created`.

## Procedure

1. **`check` first, every session.** It reports the version, the authenticated
   account, and whether that account has admin access. A non-admin token cannot
   create collections or permissions; say so instead of retrying.
2. **Look before writing.** `collections list` and `fields list <collection>` show
   what exists; field names are case-sensitive and a typo surfaces as HTTP 422.
3. **Filter server-side.** Pass `--filter` rather than listing everything and
   narrowing afterwards; `--all` pages through large result sets in blocks of 100.
4. **Granting access is four objects, not one.** Policy → permissions → role →
   `access` link. `bootstrap-agent` does the whole chain; do it by hand only when
   reusing an existing role or policy. Read
   `references/directus-11-permissions.md` before debugging a 403.
5. **Static tokens are shown once.** `token set` prints the value in its JSON
   result; hand it to the user immediately and tell them it goes in their agent's
   `.env` as `DIRECTUS_TOKEN`. It cannot be read back.
6. **Destructive commands need `--yes`.** Confirm with the user in the same turn —
   `collections delete` drops the table and every row with it.
7. **Report what changed**, with ids: a created policy id is what the next command
   needs, and the user cannot see the JSON unless it is quoted back to them.

## Pitfalls

- **A role grants nothing on its own.** In Directus 11 permissions live on a
  policy; the role only matters because `directus_access` links it to one. Creating
  a role and stopping there is the most common "it still returns 403".
- **`fields: []` means the primary key only**, not every field. That is why a read
  can succeed and return rows with just an `id`. `permissions create` defaults to
  `*` and warns on an explicitly empty `--fields`.
- **`admin_access` on a policy bypasses every permission row.** Never give it to an
  agent service account; `--app` (Studio sign-in) is also unnecessary for API use.
- **A static token cannot be set with SQL.** Directus writes it through the user
  endpoint; a direct `UPDATE` on `directus_users.token` leaves a value that never
  authenticates.
- **Tables created with raw DDL are invisible** to `/items` until Directus has a
  `directus_collections` row for them. Create through `collections create` instead.
- **Directus 10 is a different product here.** `check` reports the model and the
  policy commands refuse with the version rather than returning a confusing 404.
- **Remote instances mean the data leaves the machine.** If `directus.url` is not
  localhost, say so when the user asks to bulk-export a collection.
- **429 and 5xx retry three times** with a 1/3/6 s back-off; a persistent 429 means
  the instance is rate-limited, so slow down instead of looping.

## Verification

- `check` prints `"directus": "ok"`, a version starting with `11` or higher, and
  the account's email.
- `collections create` followed by `fields list <collection>` shows the primary key
  plus every field that was passed with `--field`.
- After `bootstrap-agent`, `access list --role <role-id>` returns one link and
  `permissions list --policy <policy-id>` returns one row per collection × action.
- The printed static token works: `check --token <value>` authenticates as the
  service user and reports `admin_access: false`.
- Tests: `scripts/run_tests.sh tests/skills/test_directus_skill.py -q`.
