---
sidebar_position: 4
title: "Remote Config"
description: "Serve each profile's config from the config plane instead of config.yaml"
---

# Remote Config

**Remote config mode** replaces `config.yaml` with the config plane: each
profile's config is the document the plane resolves for this agent instance
and that profile. The plane layers it from global, tenant, group, owner, and
agent settings, with the profile's own settings last. Administrators can lock
keys at an upper level, and the agent cannot change a locked key.

Remote config is for fleet deployments, for example Hermes Cloud. A single
machine with one user should keep `config.yaml`.

## Turning it on

Set the following in the environment or in `~/.hermes/.env`. A value in
`config.yaml` cannot select the backend, because the backend is what serves
config.

| Variable | Meaning |
|---|---|
| `HERMES_CONFIG_BACKEND=remote` | Use the config plane instead of `config.yaml` (default `file`). |
| `HERMES_CONFIG_INSTANCE_ID` | This agent's id in the plane. Hermes Cloud sets it; on-prem the operator sets it when registering the agent. Required. |
| `HERMES_CONFIG_REMOTE_URL` | The plane's base URL (default `https://config-config.nousresearch.com`). |
| `HERMES_CONFIG_REMOTE_POLL_SECONDS` | How often each process checks for changes (default `300`, minimum `30`). |

The agent authenticates as itself:

- **Hermes Cloud**: the Nous access token in the profile's `auth.json`.
- **On-prem**: an OAuth2 client-credentials token from your identity provider,
  configured in `.env` with `GATEWAY_RELAY_IDP_TOKEN_URL`,
  `GATEWAY_RELAY_IDP_CLIENT_ID`, `GATEWAY_RELAY_IDP_CLIENT_SECRET`, and
  optionally `GATEWAY_RELAY_IDP_SCOPE`.

These values, and the `HERMES_CONFIG_*` variables above, must come from
`auth.json`, `.env`, or the process environment. If a `secrets:` source
(1Password, Bitwarden, a command) supplies any of them, Hermes refuses to
start.

## What changes

- **Startup fetches config and fails closed.** Every Hermes process (gateway,
  dashboard, CLI, terminal children) fetches its profile's config when it
  loads `.env`. If the plane cannot be reached after about 30 seconds of
  retries, or refuses the agent, the process exits with an error. It never
  falls back to a local file or to defaults.
- **No local config file.** `config.yaml` is neither read nor written, and
  there is no local cache. A running process keeps the config it last fetched
  in memory. If the plane becomes unreachable while Hermes runs, the process
  keeps its in-memory config and retries on the next poll.
- **Changes arrive by polling.** Each process checks for changes every
  `HERMES_CONFIG_REMOTE_POLL_SECONDS`. A change made in the plane reaches a
  running agent within one interval.
- **Writes go to the profile's own level.** `hermes config set`, `/model`,
  the dashboard, and every other config writer send only the keys that changed,
  guarded against concurrent edits. Removing a key removes it from the profile
  level only: if an upper level also sets it, that value still applies, and
  Hermes warns you.
- **Locked keys are refused.** `hermes config set` on a locked key prints which
  level locks it and changes nothing. When a whole document is saved, locked
  keys are left out, and Hermes prints a note listing them.
- **Secrets stay out of the plane.** A secret-shaped key such as `api_key`
  accepts only a `${VAR}` reference, never the secret itself. Put the value in
  `.env` or a secret source and set the reference:

  ```bash
  hermes config set model.api_key '${OPENROUTER_API_KEY}'
  ```

- **`secrets:` comes from the plane** like any other key, so administrators can
  manage and lock secret-source settings centrally.
- **Schema migrations run in memory.** A document written by an older Hermes is
  migrated when it is read, and the result is not written back.
- **Unknown keys** in the fetched document are ignored with a warning, for
  example a key added by a newer Hermes.

## Managed scope

`/etc/hermes/config.yaml` (see [Managed Scope](./managed-scope.md)) is
**ignored** in remote mode: config and locks come only from the plane. Hermes
warns at startup, and `hermes doctor` flags the file. The managed
`/etc/hermes/.env` still applies.

## Commands that work on the config file

These commands copy or edit `config.yaml` and are refused in remote mode:

- `hermes config edit` (use `hermes config set` instead)
- profile clone and profile distribution install
- `hermes backup`, `hermes import`, and snapshot restore
- `hermes gateway --config <file>`

`hermes doctor` reports the backend instead of the file: the plane URL, the
profile, the profile level's version, the number of locks, and when the config
was last fetched.
