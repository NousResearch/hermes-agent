---
sidebar_position: 18
---

# Photon iMessage

Connect Hermes to iMessage through [Photon][photon] and the Photon Spectrum SDK.

## Architecture

Photon uses a single Hermes-owned adapter boundary:

```text
Photon Spectrum SDK
  -> plugins/platforms/photon/adapter.py
  -> Hermes gateway MessageEvent / SendResult
```

The Python adapter owns Hermes behavior: inbound event normalization, sender
authorization, `MessageEvent` creation, outbound payload construction,
`SendResult` mapping, health/status, and current-home runtime state.

The Spectrum SDK currently runs from a private Node sidecar because the
SDK is TypeScript-based. Hermes starts that sidecar over stdio from
`adapter.py`; it is not a public server and does not expose local or public HTTP
endpoints.

## Prerequisites

- A Photon account at [app.photon.codes][app].
- Node.js 20.18.1 or newer on PATH, or `PHOTON_NODE_BIN` pointing at Node.
- A phone number in E.164 format, such as `+15105550123`.

## First-Time Setup

```bash
hermes photon setup +15105550123
```

Setup always uses the Photon dashboard project name `hermes-agent`. You do not
choose a project name.

Setup will:

- validate or run Photon dashboard login
- find or create the fixed `hermes-agent` project
- store Spectrum project credentials for the current Hermes home
- find or create the operator phone user
- seed `PHOTON_HOME_CHANNEL=any;-;+E164` and
  `PHOTON_HOME_CHANNEL_NAME=You (iMessage)` when unset
- authorize the operator phone for Hermes gateway access
- install sidecar dependencies when needed
- enable `platforms.photon.enabled=true`

`PHOTON_OPERATOR_PHONE` and `PHOTON_ASSIGNED_PHONE_NUMBER` are last-setup/status
metadata. They are not used to route replies and are not the multi-phone source
of truth. Photon replies are sent back to the inbound Spectrum space, while
runtime authorization is controlled by `PHOTON_ALLOWED_USERS` and
`PHOTON_ALLOW_ALL_USERS`.

Setup preserves any existing `PHOTON_HOME_CHANNEL`, so custom cron/proactive
delivery targets are not overwritten. With the seeded operator DM, cron jobs and
other proactive sends can use `deliver=photon`; if cron is running outside the
gateway process, Hermes starts a private send-once sidecar that does not consume
the inbound Spectrum stream.

After setup, start or restart the Hermes gateway. The gateway loads
`plugins/platforms/photon/adapter.py`, and the adapter starts the private SDK
sidecar and subscribes to inbound Spectrum events.

Only one Hermes gateway process can stream a given Photon Spectrum project at a
time. If Photon status reports that the project is already in use, stop the
other gateway first, then start this gateway again. Send-once delivery for
`deliver=photon` does not take this lock because it only sends one outbound
message.

## Commands

```bash
hermes photon login
hermes photon setup +15105550123
hermes photon phones list
hermes photon phones add +15105550124
hermes photon phones remove +15105550124
hermes photon status
hermes photon reset
hermes photon reset --all
```

`hermes photon allow-phone <phone>` remains a low-level local authorization
command. It only edits Hermes sender access and does not create a Photon project
user, so prefer `hermes photon phones add <phone>` for multi-phone setup.

## Phone Management

```bash
hermes photon phones list
hermes photon phones add +15105550124
hermes photon phones remove +15105550124
```

`phones list` shows every active phone/user configured on the fixed
`hermes-agent` Photon project, including submitted phone, assigned
Photon/iMessage number when known, user ID, and Hermes authorization state.

`phones add` creates the Photon project user first, then appends the phone to
`PHOTON_ALLOWED_USERS`. If the phone already exists on the Photon project, the
command fails clearly and does not duplicate local authorization state.

`phones remove` removes the Photon project user first, then removes the phone
from `PHOTON_ALLOWED_USERS`. If the phone does not exist on the Photon project,
the command fails clearly and does not modify local authorization state.

When `PHOTON_ALLOW_ALL_USERS=true`, access remains open even after removing a
phone from `PHOTON_ALLOWED_USERS`; `phones list` and `status` make that visible.

## Status

```bash
hermes photon status
```

Status shows:

- current Hermes home and env path
- dashboard login state
- Spectrum credential validity
- fixed project name and project ID
- operator phone
- assigned Photon/iMessage number when Photon returns one
- home channel and home channel label
- concise phones summary
- sidecar dependency state
- adapter runtime health
- authorized sender state
- next step

The adapter runtime state is stored under the current Hermes home:

```text
<HERMES_HOME>/photon/adapter-runtime.json
```

If status says the adapter is not running after setup, start or restart the
gateway.

## Environment

Setup manages the primary values:

| Variable | Purpose |
| --- | --- |
| `PHOTON_PROJECT_ID` | Spectrum project ID. |
| `PHOTON_PROJECT_SECRET` | Spectrum project secret. |
| `PHOTON_PROJECT_NAME` | Always `hermes-agent` for setup. |
| `PHOTON_OPERATOR_PHONE` | Operator phone passed to setup. |
| `PHOTON_ASSIGNED_PHONE_NUMBER` | Assigned Photon/iMessage number, when known. |
| `PHOTON_ALLOWED_USERS` | Authorized senders. |
| `PHOTON_HOME_CHANNEL` | Default proactive target, usually `any;-;+E164`. |
| `PHOTON_HOME_CHANNEL_NAME` | Human label, seeded as `You (iMessage)`. |

Optional values:

| Variable | Purpose |
| --- | --- |
| `PHOTON_NODE_BIN` | Node binary override. |
| `PHOTON_API_HOST` | Spectrum API host override. |
| `PHOTON_DASHBOARD_HOST` | Dashboard API host override. |
| `PHOTON_ALLOW_ALL_USERS` | Development-only sender allowlist bypass. |

## Reset

```bash
hermes photon reset
hermes photon reset --all
```

`reset` clears local Photon project/runtime identity for the current Hermes
home. `reset --all` also clears the Photon dashboard login token after
confirmation.

Neither command deletes Photon dashboard projects or users.

[photon]: https://photon.codes/
[app]: https://app.photon.codes/
