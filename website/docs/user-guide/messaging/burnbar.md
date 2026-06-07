---
sidebar_position: 9
title: "BurnBar Cloud"
description: "Connect Hermes Agent to BurnBar Cloud through the BurnBar Hermes Gateway API"
---

# BurnBar Cloud Setup

BurnBar Cloud connects to Hermes through the BurnBar Hermes Gateway API. Hermes
runs the gateway adapter locally; BurnBar provides the mobile and cloud surface
that sends events to the adapter and receives replies.

Use this integration when you want BurnBar to act as a mobile control surface for
a local Hermes agent: send messages, receive replies, approve sensitive actions,
and route scheduled `deliver=burnbar` notifications to a home destination.

## Capabilities

| Surface | Behavior |
|---------|----------|
| **Inbound events** | Hermes polls BurnBar `/events` with a durable cursor. |
| **Replies** | Hermes sends through `/messages`. |
| **Typing** | Hermes publishes typing state through `/typing`. |
| **Attachments** | Hermes initializes signed uploads through `/attachments/init`. |
| **Oversight** | Supervised mode gates slash confirmations; autonomous mode auto-approves. |
| **Cron delivery** | `deliver=burnbar` sends to `BURNBAR_HOME_CHANNEL`. |

## Setup

Run the gateway setup wizard and choose **BurnBar Cloud**:

```bash
hermes gateway setup
hermes gateway restart
hermes gateway status
```

The setup flow starts a device-code grant. Approve the displayed code in BurnBar,
then restart the gateway. The approved token is written to your active Hermes
home `.env`.

## Configuration

The setup flow writes the required token automatically:

| Variable | Purpose |
|----------|---------|
| `BURNBAR_ACCESS_TOKEN` | Scoped bearer token from the approved device grant. |
| `BURNBAR_API_BASE_URL` | Gateway API base URL. Defaults to the BurnBar Cloud API. |
| `BURNBAR_HOME_CHANNEL` | Default destination for cron and notification delivery. |
| `BURNBAR_HOME_CHANNEL_NAME` | Human-readable label for the home destination. |
| `BURNBAR_ALLOWED_USERS` | Comma-separated BurnBar sender IDs allowed to reach Hermes. |
| `BURNBAR_ALLOW_ALL_USERS` | Set to `true` only for a trusted account/workspace. |

You can also set non-secret defaults in `config.yaml` under the BurnBar platform
entry. Secrets belong in `.env`.

## Access Control

BurnBar uses the same gateway access-control model as other Hermes messaging
platforms:

- set `BURNBAR_ALLOWED_USERS` for an explicit sender allowlist
- set `BURNBAR_ALLOW_ALL_USERS=true` only when every sender in the connected
  BurnBar account is trusted
- use `/whoami` from BurnBar to confirm the active scope and command access

## Oversight Mode

BurnBar's server-owned `/state` response controls the current oversight mode:

- `supervised` arms a phone approval gate before slash-confirm actions proceed
- `autonomous` allows the adapter to auto-approve those actions

The adapter refreshes this state while polling, so the phone remains the control
surface for changing the mode.

## Troubleshooting

Check the gateway status first:

```bash
hermes gateway status
hermes logs --level warning
```

Common causes:

- `BURNBAR_ACCESS_TOKEN` is missing or expired: rerun `hermes gateway setup`
- the home channel is unset: send from BurnBar once or set `BURNBAR_HOME_CHANNEL`
- the sender is denied: add the sender to `BURNBAR_ALLOWED_USERS` or explicitly
  enable `BURNBAR_ALLOW_ALL_USERS`

## Tests

The plugin tests load the adapter through the same plugin-loader guard used by
the Hermes gateway tests:

```bash
scripts/run_tests.sh tests/gateway/test_burnbar_plugin.py
```
