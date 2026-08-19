---
title: "Stale Override Notices"
description: "Warn or ask before an idle messaging session reuses a custom model or reasoning level."
sidebar_label: "Stale Override Notices"
---

# Stale Override Notices

A session-scoped `/model` or `/reasoning` choice can remain active long after the task that needed it. The stale override notice checks the **first ordinary message after an idle period** and discloses relevant overrides before they are reused accidentally.

The feature is disabled by default.

## Configuration

Add the policy under `gateway.stale_override_notice` in `~/.hermes/config.yaml`:

```yaml
gateway:
  stale_override_notice:
    mode: confirm
    idle_minutes: 60
    model: non_default
    reasoning: above_default
    channels:
      - home
```

Restart the messaging gateway after changing the configuration.

## Modes

| Mode | Behavior |
|---|---|
| `off` | Disable stale override notices. This is the default. |
| `info_only` | Send a notice, then process the user's message immediately with the current overrides. |
| `confirm` | Hold the message and show an interactive choice picker. The user can continue, reset the relevant override, or reset both overrides. If the prompt expires, the message is **not** submitted. |

`confirm` requires a platform adapter with the generic interactive choice-picker capability. If the picker cannot be rendered, Hermes fails open and processes the message rather than losing it. Discord, Matrix, and Telegram support the picker. In a shared session, any user authorized for that chat/topic may choose an action. In a per-user Discord or Telegram session, only the user whose message is being held may use its picker.

## Detection policies

### Model

- `non_default` — notify when an explicit session `/model` route differs from the live route the channel would otherwise use.
- `off` — ignore model overrides.

A route compares both provider and model. Channel model/provider overrides are part of the baseline, so selecting the same route explicitly does not produce a false warning. Provider fallbacks are not treated as user overrides.

### Reasoning

- `above_default` — notify only when the explicit session reasoning effort is higher than the live per-model/global default.
- `non_default` — notify whenever the explicit session effort differs from the default, including lower effort or disabled reasoning.
- `off` — ignore reasoning overrides.

The ordered effort ladder is `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra`.

## Channel scope

Omitting `channels` limits the enabled policy to `home`. Use an explicit empty list to apply it to all messaging channels. Selectors are case-insensitive:

```yaml
channels:
  - home                         # each platform's configured Home destination
  - discord:*                    # every Discord channel
  - discord:123456789            # a chat/channel and all of its threads
  - discord:123456789:987654321  # one exact thread
  - telegram:-1001234567890      # one Telegram chat
  - "*"                          # all channels
```

## Timing and bypasses

- Idle time begins when the platform adapter's delivery lifecycle finishes for the previous user-originated turn, not when the user sent the request or when model generation ended. This completion callback also runs after a failed delivery attempt, so the clock records lifecycle completion rather than guaranteed receipt by the user. A two-hour task does not become stale while it is still running.
- Internal events, slash commands, approvals, and messages steering an active turn bypass this feature.
- A due idle/daily session auto-reset takes precedence; the message follows the normal reset boundary without showing a stale-override prompt.
- The completion clock is stored with session routing metadata and survives gateway restarts.
- Notices are event-driven. Hermes does not run a cron job or periodic reminder.
