---
sidebar_position: 1
title: "Account Usage Presence"
description: "Show one provider account's remaining usage on Telegram and Discord bot identity surfaces"
---

# Account Usage Presence (Experimental)

Account-usage presence exposes one provider account's remaining usage on messaging
surfaces without sending a chat message. Hermes fetches the same
provider-neutral account usage data used by `/usage`, then fans one snapshot
out to every selected platform.

| Platform | Surface |
|---|---|
| Telegram | Default-language bot display name, for example `Hermes · Session 75%` |
| Discord | Bot activity, for example `Watching Five hour 75% remaining` |

The feature is **off by default** because these are global bot identity writes,
visible to every user of that bot rather than one conversation.

## Configure

You can enable it from `hermes setup gateway` (after selecting messaging
platforms), or add an explicit provider and platform allowlist to
`~/.hermes/config.yaml`:

```yaml
gateway:
  account_usage_presence:
    enabled: true
    provider: openai-codex
    platforms:
      - telegram
      - discord
    update_interval_seconds: 300
    stale_after_seconds: 900
    # Optional: select a provider window by its displayed label.
    # Without this, Hermes uses the first window with a percentage.
    window_label: Session
```

`provider` is fixed for the global identity. It does not follow whichever model
a conversation used most recently, so separate chats cannot make the bot name
or activity oscillate between providers. Aliases such as `auto` and `main` do
not select an explicit account and therefore leave the feature disabled. Only
`openai-codex`, `anthropic`, and `openrouter` are accepted; the configured
account must already be authenticated for `/usage` to return a percentage
window. Only `telegram` and `discord` are accepted platforms in this version.

The minimum update interval is five minutes. Hermes also performs change-only
writes: an unchanged rendered identity does not call the platform API again.
When a stale cached value is still within `stale_after_seconds`, platforms show
an explicit `(cached)` marker instead of presenting it as live.

## Failure and restoration behavior

- Provider and platform failures are **fail-open for messaging**: chat delivery
  continues even if account usage cannot be fetched or displayed.
- A transient fetch failure keeps the last live value only until
  `stale_after_seconds`. After that, the surface says usage is unavailable
  instead of presenting a stale percentage as live.
- Provider `Retry-After` metadata is honored separately from platform write
  backoff.
- Telegram ownership uses a private journal at
  `~/.hermes/state/account-usage-presence/journal.json`. Before each mutation
  Hermes records both the original baseline and the feature-owned value. On
  restore it re-reads the remote name and only writes the baseline when the
  current remote value still matches the owned value (CAS). External renames
  are preserved and the journal entry is retired.
- Unsafe or malformed journals fail closed for identity mutation until an
  operator inspects the file. Symlinks and non-regular files are rejected.
- If Telegram was offline when the feature was disabled, recovery retries after
  adapter reconnect instead of running only once at startup.
- Discord activity is connection-scoped and is not journaled. Hermes does not
  clear an activity it does not own; normal disconnect drops the presence.
- Telegram's optional `status_indicator` uses the short description, so it can
  run at the same time as account-usage presence without competing for the name.
- `gateway.multiplex_profiles` is not supported for live updates in this
  experimental version. Hermes logs a warning, performs only saved-identity
  recovery for each profile journal, and does not start account-usage updates.

## Disable

Set `enabled: false` (or remove the section) and restart the gateway. If a
Telegram ownership journal still exists, Hermes restores the original name after
the adapter reconnects. Do not delete the journal while a mutated Telegram name
still needs restoration.
