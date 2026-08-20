---
sidebar_position: 14
title: "Contact Route Registry"
description: "Resolve people to purpose-specific outbound routes without conflating identity, reachability, or authorization"
---

# Contact Route Registry

Hermes can discover reachable messaging destinations in `channel_directory.json`, and each gateway has its own authorization controls. Neither answers a different question: **which verified route should Hermes use to contact a particular person for a particular purpose?**

The profile-scoped contact registry provides that missing user-owned layer:

```text
person → purpose-specific preferred route → current destination
```

Route resolution never sends a message and never grants messaging authority.

## Keep the layers separate

| Layer | Meaning |
|---|---|
| `contacts.yaml` | Durable identity evidence and the user's route preferences |
| `channel_directory.json` | Generated cache of destinations currently reachable by configured gateways |
| Platform allowlists | Authorization to receive from or interact with particular users/chats |
| Session history | Prior conversation evidence |

A destination appearing in the generated directory does not prove who owns it or that it is preferred. An allowlist entry grants access; it is not an address book. A resolved route still requires a separate authority check before sending.

## Initialize the registry

```bash
hermes contacts init
hermes contacts path
```

The default path is:

```text
$HERMES_HOME/contacts.yaml
```

On POSIX systems, Hermes creates the file with owner-only `0600` permissions. `init` never overwrites an existing registry.

## Registry format

```yaml
schema_version: 1
policy:
  default_send: deny

contacts:
  - id: alice-example
    display_name: Alice Example
    aliases: [Alice]
    routes:
      - key: discord-dm
        platform: discord
        destination_type: dm
        destination: "discord-user-id"
        preferred_for: [internal]
        status: verified
        sendable: true
        last_verified: 2026-01-02
        constraints:
          - Verify messaging authority before sending.
```

Extra fields are permitted so a registry can retain provenance, identity evidence, and platform-specific notes. Required identifiers, aliases, route keys, route status, and purpose lists are validated.

Hermes rejects:

- duplicate contact IDs;
- an alias/name that maps to multiple contacts;
- duplicate route keys within one contact;
- malformed route fields, including a non-boolean `sendable` value;
- unsupported route states.

Supported route states are `verified`, `unverified`, and `stale`.

## Validate and inspect

```bash
hermes contacts validate
hermes contacts list
hermes contacts show "Alice"
```

Endpoint values are hidden from `list` and `show` by default. Include them only when needed:

```bash
hermes contacts show "Alice" --show-destinations
```

## Resolve without sending

Resolve by purpose:

```bash
hermes contacts resolve "Alice" --purpose internal
```

Or select a known route key explicitly:

```bash
hermes contacts resolve "Alice" --route discord-dm
```

Destination values remain hidden unless `--show-destination` is passed. Machine-readable output includes `send_performed: false` and `authorization_check: required`.

The resolver fails closed for cases such as:

- `unknown_contact`
- `ambiguous_contact`
- `route_selector_required`
- `no_preferred_route`
- `ambiguous_route`
- `stale_destination`
- `unverified_destination`
- `not_sendable`
- `stale_channel_directory`
- `destination_not_in_live_directory`
- `live_check_unavailable`

For any gateway or plugin platform represented in `channel_directory.json`, the selected destination must appear in a directory refreshed within the last ten minutes. The gateway normally refreshes it every five minutes; an older or malformed cache returns `stale_channel_directory` even if the destination still appears in the file. This cache check is reachability evidence, not proof that a gateway remains active, so the actual send path must still revalidate its adapter.

Platforms such as email require their own account/mailbox verification and are excluded from directory checks. Until a caller performs that check, the resolver returns `live_check_unavailable` rather than silently reporting a send-ready result.

## Before sending

A successful resolution means only that one source-backed route matched. Before any outbound action:

1. Confirm the communication itself is authorized.
2. Perform any platform-specific live identity or membership check not covered by the generated directory.
3. Send through Hermes's existing messaging or mailbox surface.
4. Read back the actual destination, sender identity, message ID, and content when the platform permits.

Do not automatically merge identities from similar display names, promote every discovered endpoint into the registry, or infer a preferred route merely because a destination exists.
