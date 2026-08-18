# Bot Mode and Group/User Routing Diagnostics (PR2 design spec)

> **Status:** design-only follow-up to the profile composition doctor. This
> document defines a future diagnostic surface; PR1 does not implement it.

## Problem

Bot Mode can multiplex several profiles across group and direct-user
conversations. When a message is routed to an unexpected profile, operators
need an explainable, read-only diagnostic that distinguishes configuration,
identity, and runtime routing decisions without exposing conversation content.

## Proposed interface

Extend `hermes doctor` in a future PR with an explicitly opt-in routing mode,
for example:

```text
hermes doctor --routing --profile NAME [--group GROUP_ID] [--user USER_ID]
hermes doctor --routing --all-profiles --json
```

The exact flag names remain subject to CLI review. The report should include,
for each evaluated profile:

- whether Bot Mode/group/user routing is enabled;
- configured platform and gateway presence (not credentials);
- normalized, redacted route-match dimensions (platform, group/user scope,
  allow/deny outcome, precedence class);
- selected profile and a bounded reason code (for example
  `explicit-profile`, `group-rule`, `user-rule`, `default-fallback`, or
  `no-match`);
- configuration errors and actionable remediation hints;
- an evaluation timestamp only when requested by a human-readable mode (JSON
  should otherwise be deterministic).

The implementation must use the same profile/config discovery APIs as PR1,
avoid changing active-profile state, and never start a gateway or send a
platform request. IDs should be hashed or partially redacted unless the
operator explicitly supplies a safe local diagnostic mode.

## Non-goals

- No changes to Bot Mode routing semantics, precedence, or authorization.
- No automatic repair, profile mutation, gateway restart, or message delivery.
- No transcript, message body, attachment, contact name, token, API key,
  cookie, OAuth credential, or raw `.env` content in output.
- No live network probes or platform API calls.
- No speculative routing plugin hooks or new user-facing environment variables.
- No replacement of the existing full `hermes doctor` checks.
- No snapshot tests tied to the number or order of platforms; behavior tests
  should assert routing invariants and redaction.

## Acceptance criteria

1. Existing `hermes doctor` with no new routing flags is byte-for-byte behavior
   compatible, including read-only semantics.
2. Routing diagnostics run against a temporary `HERMES_HOME` in tests using
   real imports and representative group/user configuration.
3. Given the same filesystem/config inputs, JSON output is deterministic,
   JSON-serializable, and stably ordered.
4. Tests prove that secrets and message content cannot appear in either text
   or JSON output, including values in malformed or unexpected config fields.
5. Tests cover explicit profile selection, group match, user match,
   precedence conflicts, default fallback, no-match, missing profile, and
   malformed routing configuration.
6. The command performs no writes and does not acquire gateway/network
   side-effects; this is verified with filesystem and network guards.
7. CLI reference documentation defines examples, redaction guarantees, and
   the distinction between static routing diagnostics and live gateway logs.
8. A reviewer can implement the feature without changing PR1's profile
   composition report contract.
