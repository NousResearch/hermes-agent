---
title: Credential Scope Contract
sidebar_label: Credential Scope
---

# Credential scope: request-bound opaque credential identity

Hermes may hold **several credentials for one provider** (credential pools).
A downstream observer that tracks quota exhaustion, per-account health, or
authentication failures must attribute an outcome to the credential that
actually served that request — never to the provider, model, or profile as a
whole. This page documents the supported, plugin-facing contract for that
attribution. It is **observational only**: it never changes which credential
Hermes selects, and it never exposes credential material.

## The contract

The main-loop API hooks (`pre_api_request`, `post_api_request`,
`api_request_error`) carry a `credential_scope` object:

```python
{
    "provider_id": "nvidia",          # provider that served the attempt
    "credential_scope_id": "9f2c…",   # opaque id, or None
    "identity_status": "known",       # "known" | "unknown" | "unavailable"
}
```

| Field | Meaning |
| --- | --- |
| `provider_id` | The provider of the attempt (same value as the hook's `provider`). |
| `credential_scope_id` | Host-owned opaque identifier of the selected pool entry, or `None`. |
| `identity_status` | `known` (id present), `unknown` (a credential was used but no pool entry could be attributed), `unavailable` (no credential in play — e.g. a keyless local provider). |

`identity_status` is never invented: when Hermes cannot attribute the attempt
to a pool entry, the id is `None`. Consumers must key quota/health state by
`(provider_id, credential_scope_id)` and treat `unknown`/`unavailable` as
provider-wide observations, not as a guessable account.

## Identity lifecycle

- **Per pool entry.** Every `PooledCredential` owns a random `scope_id`
  (32 hex chars) minted independently of the token — never a hash or
  fingerprint of the secret. It cannot authenticate anything and cannot be
  reversed to the credential.
- **Stable across ordinary requests and token refreshes.** Refreshes replace
  the entry's tokens in place (same entry id), so the scope survives them.
- **Replacement mints a new scope.** Removing and re-adding a credential
  creates a new entry with a new scope id; a failure recorded against the old
  scope never contaminates the new one.
- **Distinct per entry, never a pool index.** Scopes are not positional, so
  rotation and removal cannot recycle an id onto a different credential.
- **Persisted with the pool row**, so the scope is stable across processes
  after the first pool write (legacy rows are assigned one on load).

## Attempt-level attribution

Attribution is **per attempt**, correlated by the existing `api_request_id`:

```
build request  → snapshot scope (pre_api_request)
dispatch       → provider call on that credential
post / error   → same snapshot reported (post_api_request / api_request_error)
```

If the attempt fails and Hermes rotates to another credential, the failed
attempt's error hook still reports the first credential's scope; the retry
builds a new snapshot for the new credential. The snapshot lives on the agent
instance — there is no process-global credential context, so concurrent
agents cannot exchange scopes.

## Permission model

`credential_scope` is gated by the **`llm.credential_scope`** plugin
capability (consent-only; no legacy `allow_*` gate). A plugin declares it in
its manifest and the user grants it through the normal capability-consent
flow, exactly like `llm.side_question`:

```yaml
capabilities:
  - llm.credential_scope
```

Hook dispatch strips `credential_scope` from the payload of every callback
whose owning plugin was not granted the capability (callbacks registered
outside the plugin system are fail-closed). First-party observability
consumers are unaffected. Plugins that do not declare the capability keep
working unchanged — they simply never see the field. Probe with
`ctx.has_capability("llm.credential_scope")` and degrade gracefully.

## Quota observation (llm.quota_observation)

`post_api_request` and `api_request_error` also carry a `quota_observation`
object, gated by the **`llm.quota_observation`** capability (consent-only;
an independent grant from `llm.credential_scope`):

```python
{
    "schema_version": 1,
    "provider_id": "openrouter",
    "credential_scope_id": "9f2c…",   # same opaque id as credential_scope
    "identity_status": "known",
    "observed_at": 1756000001.234,
    "outcome": "ok",                   # ok | auth_failure | rate_limited | provider_error | timeout | unknown
    "status_code": 200,                # or None
    "retry_after_seconds": None,       # parsed Retry-After, or None (never "no wait")
    "rate_limit": {                    # only windows the provider actually sent
        "requests_min": {"limit": 100, "remaining": 87, "reset_seconds": 42.0},
    },                                 # None when no x-ratelimit-* headers were present
}
```

Contract rules:

- **No invented verdicts.** A generic 200 is not proof of remaining allowance;
  a 429 maps to `rate_limited`, never to "monthly allowance exhausted" (the
  taxonomy deliberately has no `quota_exhausted` class); `Retry-After` and
  `x-ratelimit-*` are reported exactly as sent — missing windows are absent,
  never zero-filled.
- **Attempt-bound.** The observation is built from the same snapshot as
  `credential_scope`, so retry rotation attributes outcomes to the credential
  that served the attempt.
- **Non-secret.** No keys, tokens or authorization headers; ids match
  `credential_scope`.

## ctx.quota_fetch (OpenRouter official key limits)

Plugins granted `llm.quota_observation` may call:

```python
result = ctx.quota_fetch(
    provider_id="openrouter",
    credential_scope_id="9f2c…",        # from an observed scope
    quota_source="official_key_limits", # the only implemented source
    timeout=2.0,                        # clamped to <= 2 s
)
```

- The host resolves the credential for **that scope's pool entry** and calls
  the documented key-info endpoint; the credential never leaves the host and
  never appears in the result.
- Success: `{"status": "ok", "provider_id": "openrouter",
  "credential_scope_id": "9f2c…", "quota_source": "official_key_limits",
  "observed_at": …, "expires_at": …, "limit": int, "remaining": int,
  "used": int}` with `remaining + used == limit`.
- Anything else — capability not granted, unsupported provider/source,
  unknown or rotated scope, provider error, timeout, off-contract response
  shape — returns `{"status": "unavailable", "reason": …}`. Fail-closed by
  contract: no quota is ever guessed.
- Other providers have no authoritative quota endpoint and report
  `unsupported_provider`; no endpoint is invented.
- Plugins that do not declare the capability keep working unchanged — calls
  return `unavailable` without touching the network.

## Scope of the contract

Auxiliary calls (`pre_auxiliary_call` / `post_auxiliary_call`) do not carry
`credential_scope` or `quota_observation` today; the contract covers the main
agent loop.

## Security properties

- No API keys, tokens, refresh tokens, cookies, or authorization headers.
- No secret-derived hashes, prefixes, suffixes, or lengths.
- No new credential-reading endpoint and no accessor on `PluginContext`;
  the scope arrives only as part of the three existing hook payloads.
- Scope ids are random and unusable for authentication.

## Example

See `plugins/credential-scope-example/` — a minimal plugin that records
per-scope attempt outcomes from the three hooks and degrades to a no-op when
the capability is not granted.
