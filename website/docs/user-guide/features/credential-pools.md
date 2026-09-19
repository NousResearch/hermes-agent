---
title: Credential Pools
description: Pool multiple API keys or OAuth tokens per provider for automatic rotation and rate limit recovery.
sidebar_label: Credential Pools
sidebar_position: 9
---

# Credential Pools

Credential pools let you register multiple API keys or OAuth tokens for the same provider. When one key hits a rate limit or billing quota, Hermes automatically rotates to the next healthy key — keeping your session alive without switching providers.

This is different from [fallback providers](./fallback-providers.md), which switch to a *different* provider entirely. Credential pools are same-provider rotation; fallback providers are cross-provider failover. Pools are tried first — if all pool keys are exhausted, *then* the fallback provider activates.

:::warning Key rotation resets the prompt cache
Provider-side prompt caches (Anthropic, OpenAI, OpenRouter) are scoped to the account/API key that made the request. When the pool rotates to a different key mid-session, the new key has no cached prefix for your conversation — the next request re-reads the full history at undiscounted input price, and rotating back later is another full re-read unless the earlier key's cache TTL is still alive. Rotation keeps your session running, which is the point, but on long conversations each rotation costs one full-price pass over the context.
:::

:::tip
Credential pools are mainly for API-key providers (OpenRouter, Anthropic). A single [Nous Portal](../../integrations/nous-portal.md) OAuth covers 300+ models, so most users don't need a pool when on Portal.
:::

## How It Works

```
Your request
  → Pick key from pool (round_robin / least_used / fill_first / random)
  → Send to provider
  → 429 rate limit?
      → Plan/usage limit reached (e.g. ChatGPT/Codex "usage limit reached")?
          → Rotate to next pool key immediately (no retry — the cap won't clear on retry)
      → Generic / transient 429?
          → Retry same key once (transient blip)
          → Second 429 → rotate to next pool key
      → All keys exhausted → fallback_model (different provider)
  → 402 billing error?
      → Immediately rotate to next pool key (1h cooldown)
  → 401 auth expired?
      → Try refreshing the token (OAuth)
      → Refresh failed → rotate to next pool key
  → 400 "model is not supported when using Codex with a ChatGPT account"?
      → Bench this key for that model only, rotate to the next key (other models stay usable)
      → Every key rejects the model → fallback_model; the model is skipped for the session
  → HTTP 200 but `response.status: failed` (ChatGPT/Codex reports usage limits this way)?
      → Same rules as above, keyed on the embedded error code/message:
        quota/billing/auth → pool rotation first, provider fallback only once the pool is exhausted;
        content-policy and other failures → no rotation
  → Success → continue normally
```

## Quick Start

If you already have an API key set in `.env`, Hermes auto-discovers it as a 1-key pool. To benefit from pooling, add more keys:

```bash
# Add a second OpenRouter key
hermes auth add openrouter --api-key sk-or-v1-your-second-key

# ...or let a browser login mint one (OpenRouter OAuth PKCE; stored as a plain API key)
hermes auth add openrouter --type oauth

# Add a second Anthropic key
hermes auth add anthropic --type api-key --api-key sk-ant-api03-your-second-key

# Add an Anthropic OAuth credential (requires Claude Max plan + extra usage credits)
hermes auth add anthropic --type oauth
# Opens browser for OAuth login
```

Check your pools:

```bash
hermes auth list
```

Output:
```
openrouter (2 credentials):
  #1  OPENROUTER_API_KEY   api_key id=ab12cd34 priority=0 env:OPENROUTER_API_KEY ←
  #2  backup-key           api_key id=ef56gh78 priority=1 manual

anthropic (3 credentials):
  #1  hermes_pkce          oauth   id=ab12cd34 priority=0 hermes_pkce ←
  #2  claude_code          oauth   id=cd34ef56 priority=1 claude_code
  #3  ANTHROPIC_API_KEY    api_key id=ef56gh78 priority=2 env:ANTHROPIC_API_KEY
```

The `←` marks the currently selected credential. `id=` is the entry id accepted by
`hermes auth remove <provider> <target>` when a label is ambiguous, and `priority=` is
the order the pool tries credentials in under the `fill_first` strategy.

## Interactive Management

Run `hermes auth` with no subcommand for an interactive wizard:

```bash
hermes auth
```

This shows your full pool status and offers a menu:

```
What would you like to do?
  1. Add a credential
  2. Remove a credential
  3. Reset cooldowns for a provider
  4. Set rotation strategy for a provider
  5. Exit
```

For providers that support both API keys and OAuth (Anthropic, Nous, Codex), the add flow asks which type:

```
anthropic supports both API keys and OAuth login.
  1. API key (paste a key from the provider dashboard)
  2. OAuth login (authenticate via browser)
Type [1/2]:
```

Each `hermes auth add openai-codex` login becomes its own pool entry, but only **different** OpenAI accounts rotate independently: two logins of the same account share one token family upstream, so OpenAI revokes the older one and the second entry adds no quota. Hermes warns at add time (`warning: this login is the same OpenAI account as openai-codex credential #N`) — log into a different account, or keep just one.

## CLI Commands

| Command | Description |
|---------|-------------|
| `hermes auth` | Interactive pool management wizard |
| `hermes auth list` | Show all pools and credentials |
| `hermes auth list <provider>` | Show a specific provider's pool |
| `hermes auth add <provider>` | Add a credential (prompts for type and key) |
| `hermes auth add <provider> --type api-key --api-key <key>` | Add an API key non-interactively |
| `hermes auth add <provider> --type oauth` | Add an OAuth credential via browser login |
| `hermes auth add <provider> --priority 0` | Add a credential and place it first in the `fill_first` order |
| `hermes auth priority <provider> <target> <n>` | Move a credential to priority `n` (0 = tried first); the rest are renumbered |
| `hermes auth remove <provider> <index>` | Remove credential by 1-based index |
| `hermes auth reset <provider>` | Clear all cooldowns/exhaustion status (applies to running sessions too: a live gateway or chat picks the reset up on its next request instead of writing its stale cooldown back) |
| `hermes auth reset <provider> <target>` | Clear the cooldown on one credential by index, id, or label |
| `hermes auth refresh <provider> [target]` | Refresh one OAuth credential's tokens and return it to rotation (proves the grant is alive; the next request re-checks quota) |

For Nous, `auth refresh` supports only the login's `device_code` singleton.
Independent Nous pool accounts are rejected before refresh; their tokens and
cooldowns are preserved. Reauthenticate with `hermes auth add nous --type oauth`
to update the singleton; this does not refresh an independent account. Other
providers retain their existing source-specific refresh support.

## Rotation Strategies

Priority positions are zero-based and clamp to the pool's ends; displayed targets
are one-based indices, entry IDs, or unambiguous exact labels. `auth add --priority`
also places an existing entry updated by reauthentication. Anthropic keeps manual
credentials ahead of seeded credentials, so the command reports the effective
position when that rule changes it. Other strategies may override priority, and
reordering does not rebind credentials already held by a running session.

Every successful pool selection increments `request_count`, regardless of strategy.
Refresh-only lookups and peeks do not count. Status reads (`hermes doctor`, the `/model`
picker's provider rows, dashboard auth cards) are peeks: they never refresh, rotate, or
bench a pool credential, so a token endpoint hiccup while the picker is open cannot hide
a provider that is still serving requests. These are selection counters, not
billing totals or a count of every inference request: a cached credential can serve
multiple requests. Counts remain in memory until the next existing pool write
(for example rotation, exhaustion, refresh, or an administrative change); this does
not add a disk write per selection.

Configure via `hermes auth` → "Set rotation strategy" or in `config.yaml`:

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```

| Strategy | Behavior |
|----------|----------|
| `fill_first` (default) | Use the first healthy key until it's exhausted, then move to the next; order is each credential's `priority` (`hermes auth priority` changes it) |
| `round_robin` | Cycle through keys evenly, rotating after each selection |
| `least_used` | Always pick the key with the lowest request count |
| `random` | Random selection among healthy keys |

### Demoting a healthy credential {#demoting-a-healthy-credential}

The pool only benches a credential *after* the provider rejects it (429/402/401). To keep a
credential you are actively using elsewhere — for example a Codex login whose weekly window
you want to save for interactive work — out of the gateway's first pick *before* it runs dry,
move it to the back of the `fill_first` order instead of removing it:

```bash
hermes auth list openai-codex                 # find the index, id or label
hermes auth priority openai-codex 1 99        # 1-based index, entry id, or exact label; large n = last
hermes auth priority openai-codex work-seat 0 # ...and back to the front later
```

`hermes auth priority <provider> <target> <priority>` reorders one credential and renumbers the
others, then persists the new order to `auth.json`. The demoted entry stays healthy: it is not
exhausted, so it is never touched by the Codex quota-reset probe and there is nothing for
`hermes auth reset` to clear; it is not refreshed on a timer, and it is still used once every
credential ahead of it is benched.
Sessions that already hold a credential keep it until they rotate; new sessions (and the next
gateway start) follow the new order.

## Error Recovery

The pool handles different errors differently:

| Error | Behavior | Cooldown |
|-------|----------|----------|
| **429 Rate Limit** | Retry same key once (transient). Second consecutive 429 rotates to next key | 1 hour |
| **402 Billing/Quota** | Immediately rotate to next key | 1 hour |
| **401 Auth Expired** | Try refreshing the OAuth token first. Rotate only if refresh fails | 5 minutes |
| **400 Codex model entitlement** (`The '<model>' model is not supported when using Codex with a ChatGPT account.`) | Bench this key for the rejected model only and rotate to the next key; other models keep using the key. Other 400s never rotate | Until `hermes auth reset` (per model; an entitlement is a plan property, not a window) |
| **All keys exhausted** | Fall through to `fallback_model` if configured | — |

Provider-supplied `reset_at` timestamps override these default cooldowns.

The `has_retried_429` flag resets on every successful API call, so a single transient 429 doesn't trigger rotation.

**Quota benches are temporary for the live session too.** When a 429/402 rotates a session off a
credential, that session checks at the start of each turn whether the benched credential is back in
rotation and moves back to it as soon as its cooldown lifts — the same choice a new session would make.
A long-running chat (the gateway keeps agents cached) therefore returns to a subscription seat once
its window reopens instead of billing the metered fallback for the rest of its life. A `401` bench
does not trigger this; an explicit `/model` switch cancels a pending switch-back.

**Anthropic 429s are per model.** Anthropic enforces its rate limits per model, so a generic 429 for
one Claude model cools that credential down for *that model only* — the same key keeps serving every
other Claude model, and `ANTHROPIC_API_KEY` / borrowed Claude Code tokens honour the same per-model
cooldown. Billing (`402`, usage-limit) and auth (`401`) failures still bench the whole credential.

**A dead OAuth login is reported, not benched.** When a refresh token is rejected for good
(`invalid_grant`, `invalid_token`, `refresh_token_reused` — the token was revoked, or another program
holding the same login rotated it first — or, for Nous, the profile holds no Portal login or token
pair to refresh with), the pool logs one WARNING naming the entry and the repair
command (`hermes auth add <provider>`), and the credential leaves rotation — marked `dead`, or dropped
when it only mirrored a token file the pool has just cleared — until you sign in again. This applies to Anthropic, Codex, xAI
and Nous OAuth logins alike. A dead credential never re-enters rotation on a timer, so a lost login
shows up once in the log instead of failing quietly every hour.

**Every Codex login in an always-on home is dead: sign in again, do not wait for adoption.** Hermes
imports the Codex CLI's `~/.codex/auth.json` automatically only to *repair a login it already has*
— when its own refresh of the `openai-codex` entry fails and `auth.adopt_external_logins` is on
(see [Borrowed CLI logins](../security.md#borrowed-cli-logins)). A pool whose Codex entries are
all `dead` (or removed) has nothing left to repair, so a headless gateway or cron profile stays
without a Codex credential until you run `hermes auth add openai-codex` in *that* home
(`hermes -p <profile> auth add openai-codex` for a named profile), which offers the Codex CLI import
interactively. Profiles that should share one login can point at the same `HERMES_HOME` instead of
each holding a copy of a single-use refresh token.

**A cooling-down or dead credential is not a blank install.** When a configured profile starts the
CLI while its only credential is benched or quarantined, startup prints the failure and, for a bench,
the remaining cooldown (or the `hermes auth add <provider>` re-login for a dead one) — the first-run
"No inference provider is configured yet" wizard is offered only when the resolver finds nothing
configured at all.

## Custom Endpoint Pools

Custom OpenAI-compatible endpoints (Together.ai, RunPod, local servers) get their own pools, keyed by the endpoint name from the `providers:` dict in config.yaml (or the legacy `custom_providers` list, which is auto-migrated).

When you set up a custom endpoint via `hermes model`, it auto-generates a name like "Together.ai" or "Local (localhost:8080)". This name becomes the pool key.

```bash
# After setting up a custom endpoint via hermes model:
hermes auth list
# Shows:
#   Together.ai (1 credential):
#     #1  config key    api_key config:Together.ai ←

# Add a second key for the same endpoint:
hermes auth add Together.ai --api-key sk-together-second-key
```

Custom endpoint pools are stored in `auth.json` under `credential_pool` with a `custom:` prefix:

```json
{
  "credential_pool": {
    "openrouter": [...],
    "custom:together.ai": [...]
  }
}
```

## Auto-Discovery

Hermes automatically discovers credentials from multiple sources and seeds the pool on startup:

| Source | Example | Auto-seeded? |
|--------|---------|-------------|
| Environment variables | `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` | Yes |
| Numbered env siblings | `OPENROUTER_API_KEY_2`, `OPENROUTER_API_KEY_3`, … | Yes (see below) |
| OAuth tokens (auth.json) | Codex device code, Nous device code | Yes |
| Claude Code credentials | `~/.claude/.credentials.json` | Yes (Anthropic) |
| Hermes PKCE OAuth | `~/.hermes/auth.json` | Yes (Anthropic) |
| Custom endpoint config | `model.api_key` in config.yaml | Yes (custom endpoints) |
| Manual entries | Added via `hermes auth add` | Persisted in auth.json |

Auto-seeded entries are updated on each pool load — if you remove an env var, its pool entry is automatically pruned. Manual entries (added via `hermes auth add`) are never auto-pruned.

### Several keys from the environment

Want more than one key for a provider without storing any of them in `auth.json`? Number them. Next to `NVIDIA_API_KEY` set `NVIDIA_API_KEY_2`, `NVIDIA_API_KEY_3`, … in your shell, `.env`, or secret manager (Bitwarden Secrets, Vault, …) and each becomes its own pool entry on the next load — no command, no config. Discovery stops at the first missing number, so a stray `_5` with no `_4` is ignored. Combine with `credential_pool_strategies` to rotate them:

```yaml
credential_pool_strategies:
  nvidia: round_robin
```

Borrowed runtime secrets (for example env vars, Bitwarden/Vault/keyring/systemd references, and custom config values) are reference-only at the `auth.json` boundary. Hermes can use the resolved value in memory for the current run, but it persists only metadata such as the source ref, label, status, request counters, and a non-reversible fingerprint. Manual entries and Hermes-owned OAuth/device-code state keep the durable tokens they need to refresh.

## Delegation & Subagent Sharing

When the agent spawns subagents via `delegate_task`, the parent's credential pool is automatically shared with children:

- **Same provider** — the child receives the parent's full pool, enabling key rotation on rate limits
- **Different provider** — the child loads that provider's own pool (if configured)
- **No pool configured** — the child falls back to the inherited single API key

This means subagents benefit from the same rate-limit resilience as the parent, with no extra configuration needed. Per-task credential leasing ensures children don't conflict with each other when rotating keys concurrently.

## Thread Safety

The credential pool uses a threading lock for all state mutations (`select()`, `mark_exhausted_and_rotate()`, `try_refresh_current()`, `mark_used()`). This ensures safe concurrent access when the gateway handles multiple chat sessions simultaneously.

Across processes (many subagents, a gateway plus a CLI, cron jobs), OAuth refreshes are serialized through a file lock on `auth.json`. When one shared OAuth grant expires under many concurrent processes, exactly one process performs the refresh; the others detect that the on-disk token no longer matches the one that failed and adopt it instead of rotating the single-use refresh token again. A process that loses the lock race keeps its entry healthy and retries — lock contention is never recorded as a credential failure.

## Architecture

For the full data flow diagram, see [`docs/credential-pool-flow.excalidraw`](https://excalidraw.com/#json=2Ycqhqpi6f12E_3ITyiwh,c7u9jSt5BwrmiVzHGbm87g) in the repository.

The credential pool integrates at the provider resolution layer:

1. **`agent/credential_pool.py`** — Pool manager: storage, selection, rotation, cooldowns; **`agent/credential_pool_admin.py`** owns locked target resolution, reset, add, removal, and priority mutations; **`agent/credential_pool_model_cooldowns.py`** owns the per-model Anthropic 429 cooldowns
2. **`hermes_cli/auth_commands.py`** — CLI commands and interactive wizard
3. **`hermes_cli/runtime_provider.py`** — Pool-aware credential resolution
4. **`agent/turn_api_error.py`** — Error recovery: 429/402/401 → pool rotation → fallback

## Storage

Pool state is stored in `~/.hermes/auth.json` under the `credential_pool` key:

```json
{
  "version": 1,
  "credential_pool": {
    "openrouter": [
      {
        "id": "abc123",
        "label": "OPENROUTER_API_KEY",
        "auth_type": "api_key",
        "priority": 0,
        "source": "env:OPENROUTER_API_KEY",
        "secret_source": "bitwarden",
        "secret_fingerprint": "sha256:12ab34cd56ef7890",
        "last_status": "ok",
        "request_count": 142
      }
    ],
    "anthropic": [
      {
        "id": "manual1",
        "label": "personal-api-key",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": "sk-ant-api03-..."
      }
    ]
  }
}
```

The OpenRouter entry above was borrowed from an external source, so the raw key is not stored in `auth.json`. The manual Anthropic entry was intentionally added to Hermes' credential store, so its token remains persistable.

An `env:` row is re-hydrated from the environment on every load, and the variable name does not have to be one Hermes declares for the provider: numbered siblings (`OPENROUTER_API_KEY_2`, see [Auto-Discovery](#auto-discovery)) appear here automatically, and a hand-written row pointing at any other variable is filled the same way, without the secret ever being written to `auth.json`.

Strategies are stored in `config.yaml` (not `auth.json`):

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```
