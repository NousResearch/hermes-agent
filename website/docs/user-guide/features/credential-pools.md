---
title: Credential Pools
description: Pool multiple API keys or OAuth tokens per provider for automatic rotation and rate limit recovery.
sidebar_label: Credential Pools
sidebar_position: 9
---

# Credential Pools

Credential pools let you register multiple API keys or OAuth tokens for the same provider. When one key hits a rate limit or billing quota, Hermes automatically rotates to the next healthy key — keeping your session alive without switching providers.

This is different from [fallback providers](./fallback-providers.md), which switch to a *different* provider entirely. Credential pools are same-provider rotation; fallback providers are cross-provider failover. Pools are tried first — if all pool keys are exhausted, *then* the fallback provider activates.

## How It Works

```
Your request
  → Pick key from pool (round_robin / least_used / fill_first / random)
  → Send to provider
  → 429 rate limit?
      → Retry same key once (transient blip)
      → Second 429 → rotate to next pool key
      → All keys exhausted → fallback_model (different provider)
  → 402 billing error?
      → Immediately rotate to next pool key (24h cooldown)
  → 401 auth expired?
      → Try refreshing the token (OAuth)
      → Refresh failed → rotate to next pool key
  → Success → continue normally
```

## Quick Start

If you already have an API key set in `.env`, Hermes auto-discovers it as a 1-key pool. To benefit from pooling, add more keys:

```bash
# Add a second OpenRouter key
hermes auth add openrouter --api-key sk-or-v1-your-second-key

# Add a second Anthropic key
hermes auth add anthropic --type api-key --api-key sk-ant-api03-your-second-key

# Add an Anthropic OAuth credential (Claude Code subscription)
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
  #1  OPENROUTER_API_KEY   api_key env:OPENROUTER_API_KEY ←
  #2  backup-key           api_key manual

anthropic (3 credentials):
  #1  hermes_pkce          oauth   hermes_pkce ←
  #2  claude_code          oauth   claude_code
  #3  ANTHROPIC_API_KEY    api_key env:ANTHROPIC_API_KEY
```

The `←` marks the currently selected credential.

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

## CLI Commands

| Command | Description |
|---------|-------------|
| `hermes auth` | Interactive pool management wizard |
| `hermes auth list` | Show all pools and credentials |
| `hermes auth list <provider>` | Show a specific provider's pool |
| `hermes auth add <provider>` | Add a credential (prompts for type and key) |
| `hermes auth add <provider> --type api-key --api-key <key>` | Add an API key non-interactively |
| `hermes auth add <provider> --type oauth` | Add an OAuth credential via browser login |
| `hermes auth remove <provider> <index>` | Remove credential by 1-based index |
| `hermes auth reset <provider>` | Clear all cooldowns/exhaustion status |

## Rotation Strategies

Configure via `hermes auth` → "Set rotation strategy" or in `config.yaml`:

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```

| Strategy | Behavior |
|----------|----------|
| `fill_first` (default) | Use the first healthy key until it's exhausted, then move to the next |
| `round_robin` | Cycle through keys evenly, rotating after each selection |
| `least_used` | Always pick the key with the lowest request count |
| `random` | Random selection among healthy keys |

## Error Recovery

The pool handles different errors differently:

| Error | Behavior | Cooldown |
|-------|----------|----------|
| **429 Rate Limit** | Retry same key once (transient). Second consecutive 429 rotates to next key | 1 hour |
| **402 Billing/Quota** | Immediately rotate to next key | 24 hours |
| **401 Auth Expired** | Try refreshing the OAuth token first. Rotate only if refresh fails | — |
| **All keys exhausted** | Fall through to `fallback_model` if configured | — |

The `has_retried_429` flag resets on every successful API call, so a single transient 429 doesn't trigger rotation.

## Custom Endpoint Pools

Custom OpenAI-compatible endpoints (Together.ai, RunPod, local servers) get their own pools, keyed by the endpoint name from `custom_providers` in config.yaml.

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
| OAuth tokens (auth.json) | Codex device code, Nous device code | Yes |
| Claude Code credentials | `~/.claude/.credentials.json` | Yes (Anthropic) |
| Hermes PKCE OAuth | `~/.hermes/auth.json` | Yes (Anthropic) |
| Custom endpoint config | `model.api_key` in config.yaml | Yes (custom endpoints) |
| Manual entries | Added via `hermes auth add` | Persisted in auth.json |

Auto-seeded entries are updated on each pool load — if you remove an env var, its pool entry is automatically pruned. Manual entries (added via `hermes auth add`) are never auto-pruned.

## Thread Safety

The credential pool uses a threading lock for all state mutations (`select()`, `mark_exhausted_and_rotate()`, `try_refresh_current()`, `mark_used()`). This ensures safe concurrent access when the gateway handles multiple chat sessions simultaneously.

## Architecture

For the full data flow diagram, see [`docs/credential-pool-flow.excalidraw`](https://excalidraw.com/#json=2Ycqhqpi6f12E_3ITyiwh,c7u9jSt5BwrmiVzHGbm87g) in the repository.

The credential pool integrates at the provider resolution layer:

1. **`agent/credential_pool.py`** — Pool manager: storage, selection, rotation, cooldowns
2. **`hermes_cli/auth_commands.py`** — CLI commands and interactive wizard
3. **`hermes_cli/runtime_provider.py`** — Pool-aware credential resolution
4. **`run_agent.py`** — Error recovery: 429/402/401 → pool rotation → fallback

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
        "access_token": "sk-or-v1-...",
        "last_status": "ok",
        "request_count": 142
      }
    ]
  },
  "credential_pool_strategies": {
    "openrouter": "round_robin"
  }
}
```

Strategies are stored in `config.yaml` (not `auth.json`):

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```
