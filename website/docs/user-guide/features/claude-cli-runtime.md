---
title: Claude CLI Runtime
sidebar_label: Claude CLI Runtime
---

# Claude CLI Runtime

Hermes can hand Anthropic Claude turns to a local [`claude -p`](https://docs.anthropic.com/en/docs/claude-code) subprocess (Claude Code CLI) instead of the Anthropic HTTP Messages API. When active, Max subscription billing and Claude Code's native session tools ride the CLI; Hermes still owns sessions, slash commands, the gateway, memory, and skill review.

## Default-when-token (no per-profile flag required)

For **provider `anthropic`** + a **Claude** model, Hermes defaults to `claude_cli` when **both** are true:

1. A Claude Code **setup token** is resolvable (`CLAUDE_CODE_OAUTH_TOKEN` via profile env, credential pool, or the canonical root `~/.hermes/.env`)
2. The **`claude` binary** is available on `PATH` (or `~/.local/bin/claude`)

So switching **any** profile to a Claude model (CLI, gateway, desktop app model picker) lands on `claude -p` base Max **without** setting `anthropic_runtime: claude_cli` on that profile. Profiles that already have their own `config.yaml` and never inherit a root-level runtime key are covered.

Environments **without** a setup token or without the `claude` binary are **unchanged**: they keep the HTTP `anthropic_messages` path exactly as before. Non-Anthropic providers (`openai-codex`, `xai-oauth`, etc.) are unaffected.

## Precedence (explicit always wins)

| Setting | Result |
|---|---|
| `model.anthropic_runtime: claude_cli` (or env `HERMES_ANTHROPIC_RUNTIME=claude_cli`) | Force `claude_cli` |
| `model.anthropic_runtime: anthropic_messages` (aliases: `http`, `api`, `messages`, `off`) | Force HTTP (`anthropic_messages`) — **explicit opt-out** |
| UNSET / `auto` + setup token + `claude` binary + Claude model | **Default** `claude_cli` |
| UNSET / `auto` without token or binary | HTTP `anthropic_messages` (unchanged) |

## Enable / opt-out

Most fleets only need the setup token once in the root `.env` (see [Fleet setup](#fleet-setup)). Then any profile can switch to a Claude model:

```bash
# Interactive model picker for a profile (desktop app does the same)
hermes -p <agent> model

# Or pin provider + model — no anthropic_runtime required when token+binary exist
hermes -p <agent> config set model.provider anthropic
hermes -p <agent> config set model.default claude-opus-4-6
```

Force the CLI runtime explicitly (optional; same as the auto default when eligible):

```yaml
model:
  provider: anthropic
  default: claude-opus-4-6
  anthropic_runtime: claude_cli
```

**Opt out** of the CLI path and keep HTTP (extra-usage / API credits):

```yaml
model:
  provider: anthropic
  default: claude-opus-4-6
  anthropic_runtime: anthropic_messages   # or http / api
```

One-session override (does not rewrite config):

```bash
# Force CLI
HERMES_ANTHROPIC_RUNTIME=claude_cli hermes -p <agent> chat

# Force HTTP even if config enables claude_cli
HERMES_ANTHROPIC_RUNTIME=anthropic_messages hermes -p <agent> chat
```

Requires the `claude` binary on `PATH` (`npm install -g @anthropic-ai/claude-code`) for the CLI path.

## Auth: non-rotating setup token

`claude_cli` injects **`CLAUDE_CODE_OAUTH_TOKEN`** into a clean child env. That value must be the **non-rotating setup token** from:

```bash
claude setup-token
```

Setup tokens look like `sk-ant-oat…` and last on the order of a year. They are **fork-safe** (like an API key): multiple Hermes profiles and other Claude Code consumers on the same login can use the same token concurrently without a shared lock or rotating-token store.

**Do not** rely on the rotating `claude /login` session under `~/.claude` for this runtime. That login is not injected into the clean `claude -p` env and is not a resolution source for `claude_cli`.

### Resolution order

First hit wins:

1. Explicit / passed token (`explicit=` argument, or agent-held setup/OAuth-shaped key)
2. Profile / process env `CLAUDE_CODE_OAUTH_TOKEN` (legacy alias: `ANTHROPIC_TOKEN`)
3. Profile anthropic credential_pool (`claude_code` / `env:CLAUDE_CODE_OAUTH_TOKEN` OAuth entries)
4. **Canonical Hermes root** `~/.hermes/.env` → `CLAUDE_CODE_OAUTH_TOKEN` (fleet fallback)
5. Never the rotating `~/.claude` login

Only if **no** source yields a token does Hermes treat the CLI path as unavailable (auto falls back to HTTP; explicit `claude_cli` raises a clear setup error at turn time).

The token is a **secret**: store it in `.env` (or the credential pool), never in `config.yaml`.

## Fleet setup

Put the setup token **once** in the platform Hermes root env file:

```bash
# ~/.hermes/.env  (canonical root — not a profile directory)
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...   # from `claude setup-token`
```

Every profile then resolves that same token on demand when profile env and credential_pool have none. You do **not** need to copy the token into each `~/.hermes/profiles/<name>/.env`, and you do **not** need `anthropic_runtime: claude_cli` on each profile's `config.yaml`.

Then switch any agent to a Claude model:

```bash
hermes -p <agent> model
# or
hermes -p <agent> config set model.provider anthropic
hermes -p <agent> config set model.default claude-opus-4-6
```

Regenerate the setup token about yearly with `claude setup-token` and update the single root `.env` line.

### Optional: per-profile override

If one profile should use a different token, set `CLAUDE_CODE_OAUTH_TOKEN` in that profile's own `.env` or credential pool — profile sources win over the canonical root.

If one profile must stay on HTTP (extra-usage API path), set:

```yaml
model:
  anthropic_runtime: anthropic_messages
```

## Concurrency

Host-global caps limit concurrent Hermes `claude -p` children so other Claude Code consumers on the same Max login still have headroom. Configure under `model.claude_cli` in `config.yaml`:

```yaml
model:
  claude_cli:
    max_concurrent: 3                 # default 3; 0 = unbounded
    acquire_timeout_seconds: 45       # wait then fall back
```

See the concurrency notes in the developer guide / Phase 2c tests for slot reaping and fallback behavior.

## What this runtime does not change

- Base Max / subscription billing still goes through Claude Code's CLI path.
- Hermes MCP tools, multi-turn session resume, host concurrency, and auxiliary-model handling stay as implemented for `claude_cli` (no HTTP Anthropic aux for `claude_cli` turns).
- Non-secret settings stay in `config.yaml`; secrets stay in `.env`.
- Profiles and hosts **without** a setup token or `claude` binary keep pure HTTP Anthropic — no forced dependency.
