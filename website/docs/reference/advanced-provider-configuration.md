---
title: Advanced Provider Configuration
sidebar_label: Advanced Provider Configuration
sidebar_position: 90
---

This reference preserves the advanced provider, timeout, OpenRouter, and
worktree examples formerly embedded in `cli-config.yaml.example`. The
example file now keeps a compact index while this page retains every
setting and explanatory comment.

```yaml
# Command-minted credentials (optional): key_cmd
# ------------------------------------------------------------------
# Enterprise gateways often issue SHORT-LIVED bearers (SSO/OIDC brokers, cloud
# IAM, internal auth proxies) rather than static API keys, so a value copied
# into .env via `key_env` is stale within the hour and every later request 401s.
# `key_cmd` names a command that PRINTS a token instead: Hermes runs it per
# request (cached until shortly before expiry), so long sessions keep working
# with no restart.
#
# Contract: print ONLY the token on stdout, either bare or as JSON with an
# "access_token" field ("expires_in" is honoured). Same shape as OAuth 2.0
# token endpoints, Claude Code's `apiKeyHelper`, `gcloud auth
# print-access-token`, and `aws ecr get-login-password`.
#
# Precedence: an explicit --api-key still wins; otherwise key_cmd is preferred
# over inline api_key / key_env on that entry.
#
# Applies to the main agent turn and to auxiliary tasks (title generation,
# context compression, vision, embedding) alike.
#
# Not to be confused with `secrets.command`, which is a different mechanism:
# that one runs a helper ONCE at startup to populate env vars for many secrets
# at the process level. Use it for a vault or keychain helper that hands back a
# KEY=VALUE blob. Use `key_cmd` when ONE provider needs a credential refreshed
# DURING a session, because a startup-time env var cannot be re-minted after it
# expires.
#
# providers:
#   my-gateway:
#     base_url: "https://gateway.internal.example.com/v1"
#     api_mode: chat_completions
#     key_cmd: "my-auth-cli print-token --profile prod"
#
# Worked example — an AI gateway that routes by model family, so one entry per
# wire format shares the same credential helper:
#
# providers:
#   dbx:                             # OpenAI-compatible (MLflow) route
#     base_url: "https://<workspace>.cloud.databricks.com/ai-gateway/mlflow/v1"
#     api_mode: chat_completions
#     model: databricks-claude-sonnet-4-6
#     key_cmd: "databricks auth token -p MY-PROFILE"
#   dbx-gpt:                         # OpenAI Responses route
#     base_url: "https://<workspace>.cloud.databricks.com/ai-gateway/openai/v1"
#     api_mode: codex_responses
#     model: databricks-gpt-5-5
#     key_cmd: "databricks auth token -p MY-PROFILE"
#   dbx-claude:                      # Anthropic Messages route
#     base_url: "https://<workspace>.cloud.databricks.com/ai-gateway/anthropic"
#     api_mode: anthropic_messages
#     model: databricks-claude-fable-5
#     key_cmd: "databricks auth token -p MY-PROFILE"

# Named provider overrides (optional)
# Use this for per-provider request timeouts, non-stream stale timeouts,
# and per-model exceptions.
# Applies to the primary turn client on every api_mode (OpenAI-wire, native
# Anthropic, and Anthropic-compatible providers), the fallback chain, and
# client rebuilds during credential rotation.  For OpenAI-wire chat
# completions (streaming and non-streaming) the configured value is also
# used as the per-request ``timeout=`` kwarg so it wins over the legacy
# HERMES_API_TIMEOUT env var (which still applies when no config is set).
# ``stale_timeout_seconds`` controls the non-streaming stale-call detector and
# wins over the legacy HERMES_API_CALL_STALE_TIMEOUT env var. Leaving these
# unset keeps the legacy defaults (HERMES_API_TIMEOUT=1800s,
# HERMES_API_CALL_STALE_TIMEOUT=90s, native Anthropic 900s). The
# implicit non-stream stale detector is auto-disabled for local endpoints
# and can scale upward for very large contexts.
#
# Not currently wired for AWS Bedrock (bedrock_converse + AnthropicBedrock
# SDK paths) — those use boto3 with its own timeout configuration.
#
# providers:
#   ollama-local:
#     request_timeout_seconds: 300   # Longer timeout for local cold-starts
#     stale_timeout_seconds: 900     # Explicitly re-enable stale detection on local endpoints
#   anthropic:
#     request_timeout_seconds: 30    # Fast-fail cloud requests
#     models:
#       claude-opus-4.6:
#         timeout_seconds: 600       # Longer timeout for extended-thinking Opus calls
#   openai-codex:
#     models:
#       gpt-5.4:
#         stale_timeout_seconds: 1800  # Longer non-stream stale timeout for slow large-context turns

# =============================================================================
# Unified Timeouts (operation deadlines)
# =============================================================================
# One place to override Hermes's internal operation deadlines (seconds).
# Keys are dotted paths resolved by agent/deadline.py:resolve_timeout().
# Precedence: this section > legacy HERMES_* env var (back-compat) > built-in
# default. 0 or a negative value disables the bound (unbounded); very large
# values are clamped to a platform-safe maximum automatically.
#
# Currently resolved keys (more paths migrate here over time — see issue #85125):
#
# timeouts:
#   tools:
#     concurrent_batch: 420   # Deadline for a parallel tool-call batch
#                             # (legacy env: HERMES_CONCURRENT_TOOL_TIMEOUT_S)
#     sequential_call: 420    # Deadline for one sequentially-executed tool call.
#                             # Defaults to concurrent_batch's value so the two
#                             # executor paths stay in sync; human waits
#                             # (approval prompts, clarify) never count against it.

# =============================================================================
# OpenRouter Provider Routing (only applies when using OpenRouter)
# =============================================================================
# Control how requests are routed across providers on OpenRouter.
# See: https://openrouter.ai/docs/guides/routing/provider-selection
#
# provider_routing:
#   # Sort strategy: "price" (default), "throughput", or "latency"
#   # Append :nitro to model name for a shortcut to throughput sorting.
#   sort: "throughput"
#
#   # Only allow these providers (provider slugs from OpenRouter)
#   # only: ["anthropic", "google"]
#
#   # Skip these providers entirely
#   # ignore: ["deepinfra", "fireworks"]
#
#   # Try providers in this order (overrides default load balancing)
#   # order: ["anthropic", "google", "together"]
#
#   # Require providers to support all parameters in your request
#   # require_parameters: true
#
#   # Data policy: "allow" (default) or "deny" to exclude providers that may store data
#   # data_collection: "deny"
#
#   # Per-model overrides: same keys, applied only when the agent is on that model
#   # (spelling-tolerant match; unset keys fall through to the flat values above).
#   # models:
#   #   "openai/gpt-6-astra":
#   #     only: ["openai"]
#   #   "anthropic/claude-fable-5.1":
#   #     only: ["anthropic"]

# =============================================================================
# OpenRouter Response Caching (only applies when using OpenRouter)
# =============================================================================
# Cache identical API responses at the OpenRouter edge for free instant replays.
# When enabled, identical requests (same model, messages, parameters) return
# cached responses with zero billing. Separate from Anthropic prompt caching.
# See: https://openrouter.ai/docs/guides/features/response-caching
#
# openrouter:
#   response_cache: true         # Enable response caching (default: true)
#   response_cache_ttl: 300      # Cache TTL in seconds, 1-86400 (default: 300)

# =============================================================================
# Git Worktree Isolation
# =============================================================================
# When enabled, each CLI session creates an isolated git worktree so multiple
# agents can work on the same repo concurrently without file collisions.
# Equivalent to always passing --worktree / -w on the command line.
#
# worktree: true    # Always create a worktree when in a git repo
# worktree: false   # Default — only create when -w flag is passed
#
# By default a new worktree branches from the freshly-fetched remote tip
# (the current branch's upstream, else the remote's default branch) so it
# starts current with the project instead of from the local clone's
# (possibly stale) HEAD. Set worktree_sync: false to branch from local HEAD
# instead — useful when offline or when you deliberately want the clone's
# exact current state as the base.
#
# worktree_sync: true   # Default — branch from the fetched remote tip
# worktree_sync: false  # Branch from local HEAD (offline / pinned base)
```
