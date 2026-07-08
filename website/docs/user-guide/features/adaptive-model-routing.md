---
title: Adaptive Model Routing
---

# Adaptive Model Routing

Adaptive model routing lets Hermes choose a configured provider/model tier for each **main agent turn** before the `AIAgent` is constructed. It is opt-in and fails closed: when routing is disabled, incomplete, or cannot resolve credentials, Hermes uses the normal configured model.

This is different from adjacent routing features:

- `provider_routing` is OpenRouter's provider selection for an already chosen model.
- `auxiliary.*` routes side tasks such as compression, vision, approvals, and title generation.
- `fallback_providers` is failure recovery after the active model/provider errors.
- `delegation.model` and cron job `model` overrides are explicit worker/job choices and keep taking precedence.

## Configuration

```yaml
model_routing:
  enabled: true
  default_tier: balanced
  log_decisions: true

  tiers:
    cheap:
      provider: openrouter
      model: google/gemini-3-flash-preview

    balanced:
      provider: openrouter
      model: anthropic/claude-sonnet-4.6

    best:
      provider: anthropic
      model: claude-opus-4-6
```

Each tier uses the same provider resolution path as normal Hermes model startup, so configured providers, credential pools, direct endpoints, and provider-specific API modes continue to work.

## Manual overrides

Natural-language overrides win over heuristics:

- `use cheap model`, `quick answer` → `cheap`
- `use best model`, `use strongest model`, `think deeply`, `use Fable` → `best`
- `use balanced model` → `balanced`

Explicit runtime choices still win over routing. For example, a session-scoped `/model` override, a cron job model override, or a delegation model override should not be silently replaced by adaptive routing.

## Built-in heuristics

The first implementation is deterministic and does not spend an extra classifier call.

- `cheap`: very short/simple turns, brief factual Q&A, short rewriting/summarization.
- `balanced`: normal tool use, moderate coding, ordinary troubleshooting, normal research.
- `best`: architecture, difficult debugging, large refactors, Hermes internals, model/provider selection, security/privacy-sensitive changes, and high-stakes medical/legal/financial reasoning.

Routing decisions are logged with the selected tier, provider/model, source, confidence, and reason.
