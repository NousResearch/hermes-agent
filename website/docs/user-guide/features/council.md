---
title: "Multi-Model Deliberation (Council)"
sidebar_label: "Council"
sidebar_position: 44
---

# Multi-Model Deliberation (`/council`)

The `/council` command runs a multi-model deliberation pipeline that produces structured implementation plans. Three different frontier models independently propose plans, an anonymized critic evaluates them, and a chairman model synthesizes the best approach into a unified output.

## Why a council?

Single-model planning has three blind spots that `/council` addresses:

1. **Model-specific strengths** — GLM leads SWE-Bench, Kimi excels at reasoning chains, MiniMax uses sparse-attention for long-context recall. A council captures the best of each.
2. **No structured output** — single models return ad-hoc formats. The council produces a standardized plan with file manifest, execution steps, risk register, and vote matrix.
3. **No peer review** — plans are never critiqued before execution, so gaps in edge cases, security, or error handling go unnoticed.

## Pipeline

The council runs a 3-stage pipeline in `agent/council.py`:

**Stage 1 — Propose:** Each configured proposer model generates an independent implementation plan. Two modes:

- **Flat mode** (fast, ~40s): each proposer is a single API call
- **Subagent mode** (thorough, ~2.5 min): each proposer runs as a Hermes `delegate_task` with `role=orchestrator`, capable of delegating subtasks to cheaper models

**Stage 2 — Critique:** An anonymized peer review evaluates all plans on correctness, completeness, and efficiency. Automatically skipped when only one plan exists (no peer comparison possible).

**Stage 3 — Chairman:** Synthesizes plans + critiques into a unified output with:

- Consensus score
- Selected approach with justification
- File manifest (file, action, rationale, proposed by)
- Execution steps (numbered)
- Risk register (risk, likelihood, mitigation, flagged by)
- Disagreements resolved
- Proposal breakdown table (per-proposer scores)

## Resilience

Cloud LLM calls fail. The council handles this intrinsically:

- **Retry with geometric backoff** — TimeoutError, empty content, and transient connection errors are retried at 5s, 15s, 30s delays. Max 3 attempts per call.
- **Content validation gate** — Output under 15 lines, 500 chars, or lacking structural markers (code blocks, `## Plan` headings) is rejected as a stub.
- **State persistence** — Full pipeline state saved before the chairman stage. Enables `/council resume` if the chairman crashes.
- **Pipeline timeout guard** — 2400s (40 min) upper bound prevents runaway execution.
- **Dynamic timeouts** — Each stage timeout scales with input size. Floor values: proposer 120s, critique 120s, chairman 300s.

## Usage

### CLI

```
/council rewrite the auth module to use JWT
```

Subcommands:

| Command | Description |
|---------|-------------|
| `/council <task>` | Run full council deliberation (default) |
| `/council plan <task>` | Explicit plan subcommand |
| `/council status` | Show current council configuration |
| `/council resume` | Resume a failed council from saved state |

### Gateway (Discord, Telegram, etc.)

Same syntax — `/council <task>` works in any messaging platform that supports the Hermes gateway.

## Configuration

All settings live under `council.*` in `~/.hermes/config.yaml`:

```yaml
council:
  # Core
  enabled: true
  default_mode: plan
  subagent_delegation: true       # false = flat API calls (faster)
  max_concurrent_calls: 2          # 0 = derive from delegation config
  pipeline_timeout_seconds: 2400   # max wall-clock for entire pipeline
  max_retries: 2                   # retry attempts per proposer/critic/chairman

  # Proposer models (N models independently generate plans)
  proposers:
    - provider: ollama-cloud
      model: glm-5.1:cloud
    - provider: ollama-cloud
      model: kimi-k2.6:cloud
    - provider: ollama-cloud
      model: minimax-m2.5:cloud

  # Subagent delegation (used when subagent_delegation: true)
  subagents:
    fast:
      provider: ollama-cloud
      model: deepseek-v4-flash:cloud
    lightweight:
      provider: ollama-cloud
      model: gemma4:31b:cloud

  # Review
  critic:
    provider: ollama-cloud
    model: minimax-m3:cloud
  chairman:
    provider: ollama-cloud
    model: minimax-m3:cloud
  peer_review: true
  anonymize_reviews: true

  # Preflight health check
  preflight:
    enabled: true
    timeout_seconds: 30
    min_proposers: 2
    check_interval_hours: 24     # 0 = always check, never cache

  # Content validation
  min_plan_lines: 15
  min_plan_chars: 500
```

### Key settings

| Setting | Default | Notes |
|---------|---------|-------|
| `subagent_delegation` | `true` | Set `false` for faster runs on small tasks |
| `max_concurrent_calls` | `2` | Parallel proposer limit; `0` auto-derives from `delegation.max_concurrent_children` |
| `pipeline_timeout_seconds` | `2400` | Last-resort guard — the pipeline aborts if it exceeds this |
| `preflight.timeout_seconds` | `30` | Per-model health-check timeout; set ≥30 for cloud providers |
| `min_plan_lines` / `min_plan_chars` | `15` / `500` | Minimum output size to accept as a valid plan |

## Resume from failures

If the chairman stage fails (timeout, API error), the pipeline state is automatically saved to `~/.hermes/council_state.json`. Resume with:

```
/council resume
```

This loads saved plans + critiques and re-runs only the chairman stage. Successful completion cleans up the state file.

## Example output

```
## Council Plan: Rewrite auth module to JWT

### Consensus Score: 8.5/10

### Selected Approach
Hybrid: OAuth2 + JWT access tokens with refresh token rotation
(Recommended by 2/3 proposers; highest avg score 8.7)

### File Manifest
| File | Action | Rationale | Proposed By |
|------|--------|-----------|-------------|
| src/auth/jwt_handler.py | Create | JWT encode/decode/verify | Proposer A |
| src/auth/middleware.py | Modify | Add token validation | Proposer B |
| src/config/auth.py | Modify | Add JWT config vars | Proposer C |

### Execution Steps
1. Create JWT handler with RS256 signing
2. Add middleware for token extraction + validation
3. Implement refresh token rotation
4. Add integration tests

### Risk Register
| Risk | Likelihood | Mitigation | Flagged By |
|------|:----------:|------------|------------|
| Token leakage in logs | Medium | Strip Authorization header from log filters | Proposer A |
| Clock skew between services | Low | Use leeway parameter in validation | Proposer C |
```

## Quality gates

- **35 unit tests** — preflight, timeouts, retry, validation, state, resume
- **11 live E2E tests** — real API calls validating full pipeline
- **Content validation** rejects stubs under 15 lines / 500 chars / no markers
- **Pipeline timeout** prevents runaway execution past 2400s