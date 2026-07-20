# Model Desk

Unified operational surface for Hermes **local + API** model/provider health
(audit backlog **M01–Q05**, version **1.0.0**, **35** features).

Keeps resolution, preflight, privacy, and fallback audit at the edge —
`agent/model_desk/` + `hermes model <cmd>` — without growing the agent loop.

## Quick start

```bash
# Import-check every module + golden invariants
hermes model features
hermes model golden
hermes model doctor

# Before chat on a local stack
hermes model preflight
hermes model ports
hermes model sync-ctx --apply

# Safe provider switch (blocks hijack pairs)
hermes model atomic-write \
  --provider custom \
  --model-id my-local \
  --base-url http://127.0.0.1:8080/v1 \
  --apply

# Inspect resolve without calling the model API
hermes model dry-run
hermes model explain
```

Desk commands are **non-interactive** (no TTY required). Plain `hermes model`
without a subcommand still opens the interactive picker.

## Feature map

### M — Core desk (M01–M10)

| ID | Feature | CLI |
|----|---------|-----|
| M01 | Aggregated desk status | `model desk [--deep]` |
| M02 | providers ↔ custom_providers unify | `model unify [--apply]` |
| M03 | Preflight hard-gate | `model preflight [--soft]` |
| M04 | Local port collision detector | `model ports` |
| M05 | n_ctx ↔ context_length sync | `model sync-ctx [--apply]` |
| M06 | Ollama tag normalizer | `model normalize-tag <id>` |
| M07 | Aux never-leave-local privacy | `model privacy` |
| M08 | Fallback audit log | `model fallback-audit` |
| M09 | Atomic provider+model+url write | `model atomic-write …` |
| M10 | api_mode self-test | `model api-mode [--live]` |

### L — Local GGUF / Ollama / LM Studio (L01–L10)

| ID | Feature | CLI |
|----|---------|-----|
| L01 | Serve plan + lock hint | `model serve-plan` |
| L02 | Local slot / backpressure | (desk deep) |
| L03 | LM Studio catalog probe | `model lmstudio` |
| L04 | Ollama list | `model ollama-list` |
| L05 | RAM budget gate | `model resources` |
| L06–L09 | Hot-swap / embeddings / spec / structured | desk modules |
| L10 | Quantization recommender | `model quant --params-b 7` |

### C — Routing & capability (C01–C10)

| ID | Feature | CLI |
|----|---------|-----|
| C01 | Smart route score | `model smart-route` |
| C02 | Auxiliary task pins | `model aux-pins` |
| C03 | Credential pool status | `model pool [--provider …]` |
| C06 | Streaming/tool parity | `model parity` |
| C07 | Vision/audio matrix | `model capabilities` |
| C09 | Dry-run resolve | `model dry-run` |

### Q — Observability (Q01–Q05)

| ID | Feature | CLI |
|----|---------|-----|
| Q01 | Provider SLO snapshot | `model slo` |
| Q02 | Chaos local-down drill | `model chaos` |
| Q03 | Golden resolve scenarios | `model golden` |
| Q04 | Why-this-provider | `model explain` |
| Q05 | Debug secret redaction | `model redact --json '…'` |

## Runtime hooks

- **Alias audit** — when `ollama` / `vllm` / `llamacpp` normalize to `custom`,
  an event is appended to `$HERMES_HOME/model_fallback_audit.jsonl`.
- **Chat soft preflight** — `hermes chat` runs Model Desk preflight at startup
  (warn-only by default). Set `HERMES_MODEL_PREFLIGHT_HARD=1` to refuse start
  when local endpoint is down or hijack is detected.

## CI

```bash
bash scripts/model_desk_ci_gate.sh
```

Workflow: `.github/workflows/model-desk-ci.yml` (path-filtered on Model Desk files).

## config.yaml

```yaml
auxiliary:
  never_leave_local: false   # or HERMES_AUX_NEVER_LEAVE_LOCAL=1

model:
  provider: custom
  default: my-local
  base_url: http://127.0.0.1:8080/v1
  api_mode: chat_completions
  context_length: 8192       # keep in sync via model sync-ctx --apply

local_llm:
  ravenx:
    enabled: true
    host: 127.0.0.1
    port: 8080
```

## Invariants

1. **Atomic selection** — never persist `custom` with an OpenRouter `base_url`,
   or `openrouter` with a loopback URL (`MODEL-ATOMIC-002`).
2. **Preflight** — hard mode blocks unreachable local endpoints
   (`MODEL-LOCAL-DOWN-001`) and local+aggregator hijack (`MODEL-HIJACK-001`).
3. **Aux privacy** — when enabled, `_resolve_auto` skips cloud chain entries
   (soft import; aux resolve never crashes if Model Desk is missing).
4. **Redaction** — dry-run / audit never echo raw API keys.
5. **Doctor** — `hermes doctor` includes `model_desk_wiring` (import + golden).
6. **Count/version** — 35 features, version `1.0.0`.

### Soft panels (deepened)

- **`hermes model spend`** (C08) — SessionDB token aggregates + credits/usage layers;
  returns `skipped: true` only when no local data exists yet.
- **`hermes model parity [--provider]`** (C06) — static matrix; pass probe via
  Python API `parity_matrix(provider, probe=True)` for local `/v1/models` reachability.
- **`hermes model spec-decode`** (L08) — draft-pair advice with live detection of
  `llama-server` / `ollama` / `vllm` binaries.

## Env bridges

| Env | Effect |
|-----|--------|
| `HERMES_AUX_NEVER_LEAVE_LOCAL` | Force aux privacy on/off |
| `HERMES_MODEL_PREFLIGHT_HARD` | Force hard preflight |

## Audit log

Fallback / atomic-write events append to:

`$HERMES_HOME/model_fallback_audit.jsonl`

Inspect with `hermes model fallback-audit --limit 50`.

## Tests

```bash
scripts/run_tests.sh tests/agent/test_model_desk.py -q
```

Golden CLI:

```bash
hermes model golden
```
