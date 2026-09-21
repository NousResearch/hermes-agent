# Expanded Provider Pool + Fallback Chain Audit
## Scope: ~/repos/KenseiAgent | Read-only | Kanban: t_b6d1bcc2
## Run date: 2026-09-03

---

## 1) Current Behaviour: pool-then-next or model-then-next?

**Verdict: model-then-next (not pool-then-next).**

Evidence:

- `hermes_cli/fallback_config.py:80-101` defines `get_fallback_chain(config)` as a straight merge of `fallback_providers` then legacy `fallback_model`, preserving config order. It returns a list of provider/model/base_url identities. There is no pool expansion here.
- `hermes_cli/runtime_provider.py` resolves each returned entry into a runtime dict. For a given provider, the pool is consumed via `CredentialPool.select()` / `load_pool(...).select()` (see `_resolve_runtime_from_pool_entry`, `_try_resolve_from_custom_pool`). That selects one credential from the pool for that provider/model pair.
- `credential_pool_strategies` (e.g. `fill_first`) controls intra-pool rotation, but the loop that drives fallback is outside the pool. The caller walks `get_fallback_chain()` and asks the runtime resolver for one try per list entry.
- Net effect: if `ollama-cloud / deepseek-v4-flash` is slot 1 and the ollama-cloud pool has 3 keys, Hermes may rotate keys across *requests* via pool policy, but it does **not** exhaust those 3 keys against `deepseek-v4-flash` before advancing to slot 2 (`ollama-cloud / deepseek-v4-pro`). The moment a request to slot 1 returns a hard failure, slot 2 is tried next.

So the current behaviour is **model-slot-then-next**, with pool selection happening per slot attempt, not per pool exhaustion.

---

## 2) Expanded Fallback Chain Per Main Profile (15 profiles)

Key:
- Provider/Main: from `model.provider` + `model.default`
- Slot: config `fallback_providers` order
- Endpoint: resolved from overlay or config
- Auth: overlay/registry default

### 2.1 wesker
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | ollama-cloud | deepseek-v4-flash | https://ollama.com/v1 | api_key |
| 1 | ollama-cloud | deepseek-v4-pro | https://ollama.com/v1 | api_key |
| 2 | ollama-cloud | deepseek-v4-flash | https://ollama.com/v1 | api_key |
| 3 | ollama-cloud | glm-5.1 | https://ollama.com/v1 | api_key |
| 4 | openrouter | google/gemma-4-31b-it:free | https://openrouter.ai/api/v1 | api_key |
| 5 | openrouter | nvidia/nemotron-3-super-120b-a12b:free | https://openrouter.ai/api/v1 | api_key |
| 6 | openrouter | meta-llama/llama-3.3-70b-instruct:free | https://openrouter.ai/api/v1 | api_key |

### 2.2 octacon
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | ollama-cloud | glm-5.1 | https://ollama.com/v1 | api_key |
| 1 | ollama-cloud | deepseek-v4-pro | https://ollama.com/v1 | api_key |
| 2 | ollama-cloud | glm-5.1 | https://ollama.com/v1 | api_key |
| 3 | ollama-cloud | kimi-k2.6 | https://ollama.com/v1 | api_key |
| 4 | openrouter | google/gemma-4-31b-it:free | https://openrouter.ai/api/v1 | api_key |
| 5 | openrouter | nvidia/nemotron-3-super-120b-a12b:free | https://openrouter.ai/api/v1 | api_key |

### 2.3 ceecee
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | opencode-go | minimax-m3 | https://opencode.ai/zen/go/v1 | aggregator/api_key |
| 1 | opencode-zen | deepseek-v4-flash | https://opencode.ai/zen/v1 | aggregator/api_key |
| 2 | opencode-go | minimax-m3 | https://opencode.ai/zen/go/v1 | aggregator/api_key |
| 3 | nous | stepfun/step-3.7-flash | https://inference-api.nousresearch.com/v1 | oauth_device_code |

### 2.4 remii
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | ollama-cloud | deepseek-v4-flash | https://ollama.com/v1 | api_key |
| 1 | openrouter | google/gemma-4-31b-it:free | https://openrouter.ai/api/v1 | api_key |
| 2 | ollama-cloud | deepseek-v4-flash | https://ollama.com/v1 | api_key |
| 3 | openrouter | nvidia/nemotron-3-super-120b-a12b:free | https://openrouter.ai/api/v1 | api_key |

### 2.5 gojo
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | ollama-cloud | deepseek-v4-flash | https://ollama.com/v1 | api_key |
| 1 | ollama-cloud | kimi-k2.6 | https://ollama.com/v1 | api_key |
| 2 | openrouter | google/gemma-4-31b-it:free | https://openrouter.ai/api/v1 | api_key |
| 3 | openrouter | nvidia/nemotron-3-super-120b-a12b:free | https://openrouter.ai/api/v1 | api_key |

### 2.6 light
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | ollama-cloud | deepseek-v4-flash | https://ollama.com/v1 | api_key |
| 1 | ollama-cloud | deepseek-v4-flash | https://ollama.com/v1 | api_key |
| 2 | openrouter | google/gemma-4-31b-it:free | https://openrouter.ai/api/v1 | api_key |
| 3 | openrouter | nvidia/nemotron-3-super-120b-a12b:free | https://openrouter.ai/api/v1 | api_key |

### 2.7 denji
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | ollama-cloud | deepseek-v4-flash | https://ollama.com/v1 | api_key |
| 1 | ollama-cloud | deepseek-v4-flash | https://ollama.com/v1 | api_key |
| 2 | ollama-cloud | kimi-k2.6 | https://ollama.com/v1 | api_key |
| 3 | openrouter | google/gemma-4-31b-it:free | https://openrouter.ai/api/v1 | api_key |
| 4 | openrouter | nvidia/nemotron-3-super-120b-a12b:free | https://openrouter.ai/api/v1 | api_key |

### 2.8 quan
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | ollama-cloud | deepseek-v4-flash | https://ollama.com/v1 | api_key |
| 1 | ollama-cloud | deepseek-v4-pro | https://ollama.com/v1 | api_key |
| 2 | ollama-cloud | glm-5.1 | https://ollama.com/v1 | api_key |
| 3 | ollama-cloud | kimi-k2.6 | https://ollama.com/v1 | api_key |
| 4 | openrouter | google/gemma-4-31b-it:free | https://openrouter.ai/api/v1 | api_key |
| 5 | openrouter | nvidia/nemotron-3-super-120b-a12b:free | https://openrouter.ai/api/v1 | api_key |

### 2.9 wesker-ops
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | ollama-cloud | gemma4:31b | https://ollama.com/v1 | api_key |
| 1 | ollama-cloud | kimi-k2.6 | https://ollama.com/v1 | api_key |
| 2 | ollama-cloud | glm-5.1 | https://ollama.com/v1 | api_key |
| 3 | nous | qwen/qwen3.6-plus | https://inference-api.nousresearch.com/v1 | oauth_device_code |
| 4 | openrouter | nvidia/nemotron-3-super-120b-a12b | https://openrouter.ai/api/v1 | api_key |

### 2.10 denji-ledger
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | ollama-cloud | gemma4:31b | https://ollama.com/v1 | api_key |
| 1 | ollama-cloud | glm-5.1 | https://ollama.com/v1 | api_key |
| 2 | ollama-cloud | kimi-k2.6 | https://ollama.com/v1 | api_key |
| 3 | nous | qwen/qwen3.6-plus | https://inference-api.nousresearch.com/v1 | oauth_device_code |
| 4 | openrouter | nvidia/nemotron-3-super-120b-a12b | https://openrouter.ai/api/v1 | api_key |

### 2.11 denji-reviewer
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | ollama-cloud | deepseek-v4-flash | https://ollama.com/v1 | api_key |
| 1 | ollama-cloud | glm-5.1 | https://ollama.com/v1 | api_key |
| 2 | ollama-cloud | kimi-k2.6 | https://ollama.com/v1 | api_key |
| 3 | nous | qwen/qwen3.6-plus | https://inference-api.nousresearch.com/v1 | oauth_device_code |
| 4 | openrouter | nvidia/nemotron-3-super-120b-a12b | https://openrouter.ai/api/v1 | api_key |

### 2.12 mrhermagi
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | ollama-cloud | glm-5.2 | https://ollama.com/v1 | api_key |
| 1 | ollama-cloud | gemma3:27b | https://ollama.com/v1 | api_key |
| 2 | ollama-cloud | glm-5.2 | https://ollama.com/v1 | api_key |

### 2.13 dezzy
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | ollama-cloud | minimax-m3 | https://ollama.com/v1 | api_key |
| 1 | ollama-cloud | deepseek-v4-flash | https://ollama.com/v1 | api_key |
| 2 | openrouter | google/gemma-4-31b-it:free | https://openrouter.ai/api/v1 | api_key |
| 3 | openrouter | nvidia/nemotron-3-super-120b-a12b:free | https://openrouter.ai/api/v1 | api_key |

### 2.14 gojo-mailbox
| Slot | Provider | Model | Endpoint | Auth |
|------|----------|-------|----------|------|
| Main | ollama-cloud | gemma4:31b | https://ollama.com/v1 | api_key |
| 1 | ollama-cloud | kimi-k2.6 | https://ollama.com/v1 | api_key |
| 2 | openrouter | google/gemma-4-31b-it:free | https://openrouter.ai/api/v1 | api_key |
| 3 | openrouter | nvidia/nemotron-3-super-120b-a12b:free | https://openrouter.ai/api/v1 | api_key |

### 2.15 content-strategist
- No `agents/content-strategist/config.yaml` present in repo. Prior audit inferred mains from existing evidence; no confirmed live chain available.

---

## 3) Ramifications via Remii Lens

### 3.1 Latency
- **Finding:** current model-then-next order adds tail latency when slots 1..N-1 fail. For profiles with duplicate slots (`ollama-cloud / deepseek-v4-flash` repeated), retries add no diversification benefit while still incurring round-trip time before advancing.
- **Source:** `hermes_cli/runtime_provider.py` resolves each slot sequentially; no parallel hedging in `get_fallback_chain` path.

### 3.2 Cost
- **Finding:** free OpenRouter fallbacks (`gemma-4-31b-it:free`, `nemotron-3-super-120b-a12b:free`) are lowest-cost last resort, but reaching them requires exhausting paid/slower slots first. If pool exhaustion had been implemented per provider, cost would have been more predictable within a slot.
- **Source:** `models.py:145-153` OPENROUTER_MODELS free-tier entries; `audits/provider-fallback-audit-2026-09-03.md`.

### 3.3 Rate-Limit / Exhaustion
- **Finding:** with `credential_pool_strategies: ollama-cloud: fill_first`, multiple keys for the same provider/model are rotated across requests, but a single failing request still consumes one key/try before the chain advances. Under burst failure, this can exhaust multiple keys on one slot before moving on if retries happen at the request layer.
- **Source:** `config.yaml.example:40-43`, `runtime_provider.py:498-644`.

### 3.4 Prompt-Cache Invalidation
- **Finding:** each slot change risks changing provider/model/base_url, which changes runtime identity and invalidates cached prefixes. pool-then-next within the same provider/model would reduce unnecessary cache invalidations by staying on one route until all keys are exhausted.
- **Source:** `AGENTS.md` per-conversation prompt caching contract.

### 3.5 Credential Pool Rotation
- **Finding:** current behaviour conflates "pool rotation" with "fallback advancement." Rotation is intra-request via pool policy; fallback advancement is inter-slot. This means rotation does not help availability when a provider is partially throttled unless multiple keys are tried in one slot.
- **Source:** `agent/credential_pool.py` `CredentialPool.select()` / `PooledCredential`.

### 3.6 Failure-Mode Loops
- **Finding:** duplicate slots in wesker (`deepseek-v4-flash` twice) and mrhermagi (`glm-5.2` twice) create no-ops that delay diversification and can mask provider-level outages.
- **Source:** configs read above.

### 3.7 Observability
- **Finding:** no expanded pool-level attempt telemetry is visible in `get_fallback_chain`. A pool-then-next strategy needs per-pool-member attempt counters; current code tracks slot-level fallback only.
- **Source:** `audits/provider-fallback-audit-2026-09-03.md` method audit; `fallback_config.py`.

---

## 4) Proposed Better Approach

Pool-then-next is not inherently harmful, but the current implementation does **not** do it. If the fleet wants to exhaust a provider pool before advancing, the cleanest change is:

### 4.1 Ranked Pools + Sticky Failover
- Replace flat slot list with ranked fallback groups.
- Each group has one provider/model identity. Before advancing to group N+1, attempt all available pool entries for group N, ordered by health/priority.
- After a successful pool entry is identified, "stick" to that entry for subsequent requests until failure threshold is crossed.

### 4.2 Hedged Tries
- For latency-critical leads, issue hedged concurrent attempts to the top 2 pool entries of the same provider/model; use first success, cancel the other. Avoids waiting for sequential pool exhaustion while still keeping a single provider identity.

### 4.3 Circuit-Breaker Per Provider/Model
- After N failures within a time window for provider/model P/M, open circuit and skip that group entirely for T seconds. Prevents burning multiple pool keys on an endpoint that is down or quota-exhausted.

### 4.4 Sticky Failover
- Once a profile has fallen back to a secondary slot, persist that choice for the session or a cooldown window. Avoids re-walking the primary chain on every request when the primary is known-degraded.

### 4.5 Suggested Config Shape (conceptual)
```yaml
fallback_groups:
  - provider: ollama-cloud
    model: deepseek-v4-flash
    pool_mode: exhaust_first   # try all pool keys before advancing
    max_attempts: 3
    circuit_breaker:
      window: 60s
      threshold: 3
    sticky: true
    sticky_ttl: 300s
```

**Net verdict:** retain Main+4, but change fallback semantics from flat model-slot list to pool-aware groups. The immediate win is removing duplicate slots and adding a circuit-breaker so a failing provider does not burn pool keys or latency before advancing.

---

*Read-only audit produced for kanban task t_b6d1bcc2. No config files modified.*
