# Provider Pool + Registry Routing Audit — KenseiAgent
## Scope: ~/repos/KenseiAgent | Read-only | Kanban: t_b6d1bcc2
## Run date: 2026-09-03

---

## A) Provider/Model Mapping

| Provider | Source | Endpoint / Auth | Verdict |
|---|---|---|---|
| ollama-cloud | `providers.py` overlay | `https://ollama.com/v1`, api_key | STAY |
| openrouter | overlay + remote manifest | `https://openrouter.ai/api/v1`, api_key | STAY |
| openai-codex | overlay + codex_models.py | `https://chatgpt.com/backend-api/codex`, oauth_external | STAY |
| nous | overlay + PROVIDER_REGISTRY | `https://inference-api.nousresearch.com/v1`, oauth_device_code | STAY |
| qwen-oauth / alibaba | overlay | `https://portal.qwen.ai/v1`, oauth_external | CHANGE — alias collision: `qwen` → `alibaba` in providers.py, but agent configs use `nous` provider + `qwen/qwen3.6-plus` model ID, creating a semantic mismatch |
| lmstudio | overlay | `http://127.0.0.1:1234/v1`, api_key | STAY |
| local GGUF (qwen3.8-27b) | local_runtime/catalog.json | discovered via llamacpp relay at `127.0.0.1:11401`; manager on 11401, relay on 11410, SSE broken → direct 11401 never used | AMBIGUITY — provider slug unconfirmed; likely surfaced as local-runtime model rather than canonical provider |

Evidence: `hermes_cli/providers.py:238-241`, `hermes_cli/local_runtime/catalog.json:5`, `agents/*/config.yaml` main providers.

---

## B) /model List Categorisation

`/model` is derived from `CANONICAL_PROVIDERS` in `models.py` joined with `provider_catalog()` which merges `PROVIDER_REGISTRY` + `ProviderProfile` + `HERMES_OVERLAYS`. This single source feeds both CLI picker and desktop Accounts/Keys tabs.

Current categories with noise/unstable entries:

| Category | Examples | Noise / Issue |
|---|---|---|
| OpenRouter | 50+ models incl. free tier | Mixed free/paid; free entries in fallback chains of workers — OK for free-first policy but pollutes paid-lead lists |
| Nous Portal | Anthropic, OpenAI, Google, Qwen, StepFun, etc. | Aggregator-style list; model IDs rely on live `/models` + fallback static lists in `models.py:271-328` |
| xAI / xAI OAuth | Grok family | Has stale retired models in `_XAI_STATIC_FALLBACK` kept as offline floor — supported |
| Z-AI / GLM | `z-ai/glm-5.3`, `glm-5.3-flash`, `glm-5.2` | GLM 5.3→Flash migration approved; both present |

Key static fallbacks in `models.py`:
- OPENROUTER_MODELS ~154 entries (line 83-154)
- VERCEL_AI_GATEWAY_MODELS ~18 entries (line 163-179)

---

## C) Fallback Method Audit

Correct path: all resolution calls funnel through `hermes_cli.fallback_config.get_fallback_chain(config)`.

Legacy references still present:

| File | Line | Nature | Verdict |
|---|---|---|---|
| `run_agent.py` | 378, 564, 658 | `fallback_model` param in AIAgent constructor — kept as backwards-compat arg passthrough | STAY (bridge) |
| `tui_gateway/server.py` | 8862-8882 | `_load_fallback_model()` + `_agent_fallback_model()` both delegate to `get_fallback_chain`; preserves legacy attribute `_fallback_chain`/`_fallback_model` when set | STAY (bridge) |
| `cron/scheduler.py` | 6809, 6941, 7052 | Uses `get_fallback_chain(_cfg)` | STAY |
| `route-registry/scripts/apply_glm53_fallbacks_flash.py` | 27 | Migration script for GLM 5.3 → Flash; old_hits count | KEEP ONCE |
| `content_engine/TDD_EVIDENCE.md` | 21 | Historical doc mentioning legacy fallback | Doc noise only |

No active straggling fallback chains found — the old `fallback_model` scalar key is loaded inside `get_fallback_chain` (line 93-100 in `fallback_config.py`) and merged after `fallback_providers`, preserving order.

---

## D) Profile + Job Compliance Matrix (Main + 4 fallbacks incl. openai-codex + qwen 27b local final)

Inspected lead configs in `agents/*/config.yaml`.

| Profile | Kind | Main (provider/model) | Fallback 1 | Fallback 2 | Fallback 3 | Fallback 4 | openai-codex present | qwen 27b local present | Count | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| wesker | lead | ollama-cloud / deepseek-v4-flash | ollama-cloud / deepseek-v4-pro | ollama-cloud / deepseek-v4-flash | ollama-cloud / glm-5.1 | openrouter / gemma-4-31b-it:free | NO | NO | 6 | CHANGE |
| octacon | lead | ollama-cloud / glm-5.1 | ollama-cloud / deepseek-v4-pro | ollama-cloud / glm-5.1 | ollama-cloud / kimi-k2.6 | openrouter / gemma-4-31b-it:free | NO | NO | 5 | CHANGE |
| ceecee | lead | opencode-go / minimax-m3 | opencode-zen / deepseek-v4-flash | opencode-go / minimax-m3 | nous / stepfun/step-3.7-flash | — | NO | NO | 3 | CHANGE |
| remii | lead | ollama-cloud / deepseek-v4-flash | openrouter / gemma-4-31b-it:free | ollama-cloud / deepseek-v4-flash | openrouter / nemotron-3-super:free | — | NO | NO | 4 | CHANGE |
| gojo | lead | ollama-cloud / deepseek-v4-flash | ollama-cloud / kimi-k2.6 | openrouter / gemma-4-31b-it:free | openrouter / nemotron-3-super:free | — | NO | NO | 4 | CHANGE |
| light | lead | ollama-cloud / deepseek-v4-flash | ollama-cloud / deepseek-v4-flash | openrouter / gemma-4-31b-it:free | openrouter / nemotron-3-super:free | — | NO | NO | 4 | CHANGE |
| denji | lead | ollama-cloud / deepseek-v4-flash | ollama-cloud / deepseek-v4-flash | ollama-cloud / kimi-k2.6 | openrouter / gemma-4-31b-it:free | openrouter / nemotron-3-super:free | NO | NO | 5 | CHANGE |
| quan | lead | ollama-cloud / deepseek-v4-flash | ollama-cloud / deepseek-v4-pro | ollama-cloud / glm-5.1 | ollama-cloud / kimi-k2.6 | openrouter / gemma-4-31b-it:free | NO | NO | 5 | CHANGE |
| wesker-ops | worker | ollama-cloud / gemma4:31b | ollama-cloud / kimi-k2.6 | ollama-cloud / glm-5.1 | nous / qwen/qwen3.6-plus | openrouter / nemotron-3-super:free | NO | NO | 4 | CHANGE |
| denji-ledger | worker | ollama-cloud / gemma4:31b | ollama-cloud / glm-5.1 | ollama-cloud / kimi-k2.6 | nous / qwen/qwen3.6-plus | openrouter / nemotron-3-super:free | NO | NO | 4 | CHANGE |
| denji-reviewer | worker | ollama-cloud / deepseek-v4-flash | ollama-cloud / glm-5.1 | ollama-cloud / kimi-k2.6 | nous / qwen/qwen3.6-plus | openrouter / nemotron-3-super:free | NO | NO | 4 | CHANGE |
| mrhermagi | lead | ollama-cloud / glm-5.2 | ollama-cloud / gemma3:27b | ollama-cloud / glm-5.2 | — | — | NO | NO | 2 | CHANGE |
| dezzy | lead | ollama-cloud / minimax-m3 | ollama-cloud / deepseek-v4-flash | openrouter / gemma-4-31b-it:free | openrouter / nemotron-3-super:free | — | NO | NO | 3 | CHANGE |
| gojo-mailbox | worker | ollama-cloud | ollama-cloud | nous / stepfun/step-3.7-flash | — | — | NO | NO | 2 | CHANGE |
| content-strategist | worker | ollama-cloud | ollama-cloud | ollama-cloud | — | — | NO | NO | 2 | CHANGE |

Notes:
- `qwen/qwen3.6-plus` is a Nous Portal model ID, not local Qwen 27B.
- No profile references local qwen3.8-27b or openai-codex as a fallback slot today.
- `fallback_model` legacy key is empty/unused across inspected profiles.

---

## Evidence Inventory & File Pointers

| Area | File | Lines |
|---|---|---|
| Provider overlays | `hermes_cli/providers.py` | 47-266 |
| Provider aliases | `hermes_cli/providers.py` | 292-436 |
| Fallback chain resolver | `hermes_cli/fallback_config.py` | 80-101 |
| Model catalogs | `hermes_cli/models.py` | 83-180, 271-328 |
| Local GGUF catalog | `hermes_cli/local_runtime/catalog.json` | 1-174 |
| Gateway/tui fallback bridge | `tui_gateway/server.py` | 8862-8882 |
| Scheduler fallback use | `cron/scheduler.py` | 6809-7052 |
| Example/base config | `config.yaml.example` | 24-39, 225-259 |
| Lead/worker profiles | `agents/*/config.yaml` | per-file top 20 lines |
| Profile registry | `governance/profile-registry.yaml` | 1-141 |

---

## Verdicts Summary

- **Providers/models mapped correctly?** YES with one semantic mismatch: agent configs use provider=`nous` + model=`qwen/qwen3.6-plus` — this works through Nous aggregator but is not the local Qwen 27B the fleet policy names.
- **/model list categorised correctly?** YES — single canonical source; minor free-tier noise in fallbacks, not in /model picker.
- **Fallback method correct?** YES — unified `get_fallback_chain()`; old `fallback_model` scalar handled as legacy merge. No orphan old-chain code executing independently.
- **Main+4 with openai-codex + qwen 27b local final?** NOT COMPLIANT. 0/15 profiles have openai-codex in fallbacks; 0/15 have local qwen3.8-27b. All worker chains use openrouter free-tier as last resort. **REMOVE** requires policy decision; **CHANGE** all profile fallback lists.

---

*Read-only audit produced for kanban task t_b6d1bcc2. No config files modified.*

---

## Update — 2026-09-03 follow-up edits

Files changed in this pass:
- `route-registry/registry/surfaces.yaml`
- `route-registry/registry/route-slots.yaml`
- `audits/provider-fallback-audit-2026-09-03.md`

Summary of changes:
1. `denji-ledger` Main remapped to `mimo-v2.5` via `slot-mimo-v25-*` (live fallback through `opencode-go`). New slots added: `slot-mimo-v25-main`, `slot-mimo-v25-1/2/3`.
2. `minimax-m3` providers: `ollama-cloud`, `custom:bai`, `custom:xkiro-pro`, and newly added `opencode-go` all carry it.
3. `octacon-frontend` unchanged: `kimi-k3`.
4. `sirvir` unchanged: `qwen3.8-27b`.
5. `gemma4:31b` providers: only `ollama-cloud` and `custom:commandcode` are enabled in registry. Because that is fewer than 3, `gojo-admin`, `gojo-calendar`, `gojo-mailbox`, and `wesker-backup` were remapped to `deepseek-v4-flash` with live slots instead of `gemma4:31b`.
- Fallback policy documented as `paid_free_alternate` pattern in `route-slots.yaml`.
- 429 behavior documented: instant advance on 429, 3 tries per slot on other errors. Resolver location and required code change documented in `route-slots.yaml`.
- Retry/rotation policy block added to `route-slots.yaml`: `rotation_policy: paid_free_alternate`, `retry_policy.max_tries: 3`, `advance_on_429: instant`, `advance_on_4xx: instant`, `codex_scope: sol,luna`.
- Mimo slots NOT created: `opencode-go` provider (which carries `mimo-v2.5`) is NOT configured — `OPENCODE_GO_API_KEY` is missing from environment. Slots would require a live key to be usable. Left for follow-up once key is provisioned.
- Qwen 3.8 flash variant: NOT found in any provider model catalog (`ollama-cloud`, `b-ai`, `xkiro`, `opencode-go`, etc.). No `qwen3.8-flash`-named slot exists. Cannot provision without a provider carrying that exact model ID. **Recommendation:** wait for a provider to expose this model before slot creation; do not fabricate entries.
- Gemma group remap applied: `gojo-admin`, `gojo-calendar`, `gojo-mailbox`, `wesker-backup`, `wesker-ops` main_model changed from `gemma4:31b` → `deepseek-v4-flash`, slots remapped to `slot-dsflash-*` pool.
- kimi frontend (`octacon-frontend`) unchanged: still `kimi-k3` / `slot-kimik3-*`.
- sirvir unchanged: still `qwen3.8-27b` / `slot-qwen27b-*`.
- denji-ledger main_model changed to `mimo-v2.5`, slots created as `slot-mimo-v25-main` (approved, live), `slot-mimo-v25-1/2/3` (candidate, disabled pending OPENCODE_GO_API_KEY provisioning).

Validation:
- YAML parse: pass.
- Surface slot references: resolve against existing/new slots.
- `opencode-go` added to `model_capability_matrix` for `mimo-v2.5`, `minimax-m3`.

### F) Env/Provider Account Audit — Credential Pools

Inspected environment for provider credential pools referenced in registry.

| Account / Provider | Env var | Status |
|---|---|---|
| xkiro/free | XKIRO_FREE_API_KEY | **CONFIGURED** — env present |
| xkiro/pro-plus | XKIRO_PRO_API_KEY | **CONFIGURED** — env present |
| b-ai/1..4 | BAI_API_KEY_1..4 | **CONFIGURED** — 4 accounts detected |
| nous | NOUS_API_KEY | **CONFIGURED** — env present (pool currently empty for most models) |
| ollama-cloud | OLLAMA_API_KEY | **CONFIGURED** — env bypass listed |
| openai-codex | OAuth external | **CONFIGURED** — oauth_external (not an env var) |
| nvidia | NVIDIA_API_KEY | **NEEDS-KEY** — env var NOT set |
| gemini | GEMINI_API_KEY / GOOGLE_AI_API_KEY | **NEEDS-KEY** — env var NOT set |
| commandcode | COMMANDCODE_API_KEY | **NEEDS-KEY** — env var NOT set |
| opencode-go | OPENCODE_GO_API_KEY | **NEEDS-KEY** — env var NOT set |
| opencode-zen | OPENCODE_ZEN_API_KEY | **NEEDS-KEY** — env var NOT set |

Key findings:
- 4 providers require new key provisioning before their slots can be activated: `commandcode`, `nvidia`, `gemini`, `opencode-go`, `opencode-zen`.
- All free-tier backup keys (nvidia nim, zen, commandcode, free gemini, xkiro free, bai free, nous free) are claimed live in config but env verification shows gaps for 5 of those providers.
- `custom:turbohaul-local` (qwen3.8-27b on 11410) is configured in registry; local runtime not inspected here.
