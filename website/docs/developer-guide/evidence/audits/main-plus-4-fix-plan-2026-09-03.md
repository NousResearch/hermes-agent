# Main+4 Fix Plan + Docs Comparison
## Scope: ~/repos/KenseiAgent | Read-only draft | Kanban: t_b6d1bcc2
## Run date: 2026-09-03

---

## 1) Resolved IDs

### openai-codex model ID (Sol vs Luna per role)
- **Source files:** `hermes_cli/codex_models.py:15-51` (`DEFAULT_CODEX_MODELS`), `hermes_cli/providers.py:63-67` (`HERMES_OVERLAYS["openai-codex"]`)
- **Confirmed IDs:** `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`
- **Role mapping (fleet policy):**
  - Leads / arch / security / final-QA → **Sol** = `openai-codex / gpt-5.6-sol`
  - Workers → **Luna** = `openai-codex / gpt-5.6-luna`
- **Provider overlay:** transport=`codex_responses`, auth=`oauth_external`, base_url=`https://chatgpt.com/backend-api/codex`
- **Notes:** Forward-compat synthesis and context variants (`-900k`) are auto-appended by `_finalize_codex_models()`. Live API discovery is account-scoped via `ChatGPT-Account-Id` header; curated floor remains offline-safe.

### local qwen3.8-27b slug
- **Source files:** `hermes_cli/local_runtime/catalog.json:5` (`id: qwen3.8-27b`), relay topology from prior audit
- **Confirmed slug:** `qwen3.8-27b`
- **Provider identity:** surfaced as `llamacpp` provider via `resolve_llamacpp_endpoint()`; relay on `127.0.0.1:11410` → manager on `11401`, SSE direct `11401` broken
- **Display label:** `Qwen3.8 27B`
- **Fallback slot target:** slot 4 (final fallback) across all profiles

---

## 2) Per-Profile Main+4 Table

Rule: preserve current Main; keep Nous entries as-is; append codex + qwen-local as slots 3-4.
Leads → Sol; workers → Luna.

| # | Profile | Kind | Main (slot 1) | Fallback 2 | Fallback 3 | Fallback 4 | openai-codex slot | qwen-local slot |
|---|---------|------|---------------|------------|------------|------------|-------------------|-----------------|
| 1 | wesker | lead | ollama-cloud / deepseek-v4-flash | ollama-cloud / deepseek-v4-pro | **openai-codex / gpt-5.6-sol** | **llamacpp / qwen3.8-27b** | slot 3 = Sol | slot 4 = local |
| 2 | octacon | lead | ollama-cloud / glm-5.1 | ollama-cloud / deepseek-v4-pro | **openai-codex / gpt-5.6-sol** | **llamacpp / qwen3.8-27b** | slot 3 = Sol | slot 4 = local |
| 3 | ceecee | lead | opencode-go / minimax-m3 | opencode-zen / deepseek-v4-flash | **openai-codex / gpt-5.6-sol** | **llamacpp / qwen3.8-27b** | slot 3 = Sol | slot 4 = local |
| 4 | remii | lead | ollama-cloud / deepseek-v4-flash | openrouter / gemma-4-31b-it:free | **openai-codex / gpt-5.6-sol** | **llamacpp / qwen3.8-27b** | slot 3 = Sol | slot 4 = local |
| 5 | gojo | lead | ollama-cloud / deepseek-v4-flash | ollama-cloud / kimi-k2.6 | **openai-codex / gpt-5.6-sol** | **llamacpp / qwen3.8-27b** | slot 3 = Sol | slot 4 = local |
| 6 | light | lead | ollama-cloud / deepseek-v4-flash | ollama-cloud / deepseek-v4-flash | **openai-codex / gpt-5.6-sol** | **llamacpp / qwen3.8-27b** | slot 3 = Sol | slot 4 = local |
| 7 | denji | lead | ollama-cloud / deepseek-v4-flash | ollama-cloud / deepseek-v4-flash | **openai-codex / gpt-5.6-sol** | **llamacpp / qwen3.8-27b** | slot 3 = Sol | slot 4 = local |
| 8 | quan | lead | ollama-cloud / deepseek-v4-flash | ollama-cloud / deepseek-v4-pro | **openai-codex / gpt-5.6-sol** | **llamacpp / qwen3.8-27b** | slot 3 = Sol | slot 4 = local |
| 9 | wesker-ops | worker | ollama-cloud / gemma4:31b | ollama-cloud / kimi-k2.6 | **openai-codex / gpt-5.6-luna** | **llamacpp / qwen3.8-27b** | slot 3 = Luna | slot 4 = local |
| 10 | denji-ledger | worker | ollama-cloud / gemma4:31b | ollama-cloud / glm-5.1 | **openai-codex / gpt-5.6-luna** | **llamacpp / qwen3.8-27b** | slot 3 = Luna | slot 4 = local |
| 11 | denji-reviewer | worker | ollama-cloud / deepseek-v4-flash | ollama-cloud / glm-5.1 | **openai-codex / gpt-5.6-luna** | **llamacpp / qwen3.8-27b** | slot 3 = Luna | slot 4 = local |
| 12 | mrhermagi | lead | ollama-cloud / glm-5.2 | ollama-cloud / gemma3:27b | **openai-codex / gpt-5.6-sol** | **llamacpp / qwen3.8-27b** | slot 3 = Sol | slot 4 = local |
| 13 | dezzy | lead | ollama-cloud / minimax-m3 | ollama-cloud / deepseek-v4-flash | **openai-codex / gpt-5.6-sol** | **llamacpp / qwen3.8-27b** | slot 3 = Sol | slot 4 = local |
| 14 | gojo-mailbox | worker | ollama-cloud / gemma4:31b | ollama-cloud / kimi-k2.6 | **openai-codex / gpt-5.6-luna** | **llamacpp / qwen3.8-27b** | slot 3 = Luna | slot 4 = local |
| 15 | content-strategist | worker | ollama-cloud / *(no model set)* | ollama-cloud / *(no fallback)* | **openai-codex / gpt-5.6-luna** | **llamacpp / qwen3.8-27b** | slot 3 = Luna | slot 4 = local |

> Note: `content-strategist` config directory not found under `agents/`; mains inferred from prior audit.

---

## 3) YAML Snippets Per Profile (Sahil yes/no)

Each block below is a self-contained config delta for one profile. `sahil_approve: yes` enables the change; `no` keeps current state.

### Profile 1 — wesker (lead)

```yaml
profile: wesker
kind: lead
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: deepseek-v4-flash
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-sol
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
  note: relay 11410 -> manager 11401; SSE 11401 broken
```

### Profile 2 — octacon (lead)

```yaml
profile: octacon
kind: lead
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: glm-5.1
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-sol
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

### Profile 3 — ceecee (lead)

```yaml
profile: ceecee
kind: lead
sahil_approve: yes
current_main:
  provider: opencode-go
  model: minimax-m3
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-sol
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

### Profile 4 — remii (lead)

```yaml
profile: remii
kind: lead
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: deepseek-v4-flash
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-sol
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

### Profile 5 — gojo (lead)

```yaml
profile: gojo
kind: lead
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: deepseek-v4-flash
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-sol
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

### Profile 6 — light (lead)

```yaml
profile: light
kind: lead
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: deepseek-v4-flash
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-sol
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

### Profile 7 — denji (lead)

```yaml
profile: denji
kind: lead
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: deepseek-v4-flash
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-sol
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

### Profile 8 — quan (lead)

```yaml
profile: quan
kind: lead
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: deepseek-v4-flash
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-sol
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

### Profile 9 — wesker-ops (worker)

```yaml
profile: wesker-ops
kind: worker
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: gemma4:31b
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-luna
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

### Profile 10 — denji-ledger (worker)

```yaml
profile: denji-ledger
kind: worker
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: gemma4:31b
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-luna
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

### Profile 11 — denji-reviewer (worker)

```yaml
profile: denji-reviewer
kind: worker
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: deepseek-v4-flash
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-luna
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

### Profile 12 — mrhermagi (lead)

```yaml
profile: mrhermagi
kind: lead
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: glm-5.2
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-sol
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

### Profile 13 — dezzy (lead)

```yaml
profile: dezzy
kind: lead
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: minimax-m3
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-sol
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

### Profile 14 — gojo-mailbox (worker)

```yaml
profile: gojo-mailbox
kind: worker
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: gemma4:31b
  preserve: true
slot_3:
  provider: openai-codex
  model: gpt-5.6-luna
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

### Profile 15 — content-strategist (worker)

```yaml
profile: content-strategist
kind: worker
sahil_approve: yes
current_main:
  provider: ollama-cloud
  model: null
  preserve: true
  note: no model currently set
slot_3:
  provider: openai-codex
  model: gpt-5.6-luna
  base_url: https://chatgpt.com/backend-api/codex
  transport: codex_responses
  auth: oauth_external
slot_4:
  provider: llamacpp
  model: qwen3.8-27b
  endpoint: http://127.0.0.1:11401
```

---

## 4) Docs-Diff: Our Implementation vs Upstream Hermes Docs

### 4a. Source under audit
- **Our implementation:** `hermes_cli/models.py`
  - `CANONICAL_PROVIDERS` list (lines 1314–1354): ordered `ProviderEntry(slug, label, tui_desc)` tuples
  - `_PROVIDER_MODELS` dict (lines 271–814): per-provider curated model IDs
  - `provider_catalog()` + `CANONICAL_PROVIDERS` feed both CLI picker (`hermes model`) and `/model` endpoint
  - Auto-extension from `providers/` plugin registry for non-api-key auth providers (lines 1361–1373)
- **Upstream reference:** https://hermes-agent.nousresearch.com/docs/integrations/providers
  - Markdown table of providers with setup instructions
  - Separate Nous Portal page, fallback-providers page

### 4b. Mismatches (REPORT ONLY — no edits)

| # | Area | Our code | Upstream docs | Verdict |
|---|------|----------|---------------|---------|
| 1 | Provider count in picker | ~50+ entries in CANONICAL_PROVIDERS including auto-extended plugin providers | Docs table shows ~30 named providers; plugin-registered providers omitted | Mismatch: our picker is broader; docs don't enumerate plugin-only providers |
| 2 | Provider grouping | `PROVIDER_GROUPS` folds 10+ groups (kimi, minimax, xai, google, openai, qwen, opencode, copilot, tencent) | Docs table is flat; no grouping | Mismatch: grouping is UX-only in our code, invisible in docs |
| 3 | openai-codex label | `"ChatGPT or Codex Subscription"` | Docs: "OpenAI Codex" with note "ChatGPT OAuth, uses Codex models" | Mismatch: our label is longer; docs use shorter table-cell form |
| 4 | Nous model list | `_PROVIDER_MODELS["nous"]` has ~45 curated entries plus live Portal augmentation (`union_with_portal_free_recommendations`, `union_with_portal_paid_recommendations`) | Docs: "300+ frontier agentic models" — no enumerated list | Mismatch: we enumerate; docs aggregate. Live augmentation is invisible in docs |
| 5 | OpenRouter free-tier entries | Curated floor `OPENROUTER_MODELS` includes `:free` variants (lines 145–153); worker fallbacks use them | Docs mention OpenRouter as "Pay-per-use API aggregator" — no free-tier enumeration | Mismatch: our code surfaces free models; docs treat OpenRouter as paid-only |
| 6 | Local GGUF / llamacpp provider | `llamacpp` resolved via `resolve_llamacpp_endpoint()`; appears as "Local" in picker when server reachable | Docs section "llama.cpp / llama-server — CPU & Metal Inference" and "LM Studio" cover local inference, but no `llamacpp` provider slug | Mismatch: our code has first-class `llamacpp` provider identity; docs don't list it as a discrete provider |
| 7 | openai-codex transport | `codex_responses` (overlay `transport="codex_responses"`) | Docs don't mention transport/wire-protocol differences between OpenAI Codex and OpenAI API | Mismatch: our code distinguishes Codex Responses API; docs show both under generic setup |
| 8 | Vercel AI Gateway slug | `ai-gateway` in `CANONICAL_PROVIDERS`, derived from `VERCEL_AI_GATEWAY_MODELS` | Docs: "AI Gateway" with `AI_GATEWAY_API_KEY`; no mention of provider slug `ai-gateway` vs `vercel` | Mismatch: our canonical slug is `ai-gateway`; docs imply `vercel` alias |
| 9 | Provider alias normalization | 100+ aliases in `ALIASES` (providers.py:292–436) and `_PROVIDER_ALIASES` (models.py:1485–1581) | Docs table lists only canonical names; no alias documentation | Mismatch: our code accepts many aliases; docs expose only canonical names |
| 10 | xAI OAuth vs xAI direct | Two separate entries: `xai` (direct API) and `xai-oauth` (SuperGrok / Premium+) | Docs: "xAI (Grok) — Responses API" and "xAI Grok OAuth (SuperGrok)" as two rows | Match: docs align with our split, though our labels differ slightly |
| 11 | Qwen family | 7 slugs: `alibaba`, `alibaba-cn`, `alibaba-coding-plan`, `alibaba-coding-plan-cn`, `alibaba-token-plan`, `alibaba-token-plan-cn`, `qwen-oauth` | Docs table shows: Qwen Cloud (Alibaba DashScope), Alibaba Cloud (Coding Plan), Qwen OAuth, MiniMax OAuth, StepFun | Mismatch: our code splits Alibaba into 6 endpoint-specific slugs; docs collapse to 2–3 entries |
| 12 | Nous provider label | `_LABEL_OVERRIDES["nous"] = "Nous Portal"` | Docs: "Nous Portal" | Match |

### 4c. Summary

Our `models.py CANONICAL_PROVIDERS + provider_catalog` is strictly more granular than upstream docs:
- More providers (plugin-extended)
- More models per provider (curated + live-augmented)
- Grouped UX not reflected in docs
- Local GGUF / `llamacpp` is a first-class provider in our code but only implied in docs
- Transport/protocol distinctions (`codex_responses` vs `openai_chat`) are implementation details absent from docs

No doc drift that blocks functionality. Upstream docs are a subset of our catalog.

---

## 5) Constraints & Notes

- **Nous provider:** preserved as-is across all 15 profiles; no removal proposed.
- **Local topology:** 11410 relay → 11401 manager; SSE direct 11401 broken; local model reached via `resolve_llamacpp_endpoint()` which auto-discovers manager.
- **Read-only:** no config files modified.
- **Batch loops over profiles:** not executed; deltas drafted per-profile above.

---

*Draft produced for kanban task t_b6d1bcc2. Awaiting Sahil approval per-profile.*
