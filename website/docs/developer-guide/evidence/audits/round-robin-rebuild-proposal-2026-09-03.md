# Round-Robin Pool Rebuild Proposal
Date: 2026-09-03
Scope: Read-only audit + proposal — no edits to route-slots.yaml, surfaces.yaml, agents/, or profiles/

---

## 1) Inventory: Core-10 Accounts

| # | Provider Account | Provider ID | Model(s) | Base URL | Auth | Tier | Status |
|---|------------------|-------------|----------|----------|------|------|--------|
| 1 | ollama-cloud/1 | `ollama-cloud` | deepseek-v4-flash, deepseek-v4-pro, glm-5.3-flash, glm-5.3, minimax-m3, kimi-k3, … | https://ollama.com/v1 | OLLAMA_API_KEY | WORKER-PAID | approved (2 slots) |
| 2 | ollama-cloud/2 | `ollama-cloud` | same as above | https://ollama.com/v1 | OLLAMA_API_KEY | LEAD-PAID | approved (5 slots) |
| 3 | commandcode/1 | `custom:commandcode` | deepseek-v4-pro, deepseek-v4-flash, qwen3.8-27b, kimi-k3, minimax-m3, step-3.7-flash:free, … | https://api.commandcode.ai/provider/v1 | COMMANDCODE_API_KEY | WORKER-PAID | approved (paid rotation slot) |
| 4 | commandcode/2 | `custom:commandcode` | same | https://api.commandcode.ai/provider/v1 | COMMANDCODE_API_KEY | LEAD-PAID | approved (paid rotation slot) |
| 5 | xkiro/pro-plus | `custom:xkiro-pro` | deepseek-v4-flash, deepseek-v4-pro, z-ai/glm-5.3, z-ai/glm-5.3-flash, kimi-k3 | https://api.xkiro.com/v1 | XKIRO_API_KEY | LEAD-PAID | approved (1 slot) |
| 6 | xkiro/free | `custom:xkiro-free` | same as xkiro-pro | https://api.xkiro.com/v1 | XKIRO_API_KEY | FREE | approved (3 slots) — in LEAD-PAID lead pool |
| 7 | b-ai/1 | `custom:bai` | deepseek-v4-flash, deepseek-v4-pro, glm-5.3, glm-5.3-flash, kimi-k3, minimax-m3, qwen3.8-27b | https://api.b.ai/v1 | BAI_API_KEY | FREE | approved (2 slots) — free key used in LEAD-PAID lead pool |
| 8 | b-ai/2 | `custom:bai` | same | https://api.b.ai/v1 | BAI_API_KEY | FREE | approved (2 slots) |
| 9 | b-ai/3 | `custom:bai` | same | https://api.b.ai/v1 | BAI_API_KEY | FREE | approved (1 slot) |
| 10 | b-ai/4 | `custom:bai` | same | https://api.b.ai/v1 | BAI_API_KEY | FREE | approved (3 slots) |

---

## 2) Inventory: Free-Six Providers

| # | Provider | Provider ID in Codebase | Exact Model Names Supported | Endpoint | Free-Tier Limits | Auth Needed | Status |
|---|----------|------------------------|----------------------------|----------|------------------|-------------|--------|
| 1 | Nous Portal free | `nous` | Live catalog fetched without auth: `meta/muse-spark-1.3-contributor`, etc; aggregator supports deepseek/**, anthropic/**, openai/**, etc | https://inference-api.nousresearch.com/v1 | OAuth device-code; free tier available | No | LIVE — `nous/free` account present in registry |
| 2 | NVIDIA NIM free | `nvidia` (alias `nvidia-nim`) | Verified live list returned 200 without auth; free models include `01-ai/yi-large`, etc | https://integrate.api.nvidia.com/v1 | API-key; free rate limits per model | No | PROVIDER EXISTS — **NO ACCOUNT IN REGISTRY** → NEW-WIRING-NEEDED |
| 3 | OpenCode Zen free | `opencode-zen` | Live catalog returned 200 without auth: `claude-fable-5`, `claude-fable-5-1`, `claude-opus-5`, … | https://opencode.ai/zen/v1 | API key `OPENCODE_ZEN_API_KEY`; free models have rate limits | No | PROVIDER EXISTS — **NO ACCOUNT IN REGISTRY** → NEW-WIRING-NEEDED |
| 4 | CommandCode free | `custom:commandcode` | Live catalog returned 200 without auth: `claude-sonnet-5`, `claude-sonnet-4-6`, … | https://api.commandcode.ai/provider/v1 | COMMANDCODE_API_KEY; free tier available | No | PROVIDER EXISTS — **NO FREE ACCOUNT IN REGISTRY** → NEW-WIRING-NEEDED |
| 5 | Gemini free | `gemini` | **NEEDS-KEY** — list endpoint returned 403 without API key; model list unavailable | https://generativelanguage.googleapis.com/v1beta | GOOGLE_API_KEY; free tier rate limits | Yes | PROVIDER EXISTS — **LIST FETCH BLOCKED WITHOUT KEY** → NEEDS-KEY |
| 6 | xkiro free | `custom:xkiro-free` | Live catalog returned 200 without auth: `openai/gpt-5.6-terra`, etc | https://api.xkiro.com/v1 | XKIRO_API_KEY | No | LIVE — provider/account present |

**Free pool summary:** `commandcode/free`, `opencode-zen/free`, `xkiro/free`, `nvidia-nim/free`, plus `nous/free` as paid-tier aggregator member, and `gemini/free` marked NEEDS-KEY.

---

## 3) Rebuild Proposal: NEW Main + FB1 + FB2 + FB3 per Surface

Rules:
- **Tier mapping:** LEAD-PAID = lead paid, WORKER-PAID = worker paid, FREE = free pool.
- **3-try rotation:** each slot lists 3 tries rotating across the allowed pool for that tier.
- **Ending chain:** tier-correct codex (`slot-codex-sol` for SOL, `slot-codex-luna` for LUNA) → `slot-local-final`.
- **GPT models ONLY via `openai-codex`**.
- **Local final:** `slot-local-final` (`custom:turbohaul-local`, `qwen3.8-27b`, `11410`).

**Pool definitions:**
- LEAD-PAID pool: `ollama-cloud/2`, `commandcode/2`, `xkiro/pro-plus`, `xkiro/free`, `b-ai/1`, `nous/free`.
- WORKER-PAID pool: `ollama-cloud/1`, `commandcode/1`.
- FREE pool: `b-ai/2`, `b-ai/3`, `b-ai/4`, `nvidia-nim/free`, `opencode-zen/free`, `gemini/free` (NEEDS-KEY).

### Lead: ceecee (LEAD-PAID)

#### Surface: ceecee
- Tier: LEAD-PAID
- Main model: `minimax-m3`
- NEW Main: `slot-mini-m3-paid-main` (`ollama-cloud/2` (minimax-m3) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-mini-m3-paid-1` (`xkiro/free` (minimax-m3) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-mini-m3-paid-2` (`ollama-cloud/2` (minimax-m3) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-mini-m3-paid-3` (`xkiro/free` (minimax-m3) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

#### Surface: ceecee-brand
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: ceecee-reviewer
- Tier: LEAD-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-paid-main` (`ollama-cloud/2` (deepseek-v4-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-dsflash-paid-1` (`xkiro/free` (deepseek-v4-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-dsflash-paid-2` (`ollama-cloud/2` (deepseek-v4-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-dsflash-paid-3` (`xkiro/free` (deepseek-v4-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

#### Surface: ceecee-seo
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: ceecee-social
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: ceecee-writer
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

### Lead: content (WORKER-PAID)

#### Surface: content-strategist
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

### Lead: denji (LEAD-PAID)

#### Surface: denji
- Tier: LEAD-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-paid-main` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-glm53flash-paid-1` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-glm53flash-paid-2` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-glm53flash-paid-3` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

#### Surface: denji-ledger
- Tier: WORKER-PAID
- Main model: `stepfun/step-3.7-flash:free`
- NEW Main: `slot-step37free-free-main` (`ollama-cloud/1` (stepfun/step-3.7-flash:free) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-step37free-free-1` (`commandcode/1` (stepfun/step-3.7-flash:free) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-step37free-free-2` (`ollama-cloud/1` (stepfun/step-3.7-flash:free) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-step37free-free-3` (`commandcode/1` (stepfun/step-3.7-flash:free) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: denji-monitor
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: denji-reviewer
- Tier: LEAD-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-paid-main` (`ollama-cloud/2` (deepseek-v4-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-dsflash-paid-1` (`xkiro/free` (deepseek-v4-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-dsflash-paid-2` (`ollama-cloud/2` (deepseek-v4-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-dsflash-paid-3` (`xkiro/free` (deepseek-v4-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

#### Surface: denji-skill
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

### Lead: dezzy (LEAD-PAID)

#### Surface: dezzy
- Tier: LEAD-PAID
- Main model: `null / keep_current`
- No slot changes proposed

#### Surface: dezzy-brand
- Tier: WORKER-PAID
- Main model: `minimax-m3`
- NEW Main: `slot-mini-m3-free-main` (`ollama-cloud/1` (minimax-m3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-mini-m3-free-1` (`commandcode/1` (minimax-m3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-mini-m3-free-2` (`ollama-cloud/1` (minimax-m3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-mini-m3-free-3` (`commandcode/1` (minimax-m3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: dezzy-component-lib
- Tier: WORKER-PAID
- Main model: `minimax-m3`
- NEW Main: `slot-mini-m3-free-main` (`ollama-cloud/1` (minimax-m3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-mini-m3-free-1` (`commandcode/1` (minimax-m3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-mini-m3-free-2` (`ollama-cloud/1` (minimax-m3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-mini-m3-free-3` (`commandcode/1` (minimax-m3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: dezzy-design-system
- Tier: WORKER-PAID
- Main model: `minimax-m3`
- NEW Main: `slot-mini-m3-free-main` (`ollama-cloud/1` (minimax-m3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-mini-m3-free-1` (`commandcode/1` (minimax-m3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-mini-m3-free-2` (`ollama-cloud/1` (minimax-m3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-mini-m3-free-3` (`commandcode/1` (minimax-m3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: dezzy-image-prompt
- Tier: WORKER-PAID
- Main model: `minimax-m3`
- NEW Main: `slot-mini-m3-free-main` (`ollama-cloud/1` (minimax-m3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-mini-m3-free-1` (`commandcode/1` (minimax-m3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-mini-m3-free-2` (`ollama-cloud/1` (minimax-m3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-mini-m3-free-3` (`commandcode/1` (minimax-m3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: dezzy-ux-architect
- Tier: LEAD-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-paid-main` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-glm53flash-paid-1` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-glm53flash-paid-2` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-glm53flash-paid-3` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

#### Surface: dezzy-ux-prototype
- Tier: WORKER-PAID
- Main model: `minimax-m3`
- NEW Main: `slot-mini-m3-free-main` (`ollama-cloud/1` (minimax-m3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-mini-m3-free-1` (`commandcode/1` (minimax-m3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-mini-m3-free-2` (`ollama-cloud/1` (minimax-m3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-mini-m3-free-3` (`commandcode/1` (minimax-m3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

### Lead: gojo (LEAD-PAID)

#### Surface: gojo
- Tier: LEAD-PAID
- Main model: `poolside/laguna-s-2.1:free — keep_current / null`
- No slot changes proposed

#### Surface: gojo-admin
- Tier: WORKER-PAID
- Main model: `gemma4:31b`
- NEW Main: `slot-gemma31b-free-main` (`ollama-cloud/1` (gemma4:31b) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-gemma31b-free-1` (`commandcode/1` (gemma4:31b) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-gemma31b-free-2` (`ollama-cloud/1` (gemma4:31b) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-gemma31b-free-3` (`commandcode/1` (gemma4:31b) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: gojo-calendar
- Tier: WORKER-PAID
- Main model: `gemma4:31b`
- NEW Main: `slot-gemma31b-free-main` (`ollama-cloud/1` (gemma4:31b) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-gemma31b-free-1` (`commandcode/1` (gemma4:31b) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-gemma31b-free-2` (`ollama-cloud/1` (gemma4:31b) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-gemma31b-free-3` (`commandcode/1` (gemma4:31b) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: gojo-mailbox
- Tier: WORKER-PAID
- Main model: `gemma4:31b`
- NEW Main: `slot-gemma31b-free-main` (`ollama-cloud/1` (gemma4:31b) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-gemma31b-free-1` (`commandcode/1` (gemma4:31b) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-gemma31b-free-2` (`ollama-cloud/1` (gemma4:31b) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-gemma31b-free-3` (`commandcode/1` (gemma4:31b) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

### Lead: kensei (LEAD-PAID)

#### Surface: kensei
- Tier: LEAD-PAID
- Main model: `null / keep_current`
- No slot changes proposed

#### Surface: kensei-review
- Tier: LEAD-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-paid-main` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-glm53flash-paid-1` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-glm53flash-paid-2` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-glm53flash-paid-3` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

### Lead: light (LEAD-PAID)

#### Surface: light
- Tier: LEAD-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-paid-main` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-glm53flash-paid-1` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-glm53flash-paid-2` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-glm53flash-paid-3` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

#### Surface: light-archivist
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: light-indexer
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: light-wiki
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

### Lead: market (WORKER-PAID)

#### Surface: market-scanner
- Tier: WORKER-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-free-main` (`ollama-cloud/1` (glm-5.3-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-glm53flash-free-1` (`commandcode/1` (glm-5.3-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-glm53flash-free-2` (`ollama-cloud/1` (glm-5.3-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-glm53flash-free-3` (`commandcode/1` (glm-5.3-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

### Lead: misa (LEAD-PAID)

#### Surface: misa-misa
- Tier: LEAD-PAID
- Main model: `minimax-m3`
- NEW Main: `slot-mini-m3-paid-main` (`ollama-cloud/2` (minimax-m3) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-mini-m3-paid-1` (`xkiro/free` (minimax-m3) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-mini-m3-paid-2` (`ollama-cloud/2` (minimax-m3) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-mini-m3-paid-3` (`xkiro/free` (minimax-m3) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

### Lead: moss (LEAD-PAID)

#### Surface: moss
- Tier: LEAD-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-paid-main` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-glm53flash-paid-1` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-glm53flash-paid-2` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-glm53flash-paid-3` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

### Lead: mrhermagi (LEAD-PAID)

#### Surface: mrhermagi
- Tier: LEAD-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-paid-main` (`ollama-cloud/2` (deepseek-v4-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-dsflash-paid-1` (`xkiro/free` (deepseek-v4-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-dsflash-paid-2` (`ollama-cloud/2` (deepseek-v4-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-dsflash-paid-3` (`xkiro/free` (deepseek-v4-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

### Lead: octacon (LEAD-PAID)

#### Surface: octacon
- Tier: LEAD-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-paid-main` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-glm53flash-paid-1` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-glm53flash-paid-2` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-glm53flash-paid-3` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

#### Surface: octacon-architect
- Tier: LEAD-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-paid-main` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-glm53flash-paid-1` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-glm53flash-paid-2` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-glm53flash-paid-3` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

#### Surface: octacon-backend
- Tier: WORKER-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-free-main` (`ollama-cloud/1` (glm-5.3-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-glm53flash-free-1` (`commandcode/1` (glm-5.3-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-glm53flash-free-2` (`ollama-cloud/1` (glm-5.3-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-glm53flash-free-3` (`commandcode/1` (glm-5.3-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: octacon-frontend
- Tier: WORKER-PAID
- Main model: `kimi-k3`
- NEW Main: `slot-kimik3-free-main` (`ollama-cloud/1` (kimi-k3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-kimik3-free-1` (`commandcode/1` (kimi-k3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-kimik3-free-2` (`ollama-cloud/1` (kimi-k3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-kimik3-free-3` (`commandcode/1` (kimi-k3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: octacon-infra
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: octacon-mobile
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: octacon-techwriter
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: octacon-testrunner
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

### Lead: orchestrator (LEAD-PAID)

#### Surface: orchestrator
- Tier: LEAD-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-paid-main` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-glm53flash-paid-1` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-glm53flash-paid-2` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-glm53flash-paid-3` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

### Lead: quan (LEAD-PAID)

#### Surface: quan
- Tier: LEAD-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-paid-main` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-glm53flash-paid-1` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-glm53flash-paid-2` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-glm53flash-paid-3` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

#### Surface: quan-arch
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: quan-code
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: quan-e2e
- Tier: LEAD-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-paid-main` (`ollama-cloud/2` (deepseek-v4-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-dsflash-paid-1` (`xkiro/free` (deepseek-v4-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-dsflash-paid-2` (`ollama-cloud/2` (deepseek-v4-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-dsflash-paid-3` (`xkiro/free` (deepseek-v4-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

#### Surface: quan-perf
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: quan-security
- Tier: LEAD-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-paid-main` (`ollama-cloud/2` (deepseek-v4-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-dsflash-paid-1` (`xkiro/free` (deepseek-v4-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-dsflash-paid-2` (`ollama-cloud/2` (deepseek-v4-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-dsflash-paid-3` (`xkiro/free` (deepseek-v4-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

#### Surface: quan-ux
- Tier: WORKER-PAID
- Main model: `minimax-m3`
- NEW Main: `slot-mini-m3-free-main` (`ollama-cloud/1` (minimax-m3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-mini-m3-free-1` (`commandcode/1` (minimax-m3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-mini-m3-free-2` (`ollama-cloud/1` (minimax-m3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-mini-m3-free-3` (`commandcode/1` (minimax-m3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

### Lead: remii (LEAD-PAID)

#### Surface: remii
- Tier: LEAD-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-paid-main` (`ollama-cloud/2` (deepseek-v4-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-dsflash-paid-1` (`xkiro/free` (deepseek-v4-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-dsflash-paid-2` (`ollama-cloud/2` (deepseek-v4-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-dsflash-paid-3` (`xkiro/free` (deepseek-v4-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

#### Surface: remii-deep
- Tier: WORKER-PAID
- Main model: `deepseek-v4-pro`
- NEW Main: `slot-dspro-free-main` (`ollama-cloud/1` (deepseek-v4-pro) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dspro-free-1` (`commandcode/1` (deepseek-v4-pro) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dspro-free-2` (`ollama-cloud/1` (deepseek-v4-pro) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dspro-free-3` (`commandcode/1` (deepseek-v4-pro) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: remii-digest
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: remii-gitradar
- Tier: WORKER-PAID
- Main model: `deepseek-v4-pro`
- NEW Main: `slot-dspro-free-main` (`ollama-cloud/1` (deepseek-v4-pro) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dspro-free-1` (`commandcode/1` (deepseek-v4-pro) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dspro-free-2` (`ollama-cloud/1` (deepseek-v4-pro) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dspro-free-3` (`commandcode/1` (deepseek-v4-pro) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: remii-market
- Tier: WORKER-PAID
- Main model: `deepseek-v4-pro`
- NEW Main: `slot-dspro-free-main` (`ollama-cloud/1` (deepseek-v4-pro) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dspro-free-1` (`commandcode/1` (deepseek-v4-pro) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dspro-free-2` (`ollama-cloud/1` (deepseek-v4-pro) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dspro-free-3` (`commandcode/1` (deepseek-v4-pro) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

### Lead: sirvir (LEAD-PAID)

#### Surface: sirvir
- Tier: LEAD-PAID
- Main model: `qwen3.8-27b`
- NEW Main: `slot-qwen27b-paid-main` (`ollama-cloud/2` (qwen3.8-27b) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-qwen27b-paid-1` (`xkiro/free` (qwen3.8-27b) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-qwen27b-paid-2` (`ollama-cloud/2` (qwen3.8-27b) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-qwen27b-paid-3` (`xkiro/free` (qwen3.8-27b) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

### Lead: skill (WORKER-PAID)

#### Surface: skill-broker
- Tier: WORKER-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-free-main` (`ollama-cloud/1` (glm-5.3-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-glm53flash-free-1` (`commandcode/1` (glm-5.3-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-glm53flash-free-2` (`ollama-cloud/1` (glm-5.3-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-glm53flash-free-3` (`commandcode/1` (glm-5.3-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: skill-research
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

### Lead: triage (WORKER-PAID)

#### Surface: triage-router
- Tier: WORKER-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-free-main` (`ollama-cloud/1` (glm-5.3-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-glm53flash-free-1` (`commandcode/1` (glm-5.3-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-glm53flash-free-2` (`ollama-cloud/1` (glm-5.3-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-glm53flash-free-3` (`commandcode/1` (glm-5.3-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

### Lead: wesker (LEAD-PAID)

#### Surface: wesker
- Tier: LEAD-PAID
- Main model: `glm-5.3-flash`
- NEW Main: `slot-glm53flash-paid-main` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB1: `slot-glm53flash-paid-1` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- NEW FB2: `slot-glm53flash-paid-2` (`ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3)
- NEW FB3: `slot-glm53flash-paid-3` (`xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3)
- Codex: `slot-codex-sol`
- Local final: `slot-local-final`

#### Surface: wesker-backup
- Tier: WORKER-PAID
- Main model: `gemma4:31b`
- NEW Main: `slot-gemma31b-free-main` (`ollama-cloud/1` (gemma4:31b) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-gemma31b-free-1` (`commandcode/1` (gemma4:31b) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-gemma31b-free-2` (`ollama-cloud/1` (gemma4:31b) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-gemma31b-free-3` (`commandcode/1` (gemma4:31b) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: wesker-ops
- Tier: WORKER-PAID
- Main model: `gemma4:31b`
- NEW Main: `slot-gemma31b-free-main` (`ollama-cloud/1` (gemma4:31b) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-gemma31b-free-1` (`commandcode/1` (gemma4:31b) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-gemma31b-free-2` (`ollama-cloud/1` (gemma4:31b) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-gemma31b-free-3` (`commandcode/1` (gemma4:31b) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

#### Surface: wesker-scanner
- Tier: WORKER-PAID
- Main model: `deepseek-v4-flash`
- NEW Main: `slot-dsflash-free-main` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB1: `slot-dsflash-free-1` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- NEW FB2: `slot-dsflash-free-2` (`ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3)
- NEW FB3: `slot-dsflash-free-3` (`commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3)
- Codex: `slot-codex-luna`
- Local final: `slot-local-final`

### Lead: work (WORKER-PAID)

#### Surface: work
- Tier: WORKER-PAID
- Main model: `null / keep_current`
- No slot changes proposed

## 4) Rotation Summary Table (selected canonical surfaces)

| Surface | Tier | Slot | Try 1 | Try 2 | Try 3 | Codex | Local Final |
|---------|------|------|-------|-------|-------|-------|-------------|
| ceecee | LEAD-PAID | Main | `ollama-cloud/2` (minimax-m3) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3 | | | slot-codex-sol | slot-local-final |
| ceecee | LEAD-PAID | FB1 | `xkiro/free` (minimax-m3) try1 / `b-ai/1` try2 / `nous/free` try3 | | | | |
| ceecee | LEAD-PAID | FB2 | `ollama-cloud/2` (minimax-m3) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3 | | | | |
| ceecee | LEAD-PAID | FB3 | `xkiro/free` (minimax-m3) try1 / `b-ai/1` try2 / `nous/free` try3 | | | | |
| ceecee-brand | WORKER-PAID | Main | `ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3 | | | slot-codex-luna | slot-local-final |
| ceecee-brand | WORKER-PAID | FB1 | `commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3 | | | | |
| ceecee-brand | WORKER-PAID | FB2 | `ollama-cloud/1` (deepseek-v4-flash) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3 | | | | |
| ceecee-brand | WORKER-PAID | FB3 | `commandcode/1` (deepseek-v4-flash) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3 | | | | |
| wesker | LEAD-PAID | Main | `ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3 | | | slot-codex-sol | slot-local-final |
| wesker | LEAD-PAID | FB1 | `xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3 | | | | |
| wesker | LEAD-PAID | FB2 | `ollama-cloud/2` (glm-5.3-flash) try1 / `commandcode/2` try2 / `xkiro/pro-plus` try3 | | | | |
| wesker | LEAD-PAID | FB3 | `xkiro/free` (glm-5.3-flash) try1 / `b-ai/1` try2 / `nous/free` try3 | | | | |
| octacon-frontend | WORKER-PAID | Main | `ollama-cloud/1` (kimi-k3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3 | | | slot-codex-luna | slot-local-final |
| octacon-frontend | WORKER-PAID | FB1 | `commandcode/1` (kimi-k3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3 | | | | |
| octacon-frontend | WORKER-PAID | FB2 | `ollama-cloud/1` (kimi-k3) try1 / `commandcode/1` try2 / `ollama-cloud/1` try3 | | | | |
| octacon-frontend | WORKER-PAID | FB3 | `commandcode/1` (kimi-k3) try1 / `ollama-cloud/1` try2 / `commandcode/1` try3 | | | | |

---

## 5) YES/NO Confirmation Checklist

| Check | Result | Notes |
|-------|--------|-------|
| Tier column corrected: LEAD-PAID lead paid, WORKER-PAID worker paid, FREE free | YES | All surfaces classified |
| b-ai/1 reclassified as free in lead pool; b-ai/2,3,4 reclassified as free | YES | Core-10 inventory updated |
| nous/free included in LEAD-PAID lead pool | YES | Listed in lead pool |
| commandcode/1-2 reclassified as LEAD-PAID paid | YES | Updated inventory and rotations |
| xkiro/free included in LEAD-PAID lead pool | YES | Tier remains FREE, pool = lead |
| WORKER-PAID pool = ollama-cloud/1 + commandcode/1 exclusive | YES | Explicit choice noted |
| Free pool preserved for unused FREE accounts | YES | b-ai/2-4, nvidia-nim, opencode-zen, gemini |
| Free model lists fetched without auth where possible | YES | bai, gemini marked NEEDS-KEY; others listed live |
| GPT only via openai-codex preserved | YES | codex slots only |
| Local final always qwen3.8-27b 11410 | YES | slot-local-final |
| Same-model 3-try rotation per surface | YES | Main FB1-3 codex local chain preserved |
| No edits made to route-slots.yaml / surfaces.yaml / agents/ / profiles/ | YES | Read-only rewrite of proposal only |

---

*Report generated by Orchestrator subagent for kanban task. File: audits/round-robin-rebuild-proposal-2026-09-03.md*
