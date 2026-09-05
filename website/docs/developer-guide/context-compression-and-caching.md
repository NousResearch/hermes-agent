# Context Compression and Caching

Hermes Agent uses a dual compression system and Anthropic prompt caching to
manage context window usage efficiently across long conversations.

Source files: `agent/context_engine.py` (ABC), `agent/context_compressor.py` (default engine),
`agent/prompt_caching.py`, `gateway/run_turn.py` (session hygiene), `agent/compression_facade.py` (search for `_compress_context`)


## Pluggable Context Engine

Context management is built on the `ContextEngine` ABC (`agent/context_engine.py`). The built-in `ContextCompressor` is the default implementation, but plugins can replace it with alternative engines (e.g., Lossless Context Management).

```yaml
context:
  engine: "compressor"    # default — built-in lossy summarization
  engine: "lcm"           # example — plugin providing lossless context
```

The engine is responsible for:
- Deciding when compaction should fire (`should_compress()`)
- Performing compaction (`compress()`)
- Optionally exposing tools the agent can call (e.g., `lcm_grep`)
- Tracking token usage from API responses

Selection is config-driven via `context.engine` in `config.yaml`. The resolution order:
1. Check `plugins/context_engine/<name>/` directory
2. Check general plugin system (`register_context_engine()`)
3. Fall back to built-in `ContextCompressor`

Plugin engines are **never auto-activated** — the user must explicitly set `context.engine` to the plugin's name. The default `"compressor"` always uses the built-in.

Configure via `hermes plugins` → Provider Plugins → Context Engine, or edit `config.yaml` directly.

For building a context engine plugin, see [Context Engine Plugins](/developer-guide/context-engine-plugin).

## Dual Compression System

Hermes has two separate compression layers that operate independently:

```
                     ┌──────────────────────────┐
  Incoming message   │   Gateway Session Hygiene │  Fires at 85% of context
  ─────────────────► │   (pre-agent, rough est.) │  Safety net for large sessions
                     └─────────────┬────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │   Agent ContextCompressor │  Fires at 50% of context (default)
                     │   (in-loop, real tokens)  │  Normal context management
                     └──────────────────────────┘
```

### 1. Gateway Session Hygiene (85% threshold)

Located in `gateway/run_turn.py` (search for `Session hygiene`). This is a **safety net** that
runs before the agent processes a message. It prevents API failures when sessions
grow too large between turns (e.g., overnight accumulation in Telegram/Discord).

- **Threshold**: Fixed at 85% of model context length
- **Token source**: Prefers actual API-reported tokens from last turn; falls back
  to rough character-based estimate (`estimate_messages_tokens_rough`)
- **Fires**: Only when `len(history) >= 4` and compression is enabled
- **Purpose**: Catch sessions that escaped the agent's own compressor

The gateway hygiene threshold is intentionally higher than the agent's compressor.
Setting it at 50% (same as the agent) caused premature compression on every turn
in long gateway sessions.

### 2. Agent ContextCompressor (50% threshold, configurable)

Located in `agent/context_compressor.py`. This is the **primary compression
system** that runs inside the agent's tool loop with access to accurate,
API-reported token counts.

#### Failure cooldown and provider-proven overflow

A failed or stalled summary attempt arms a per-session **failure cooldown**
(escalating 60s → 300s → 900s, persisted in `state.db`). While it is armed,
ordinary threshold-triggered compaction is deferred so a broken summary backend
does not re-fire every turn. Two paths run a real attempt anyway:

- Manual `/compress` (`force=True`) — clears the cooldown and retries.
- **Provider-proven overflow** — when the provider itself rejects the request
  with a context-length error, the recovery pass ignores the cooldown for one
  bounded attempt (`max_compression_attempts`) without clearing it. Deferring
  here would wedge the session: every turn would bounce off the provider and
  the next failure would extend the ladder (#100661). If that attempt fails,
  the cooldown is recorded normally.


## Configuration

All compression settings are read from `config.yaml` under the `compression` key:

```yaml
compression:
  enabled: true              # Enable/disable compression (default: true)
  threshold: 0.50            # Fraction of context window (default: 0.50 = 50%)
  # model_thresholds:        # Per-model threshold overrides (substring match,
  #   "glm-5.2": 0.40        # longest key wins). See "Per-model threshold
  #   "claude-sonnet": 0.35  # overrides" below.
  target_ratio: 0.20         # How much of threshold to keep as tail (default: 0.20)
  tail_mode: lean            # Tail retention policy: lean | legacy (default: lean)
  protect_last_n: 20         # Minimum protected tail messages (default: 20)
  min_tail_user_messages: 1  # Real user messages guaranteed in the tail (default: 1)
  codex_gpt55_autoraise: true  # gpt-5.5 on Codex OAuth: raise trigger to 85% (default: true)
  codex_gpt55_autoraise_notice: true  # Show the one-time autoraise notice (default: true)
  codex_app_server_auto: native  # native|hermes|off for Codex app-server thread compaction
  codex_responses_native: false  # Opt-in server compaction: gpt-5.6 on OpenAI/Codex; Astra on Codex OAuth
  codex_responses_compact_threshold: null  # Automatic server compaction trigger
  in_place: true             # Compact on the same session id, no rotation (default: true)

# Summarization model/provider configured under auxiliary:
auxiliary:
  compression:
    model: null              # Override model for summaries (default: auto-detect)
    provider: auto           # Provider: "auto", "openrouter", "nous", "main", etc.
    base_url: null           # Custom OpenAI-compatible endpoint
```

### Parameter Details

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `threshold` | `0.50` | 0.0-1.0 | Compression triggers when prompt tokens ≥ `threshold × context_length` |
| `model_thresholds` | `{}` | map | Per-model overrides of `threshold`. Keys are substring-matched against the model name (longest match wins). The small-context floor still applies on top (see below) |
| `target_ratio` | `0.20` | 0.10-0.80 | Controls tail protection token budget: `threshold_tokens × target_ratio` (legacy mode only — `lean` uses its own clamp) |
| `tail_mode` | `lean` | `lean`, `legacy` | Tail retention policy. `legacy` keeps a `target_ratio`-sized verbatim tail (~100K+ tokens on big-window models). `lean` keeps a clamped tail of `2.5% × context window` (10K floor, 25K cap) and instead carries continuity in the summary: a detailed identifier-preserving session log (produced by the same single summary request — lean compaction makes exactly one auxiliary LLM call per attempt), a mechanically extracted anchor index (PR numbers, SHAs, paths, error strings — regex, never paraphrased), every real user message quoted verbatim (newest-first budget), and a `session_search` recovery pointer so the agent can re-access anything summarized away. Oversized regions are evenly sampled into the summarizer input (with explicit elision markers) rather than triggering extra calls. Result on 500K-token real sessions: ~49K retained vs ~162K, with higher recall when paired with recovery (see `evals/compaction/results/`). Old tool results inside the lean tail are demoted to one-line stubs carrying a recovery pointer |
| `protect_last_n` | `20` | ≥1 | Minimum number of recent messages always preserved |
| `min_tail_user_messages` | `1` | ≥1 | Minimum number of REAL (actionable) user messages guaranteed to survive in the uncompressed tail. `1` = the existing single last-user anchor (behavior-preserving default). Raise to e.g. `3` to keep the last 3 real user turns verbatim even when bulky tool outputs fill the tail token budget. Blank platform echoes, compaction handoffs, and synthetic continuation rows never count toward N. The guarantee wins over the tail token budget — the tail may exceed the budget when the anchor pulls the cut back |
| `protect_first_n` | `3` | (hardcoded) | System prompt + first exchange always preserved |
| `idle_compact_after_seconds` | `0` | ≥0 seconds | Opt-in: compact up front when a session resumes after this many seconds idle (0 = disabled). Skips when context ≤ threshold × target_ratio; honors cooldown/anti-thrash/lock guards |
| `codex_gpt55_autoraise` | `true` | bool | Raise the trigger to 85% for gpt-5.5 on the ChatGPT Codex OAuth route (see below). Set `false` to keep the global `threshold` |
| `codex_gpt55_autoraise_notice` | `true` | bool | Show the one-time Codex gpt-5.5 autoraise notice. Set `false` to keep the 85% autoraise but suppress the banner |
| `codex_app_server_auto` | `native` | `native`, `hermes`, `off` | Thread-compaction mode for Codex app-server sessions (see below) |
| `codex_responses_native` | `false` | bool | Opt in to OpenAI's server-side compaction on the Responses API. Engages for gpt-5.6-family models on the direct OpenAI API or a ChatGPT Codex subscription, and exact `gpt-6-astra` on official Codex OAuth (see below) |
| `codex_responses_compact_threshold` | `null` | `null` or positive integer | `null` follows the resolved local compression trigger with an 8,192 token safety margin. A positive integer remains absolute and only clamps downward when required. Invalid values use automatic behavior. Automatic mode falls back to `200000` when no usable local trigger exists |
| `in_place` | `true` | bool | Compact on the same session id instead of rotating to a new one (see below) |

### In-place compaction (single stable session id)

With `compression.in_place: true` (the default), a compaction **rewrites the live message list on the same session id**: the system prompt is rebuilt, the summarized middle is swapped in, and the pre-compaction turns are soft-archived under the same id (`active=0, compacted=1` in the session store) — still searchable via `session_search` and recoverable, never deleted. There is no `parent_session_id` chain and no `name #N` renumbering; one conversation keeps one durable id for its whole life. This eliminated the session-rotation bug cluster (lost `/goal` state, orphaned sessions, search gaps across boundaries).

Consumers observe the mode rather than diffing session ids:

- The `session:compress` event carries `in_place: true/false` and `old_session_id` (empty string in in-place mode, since there is no old id).
- The gateway re-baselines transcript handling from the agent's rotation-independent `_last_compaction_in_place` flag, not from an id-change diff.

Set `in_place: false` to restore the legacy rotating path, where each compaction commits a new session id linked to the previous one via `parent_session_id`.

### Per-model threshold overrides

`compression.model_thresholds` lets you trigger compaction at different points
depending on the active model — useful when you swap between models with very
different context windows (e.g. a 1M-context model can compress later while a
128K model should compress earlier):

```yaml
compression:
  threshold: 0.50
  model_thresholds:
    "glm-5.2": 0.40
    "glm-5.2-1M": 0.25
    "claude-sonnet": 0.35
```

Resolution rules:

- Keys are **substring-matched** against the model name; the **longest
  matching key wins** (`glm-5.2-1M` beats `glm-5.2` for model `glm-5.2-1M`).
- When no key matches (or the map is empty), the global `threshold` applies.
- The override is re-resolved on every `/model` switch; switching to a model
  with no matching key falls back to the global `threshold`.
- The **small-context floor still applies on top** of overrides (raise-only):
  models with context windows below 512K are floored at `0.75`, so an
  override below the floor is raised to `0.75`, while an override above it
  (e.g. `0.80`) wins.

Plugin context engines can reuse the same resolution logic via
`from agent.context_compressor import resolve_model_threshold`; engines that
override `update_model()` own their own compaction policy and may ignore the
map.

### Codex gpt-5.5 threshold autoraise

The ChatGPT Codex OAuth backend hard-caps gpt-5.5 at a **272K** context window
(the same slug exposes 1.05M on OpenAI's direct API and OpenRouter, and 400K on
GitHub Copilot). At the default 50% trigger, compaction would fire at ~136K —
half the window the model can actually use. When the active route is Codex
OAuth (`provider: openai-codex`) and the model is gpt-5.5, Hermes raises the
trigger to **85%** (~231K) and shows a notice with the opt-out command. The
notice is shown once per profile — a marker under `$HERMES_HOME`
(`.codex_gpt55_autoraise_notice`) records that it ran, so repeated agent/session
inits (e.g. every inbound gateway message) don't re-emit it; if the raised
threshold later changes it re-notifies once. Only this exact route is affected;
gpt-5.5 on any other provider keeps your global `threshold`. To opt back down to
the global value:

```bash
hermes config set compression.codex_gpt55_autoraise false
```

To keep the 85% autoraise but hide only the one-time notice:

```bash
hermes config set compression.codex_gpt55_autoraise_notice false
```

### Codex large-context `-900k` picker variants (opt-in)

The ChatGPT Codex backend *advertises* a 272K window for the gpt-5.4 and
gpt-5.6 (Sol/Terra/Luna) families, but actually accepts ~911K input tokens
for ChatGPT-subscription accounts (live-verified Aug 2026). Hermes keeps the
**advertised 272K as the default** for the base slugs — a bigger window means
more tokens per request and much faster subscription-usage burn, so the large
window is strictly opt-in.

To use the large window, pick the explicit `-900k` variant in `/model` (e.g.
`gpt-5.6-sol-900k`, `gpt-5.6-terra-900k`, `gpt-5.6-luna-900k`,
`gpt-5.4-900k`). These are Hermes-side aliases: the suffix is stripped before
the model id is sent to the backend, and pricing/usage accounting treats them
as the base model. Slugs that genuinely enforce 272K (gpt-5.5, gpt-5.4-mini)
have no `-900k` variant.

Compaction thresholds follow the window: base slugs (272K) get the **85%
autoraise** described above, while `-900k` variants keep your global
`compression.threshold` (default 50%, ~450K) — the autoraise exists to stop
wasting a small window, which a 900K window doesn't need.

### Codex app-server thread compaction

Codex app-server sessions (`api_mode: codex_app_server` — the codex CLI/agent
runtime) are different from every other route: the codex agent owns the backing
thread context, so Hermes' auxiliary summarizer cannot shrink it — rewriting the
local transcript mirror leaves the real thread growing unbounded until a hard
context reset. For this runtime, compaction goes through the app-server's own
mechanism instead:

- Manual compaction (`/compress`) asks the app-server to compact the thread
  (`thread/compact/start`) and waits for the compaction turn to complete.
- Automatic compaction is controlled by `compression.codex_app_server_auto`:
  the default `native` lets the app-server decide when to compact and Hermes
  records the resulting compaction events (compression counters, session
  events). Set `hermes` to let Hermes' compression threshold initiate
  app-server compaction, or `off` to disable Hermes-initiated automatic
  compaction entirely (codex may still compact natively).

Hermes' local transcript is never rewritten on this runtime — state.db records
the compaction boundary while the visible transcript stays intact. All other
routes (including Codex OAuth chat sessions) keep Hermes' summary compressor.

### Native Responses compaction (gpt-5.6 and Astra on supported routes)

OpenAI's Responses API supports server-side compaction: when a request includes
`context_management: [{type: "compaction", compact_threshold: N}]` and the
rendered input crosses N tokens, the server prunes older context into an opaque
encrypted `compaction` output item. Hermes captures that item into the
assistant message's existing replay sidecar and sends it back on subsequent
turns, standing in for the pruned history — long-horizon recall without a
client-side summary pass, and ZDR-friendly (`store: false`, no
`previous_response_id`).

Opt in with `compression.codex_responses_native: true`. The gate is deliberately
narrow, re-checked on every request:

- **Models:** the gpt-5.6 family, plus exact `gpt-6-astra` on official Codex
  subscription OAuth. Astra on the direct API, Astra variants and other GPT-6
  models are excluded. gpt-5.1/5.2 return HTTP 500 or stall the stream when the
  field is present (no structured rejection to downgrade on, verified live Aug 2026).
- **Routes:** `api.openai.com` (OpenAI API key) or the ChatGPT Codex backend
  (Codex subscription OAuth) only. xAI, GitHub/Copilot, OpenRouter, relays, and
  local servers never see the field.

For Astra, both the resolved `openai-codex` provider and an official HTTPS
`chatgpt.com/backend-api/codex` endpoint are required. A trusted proxy override
does not enable Astra compaction. This uses the existing automatic
`context_management` path and does not add `configuration_update` history.

Everything else about compression is unchanged: the local compressor stays
armed as the fallback owner (the native threshold is clamped ~8K tokens below
the local trigger so the server compacts first), and a structured provider
rejection of the field disables native compaction for the session and retries
the request without it. Switching the session to a non-eligible model or route
simply stops the field from being sent — captured checkpoints are dropped from
replay by the existing cross-issuer guard when the endpoint changes.

By default, `compression.codex_responses_compact_threshold: null` derives the
native threshold from the resolved local trigger. For example, a local trigger
of 765,000 selects 756,808. Set a positive integer to preserve an absolute
threshold such as 200,000. Invalid values select automatic behavior. If no
usable local trigger exists, automatic mode uses 200,000. The provider minimum
is 1,024 tokens, so an unusually small local trigger at or below that floor
cannot preserve strict native first ordering.

### Computed Values (for a 200K context model at defaults)

```
context_length       = 200,000
threshold_tokens     = 200,000 × 0.50 = 100,000
tail_token_budget    = 100,000 × 0.20 = 20,000
max_summary_tokens   = min(200,000 × 0.05, 12,000) = 10,000
```

:::note Threshold is derived from the MAIN model's context window
`threshold_tokens` is always `threshold × context_length`, where `context_length`
is the **main agent model's** context window — never the auxiliary/summary
model's. On a 262,144-token model at the default `0.50`, the threshold is
`262,144 × 0.50 = 131,072`. That number being close to a common "128K context"
is a coincidence of the percentage, not a sign that the auxiliary model's window
is the trigger. The auxiliary model's context window is a separate concern — see
the "Summary model context length" warning below for how it affects whether a
summary can be produced, not when compression fires.
:::


## Compression Algorithm

The `ContextCompressor.compress()` method follows a 4-phase algorithm:

### Phase 1: Prune Old Tool Results (cheap, no LLM call)

Old tool results (>200 chars) outside the protected tail are replaced with:
```
[Old tool output cleared to save context space]
```

This is a cheap pre-pass that saves significant tokens from verbose tool
outputs (file contents, terminal output, search results).

### Phase 2: Determine Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│  Message list                                               │
│                                                             │
│  [0..2]  ← protect_first_n (system + first exchange)        │
│  [3..N]  ← middle turns → SUMMARIZED                        │
│  [N..end] ← tail (by token budget OR protect_last_n)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Tail protection is **token-budget based**: walks backward from the end,
accumulating tokens until the budget is exhausted. Falls back to the fixed
`protect_last_n` count if the budget would protect fewer messages.

Boundaries are aligned to avoid splitting tool_call/tool_result groups.
The `_align_boundary_backward()` method walks past consecutive tool results
to find the parent assistant message, keeping groups intact.

### Phase 3: Generate Structured Summary

:::warning Summary model context length
The summary model must have a context window **at least as large** as the main agent model's. The entire middle section is sent to the summary model in a single `call_llm(task="compression")` call. If the summary model's context is smaller, the API returns a context-length error — `_generate_summary()` catches it, logs a warning, and returns `None`. The compressor then drops the middle turns **without a summary**, silently losing conversation context. This is the most common cause of degraded compaction quality.
:::

The middle turns are summarized using the auxiliary LLM with a structured
template:

```
## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Progress
### Done
[Completed work — specific file paths, commands run, results]
### In Progress
[Work currently underway]
### Blocked
[Any blockers or issues encountered]

## Key Decisions
[Important technical decisions and why]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Next Steps
[What needs to happen next]

## Critical Context
[Specific values, error messages, configuration details]
```

Summary budget scales with the amount of content being compressed:
- Formula: `content_tokens × 0.20` (the `_SUMMARY_RATIO` constant)
- Minimum: 2,000 tokens
- Maximum: `min(context_length × 0.05, 12,000)` tokens

### Phase 4: Assemble Compressed Messages

The compressed message list is:
1. Head messages (with a note appended to system prompt on first compression)
2. Summary message (role chosen to avoid consecutive same-role violations)
3. Tail messages (unmodified)

Orphaned tool_call/tool_result pairs are cleaned up by `_sanitize_tool_pairs()`:
- Tool results referencing removed calls → removed
- Tool calls whose results were removed → stub result injected

### Iterative Re-compression

On subsequent compressions, the previous summary is passed to the LLM with
instructions to **update** it rather than summarize from scratch. This preserves
information across multiple compactions — items move from "In Progress" to "Done",
new progress is added, and obsolete information is removed.

The `_previous_summary` field on the compressor instance stores the last summary
text for this purpose.


## Before/After Example

### Before Compression (45 messages, ~95K tokens)

```
[0] system:    "You are a helpful assistant..." (system prompt)
[1] user:      "Help me set up a FastAPI project"
[2] assistant: <tool_call> terminal: mkdir project </tool_call>
[3] tool:      "directory created"
[4] assistant: <tool_call> write_file: main.py </tool_call>
[5] tool:      "file written (2.3KB)"
    ... 30 more turns of file editing, testing, debugging ...
[38] assistant: <tool_call> terminal: pytest </tool_call>
[39] tool:      "8 passed, 2 failed\n..."  (5KB output)
[40] user:      "Fix the failing tests"
[41] assistant: <tool_call> read_file: tests/test_api.py </tool_call>
[42] tool:      "import pytest\n..."  (3KB)
[43] assistant: "I see the issue with the test fixtures..."
[44] user:      "Great, also add error handling"
```

### After Compression (25 messages, ~45K tokens)

```
[0] system:    "You are a helpful assistant...
               [Note: Some earlier conversation turns have been compacted...]"
[1] user:      "Help me set up a FastAPI project"
[2] assistant: "[CONTEXT COMPACTION] Earlier turns were compacted...

               ## Goal
               Set up a FastAPI project with tests and error handling

               ## Progress
               ### Done
               - Created project structure: main.py, tests/, requirements.txt
               - Implemented 5 API endpoints in main.py
               - Wrote 10 test cases in tests/test_api.py
               - 8/10 tests passing

               ### In Progress
               - Fixing 2 failing tests (test_create_user, test_delete_user)

               ## Relevant Files
               - main.py — FastAPI app with 5 endpoints
               - tests/test_api.py — 10 test cases
               - requirements.txt — fastapi, pytest, httpx

               ## Next Steps
               - Fix failing test fixtures
               - Add error handling"
[3] user:      "Fix the failing tests"
[4] assistant: <tool_call> read_file: tests/test_api.py </tool_call>
[5] tool:      "import pytest\n..."
[6] assistant: "I see the issue with the test fixtures..."
[7] user:      "Great, also add error handling"
```


## Prompt Caching (Anthropic)

Source: `agent/prompt_caching.py`

Reduces input token costs by ~75% on multi-turn conversations by caching the
conversation prefix. Uses Anthropic's `cache_control` breakpoints.

### Strategy: system_and_3

Anthropic allows a maximum of 4 `cache_control` breakpoints per request. Hermes
uses the "system_and_3" strategy:

```
Breakpoint 1: System prompt           (stable across all turns)
Breakpoint 2: 3rd-to-last non-system message  ─┐
Breakpoint 3: 2nd-to-last non-system message   ├─ Rolling window
Breakpoint 4: Last non-system message          ─┘
```

### How It Works

`apply_anthropic_cache_control()` deep-copies the messages and injects
`cache_control` markers:

```python
# Cache marker format
marker = {"type": "ephemeral"}
# Or for 1-hour TTL:
marker = {"type": "ephemeral", "ttl": "1h"}
```

The marker is applied differently based on content type:

| Content Type | Where Marker Goes |
|-------------|-------------------|
| String content | Converted to `[{"type": "text", "text": ..., "cache_control": ...}]` |
| List content | Added to the last element's dict |
| None/empty | Added as `msg["cache_control"]` |
| Tool messages | Added as `msg["cache_control"]` (native Anthropic only) |

### Cache-Aware Design Patterns

1. **Stable system prompt**: The system prompt is breakpoint 1 and cached across
   all turns. Avoid mutating it mid-conversation (compression appends a note
   only on the first compaction).

2. **Message ordering matters**: Cache hits require prefix matching. Adding or
   removing messages in the middle invalidates the cache for everything after.

3. **Compression cache interaction**: After compression, the cache is invalidated
   for the compressed region but the system prompt cache survives. The rolling
   3-message window re-establishes caching within 1-2 turns.

4. **TTL selection**: Default is `5m` (5 minutes). Use `1h` for long-running
   sessions where the user takes breaks between turns.

5. **Model identity is part of the cache key**: Provider-side caches are scoped
   to the model (and account/API key) serving the request. Any mid-conversation
   model change — an explicit `/model` switch, primary-model fallback, or a
   credential-pool rotation onto a different account — means the next request
   gets zero cache hits and re-reads the full conversation at undiscounted
   input price. This is inherent to how provider caches work, not something
   Hermes can avoid; user-facing docs for `/model`, fallback providers, and
   credential pools carry cost warnings for this reason. Don't add features
   that silently swap the model or credentials mid-session.

### Enabling Prompt Caching

Prompt caching is automatically enabled when:
- The model is an Anthropic Claude model (detected by model name)
- The provider supports `cache_control` (native Anthropic API or OpenRouter)

```yaml
# config.yaml — TTL is configurable (must be "5m" or "1h")
prompt_caching:
  cache_ttl: "5m"
```

The CLI shows caching status at startup:
```
💾 Prompt caching: ENABLED (Claude via OpenRouter, 5m TTL)
```


## Context Pressure Warnings

Intermediate context-pressure warnings have been removed (see the iteration-budget block in `agent/turn_iteration_prep.py`, which notes: "No intermediate pressure warnings — they caused models to 'give up' prematurely on complex tasks"). Compression fires when prompt tokens reach the configured `compression.threshold` (default 50%) with no prior warning step; gateway session hygiene fires as the secondary safety net at 85% of the model's context window.
