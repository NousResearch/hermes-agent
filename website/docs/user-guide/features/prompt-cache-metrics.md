# Prompt cache & session cost signals

Hermes treats **per-conversation prompt caching as sacred**. Mid-turn changes to
the stable system prompt or toolset invalidate the cached prefix and multiply
cost. Volatile signals (Herens, Brain Networks, hunt context) are injected into
the **user message** via `pre_llm_call` — never into the frozen system tier.

## Anthropic layout

Layout id: `system_and_3` (hard cap **4** `cache_control` breakpoints).

```python
from agent.prompt_caching import describe_cache_layout, apply_anthropic_cache_control

describe_cache_layout()
# → layout, max_breakpoints=4, cache_safe_rules
```

OpenAI-compatible providers often report automatic prefix caching via
`prompt_tokens_details.cached_tokens`. The conversation loop normalizes all
shapes into `CanonicalUsage.cache_read_tokens` / `cache_write_tokens` and
accumulates them on the agent session.

## `/status` cache line

CLI `/status` prints a one-line efficiency summary from
`agent.cache_metrics.format_cache_status_line(agent)`:

```text
Cache: 12,400 read / 18,000 prompt (68.9% hit, 800 written) [good]
```

Efficiency bands: `excellent` ≥70%, `good` ≥40%, else `low`.

## Recommendations

`cache_efficiency_report(agent)` returns actionable hints when hit rate is low
or write/read ratio looks unstable (usually from mid-conversation prefix
rewrites). Prefer `/focus` + deferred skill/tool changes (`--now` only when
intentional) over swapping toolsets live.
