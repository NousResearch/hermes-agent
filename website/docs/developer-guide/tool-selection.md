---
title: "Plugin-selected Tool Working Sets"
description: "Opt-in, host-owned tool selection before schema conversion and prompt caching"
---

# Plugin-selected tool working sets

Tool selection is an **explicit schema-prefix caching tradeoff**, not the default
Hermes tool policy. It lets an installed routing plugin select a small working set
from the tools already authorized for the session. Hermes remains the schema owner
and execution policy owner. No vendor-specific router is bundled in the host.

```yaml
tools:
  tool_search:
    enabled: "on"
    defer: all
    listing: "off"
    selection:
      enabled: true
      max_tools: 8
      max_schema_tokens: 4096
```

`defer: all` is a literal string alternative to the existing list. Every eligible
registry or dynamic task tool can defer, including core, GUI, memory-provider,
context-engine and authorized `message_agent` tools. Only `tool_search`,
`tool_describe` and `tool_call` remain ambient. Session toolset restrictions,
provider availability, one-shot pruning and side-agent restrictions still apply.
Remote connector discovery continues through the existing connection grant.

With selection enabled, the request contains those three bridges plus the exact
canonical schemas of selected names. The bridges remain available even when the
router misses a capability. With `selection.enabled: false`, the chosen native
deferral policy applies without routing. Other profiles and default configurations
are unaffected. Restart a session after changing policy settings.

## Public callback contract

```python
def register(ctx):
    ctx.register_middleware("tool_selection", select_tools)


def select_tools(**kwargs):
    if kwargs["tool_selection_schema_version"] != "hermes.tool-selection.v1":
        return None
    # Rank locally, or use explicitly authorized, bounded remote inference.
    # Return ONLY names from kwargs["catalog"], never replacement schemas.
    names = rank_tools(kwargs["task_context"], kwargs["catalog"], kwargs["budget"])
    return {
        "selected_tools": names,
        "catalog_revision": kwargs["catalog_revision"],
        "source": "my-router",
        "reason": "matched",
    }
```

The callback receives keyword arguments:

| Field | Value |
| --- | --- |
| `tool_selection_schema_version` | `hermes.tool-selection.v1` |
| `catalog` | List of `{"name": str, "description": str, "schema": canonical_chat_function_wrapper}` |
| `catalog_revision` | Lowercase SHA-256 hex of the canonical catalog: sorted by name, JSON with sorted keys, compact separators, `ensure_ascii=False`, UTF-8 |
| `task_context` | `{"current_user_message": str, "recent_messages": [{"role": "user" or "assistant", "content": str}], "partial": bool}` |
| `session_id`, `turn_id` | The native session and actual user-turn identities (same turn as `pre_llm_call`) |
| `profile_name` | Active profile name, matching `ctx.profile_name`; `default` for the default home |
| `budget` | Exactly `{"max_tools": int, "max_schema_tokens": int}` |
| `telemetry_schema_version`, `middleware_schema_version` | Existing observer/middleware version context |

`schema` is the full `{"type": "function", "function": {...}}` wrapper, before
provider conversion, reserved-name aliases and cache decoration. Mutating callback
inputs does not alter host schemas, history or other callbacks' inputs.

The current input is the actual `original_user_message`, **not** a reconstructed
summary or tool output. Prior context contains at most four plain user/assistant
messages, excluding the current input, system messages, tool bodies, assistant
tool-call/reasoning rows and known synthetic compression/recovery messages. Prior
text totals at most 32,768 Unicode codepoints. The current input is capped at
131,072 codepoints. Any omission sets `partial: true`. Invalid, oversized or
non-string current input makes context unavailable: the callback is not invoked
and the host uses discovery only. Oversized/invalid prior rows are omitted, not
forwarded unchecked. Neither task text nor exception text from selection is logged.

This is **not an egress grant or secret detector**. Plugins must separately enforce
operator consent, reject known secrets, bound/redact text, and export only the
minimum necessary excerpts. Do not log raw task text in a plugin.

## Admission, lifetime and fallback

- The routing result is `None` or the exact shape shown above. `source` and `reason`
  must be strings, at most 128 and 512 codepoints respectively.
- Exactly one non-`None` proposal is accepted. Multiple proposals, any callback
  exception, stale revision, unknown/duplicate names, malformed result or excess
  budget produce **discovery only**, never a union or an all-tools fallback.
- At most eight callbacks are considered. More owners refuse selection without
  running them. The combined callback window is **750 ms, cooperative**: callbacks
  must bound their own work and I/O. Late results are discarded. Arbitrary Python
  plugins can still block the caller; the host cannot preempt them and does not
  create detached timeout threads. Selection runs outside provider conversion.
- The selected set is stable for a user turn and catalog revision, including
  request retries. A later turn replaces it; schemas never accumulate. Fresh
  registry/dynamic inventory or changed budgets invalidate the working-set cache.
  A catalog change during the callback rejects that stale decision.
- The host computes the schema budget with the existing compact-JSON **chars/4
  heuristic**, not exact tokenizer counts. The bridge overhead is outside the
  task-schema budget. Limits clamp to 0–64 task tools and 0–65,536 estimated schema
  tokens. The inventory is capped at 4,096 tools and 4 Mi codepoints of serialized
  schemas; unavailable, invalid or oversized inventories authorize no task tools.
- Request-visible names and authorized inventory are distinct. Discovery,
  deferred argument validation and dispatch use the same host-owned session
  inventory, including legitimate dynamic tools. Caller-supplied tool arguments
  cannot supply an inventory. `execute_code` retains authorized helper reachability
  even when helpers are not selected; an empty authorized intersection stays empty.
- Selected and bridge-unwrapped calls take the ordinary native hooks, guardrails,
  approvals, inline/provider handlers and result processing. Selection grants no
  execution shortcut. Scope is refreshed before dispatch, including after policy
  hooks; revocation never falls back to stale `agent.tools`.

## Compatibility boundaries

`defer: all` is currently refused for `codex_app_server` and persistent MoA because
those paths do not use the same host-owned request assembly boundary. The normal
Chat Completions, Codex/Responses, Anthropic Messages and Bedrock request assembly
paths support working sets.

An authorized explicit function `tool_choice` is retained in the offered set,
within the task budget. Unknown targets and unsupported modes are refused, never
silently changed to `auto`. Responses requires its native flat
`{"type":"function","name":"..."}` shape; Chat Completions uses its native
function shape. Explicit bridge `tool_search` choices (provider alias ambiguity),
non-function choice objects, forced-function choices on other transports and
`request_overrides.tools` are refused. Ordinary `auto`, `none` and `required` choice
strings are preserved. Later request middleware remains its existing trusted,
request-only surface: this feature does not change legacy `llm_request` behavior.

## Cache cost

Default schema-prefix caching is unchanged. Opting in deliberately replaces task
schemas between turns, which can invalidate a provider's prefix cache and change
its content-addressed cache key. Same-turn stability avoids gratuitous churn, but
does not promise cross-turn cache hits. Canonical history and the session system
prompt are not rewritten. Provider aliases and cache markers are computed **after**
working-set selection, so the first serialized request and its cache key describe
the same tools. Choose this mode only when the smaller schema surface is worth the
cache tradeoff for your workload.
