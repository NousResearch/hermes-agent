---
sidebar_position: 11
title: "Plugin LLM Access"
description: "Run a chat or structured completion from inside a plugin via ctx.llm — host-owned auth, fail-closed trust gate, optional JSON Schema validation."
---

# Plugin LLM Access

Most plugin work in Hermes is about *extending* the agent — adding a
tool, a skill, a memory backend, a gateway adapter. `ctx.llm` is for
the cases where a plugin needs to *make a decision on its own*. A
hook that ranks an inbound message before the agent ever sees it. A
gateway pre-filter that routes some events to one queue and the rest
to another. A slash command that turns a paste into a typed record.
A scheduled job that summarises overnight activity into a single
field of a status board.

These are jobs the agent shouldn't be in the loop on. They want one
LLM call, a typed answer, and to be done.

```python
result = ctx.llm.complete_structured(
    instructions="Score this support reply for urgency (0–1) and pick a category.",
    input=[{"type": "text", "text": message_body}],
    json_schema=TRIAGE_SCHEMA,
    purpose="support.triage",
    temperature=0.0,
    max_tokens=128,
)

if result.parsed["urgency"] > 0.8:
    await dispatch_to_oncall(result.parsed["category"], message_body)
```

That's the surface. No SDK initialisation, no "which provider should
I use," no API key — `ctx.llm` borrows whatever the user has running
right now. When they switch from OpenRouter to native Anthropic, the
plugin follows them automatically. When their primary provider 5xxs,
the host's fallback chain kicks in before the plugin sees the
failure. When the input contains an image and the active text model
is text-only, the host transparently routes to the configured vision
model and back.

## Three things that make this lane worth using

* **Bounded.** Single sync or async call. No streaming, no tool
  loops, no conversation state to manage. State the input, get the
  result, return.
* **Host-owned credentials.** OAuth tokens, refresh flows, the
  credential pool, per-task aux overrides — every credential
  concept Hermes already has applies. The plugin never sees a
  token; the host attributes the call back through `result.audit`.
* **Fail-closed trust.** A plugin you've never configured cannot
  pick its own model, run against another agent, or select a
  different stored credential. The default posture is "use what the
  user is using." Operators opt in to specific overrides, per
  plugin, in `config.yaml`.

## Quick start

```python
def register(ctx):
    ctx.register_command(
        name="paste-to-tasks",
        handler=lambda raw: _paste_to_tasks(ctx, raw),
        description="Turn freeform meeting notes into structured tasks.",
        args_hint="<text>",
    )


_TASKS_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "owner":  {"type": "string"},
                    "action": {"type": "string"},
                    "due":    {"type": "string", "description": "ISO date or empty"},
                },
                "required": ["action"],
            },
        },
    },
    "required": ["tasks"],
}


def _paste_to_tasks(ctx, raw_args: str) -> str:
    if not raw_args.strip():
        return "Usage: /paste-to-tasks <meeting notes>"
    result = ctx.llm.complete_structured(
        instructions=(
            "Extract concrete action items from these meeting notes. "
            "One task per actionable line. If no owner is named, leave 'owner' blank."
        ),
        input=[{"type": "text", "text": raw_args}],
        json_schema=_TASKS_SCHEMA,
        schema_name="meeting.tasks",
        purpose="paste-to-tasks",
        temperature=0.0,
        max_tokens=512,
    )
    if result.parsed is None:
        return f"Couldn't parse a response. Raw output:\n{result.text}"
    lines = [f"- [{t.get('owner') or '?'}] {t['action']}" for t in result.parsed["tasks"]]
    return "\n".join(lines) or "(no tasks found)"
```

`result.parsed` is a Python dict matching the schema when everything
worked, or `None` when the model couldn't produce valid JSON — the
raw text is always preserved on `result.text` for fallback handling.
The provider, model, token usage, and audit info come back attached
to the same result object.

A second worked example with image input ships in
[`plugins/plugin-llm-example/`](https://github.com/NousResearch/hermes-agent/tree/main/plugins/plugin-llm-example).

## API surface

`ctx.llm` is an instance of `agent.plugin_llm.PluginLlm`. Two
methods, plus async siblings.

### `complete()`

```python
result = ctx.llm.complete(
    messages=[{"role": "user", "content": "Hi"}],
    model=None,           # optional, gated
    temperature=None,
    max_tokens=None,
    timeout=None,         # seconds
    agent_id=None,        # optional, gated
    profile=None,         # optional, gated
    purpose="optional-audit-string",
)
# → PluginLlmCompleteResult(text, provider, model, agent_id, usage, audit)
```

Plain text completion. `messages` is the standard OpenAI shape.
`result.usage` exposes token + cache counts when the provider
returns them.

### `complete_structured()`

```python
result = ctx.llm.complete_structured(
    instructions="What you want extracted.",
    input=[
        {"type": "text",  "text": "..."},
        {"type": "image", "data": b"...", "mime_type": "image/png"},
        {"type": "image", "url":  "https://..."},
    ],
    json_schema={...},    # optional — triggers parsed result + validation
    json_mode=False,      # set True without a schema to ask for JSON anyway
    schema_name=None,     # optional human-readable schema name
    system_prompt=None,
    model=None,
    temperature=None,
    max_tokens=None,
    timeout=None,
    agent_id=None,
    profile=None,
    purpose=None,
)
# → PluginLlmStructuredResult(text, provider, model, agent_id,
#                             usage, parsed, content_type, audit)
```

The structured lane. Inputs are typed text or image blocks (raw
bytes get base64 encoded as a `data:` URL automatically). When
`json_schema` or `json_mode=True` is supplied, the host requests
JSON output via `response_format`, parses it locally as a fallback,
and validates against your schema if `jsonschema` is installed.

* `result.content_type == "json"` — `result.parsed` is a Python
  object that matches your schema.
* `result.content_type == "text"` — parsing or validation failed;
  inspect `result.text` for the raw model response.

### Async

```python
result = await ctx.llm.acomplete(messages=...)
result = await ctx.llm.acomplete_structured(instructions=..., input=...)
```

Same arguments and result types. Use these from gateway adapters,
async hooks, or any plugin code already running on an asyncio loop.

### Result attributes

```python
@dataclass
class PluginLlmStructuredResult:
    text: str                    # raw text, always populated
    provider: str                # e.g. "openai", "anthropic"
    model: str                   # e.g. "gpt-4o", "claude-3-5-sonnet"
    agent_id: str                # whose model/auth was used
    usage: PluginLlmUsage        # tokens + cache + cost estimate
    parsed: Optional[Any]        # JSON dict when content_type == "json"
    content_type: str            # "json" or "text"
    audit: Dict[str, Any]        # plugin_id, purpose, profile, schema_name
```

`usage` carries `input_tokens`, `output_tokens`, `total_tokens`,
`cache_read_tokens`, `cache_write_tokens`, and `cost_usd` when the
provider returns those fields.

## Trust gate

The default behaviour is fail-closed. With no `plugins.entries`
config block, a plugin can:

* run `complete()` / `complete_structured()` with the user's active
  model and auth,
* set request-shaping arguments (`temperature`, `max_tokens`,
  `timeout`, `system_prompt`, `purpose`),

…and that's it. `model=`, `agent_id=`, and `profile=` arguments
raise `PluginLlmTrustError` until the operator opts in.

To trust a plugin, add a `plugins.entries.<plugin-id>.llm` block to
`~/.hermes/config.yaml`:

```yaml
plugins:
  entries:
    my-plugin:
      llm:
        # Allow the plugin to ask for a specific model.
        allow_model_override: true

        # Optionally restrict which models. Use ["*"] for any.
        allowed_models:
          - openai/gpt-4o-mini
          - anthropic/claude-3-5-haiku

        # Allow cross-agent calls (rare).
        allow_agent_id_override: false

        # Allow the plugin to request a specific stored auth profile
        # (e.g. a different OAuth account on the same provider).
        allow_profile_override: false
```

The plugin id is the manifest `name:` field for flat plugins, or the
path-derived key for nested plugins (`image_gen/openai`,
`memory/honcho`, etc.).

### What the gate enforces

| Override               | Default | Config key                     |
| ---------------------- | ------- | ------------------------------ |
| `model="..."`          | denied  | `allow_model_override: true`   |
| `model="..."` allowlist| —       | `allowed_models: [...]`        |
| `agent_id="..."`       | denied  | `allow_agent_id_override: true`|
| `profile="..."`        | denied  | `allow_profile_override: true` |
| `model="x@profile"`    | denied  | requires both flags above      |

The `model@profile` shorthand goes through the same gate as
explicit `profile=`, so an embedded suffix can't bypass the
auth-profile policy. Conflicting explicit and embedded profiles
fail closed.

### What the gate does NOT need to enforce

* `temperature`, `max_tokens`, `timeout`, `system_prompt`,
  `purpose`, `schema_name`, `json_schema` — request-shaping
  arguments are always allowed; they don't pick credentials or
  models.
* The default deny posture means an unconfigured plugin can still do
  useful work — it just runs against the active model. Operators
  only need to think about `plugins.entries` for plugins that want
  finer routing.

## What the host owns

A complete list of the things `ctx.llm` does for the plugin so you
don't have to:

* **Provider resolution.** Reads `model.provider` + `model.model`
  from the user's config (or the explicit override when trusted).
* **Auth.** Pulls API keys, OAuth tokens, or refresh tokens from
  `~/.hermes/auth.json` / env, including the credential pool when
  one is configured. The plugin never sees them.
* **Vision routing.** When image input is supplied and the user's
  active text model is text-only, the host falls back to the
  configured vision model automatically.
* **Fallback chain.** If the user's primary provider 5xxs or 429s,
  the request goes through Hermes' usual aggregator-aware fallback
  before it returns an error to the plugin.
* **Timeout.** Honours your `timeout=` argument, falling back to
  `auxiliary.<task>.timeout` config or the global aux default.
* **JSON shaping.** Sends `response_format` to the provider when
  you ask for JSON, then re-parses locally from a code-fenced
  response if the provider returned one.
* **Schema validation.** Validates against your `json_schema` when
  `jsonschema` is installed; logs a debug line and skips strict
  validation otherwise.
* **Audit log.** Each call writes one INFO line to `agent.log` with
  the plugin id, provider/model, purpose, and token totals.

## What the plugin owns

* **Inputs.** Build the right `instructions` and `input` blocks for
  the job. Bytes vs URLs, image vs text — your call.
* **Schema.** Whatever shape you want back. The host doesn't infer
  it for you.
* **Error handling.** `complete_structured()` raises `ValueError` on
  empty inputs and on schema-validation failure. `PluginLlmTrustError`
  fires when the trust gate denies an override. Anything else
  (provider 5xx, no credentials configured, timeout) raises whatever
  `auxiliary_client.call_llm()` raises.
* **Cost.** The plugin runs against the user's paid provider. Don't
  loop on `complete_structured()` for every gateway message without
  thinking about token spend.

## Where this fits in the plugin surface

Existing `ctx.*` methods extend an existing Hermes subsystem:

| `ctx.register_tool` | adds a tool the agent can call |
| `ctx.register_platform` | wires a new gateway adapter |
| `ctx.register_image_gen_provider` | replaces an image-gen backend |
| `ctx.register_memory_provider` | replaces the memory backend |
| `ctx.register_context_engine` | replaces the context compressor |
| `ctx.register_hook` | observes a lifecycle event |

`ctx.llm` is the first surface that lets a plugin run the same model
the user is talking to, *out of band*, without any of the above.
That's its only job. If your plugin needs to register a tool the
agent invokes, use `register_tool`. If it needs to react to a
lifecycle event, use `register_hook`. If it needs to make its own
decision based on its own LLM call — `ctx.llm`.

## Reference

* Implementation: [`agent/plugin_llm.py`](https://github.com/NousResearch/hermes-agent/blob/main/agent/plugin_llm.py)
* Tests: [`tests/agent/test_plugin_llm.py`](https://github.com/NousResearch/hermes-agent/blob/main/tests/agent/test_plugin_llm.py)
* Worked example: [`plugins/plugin-llm-example/`](https://github.com/NousResearch/hermes-agent/tree/main/plugins/plugin-llm-example)
* Auxiliary client (the engine under the hood): see
  [Provider Runtime](/docs/developer-guide/provider-runtime).
