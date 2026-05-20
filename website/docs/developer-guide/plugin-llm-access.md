---
sidebar_position: 11
title: "Plugin LLM Access"
description: "Run any LLM call from inside a plugin via ctx.llm — chat or structured, sync or async. Host-owned auth, fail-closed trust gate, optional JSON Schema validation."
---

# Plugin LLM Access

`ctx.llm` is the supported way for a plugin to make an LLM call.
Chat completion, structured extraction, sync, async, with or without
images — same surface, same trust gate, same host-owned credentials.

Plugins reach for this when they need to do something that involves
the model but isn't part of the agent's conversation. A hook that
rewrites a tool error into something a non-engineer can read. A
gateway adapter that translates an inbound message before queuing
it. A slash command that summarises a long paste. A scheduled job
that scores yesterday's activity and writes one line to a status
board. A pre-filter that decides whether a message is worth waking
the agent up for at all.

These are jobs the agent shouldn't be in the loop on. They want one
LLM call, a typed answer, and to be done.

## The smallest possible call

```python
result = ctx.llm.complete(messages=[{"role": "user", "content": "ping"}])
return result.text
```

That's the whole API in one line. No keys, no provider config, no
SDK initialisation. The plugin runs against whatever provider and
model the user is currently using — when they switch providers, the
plugin follows them automatically.

## A more complete chat example

```python
result = ctx.llm.complete(
    messages=[
        {"role": "system", "content": "Rewrite errors as one short sentence a non-engineer can act on."},
        {"role": "user",   "content": traceback_text},
    ],
    max_tokens=64,
    purpose="hooks.error-rewrite",
)
return result.text
```

`purpose` is a free-form audit string — it shows up in `agent.log`
and in `result.audit` so operators can see which plugin made which
call. Optional but recommended for anything that fires often.

## Structured output

When the plugin needs a typed answer, switch to the structured lane:

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

The host requests JSON output from the provider, parses it locally
as a fallback, validates against your schema if `jsonschema` is
installed, and hands back a Python object on `result.parsed`. If the
model couldn't produce valid JSON, `result.parsed` is `None` and
`result.text` carries the raw response.

## What this lane gives you

* **One call, four shapes.** `complete()` for chat,
  `complete_structured()` for typed JSON, `acomplete()` and
  `acomplete_structured()` for asyncio. Same arguments, same result
  objects.
* **Host-owned credentials.** OAuth tokens, refresh flows, the
  credential pool, per-task aux overrides — every credential
  concept Hermes already has applies. The plugin never sees a
  token; the host attributes the call back through `result.audit`.
* **Bounded.** Single sync or async call. No streaming, no tool
  loops, no conversation state to manage. State the input, get the
  result, return.
* **Fail-closed trust.** A plugin you've never configured cannot
  pick its own provider, model, agent, or stored credential. The
  default posture is "use what the user is using." Operators opt in
  to specific overrides, per plugin, in `config.yaml`.

## Quick start

Two complete plugins below — one chat, one structured. Both ship
inside a single `register(ctx)` function and need zero outside
configuration to run against whatever model the user has active.

### Chat completion — `/tldr`

```python
def register(ctx):
    ctx.register_command(
        name="tldr",
        handler=lambda raw: _tldr(ctx, raw),
        description="Summarise the supplied text in one paragraph.",
        args_hint="<text>",
    )


def _tldr(ctx, raw_args: str) -> str:
    text = raw_args.strip()
    if not text:
        return "Usage: /tldr <text to summarise>"
    result = ctx.llm.complete(
        messages=[
            {"role": "system",
             "content": "Summarise the user's text in one tight paragraph. No preamble."},
            {"role": "user", "content": text},
        ],
        max_tokens=256,
        temperature=0.3,
        purpose="tldr",
    )
    return result.text
```

`result.text` is the model's response; `result.usage` carries token
counts; `result.provider` and `result.model` carry attribution.

### Structured extraction — `/paste-to-tasks`

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

A third worked example, this time with image input, lives in the
[`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-example)
repo (companion repo for reference plugins — not bundled with
hermes-agent itself). For the async surface (`acomplete()` /
`acomplete_structured()` with `asyncio.gather()`), see
[`plugin-llm-async-example`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-async-example)
in the same repo.

## When to use which

| You want… | Reach for |
|---|---|
| A free-form text response (translation, summary, rewrite, generation) | `complete()` |
| A multi-turn prompt (system + few-shot examples + user) | `complete()` |
| A typed dict back, validated against a schema | `complete_structured()` |
| Image-or-text input with a typed dict back | `complete_structured()` |
| The same call from async code (gateway adapters, async hooks) | `acomplete()` / `acomplete_structured()` |

Everything else — provider selection, model resolution, auth, fallback,
timeout, vision routing — is the same across all four.

## API surface

`ctx.llm` is an instance of `agent.plugin_llm.PluginLlm`.

### `complete()`

```python
result = ctx.llm.complete(
    messages=[{"role": "user", "content": "Hi"}],
    provider=None,         # optional, gated — Hermes provider id (e.g. "openrouter")
    model=None,            # optional, gated — whatever string that provider expects
    temperature=None,
    max_tokens=None,
    timeout=None,          # seconds
    agent_id=None,         # optional, gated
    profile=None,          # optional, gated — explicit auth-profile name
    purpose="optional-audit-string",
)
# → PluginLlmCompleteResult(text, provider, model, agent_id, usage, audit)
```

Plain chat completion. `messages` is the standard OpenAI shape — a
list of `{"role": "...", "content": "..."}` dicts. Multi-turn
prompts (system + few-shot user/assistant pairs + final user) work
exactly as they would with the OpenAI SDK.

`provider=` and `model=` are independent and follow the same shape
as the host's main config (`model.provider` + `model.model`). Set
just `model=` to use the user's active provider with a different
model on it. Set both to switch providers entirely. Either argument
without operator opt-in raises `PluginLlmTrustError`.

### `complete_structured()`

```python
result = ctx.llm.complete_structured(
    instructions="What you want extracted.",
    input=[
        {"type": "text",  "text": "..."},
        {"type": "image", "data": b"...", "mime_type": "image/png"},
        {"type": "image", "url":  "https://..."},
    ],
    json_schema={...},     # optional — triggers parsed result + validation
    json_mode=False,       # set True without a schema to ask for JSON anyway
    schema_name=None,      # optional human-readable schema name
    system_prompt=None,
    provider=None,         # optional, gated
    model=None,            # optional, gated
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

Inputs are typed text or image blocks (raw bytes get base64 encoded
as a `data:` URL automatically). When `json_schema` or
`json_mode=True` is supplied, the host requests JSON output via
`response_format`, parses it locally as a fallback, and validates
against your schema if `jsonschema` is installed.

* `result.content_type == "json"` — `result.parsed` is a Python
  object that matches your schema.
* `result.content_type == "text"` — parsing or validation failed;
  inspect `result.text` for the raw model response.

### Async

```python
result = await ctx.llm.acomplete(messages=...)
result = await ctx.llm.acomplete_structured(instructions=..., input=...)
```

Same arguments and result types as their sync counterparts. Use
these from gateway adapters, async hooks, or any plugin code
already running on an asyncio loop.

### Result attributes

```python
@dataclass
class PluginLlmCompleteResult:
    text: str                    # the assistant's response
    provider: str                # e.g. "openrouter", "anthropic"
    model: str                   # whatever the provider returned for this call
    agent_id: str                # whose model/auth was used
    usage: PluginLlmUsage        # tokens + cache + cost estimate
    audit: Dict[str, Any]        # plugin_id, purpose, profile

@dataclass
class PluginLlmStructuredResult(PluginLlmCompleteResult):
    parsed: Optional[Any]        # JSON object when content_type == "json"
    content_type: str            # "json" or "text"
    # audit also carries schema_name when supplied
```

`usage` carries `input_tokens`, `output_tokens`, `total_tokens`,
`cache_read_tokens`, `cache_write_tokens`, and `cost_usd` when the
provider returns those fields.

## Trust gate

The default behaviour is fail-closed. With no `plugins.entries`
config block, a plugin can:

* run any of the four methods against the user's active provider
  and model,
* set request-shaping arguments (`temperature`, `max_tokens`,
  `timeout`, `system_prompt`, `purpose`, `messages`, `instructions`,
  `input`, `json_schema`),

…and that's it. `provider=`, `model=`, `agent_id=`, and `profile=`
arguments raise `PluginLlmTrustError` until the operator opts in.

**Most plugins never need this section.** A plugin that just calls
`ctx.llm.complete(messages=...)` with no overrides runs against
whatever the user has active and works zero-config. The block below
is only relevant when a plugin specifically wants to pin to a
different model or provider than the user.

```yaml
plugins:
  entries:
    my-plugin:
      llm:
        # Allow this plugin to choose a different Hermes provider
        # (must be one Hermes already knows about — same names as
        # `hermes model` and config.yaml model.provider).
        allow_provider_override: true

        # Optionally restrict which providers. Use ["*"] for any.
        allowed_providers:
          - openrouter
          - anthropic

        # Allow this plugin to ask for a specific model.
        allow_model_override: true

        # Optionally restrict which models. Use ["*"] for any.
        # Models are matched literally against whatever string the
        # plugin sends — Hermes does not look anything up.
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

| Override        | Default | Config key                       |
| --------------- | ------- | -------------------------------- |
| `provider=`     | denied  | `allow_provider_override: true`  |
| ↳ allowlist     | —       | `allowed_providers: [...]`       |
| `model=`        | denied  | `allow_model_override: true`     |
| ↳ allowlist     | —       | `allowed_models: [...]`          |
| `agent_id=`     | denied  | `allow_agent_id_override: true`  |
| `profile=`      | denied  | `allow_profile_override: true`   |

Each override is independently gated. Granting `allow_model_override`
does **not** also grant `allow_provider_override` — a plugin trusted
to pick a model is still pinned to the user's active provider unless
it gets the provider gate as well.

### What the gate does NOT need to enforce

* Request-shaping arguments — `temperature`, `max_tokens`,
  `timeout`, `system_prompt`, `purpose`, `messages`, `instructions`,
  `input`, `json_schema`, `schema_name`, `json_mode` — are always
  allowed; they don't pick credentials or routes.
* The default deny posture means an unconfigured plugin can still do
  useful work — it just runs against the active provider and model.
  Operators only need to think about `plugins.entries` for plugins
  that want finer routing.

## What the host owns

A complete list of the things `ctx.llm` does for the plugin so you
don't have to:

* **Provider resolution.** Reads `model.provider` + `model.model`
  from the user's config (or the explicit overrides when trusted).
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

* **Request shape.** `messages` for chat, `instructions` + `input`
  for structured. The plugin builds the prompt; the host runs it.
* **Schema.** Whatever shape you want back. The host doesn't infer
  it for you.
* **Error handling.** `complete_structured()` raises `ValueError` on
  empty inputs and on schema-validation failure. `PluginLlmTrustError`
  fires when the trust gate denies an override. Anything else
  (provider 5xx, no credentials configured, timeout) raises whatever
  `auxiliary_client.call_llm()` raises.
* **Cost.** Every call runs against the user's paid provider. Don't
  loop on `complete()` for every gateway message without thinking
  about token spend.

## Where this fits in the plugin surface

Existing `ctx.*` methods extend an existing Hermes subsystem:

| `ctx.register_tool` | adds a tool the agent can call |
| `ctx.register_platform` | wires a new gateway adapter |
| `ctx.register_image_gen_provider` | replaces an image-gen backend |
| `ctx.register_memory_provider` | replaces the memory backend |
| `ctx.register_context_engine` | replaces the context compressor |
| `ctx.register_hook` | observes a lifecycle event |

`ctx.llm` is the first surface that lets a plugin run the same
model the user is talking to, *out of band*, without any of the
above. That's its only job. If your plugin needs to register a
tool the agent invokes, use `register_tool`. If it needs to react
to a lifecycle event, use `register_hook`. If it needs to make its
own model call — for any reason, structured or not — `ctx.llm`.

## Reference

* Implementation: [`agent/plugin_llm.py`](https://github.com/NousResearch/hermes-agent/blob/main/agent/plugin_llm.py)
* Tests: [`tests/agent/test_plugin_llm.py`](https://github.com/NousResearch/hermes-agent/blob/main/tests/agent/test_plugin_llm.py)
* Reference plugins (companion repo):
  * [`plugin-llm-example`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-example) — sync structured extraction with image input
  * [`plugin-llm-async-example`](https://github.com/NousResearch/hermes-example-plugins/tree/main/plugin-llm-async-example) — async with `asyncio.gather()`
* Auxiliary client (the engine under the hood): see
  [Provider Runtime](/docs/developer-guide/provider-runtime).
