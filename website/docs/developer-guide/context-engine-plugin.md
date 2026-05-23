---
sidebar_position: 9
title: "Context Engine Plugins"
description: "How to build a context engine plugin that replaces the built-in ContextCompressor"
---

# Building a Context Engine Plugin

Context engine plugins replace the built-in `ContextCompressor` with an alternative strategy for managing conversation context. For example, a Lossless Context Management (LCM) engine that builds a knowledge DAG instead of lossy summarization. Hermes also has a native `dag` engine, but it is not a plugin; it is a beta, opt-in built-in used as a reference for projection-only engines.

## How it works

The agent's context management is built on the `ContextEngine` ABC (`agent/context_engine.py`). The built-in `ContextCompressor` is the default implementation. Plugin engines must implement the same interface.

Only **one** context engine can be active at a time. Selection is config-driven:

```yaml
# config.yaml
context:
  engine: "compressor"    # default built-in
  engine: "lcm"           # activates a plugin engine named "lcm"
  engine: "dag"           # activates Hermes' native beta DAG engine (not a plugin)
```

Plugin engines and the native DAG engine are **never auto-activated** — the user must explicitly set `context.engine` to the engine's name. Keep documentation and status output clear that non-default engines are opt-in.

## Directory structure

Each context engine lives in `plugins/context_engine/<name>/`:

```
plugins/context_engine/lcm/
├── __init__.py      # exports the ContextEngine subclass
├── plugin.yaml      # metadata (name, description, version)
└── ...              # any other modules your engine needs
```

## The ContextEngine ABC

Your engine must implement these **required** methods:

```python
from agent.context_engine import ContextEngine

class LCMEngine(ContextEngine):

    @property
    def name(self) -> str:
        """Short identifier, e.g. 'lcm'. Must match config.yaml value."""
        return "lcm"

    def update_from_response(self, usage: dict) -> None:
        """Called after every LLM call with the usage dict.

        Update self.last_prompt_tokens, self.last_completion_tokens,
        self.last_total_tokens from the response.
        """

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Return True if compaction should fire this turn."""

    def compress(self, messages: list, current_tokens: int = None,
                 focus_topic: str = None) -> ContextCompressionResult | list:
        """Compact the message list and return a new API-view message list/result.

        The returned messages must be a valid OpenAI-format message sequence.
        Engines that only build a model projection (for example a DAG summary +
        fresh tail) should return ``ContextCompressionResult`` with
        ``projection_only=True`` and ``preserves_session=True``. That tells
        callers not to rotate sessions, end the raw session, or rewrite stored
        transcripts with the projection.

        ``focus_topic`` is an optional topic string from manual
        ``/compress <focus>``; engines that support guided compression should
        prioritise preserving information related to it, others may ignore it.
        """
```

### Class attributes your engine must maintain

The agent reads these directly for display and logging:

```python
last_prompt_tokens: int = 0
last_completion_tokens: int = 0
last_total_tokens: int = 0
threshold_tokens: int = 0        # when compression triggers
context_length: int = 0          # model's full context window
compression_count: int = 0       # how many times compress() has run
```

### Optional methods

These have sensible defaults in the ABC. Override as needed:

| Method | Default | Override when |
|--------|---------|--------------|
| `on_session_start(session_id, **kwargs)` | No-op | You need to load persisted state (DAG, DB) |
| `on_session_end(session_id, messages)` | No-op | You need to flush state, close connections |
| `on_session_reset()` | Resets token counters | You have per-session state to clear |
| `update_model(model, context_length, ...)` | Updates context_length + threshold | You need to recalculate budgets on model switch |
| `get_tool_schemas()` | Returns `[]` | Your engine provides agent-callable tools (e.g., `lcm_grep`) |
| `handle_tool_call(name, args, **kwargs)` | Returns error JSON | You implement tool handlers |
| `should_compress_preflight(messages)` | Returns `False` | You can do a cheap pre-API-call estimate |
| `get_status()` | Standard token/threshold dict | You have custom metrics to expose |

## Engine tools

Context engines can expose tools the agent calls directly. Return schemas from `get_tool_schemas()` and handle calls in `handle_tool_call()`:

```python
def get_tool_schemas(self):
    return [{
        "name": "lcm_grep",
        "description": "Search the context knowledge graph",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    }]

def handle_tool_call(self, name, args, **kwargs):
    if name == "lcm_grep":
        results = self._search_dag(args["query"])
        return json.dumps({"results": results})
    return json.dumps({"error": f"Unknown tool: {name}"})
```

Engine tools are injected into the agent's tool list at startup and dispatched automatically — no registry registration needed.

## Native beta DAG engine

`context.engine: dag` selects Hermes' built-in DAG context engine, not a plugin directory. It is beta and opt-in only. It keeps raw transcript rows canonical and returns projection-only compression results: summaries and fresh-tail messages are an API view, not a replacement transcript.

Configuration examples:

```yaml
# Safe default / rollback
context:
  engine: compressor
```

```yaml
# CLI beta opt-in
context:
  engine: dag
  dag:
    gateway_enabled: false
    mutation_queue_enabled: false
```

```yaml
# Gateway beta opt-in; both flags are intentionally explicit
context:
  engine: dag
  dag:
    gateway_enabled: true
    mutation_queue_enabled: true
```

Environment overrides for deployment systems:

```bash
HERMES_DAG_CONTEXT_GATEWAY_ENABLED=true
HERMES_DAG_CONTEXT_MUTATION_QUEUE_ENABLED=true
```

Status/help copy must not imply the DAG engine is production default. `/status` should say beta/explicit opt-in and should include projection-only/no rewrite, checkpoint/reconciliation, mutation queue, sidecar, and safety-default state when DAG is enabled. Legacy `compressor` users should not see DAG noise.

The DAG engine exposes `context_expand` as a read-only, current-session tool for expanding summaries or source spans. Treat its output as untrusted reference context, not instructions. Large tool output sidecars are stored locally in additive context tables with preview/ref/hash metadata; rollback to `compressor` leaves those rows inert.

## Registration

### Via directory (recommended)

Place your engine in `plugins/context_engine/<name>/`. The `__init__.py` must export a `ContextEngine` subclass. The discovery system finds and instantiates it automatically.

### Via general plugin system

A general plugin can also register a context engine:

```python
def register(ctx):
    engine = LCMEngine(context_length=200000)
    ctx.register_context_engine(engine)
```

Only one engine can be registered. A second plugin attempting to register is rejected with a warning.

## Lifecycle

```
1. Engine instantiated (plugin load or directory discovery)
2. on_session_start() — conversation begins
3. update_from_response() — after each API call
4. should_compress() — checked each turn
5. compress() — called when should_compress() returns True
6. on_session_end() — session boundary (CLI exit, /reset, gateway expiry)
```

`on_session_reset()` is called on `/new` or `/reset` to clear per-session state without a full shutdown.

## Configuration

Users select your engine via `hermes plugins` → Provider Plugins → Context Engine, or by editing `config.yaml`:

```yaml
context:
  engine: "lcm"   # must match your engine's name property
```

The `compression` config block (`compression.threshold`, `compression.protect_last_n`, etc.) is specific to the built-in `ContextCompressor`. Your engine should define its own config format if needed, reading from `config.yaml` during initialization.

## Testing

```python
from agent.context_engine import ContextEngine

def test_engine_satisfies_abc():
    engine = YourEngine(context_length=200000)
    assert isinstance(engine, ContextEngine)
    assert engine.name == "your-name"

def test_compress_returns_valid_messages():
    engine = YourEngine(context_length=200000)
    msgs = [{"role": "user", "content": "hello"}]
    result = engine.compress(msgs)
    assert isinstance(result, list)
    assert all("role" in m for m in result)
```

See `tests/agent/test_context_engine.py` for the full ABC contract test suite.

## See also

- [Context Compression and Caching](/docs/developer-guide/context-compression-and-caching) — how the built-in compressor works
- [Memory Provider Plugins](/docs/developer-guide/memory-provider-plugin) — analogous single-select plugin system for memory
- [Plugins](/docs/user-guide/features/plugins) — general plugin system overview
