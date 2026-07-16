---
sidebar_position: 9
title: "Context Engine Plugins"
description: "How to build a context engine plugin that replaces the built-in ContextCompressor"
---

# Building a Context Engine Plugin

Context engine plugins replace the built-in `ContextCompressor` with an alternative strategy for managing conversation context. For example, a Lossless Context Management (LCM) engine that builds a knowledge DAG instead of lossy summarization.

## How it works

The agent's context management is built on the `ContextEngine` ABC (`agent/context_engine.py`). The built-in `ContextCompressor` is the default implementation. Plugin engines must implement the same interface.

Only **one** context engine can be active at a time. Selection is config-driven:

```yaml
# config.yaml
context:
  engine: "lcm"  # activates a plugin engine named "lcm"; default: "compressor"
```

Plugin engines are **never auto-activated** — the user must explicitly set
`context.engine` to the plugin's name.

## Plugin structure

Third-party context engines should be standalone Hermes plugins with the
manifest and entry point at the repository root:

```
hermes-context-lcm/
├── __init__.py       # register(ctx) entry point
├── plugin.yaml       # plugin metadata
├── engine.py         # ContextEngine implementation
└── after-install.md  # optional setup instructions
```

For example:

```yaml title="plugin.yaml"
manifest_version: 1
name: context-lcm
version: "0.1.0"
description: Lossless context management for Hermes Agent
```

The in-tree `plugins/context_engine/<name>/` layout is reserved for engines
bundled with Hermes. It is useful when contributing to Hermes itself, but users
should not have to modify a managed Hermes checkout to install a third-party
engine.

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
                 focus_topic: str = None) -> list:
        """Compact the message list and return a new (possibly shorter) list.

        The returned list must be a valid OpenAI-format message sequence.

        ``focus_topic`` is an optional topic string from manual
        ``/compress <focus>``; engines that support guided compression should
        prioritise preserving information related to it, others may ignore it.
        """
```

Hermes passes optional compression keywords only when the engine declares
them. Engines using the current documented signature receive `focus_topic`;
older engines without it remain compatible. A `TypeError` raised inside the
engine is treated as an implementation error and is not retried with a
different signature.

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

## Registration

### Via a standalone plugin (recommended for third parties)

Export a `register(ctx)` function from the repository's root `__init__.py` and
register one engine instance:

```python title="__init__.py"
from .engine import LCMEngine

def register(ctx):
    ctx.register_context_engine(LCMEngine(context_length=200000))
```

Plugin loading and engine selection are separate. Enabling the plugin makes
the implementation available; setting `context.engine` selects it.

### Via the bundled-engine directory

Place your engine in `plugins/context_engine/<name>/`. The `__init__.py` must
export a `ContextEngine` subclass. The discovery system finds and instantiates
it automatically.

Use this layout only for an engine intended to ship as part of Hermes itself.

Only one engine can be registered. A second plugin attempting to register one
is rejected with a warning.

## Install and activate a third-party engine

Install from a Git repository and enable the plugin in the active Hermes
profile:

```bash
hermes plugins install owner/hermes-context-lcm --enable
hermes config set context.engine lcm
hermes gateway restart
```

The config value must match the engine's `name` property, not necessarily the
Git repository or manifest name. Start a new session after changing engines;
existing sessions may retain the engine instance created at session startup.

If the plugin provides its own setup command, run it after installation instead
of setting its config by hand. Use `hermes plugins list` to confirm that the
plugin is installed and enabled.

Updates do not modify the selected engine, but a running gateway must reload
the plugin code:

```bash
hermes plugins update context-lcm
hermes gateway restart
```

Before disabling or removing an active context engine, select the built-in
compressor so Hermes is never configured to use an unavailable engine:

```bash
hermes config set context.engine compressor
hermes plugins disable context-lcm
hermes plugins remove context-lcm
hermes gateway restart
```

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

- [Context Compression and Caching](/developer-guide/context-compression-and-caching) — how the built-in compressor works
- [Memory Provider Plugins](/developer-guide/memory-provider-plugin) — analogous single-select plugin system for memory
- [Plugins](/user-guide/features/plugins) — general plugin system overview
