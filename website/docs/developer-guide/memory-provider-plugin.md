---
sidebar_position: 8
title: "Memory Provider Plugins"
description: "How to build a memory provider plugin for Hermes Agent"
---

# Building a Memory Provider Plugin

Memory provider plugins give Hermes Agent persistent, cross-session knowledge beyond the built-in MEMORY.md and USER.md. This guide covers how to build one.

:::tip
Memory providers are one of two **provider plugin** types. The other is [Context Engine Plugins](/docs/developer-guide/context-engine-plugin), which replace the built-in context compressor. Both follow the same pattern: single-select, config-driven, managed via `hermes plugins`.
:::

## Directory Structure

Each memory provider lives in `plugins/memory/<name>/`:

```
plugins/memory/my-provider/
├── __init__.py      # MemoryProvider implementation + register() entry point
├── plugin.yaml      # Metadata (name, description, hooks)
└── README.md        # Setup instructions, config reference, tools
```

## The MemoryProvider ABC

Your plugin implements the `MemoryProvider` abstract base class from `agent/memory_provider.py`:

```python
from agent.memory_provider import MemoryProvider

class MyMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "my-provider"

    def is_available(self) -> bool:
        """Check if this provider can activate. NO network calls."""
        return bool(os.environ.get("MY_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        """Called once at agent startup.

        kwargs always includes:
          hermes_home (str): Active HERMES_HOME path. Use for storage.
        """
        self._api_key = os.environ.get("MY_API_KEY", "")
        self._session_id = session_id

    # ... implement remaining methods
```

## Required Methods

### Core Lifecycle

| Method | When Called | Must Implement? |
|--------|-----------|-----------------|
| `name` (property) | Always | **Yes** |
| `is_available()` | Agent init, before activation | **Yes** — no network calls |
| `initialize(session_id, **kwargs)` | Agent startup | **Yes** |
| `get_tool_schemas()` | After init, for tool injection | **Yes** |
| `handle_tool_call(name, args)` | When agent uses your tools | **Yes** (if you have tools) |

### Config

| Method | Purpose | Must Implement? |
|--------|---------|-----------------|
| `get_config_schema()` | Declare config fields for `hermes memory setup` | **Yes** |
| `save_config(values, hermes_home)` | Write non-secret config to native location | **Yes** (unless env-var-only) |

### Optional Hooks

| Method | When Called | Use Case |
|--------|-----------|----------|
| `system_prompt_block()` | System prompt assembly | Static provider info |
| `prefetch(query)` | Before each API call | Return recalled context |
| `queue_prefetch(query)` | After each turn | Pre-warm for next turn |
| `sync_turn(user, assistant)` | After each completed turn | Persist conversation |
| `on_session_end(messages)` | Conversation ends | Final extraction/flush |
| `on_pre_compress(messages)` | Before context compression | Save insights before discard |
| `on_memory_write(action, target, content)` | Built-in memory writes | Mirror to your backend |
| `shutdown()` | Process exit | Clean up connections |

## Config Schema

`get_config_schema()` returns a list of field descriptors used by `hermes memory setup`:

```python
def get_config_schema(self):
    return [
        {
            "key": "api_key",
            "description": "My Provider API key",
            "secret": True,           # → written to .env
            "required": True,
            "env_var": "MY_API_KEY",   # explicit env var name
            "url": "https://my-provider.com/keys",  # where to get it
        },
        {
            "key": "region",
            "description": "Server region",
            "default": "us-east",
            "choices": ["us-east", "eu-west", "ap-south"],
        },
        {
            "key": "project",
            "description": "Project identifier",
            "default": "hermes",
        },
    ]
```

Fields with `secret: True` and `env_var` go to `.env`. Non-secret fields are passed to `save_config()`.

:::tip Minimal vs Full Schema
Every field in `get_config_schema()` is prompted during `hermes memory setup`. Providers with many options should keep the schema minimal — only include fields the user **must** configure (API key, required credentials). Document optional settings in a config file reference (e.g. `$HERMES_HOME/myprovider.json`) rather than prompting for them all during setup. This keeps the setup wizard fast while still supporting advanced configuration. See the Supermemory provider for an example — it only prompts for the API key; all other options live in `supermemory.json`.
:::

## Save Config

```python
def save_config(self, values: dict, hermes_home: str) -> None:
    """Write non-secret config to your native location."""
    import json
    from pathlib import Path
    config_path = Path(hermes_home) / "my-provider.json"
    config_path.write_text(json.dumps(values, indent=2))
```

For env-var-only providers, leave the default no-op.

## Plugin Entry Point

```python
def register(ctx) -> None:
    """Called by the memory plugin discovery system."""
    ctx.register_memory_provider(MyMemoryProvider())
```

## plugin.yaml

```yaml
name: my-provider
version: 1.0.0
description: "Short description of what this provider does."
hooks:
  - on_session_end    # list hooks you implement
```

## Threading Contract

**`sync_turn()` MUST be non-blocking.** If your backend has latency (API calls, LLM processing), run the work in a daemon thread:

```python
def sync_turn(self, user_content, assistant_content):
    def _sync():
        try:
            self._api.ingest(user_content, assistant_content)
        except Exception as e:
            logger.warning("Sync failed: %s", e)

    if self._sync_thread and self._sync_thread.is_alive():
        self._sync_thread.join(timeout=5.0)
    self._sync_thread = threading.Thread(target=_sync, daemon=True)
    self._sync_thread.start()
```

## Profile Isolation

All storage paths **must** use the `hermes_home` kwarg from `initialize()`, not hardcoded `~/.hermes`:

```python
# CORRECT — profile-scoped
from hermes_constants import get_hermes_home
data_dir = get_hermes_home() / "my-provider"

# WRONG — shared across all profiles
data_dir = Path("~/.hermes/my-provider").expanduser()
```

## Testing

See `tests/agent/test_memory_plugin_e2e.py` for the complete E2E testing pattern using a real SQLite provider.

```python
from agent.memory_manager import MemoryManager

mgr = MemoryManager()
mgr.add_provider(my_provider)
mgr.initialize_all(session_id="test-1", platform="cli")

# Test tool routing
result = mgr.handle_tool_call("my_tool", {"action": "add", "content": "test"})

# Test lifecycle
mgr.sync_all("user msg", "assistant msg")
mgr.on_session_end([])
mgr.shutdown_all()
```

## Adding CLI Commands

Memory provider plugins can register their own CLI subcommand tree (e.g. `hermes my-provider status`, `hermes my-provider config`). This uses a convention-based discovery system — no changes to core files needed.

### How it works

1. Add a `cli.py` file to your plugin directory
2. Define a `register_cli(subparser)` function that builds the argparse tree
3. The memory plugin system discovers it at startup via `discover_plugin_cli_commands()`
4. Your commands appear under `hermes <provider-name> <subcommand>`

**Active-provider gating:** Your CLI commands only appear when your provider is the active `memory.provider` in config. If a user hasn't configured your provider, your commands won't show in `hermes --help`.

### Example

```python
# plugins/memory/my-provider/cli.py

def my_command(args):
    """Handler dispatched by argparse."""
    sub = getattr(args, "my_command", None)
    if sub == "status":
        print("Provider is active and connected.")
    elif sub == "config":
        print("Showing config...")
    else:
        print("Usage: hermes my-provider <status|config>")

def register_cli(subparser) -> None:
    """Build the hermes my-provider argparse tree.

    Called by discover_plugin_cli_commands() at argparse setup time.
    """
    subs = subparser.add_subparsers(dest="my_command")
    subs.add_parser("status", help="Show provider status")
    subs.add_parser("config", help="Show provider config")
    subparser.set_defaults(func=my_command)
```

### Reference implementation

See `plugins/memory/honcho/cli.py` for a full example with 13 subcommands, cross-profile management (`--target-profile`), and config read/write.

### Directory structure with CLI

```
plugins/memory/my-provider/
├── __init__.py      # MemoryProvider implementation + register()
├── plugin.yaml      # Metadata
├── cli.py           # register_cli(subparser) — CLI commands
└── README.md        # Setup instructions
```

## Single Provider Rule

Only **one** external memory provider can be active at a time. If a user tries to register a second, the MemoryManager rejects it with a warning. This prevents tool schema bloat and conflicting backends.
