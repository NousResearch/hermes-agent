---
sidebar_position: 8
title: "Memory Provider Plugins"
description: "How to build a memory provider plugin for Hermes Agent"
---

# Building a Memory Provider Plugin

Memory provider plugins give Hermes Agent persistent, cross-session knowledge beyond the built-in MEMORY.md and USER.md. This guide covers how to build one.

:::tip
Memory providers are one of two **provider plugin** types. The other is [Context Engine Plugins](/developer-guide/context-engine-plugin), which replace the built-in context compressor. Both follow the same pattern: single-select, config-driven, managed via `hermes plugins`.
:::

## Installation Layouts

Hermes discovers memory providers from four sources, in this precedence order:

| Source | Location | Notes |
|---|---|---|
| Bundled | `plugins/memory/<name>/` | Ships with Hermes. Closed to new providers — see [CONTRIBUTING](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md). |
| User | `$HERMES_HOME/plugins/<name>/` | Dropped in by the user, per profile. |
| Project | `./.hermes/plugins/<name>/` | Opt-in via `HERMES_ENABLE_PROJECT_PLUGINS=1`. |
| Package | `hermes_agent.memory_providers` entry point | `pip install`, nothing to copy. |

Earlier sources win on a name collision, so a directory dropped into a working
tree can never shadow a shipped provider.

:::note
This is the reverse of the general plugin system's later-wins order. A memory
provider is activated by *name* (`memory.provider`), so shadowing would
silently redirect the agent's memory rather than merely override a tool.
:::

Discovery only *enumerates* — it never imports a provider. Nothing runs until
`memory.provider` names it.

### Directory Provider

A directory provider lives in `plugins/memory/<name>/` when bundled with
Hermes, in `$HERMES_HOME/plugins/<name>/` when installed by a user, or in
`./.hermes/plugins/<name>/` for a project-local one:

```
plugins/memory/my-provider/
├── __init__.py      # MemoryProvider implementation + register() entry point
├── plugin.yaml      # Metadata (name, description, hooks)
└── README.md        # Setup instructions, config reference, tools
```

### Packaged Provider

A pip-installed provider publishes an entry point in the
`hermes_agent.memory_providers` group. The entry-point name is the provider
name users select in `memory.provider`; its value points to the provider's
`register(ctx)` function:

```toml title="pyproject.toml"
[project.entry-points."hermes_agent.memory_providers"]
my-provider = "my_provider:register"
```

Point the entry point at the **package**, or at a `register(ctx)` inside it, and
keep your implementation, skills, and other resources in the normal Python
package layout. No copy under `$HERMES_HOME/plugins/` is required.

A package entry point gets everything a directory install does, including the
two files Hermes reads from disk rather than importing — `config_schema.py`
(the dashboard config panel) and `cli.py` (your `hermes <provider>`
subcommands). Both are found next to your package's `__init__.py`, so point the
entry point at a package rather than a single module if you ship either.

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
| `handle_tool_call(tool_name, args, **kwargs)` | When agent uses your tools | **Yes** (if you have tools) |

### Config

| Method | Purpose | Must Implement? |
|--------|---------|-----------------|
| `get_config_schema()` | Declare config fields for `hermes memory setup` | **Yes** |
| `save_config(values, hermes_home)` | Write non-secret config to native location | **Yes** (unless env-var-only) |

### Optional Hooks

| Method | When Called | Use Case |
|--------|-----------|----------|
| `system_prompt_block()` | System prompt assembly | Static provider info |
| `prefetch(query, *, session_id="")` | Before each API call | Return recalled context |
| `queue_prefetch(query, *, session_id="")` | After each turn | Pre-warm for next turn |
| `sync_turn(user, assistant, *, session_id="", messages=None)` | After each completed turn | Persist conversation |
| `on_session_end(messages)` | Conversation ends | Final extraction/flush |
| `on_pre_compress(messages)` | Before context compression | Save insights before discard |
| `on_memory_write(action, target, content)` | Built-in memory writes | Mirror to your backend |
| `shutdown()` | Process exit | Clean up connections |

## Pre-Compress Checkpoints (fail-closed)

`on_pre_compress()` is best-effort by default: if your provider raises, the
host logs the failure and compression proceeds. That is the right default for
insight extraction — and the wrong one for a provider whose job is to archive
transcript evidence to a durable store *before* the lossy rewrite. For that
case the host offers an opt-in checkpoint contract (API v2):

```python
from agent.memory_provider import MemoryProvider

class MyArchivingProvider(MemoryProvider):
    # Opt in: every successful on_pre_compress() return means the durable
    # checkpoint is committed. Raise on any failure — do not return partial
    # success. Version 1 (the inherited default) is the implicit historical
    # contract: best-effort semantics, raw message list.
    pre_compress_checkpoint_api_version = 2

    def on_pre_compress(self, messages):
        ids = self._archive(messages)   # must be durable before returning
        return f"checkpoint: {ids}"     # forwarded into the summary prompt
```

Operators enable enforcement per deployment:

```yaml
compression:
  checkpoint_required: true   # default: false
```

With the gate on, compression **fails closed** before any lossy rewrite unless
an active provider advertising the API completed its checkpoint: the
uncompressed transcript is preserved, the compaction attempt errors with
`BLOCKED_MISSING_PREREQUISITE`, and it can be retried once your store
recovers. With the gate off (default), nothing changes for existing providers.

The gate binds to every compaction authority, not just the Hermes
summarizer: server-side native compaction (`compression.codex_responses_native`)
is suppressed while the gate is armed so the checkpoint-aware Hermes
compressor stays authoritative, and the `codex_app_server` API mode is
refused at agent init — the codex agent compacts its own thread with no
truthful pre-compaction boundary, so a required checkpoint cannot be
guaranteed there.

What your provider receives depends on its declared API version. Version 1
providers (the implicit default — every pre-existing provider) keep the
historical contract: the raw message list, exactly as before. Version 2
checkpoint providers receive normalized direct evidence instead:
user/assistant text rows only — tool results, system messages, the
`tool_calls` payload of assistant messages (their prose is kept), and prior
compaction summaries are filtered host-side. Prior summaries are recognized
via a persistent `_compressed_summary` message marker that survives process
restarts, so a resumed session never feeds derivative summaries back into
your archive.

**Checkpoints must be idempotent.** After a fail-closed block, the next
compaction attempt calls `on_pre_compress()` again with the same transcript —
and a transcript that grew only slightly produces largely overlapping
evidence. Key your archive writes by content (for example a transcript
digest) and upsert, so retries and overlaps deduplicate instead of
accumulating duplicate archives.

Contract tests: `tests/agent/test_pre_compress_checkpoint_contract.py`.

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

A provider may also expose read-only skills from the same callback. Skills are
qualified by the entry-point name and are loaded only when that memory provider
is active:

```python
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"

def register(ctx) -> None:
    ctx.register_memory_provider(MyMemoryProvider())
    ctx.register_skill(
        "maintenance",
        SKILLS_DIR / "maintenance" / "SKILL.md",
        "Maintain the provider's memory store",
    )
```

With the `my-provider` entry point active, the skill is available as
`my-provider:maintenance` through `skill_view()`.

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
def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
    def _sync():
        try:
            self._api.ingest(user_content, assistant_content, session_id=session_id, messages=messages)
        except Exception as e:
            logger.warning("Sync failed: %s", e)

    if self._sync_thread and self._sync_thread.is_alive():
        self._sync_thread.join(timeout=5.0)
    self._sync_thread = threading.Thread(target=_sync, daemon=True)
    self._sync_thread.start()
```

`messages` is optional OpenAI-style conversation context as of the completed
turn. When present, it includes user/assistant messages, assistant tool calls,
and tool result messages. Providers that do not need raw turn context can omit
the `messages` parameter; Hermes will continue calling them with the legacy
signature.

Cloud providers should document what parts of `messages` are sent off-device.
Tool calls and tool results may contain file paths, command output, or other
workspace data.

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

See `tests/agent/test_memory_provider.py` and adjacent memory tests (`tests/agent/test_memory_session_switch.py`, `tests/agent/test_memory_user_id.py`, `tests/run_agent/test_memory_provider_init.py`) for end-to-end patterns.

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
