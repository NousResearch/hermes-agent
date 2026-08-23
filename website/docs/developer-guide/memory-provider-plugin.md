---
sidebar_position: 8
title: "Memory Provider Plugins"
description: "How to build a memory provider plugin for Hermes Agent"
---

# Building a Memory Provider Plugin

Memory provider plugins give Hermes Agent persistent, cross-session knowledge beyond the built-in MEMORY.md and USER.md. This guide covers how to build one.

:::tip
Memory providers are one of two **provider plugin** types. The other is [Context Engine Plugins](./context-engine-plugin.md), which replace the built-in context compressor. Both follow the same pattern: single-select, config-driven, managed via `hermes plugins`.
:::

## Installation Layouts

Hermes discovers memory providers from four sources, in this precedence order:

| Source | Location | Notes |
|---|---|---|
| Bundled | `plugins/memory/<name>/` | Ships with Hermes. Closed to new providers — see [CONTRIBUTING](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md). |
| User | `$HERMES_HOME/plugins/<name>/` | Dropped in by the user, per profile. |
| Project | `./.hermes/plugins/<name>/` | Opt-in via `HERMES_ENABLE_PROJECT_PLUGINS=1`. |
| Package | `hermes_agent.memory_providers` entry point | Distribution supplied by the installation owner; nothing to copy. |

Earlier sources win on a name collision, so a directory dropped into a working
tree can never shadow a shipped provider.

:::note
This is the reverse of the general plugin system's later-wins order. A memory
provider is activated by *name* (`memory.provider`), so shadowing would
silently redirect the agent's memory rather than merely override a tool.
:::

Discovery only *enumerates* — it never imports a provider. Nothing runs until
`memory.provider` names it.

Entry-point discovery does not install packages. Do not inject a provider into
Hermes's selected environment with pip. On PM-managed installations, ship a
directory provider with declared Python dependencies; plugin admission and
`hermes memory setup` prepare them through PM before use. Owner-managed builds
(such as Nix) can include an entry-point distribution declaratively.

CLI and dashboard setup share candidate preparation. PM includes the provider's
`pyproject.toml` or legacy `pip_dependencies` / `python_dependencies` alongside
the active plugin union; an importable module does not bypass declared version
constraints. Dashboard readiness checks the same inputs without installing
anything. A successful preparation may require restarting Hermes before the
running process can use the selected dependency generation. External sidecar
checks and setup commands remain separate from the Python union.

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

### Initialization context

`AIAgent` passes session context through `MemoryManager.initialize_all()` to
`initialize(session_id, **kwargs)`. Accept `**kwargs` and tolerate missing optional
fields; callers may initialize a provider without an agent or a session database.

| Keyword | Meaning |
|---|---|
| `hermes_home` | Active profile's storage directory. |
| `platform` | Session surface, such as `cli`, `gui`, `acp`, or `telegram`. |
| `session_title` | Stored session title, when available. A display label is not necessarily a user-selected identity. |
| `session_title_source` | Stored title provenance, when available: `derived`, `llm`, or `user`. Automatic sources must not be mistaken for explicit identity overrides; missing provenance retains a provider's legacy behavior. Shared constants live in `hermes_state_common.py`. |
| `cwd` | Non-empty logical workspace supplied as `AIAgent(cwd=...)`, available before provider initialization. Omitted for `None` or an empty string. |
| `gateway_session_key` | Stable messaging-chat identity for per-chat session isolation. |
| `user_id`, `user_id_alt`, `user_name`, `chat_id` | Gateway identity fields, included when present. |
| `agent_identity` | Active profile name, when available. |
| `agent_workspace`, `agent_context` | Runtime agent scope. `agent_workspace` is `hermes`; `agent_context` is `cron` for scheduler runs, `subagent` for `delegate_task` children, else `primary` — skip automatic writes for the non-primary values. |

Do not assume `os.getcwd()` identifies the conversation's workspace: one Desktop
or gateway backend can serve several sessions. If `cwd` is absent and directory
routing is needed, `agent.runtime_cwd.resolve_agent_cwd()` honors the session cwd
context, then scoped `terminal.cwd` (carried internally as `TERMINAL_CWD`), then
the launch directory. Construction-time workspace metadata does not require
changing the process cwd or rebuilding an existing conversation's system prompt.

Desktop/TUI workspace changes synchronize the live agent's `session_cwd` through
`tui_gateway/session_workdir.py::_register_session_cwd`, including when a deferred
agent is attached after a workspace move. This lets a first or restarted Codex
app-server session use the current workspace instead of its construction-time
cwd. It does not move an already-running Codex thread, reinitialize memory
providers, change an existing Honcho session identity, or invalidate the cached
system prompt. Provider initialization still receives the construction-time
workspace; absent or empty cwd remains unpinned.

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
| `prefetch(query, *, session_id="")` | Before each API call | Return recalled context as `str`, or `MemoryPrefetchResult` with optional observations |
| `queue_prefetch(query, *, session_id="")` | After each turn | Pre-warm for next turn |
| `sync_turn(user, assistant, *, session_id="", messages=None)` | After each completed turn | Persist conversation |
| `on_session_end(messages)` | Conversation ends | Final extraction/flush |
| `on_pre_compress(messages)` | Before context compression | Save insights before discard |
| `on_memory_write(action, target, content, metadata=None)` | Built-in memory writes | Mirror to your backend |
| `shutdown()` | Process exit | Clean up connections |

For native `replace` and `remove`, `metadata["previous_content"]` contains the full
entry selected under the native-store lock. Notifications are emitted only after
the complete write or batch succeeds. Batch notifications preserve operation order;
each operation's previous content reflects earlier operations in that batch.
`old_text` is the caller's search text, not the identity of the changed entry.
Older Hermes versions can omit `previous_content`. Providers that require exact
identity should skip destructive mirroring when it is absent.

### Oversized prefetch results

External `prefetch()` results above the configured spill threshold are written
to a private spill file and replaced with the configured head/tail preview.
The preview includes the path so the agent can read the full result when it is
actually needed. Results at or below the threshold are returned unchanged.

This uses the shared `hooks.output_spill` settings (`10,000` characters by
default); see [Plugins — oversized-context spill](./plugins/index.md#oversized-context-spill).

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

    def on_pre_compress(self, messages, *, require_checkpoint=False):
        # require_checkpoint mirrors the operator's checkpoint_required
        # setting: True means a raise here blocks the lossy rewrite.
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

None of the providers bundled with Hermes advertise checkpoint API v2 — the
contract is opt-in and exists for third-party archiving providers. Enabling
`checkpoint_required` without one therefore blocks every compression attempt
(manual and automatic): agent init logs a warning naming the active provider,
and each refusal names `compression.checkpoint_required` as the key to disable.

The gate binds to every compaction authority, not just the Hermes
summarizer: server-side native compaction (`compression.codex_responses_native`)
is suppressed while the gate is armed, post-turn micro-compaction
(`compression.micro_compact`) is forced off at agent init (it absorbs old
exchanges into a rolling summary with no checkpoint hook in its path), and
the `codex_app_server` API mode is refused at agent init — the codex agent
compacts its own thread with no truthful pre-compaction boundary, so a
required checkpoint cannot be guaranteed there. The checkpoint-aware Hermes
compressor stays the only lossy authority.

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
### Structured prefetch results

Returning a plain `str` remains fully supported and is the compatibility path.
Providers that have a concrete in-process consumer may instead return the small
immutable `MemoryPrefetchResult` / `MemoryObservation` pair:

```python
from agent.memory_provider import MemoryObservation, MemoryPrefetchResult


def prefetch(self, query: str, *, session_id: str = ""):
    context = self._format_context(query)
    return MemoryPrefetchResult(
        context=context,
        observations=(
            MemoryObservation(
                source_kind="context_snapshot",
                schema="my-provider.context_snapshot",
                version=1,
                payload={"available": bool(context)},
            ),
        ),
    )
```

`source_kind`, `schema`, `version`, `provider`, and the opaque JSON-safe
`payload` are the only core envelope fields. Providers may leave `provider`
empty; Hermes binds it to the registered provider name and rejects a mismatch.
Hermes accepts at most `MAX_MEMORY_OBSERVATIONS` envelopes per operation. Each
payload is recursively limited to JSON values, bounded strings, collection
sizes, nesting depth, and encoded bytes; malformed or oversized envelopes are
dropped while that provider's formatted context is retained. A provider
exception keeps the existing fail-isolated behavior and may omit that
provider's context, as it did for string prefetch failures.

The `memory_prefetch` plugin hook is an opt-in, observer-only boundary. It is
queued asynchronously only when a prefetch produced at least one valid
structured observation and receives these keyword fields:

- `observations`: the immutable tuple of validated `MemoryObservation` envelopes;
  each envelope is provider-bound and its JSON payload is recursively frozen.
- `query`: the exact clean prefetch query for this operation. This is sensitive
  and may contain raw user input; it is retained because an exact-operation
  consumer needs it.
- `session_id`: the operation's session identifier.
- `task_id`: the canonical task identifier for the owning turn, or `None` when
  a trusted direct caller did not provide one.
- `turn_id`: the canonical immutable turn identifier for the owning turn, or
  `None` when a trusted direct caller did not provide one.
- `context_sha256`: the lowercase hexadecimal SHA-256 digest of the final
  merged context's exact UTF-8 bytes.
- `context_byte_length`: the byte length of those exact UTF-8 bytes.

The hook never receives `MemoryPrefetchResult` or any other object from which
merged context can be read. Raw recalled/source content may therefore reach the
hook only when the provider explicitly authors it into that provider's
observation payload. Legacy/string-only provider context never reaches this
hook. Hook return values cannot transform context, and callback errors are
isolated by the host-owned observer dispatcher. Hermes provides no telemetry,
network delivery, or storage for this event. Delivery uses one bounded FIFO of
`1024` pending events per callback; enqueue never waits, and a full queue drops
the oldest pending event. Callback exceptions are logged and isolated. During
shutdown Hermes places a stop sentinel and waits at most the dispatcher's
bounded drain timeout (one second by default); a stuck callback is left on its
daemon worker rather than delaying process teardown. Consumers must therefore
treat this observer as best-effort and tolerate delay or drop.

The operation-bound observation tuple and explicit `task_id`/`turn_id` avoid
query, hash, call-order, mutable provider `last_*`, or session-cache inference
and therefore do not share observations between concurrent turns.
`memory_prefetch` binds to the owning turn only: it intentionally has no
`api_request_id`, `api_call_count`, or `retry_count`, because one turn may
issue several provider requests and the prefetch occurs before a single final
request can be known. Direct callers may still use `prefetch_all()` /
`prefetch_all_result()` without correlation; their structured event is explicit
about unavailable task/turn IDs (`None`) and does not invent IDs in the memory
subsystem.

A provider-neutral external observer can consume the contract without knowing
the provider's internal schema:

```python
from collections import deque


class ObservationCollector:
    def __init__(self):
        self.events = deque(maxlen=128)

    def record(self, *, observations, session_id, task_id, turn_id, **metadata):
        self.events.append({
            "session_id": session_id,
            "task_id": task_id,
            "turn_id": turn_id,
            "observations": observations,
            "context_sha256": metadata["context_sha256"],
            "context_byte_length": metadata["context_byte_length"],
        })


collector = ObservationCollector()


def register(ctx):
    ctx.register_hook("memory_prefetch", collector.record)
```

The collector receives the exact operation envelope and can hand it to its own
bounded audit/evaluation pipeline; Hermes does not choose that pipeline, persist
it, or send it anywhere.

### Correlation ownership matrix

The canonical owner is the agent turn prologue in
`agent/turn_context.py`. It creates `effective_task_id` (the caller-supplied
`task_id`, or the core's normal task ID) and the immutable `turn_id` before
`pre_llm_call`, before the memory prefetch operation, and before the first API
request. The values are locals returned in `TurnContext`; memory receives them
as explicit arguments rather than reading agent/provider mutable state. The
provider call remains part of the synchronous prefetch operation; observer
delivery is queued asynchronously afterward.

| Boundary | Owner and timing | Canonical correlation |
|---|---|---|
| `pre_llm_call` | `agent/turn_context.py`, once after turn IDs are created | `session_id`, `task_id`, `turn_id` |
| `memory_prefetch` | Same prologue, `prefetch_all()` for the current turn; observer delivery is queued asynchronously | Same `session_id`, `task_id`, `turn_id`; no request ID |
| `pre_api_request`, `post_api_request`, `api_request_error` | `agent/conversation_loop.py`, around each provider attempt | Same `task_id`/`turn_id` plus the outer model-call `api_request_id`/`api_call_count`; retries of that request retain those IDs and are distinguished by `retry_count` |
| `post_llm_call` | `agent/turn_finalizer.py`, successful final-answer ownership | Same `session_id`, `task_id`, `turn_id` |
| canonical `on_session_end` | `agent/turn_finalizer.py`, every `run_conversation()` finalization | Same turn IDs and outcome; it is not a session teardown ID |
| interruption fallback | CLI `_emit_interrupted_session_end()` may emit a reduced safety payload from the active agent slots; TUI teardown emits session-only legacy fields | Not a memory-correlation owner; no new IDs are inferred |
| `on_session_finalize` / provider shutdown | CLI, gateway, or TUI session teardown | Session lifecycle only; not a substitute for a turn correlation ID |

All product surfaces (CLI, gateway, TUI gateway, desktop's gateway client,
cron, and delegated/background agents) converge on `AIAgent.run_conversation()`
and this prologue. The only other `prefetch_all*` callers are the public
manager compatibility tests/direct callers; those have no final-answer owner
and therefore receive explicit unavailable (`None`) task/turn fields. The
post-turn `queue_prefetch_all()` path only warms the next provider turn and
does not emit a structured `memory_prefetch` event. No correlation is inferred
from query text, context hashes, call order, session caches, `last_*` state, or
ambient context variables.

The current Honcho provider remains on its existing formatted-string path in
this change. A future Honcho mapping should use only fields actually returned
by `peer.context` (for example, component availability), should not label the
data as ranked retrieval, and should not claim a dialectic model identity that
the provider does not expose. The generic contract is exercised by the test
fixture provider rather than inventing Honcho provenance here.

## Setup UX — what a standalone provider keeps

Every setup surface Hermes gives a bundled provider is driven by files in the provider's
own directory, so a provider installed from the plugin catalog keeps all of them:

| Surface | What the provider ships |
|---|---|
| Desktop → Capabilities → Tools → Memory (config panel) | `config_schema.py` (below) |
| `hermes memory setup` wizard | `get_config_schema()` declares the fields the wizard prompts for, `save_config(config, hermes_home)` persists them, `post_setup(hermes_home, config)` runs afterwards for anything interactive (OAuth, first sync); `get_status_config()` feeds `hermes memory status` |
| `hermes <provider> …` subcommands | `cli.py` with `register_cli(subparser)` ([Adding CLI Commands](#adding-cli-commands)) |
| Python dependencies | `pyproject.toml` `[project] dependencies` (or `python_dependencies` in `plugin.yaml`); installed under Hermes' own pins at install time and re-applied across `hermes update` |

Your provider's name, `memory.<name>` config section, data directory and tool names are the
contract with existing users. A provider that moves out of core keeps all four; Hermes then
installs the catalog plugin automatically for anyone whose `memory.provider` still names it
(on `hermes update`, and once at agent start when `security.allow_lazy_installs` is on).

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

**`sync_turn()` MUST be non-blocking.** If your backend has latency (API calls, LLM processing), run the work in a daemon thread — spawned with `agent.memory_provider.spawn_context_thread`, never a bare `threading.Thread`. Profile isolation (the active `HERMES_HOME`, the per-turn secret scope) lives in `contextvars`, and a plain thread starts with an empty context: under multiplexed profiles it would silently write into the *default* profile's store, and `get_secret()` fails closed there.

```python
from agent.memory_provider import spawn_context_thread

def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
    def _sync():
        try:
            self._api.ingest(user_content, assistant_content, session_id=session_id, messages=messages)
        except Exception as e:
            logger.warning("Sync failed: %s", e)

    if self._sync_thread and self._sync_thread.is_alive():
        self._sync_thread.join(timeout=5.0)
    self._sync_thread = spawn_context_thread(_sync, name="myprovider-sync")
    self._sync_thread.start()
```

The same applies to prefetch and writer threads. Small JSON config sidecars (`$HERMES_HOME/<provider>.json`) are read with `utils.read_json_or_empty` and written with `utils.atomic_json_write`; anything under `config.yaml` goes through `hermes_cli.config.save_config(..., merge_existing=True)`.

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

See `tests/agent/test_memory_provider.py` and adjacent memory tests (`tests/agent/test_memory_session_switch.py`, `tests/agent/test_memory_user_id.py`, `tests/agent/test_memory_provider_init.py`) for end-to-end patterns.

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

## `HERMES_HOME` survival contract (what wrappers can rely on)

For wrapper-style providers that keep their runtime in a sidecar venv outside Hermes-managed Python (no dependency surface — no `pyproject.toml`, `pip_dependencies`, or `python_dependencies` — at the scanned plugin root; a `pyproject.toml` belonging solely to an external or nested sidecar is not scanned):

- **Location.** `$HERMES_HOME/plugins/<name>/` is the profile-scoped plugin location, and `HERMES_HOME` follows the active context override, then `$HERMES_HOME`, then the platform default. Propagate `HERMES_HOME` when launching the wrapper or sidecar so profile isolation holds; `MemoryManager.initialize_all` injects the active `hermes_home` into every provider.
- **Survival.** Ordinary Hermes updates — including managed-venv rebuild/replacement by pm — do not delete or rewrite `$HERMES_HOME/plugins/**`. An installed wrapper directory and its marker file (e.g. `mnemosyne-wrapper.json`) survive. Explicit plugin updates and deletion flows (`hermes uninstall`, `hermes plugins remove`, profile deletion, user deletion) are excluded from this guarantee.
- **Sidecar isolation.** A plugin root with no dependency surface never joins the pm workspace dependency union; a resync or venv rebuild neither provisions deps for it nor touches its tree.
- **Conflicts.** For native shared-venv plugins, an unsatisfiable dependency union fails loudly: the candidate plugin stays unenabled and unimported (the admission authority refuses before publishing config, reporting the plugin identity plus the resolver's reason, with a re-enable/retry path and a machine-readable pm receipt). Dependency resolution does not automatically disable other plugins or run a bisect. Explicit plugin updates, removal, and independent security gates are separate operations.

