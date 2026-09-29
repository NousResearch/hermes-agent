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
  engine: "compressor"    # default built-in
  engine: "lcm"           # activates a plugin engine named "lcm"
```

Plugin engines are **never auto-activated** — the user must explicitly set `context.engine` to the plugin's name.

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

        The legacy keys ``prompt_tokens``, ``completion_tokens``,
        ``total_tokens`` are always present. Newer hosts also include
        canonical buckets ``input_tokens``, ``output_tokens``,
        ``cache_read_tokens``, ``cache_write_tokens``, ``reasoning_tokens`` —
        treat them as optional for back-compat with older hosts.
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
| `carry_over_new_session_context(old_session_id, new_session_id)` | No-op | You need to migrate retained state from the old session to the new one (e.g. DAG handoff when `/new` runs with `carry_over=True`) |
| `update_model(model, context_length, ...)` | Updates context_length + threshold | You need to recalculate budgets on model switch |
| `get_tool_schemas()` | Returns `[]` | Your engine provides agent-callable tools (e.g., `lcm_grep`) |
| `handle_tool_call(name, args, **kwargs)` | Returns error JSON | You implement tool handlers |
| `should_compress_preflight(messages)` | Returns `False` | You can do a cheap pre-API-call estimate |
| `get_status()` | Standard token/threshold dict | You have custom metrics to expose |
| `select_context(request_messages, *, conversation_messages, incoming_message, budget_tokens)` | Returns `None` (no-op) | You select/route which context enters **this** request (retrieval, topic routing) — see below |
| `on_turn_complete(messages, usage=None, **kwargs)` | No-op | You ingest/index/observe the finished turn — see below |
| `clone_for_agent()` | `copy.deepcopy(self)` | Your engine holds uncopyable state (locks, SQLite/DB connections) — see [Via general plugin system](#via-general-plugin-system) |

## Per-turn context selection and observation

`compress()` answers "context is too long → make it shorter". Two optional,
no-op-default hooks cover the orthogonal *selection / observation* axis, so an
engine no longer has to force `should_compress()` to `True` and abuse
`compress()` as a per-turn callback:

```python
def select_context(self, request_messages, *, conversation_messages=None,
                   incoming_message=None, budget_tokens=0):
    """Choose/replace the context for THIS request, before dispatch.

    Return a new message list to use for this one provider call (retrieval,
    topic routing, role/branch switching), or None to leave it unchanged.
    Request-only: the persisted conversation history is never mutated.
    """

def on_turn_complete(self, messages, usage=None, **kwargs):
    """Observe a finished turn after the assistant/tool loop completes.

    Receives a shallow copy of the finalized transcript plus the turn's
    canonical usage dict (or None if no provider response was reached), so the
    engine can ingest/index/summarize for the next select_context(). The return
    value is ignored.
    """
```

Contract:

- **No-op by default, fail-open.** Both default to `return None`. A missing hook, an exception, or an invalid return value leaves the request untouched — so a failing engine is never worse than not installing one. The host also identity-checks for the inherited ABC default and skips it entirely, so non-implementing engines (including the built-in compressor) pay no per-request work at all.
- **`select_context()` is request-only.** The returned list replaces the messages for a single provider call; persisted history is never written. Returning `None`, `[]`, a non-list, or a list containing non-dicts all fall open to the unmodified request.
- **Ordering / cache stability.** The hook runs **before** prompt cache-control and every request sanitizer, so (a) a replacement still passes the same validation as any request, and (b) the no-op default leaves the request byte-identical — prompt-cache behaviour is unchanged for non-implementing engines. An engine that replaces the list changes only its own cache prefix. Evaluated per provider request (re-runs on retries).
- **`on_turn_complete()`** is post-turn observation only; treat `messages` as read-only. **Coverage is best-effort:** it fires from the standard turn-finalization seam. Some abnormal early-return paths in the loop (e.g. a content-policy block or a provider terminal failure) persist and return without routing through finalization, so they do not currently emit this hook — treat it as a best-effort observation for completed turns, not a guaranteed callback for every early exit. Unifying all terminal paths behind one finalization seam is a separate follow-up.

### Stable message identity: `message_uid`

Every message the host has persisted carries `message_uid`: a 32-hex id
(`uuid4().hex`) minted once, at the row's first insert, and stored in
`messages.message_uid`. It is the key to use when an engine keeps its own
per-message state (a verbatim store, a summary DAG, per-message embeddings)
and needs to recognise a message it has already seen. The physical row id
(`_row_id`) is not that key: it is re-issued by every copy and only present on
some restore paths.

What the host guarantees:

- **Present on every engine surface** once a row exists: the `compress()`
  input list, `on_turn_complete()` clones, `post_llm_call`'s
  `conversation_history`, `on_session_end()` messages, and every restored
  history (CLI, TUI, ACP, gateway, compression's durable-snapshot adoption). It
  is restored unconditionally, unlike `_row_id`. The one exception is a row
  written before the column existed: on a large store the upgrade mints those
  over the next few opens, so a legacy row can briefly arrive without one.
  Treat a missing uid as "no identity yet", never as an error.
- **Kept across every host copy of the same logical message:** in-place
  compaction generations and their concurrent-tail clones, rotation-child
  handoff copies and foreign-tail clones, `replace_messages` re-issues,
  rewind, export/import, and `/branch` / Desktop branch copies (the child
  session's copied rows keep the parent's uids).
- **Kept across content rewrites of the same row:** the persist override, the
  sanitizer's row-addressed rewrite, the interrupted-stream fill. Treat
  `(message_uid, content)` as a *version* of the message; never fail closed on
  a content change under a known uid.
- **Merges keep the first constituent's uid and record the rest.** Whenever
  the host folds one durable message into another (alternation repair's
  consecutive-user and consecutive-assistant merges, the compressor restating
  an in-flight task onto its summary carrier, the real user anchor folded into
  a trailing scaffolding turn, micro-compaction's adjacent-user merge), the
  composite keeps the uid of the constituent whose text comes first and
  records the others in `_absorbed_message_uids` (text order). The list is
  persisted on the survivor's row (`messages.absorbed_message_uids`) and
  restored with it, so after a restart an engine still sees that `A\n\nB` is
  the host's fold of `A` and `B` rather than a new message. Merges made while
  restoring a history (`repair_alternation=True`) record the witness the same
  way. A row the host discards rather than folds (a provisional verification
  candidate superseded by the final answer, or an assistant turn whose text
  sits beside multimodal content, which is never joined) is not recorded: the
  witness names text that lives on in the composite, nothing else.
- **Engine-authored rows keep the uid the engine sets.** If your `compress()`
  output pre-stamps `message_uid` on a summary carrier (or any row it emits),
  the host writes that value through the commit, both in place and on
  rotation, and every later restore and copy returns it; rows without one are
  minted at insert. An engine can therefore recognise its own rows by uid.
- **A uid names one logical message, not one row.** Copies of a message share
  it by design, and a host path that re-appends an edited or merged message
  next to a still-active earlier row (a persist override on a restored list,
  a merged survivor flushed as a new row) leaves two active rows with one uid.
  Never assume per-row uniqueness in the active set; key your own state on
  the uid and treat the later row as the current version.
- **Tool calls get per-occurrence ids too.** Provider tool-call ids repeat
  (Hermes mints deterministic `call_<12hex>` ids for identical calls, and models
  reuse ids), so an assistant message carries `_tool_call_uids`, a
  `{tool_call_id: uid}` map for its `tool_calls`, and each tool-result message
  carries the matching `_tool_call_uid`. Calls that repeat a provider id inside
  one response share one uid (every result carries it, so no call looks
  unanswered). When a fold leaves one message holding calls from two turns
  that share an id, that id maps to a list of uids, one per occurrence in
  `tool_calls` order. The provider-facing `id` inside
  `tool_calls` is untouched. Both are minted at the assistant row's first
  insert, paired onto the result when it is flushed (same batch, or from the
  live list when the result lands in a later flush) and on restore (from the
  preceding assistant row), persisted (`messages.tool_call_uids`,
  `messages.tool_call_uid`), and kept across the same copy and rewrite paths
  as `message_uid`. A result whose call was never persisted with a uid has none.
- **Never on the wire.** `message_uid`, `_absorbed_message_uids`,
  `_tool_call_uids` and `_tool_call_uid` are in
  `PERSISTENCE_ONLY_MESSAGE_FIELDS`: stripped from every outgoing provider
  copy and ignored by the token estimator. Engines that never read them are
  unaffected.
- **Absent only before the row exists.** The current turn's user message has
  no uid during a preflight `compress()` that runs before the turn-start
  flush; the same dict object receives it at that flush. Stores upgraded from
  an older schema backfill a uid onto every existing row once (schema v31),
  and an insert trigger mints one for any row an older build writes into a
  v31 store afterwards (that build's live dicts still lack it until restored).

### When to use these hooks — and when NOT to

- **Implement `select_context()` only when your engine must *replace* the
  per-request context** — retrieval-augmented selection, topic/branch routing,
  role switching. It is the only verb that can swap which messages enter a
  request: the `pre_llm_call` plugin hook is inject-only by documented design
  (it appends to the user message and never rewrites the list, to preserve the
  prompt-cache prefix). If you don't need replacement, don't implement it.
- **If your plugin only needs post-turn observation / ingestion** (indexing,
  memory sync, analytics), implement a **memory provider** (`sync_turn()` —
  see [Memory Provider Plugins](./memory-provider-plugin.md)) instead of a
  context engine. A context engine takes ownership of the session's compaction
  policy; a memory provider observes turns without owning anything.
  `on_turn_complete()` exists as the observation mirror for engines that
  *already* need `select_context()` — so the same component can learn from the
  turn it just routed — not as a general-purpose turn callback.
- **Prompt-cache impact of a real `select_context()`.** A non-no-op selection
  naturally changes the prompt-cache prefix for the turns where it changes the
  selection — that request's prefix no longer matches the provider's cached
  prefix, so those turns re-write cache instead of reading it. Engines should
  return **stable selections when nothing has changed** (same object or an
  equal list), and only reshape the context when the routing decision actually
  differs; a selection that shuffles per turn silently forfeits cache reuse
  every turn.

**`on_session_start` kwargs.** When the host transitions sessions (`/new`, resume, compression split), it forwards optional kwargs including `conversation_id` (stable identity that survives `session_id` rotation across `/new` / resume / compression splits — useful for keying persisted state), `old_session_id`, `carry_over_context`, `platform`, `model`, and `context_length`. Engines should treat each as optional and ignore unknown keys for forward compatibility.

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

### Slash commands

Engines registered via the `register(ctx)` pattern (see "Via general plugin system" below) can additionally expose slash commands by calling `ctx.register_command(name, handler, description=...)` during registration. Conflicts with built-in commands or with another plugin's command of the same name are silently rejected with a warning — built-ins win.

```python
def register(ctx):
    engine = LCMEngine(context_length=200000)
    ctx.register_context_engine(engine)
    ctx.register_command("lcm", engine.handle_lcm_command, description="Inspect LCM state")
```

## Registration

### Via directory (recommended)

Place your engine in `plugins/context_engine/<name>/` (bundled) or `~/.hermes/plugins/<name>/` (user-installed; `$HERMES_HOME/plugins/<name>/`). The `__init__.py` must export a `ContextEngine` subclass or a `register(ctx)` that calls `ctx.register_context_engine(...)`. Setting `context.engine: <name>` is the activation — a user-installed engine does not need a `plugins.enabled` entry. Bundled names win on collision.

### Via general plugin system

A general plugin can also register a context engine:

```python
def register(ctx):
    engine = LCMEngine(context_length=200000)
    ctx.register_context_engine(engine)
```

Only one engine can be registered. A second plugin attempting to register is rejected with a warning.

The registered instance is shared process-wide, but every `AIAgent` (parent, subagents, gateway
sessions) needs its own engine so a child's `update_model()` cannot mutate the parent's budget.
Hermes therefore calls `engine.clone_for_agent()` on the registered instance at each agent init.
The default is `copy.deepcopy(self)`; override it when the engine holds state that cannot be
deep-copied (locks, SQLite or HTTP connections) and return a fresh engine sharing the durable
backend while copying only the mutable budget fields. If the clone raises, the agent falls back to
the built-in compressor and logs `Context engine 'X' could not be safely copied for this agent`.

```python
def clone_for_agent(self):
    clone = LCMEngine(db_path=self.db_path)  # reopens its own connection
    clone.threshold_percent = self.threshold_percent
    return clone
```

## Lifecycle

```
1. Engine instantiated (plugin load or directory discovery)
2. on_session_start() — conversation begins
3. update_from_response() — after each API call
4. should_compress() — checked each turn
5. compress() — called when should_compress() returns True
6. on_session_end() — session boundary (CLI exit, /reset, gateway shutdown)
```

`on_session_reset()` is called on `/new` or `/reset` to clear per-session state without a full shutdown.

## Configuration

Users select your engine via `hermes plugins` → Provider Plugins → Context Engine, or by editing `config.yaml`:

```yaml
context:
  engine: "lcm"   # must match your engine's name property
```

The `compression` config block (`compression.threshold`, `compression.protect_last_n`, etc.) is specific to the built-in `ContextCompressor`, with one explicit exception: `compression.model_thresholds` (per-model threshold overrides) is part of the context-engine contract. The host assigns the resolved map to `engine.model_thresholds` *before* the initial `update_model()` call, and the base-class `update_model()` applies it (longest substring match, falling back to the engine's configured threshold). Engines that override `update_model()` own their own compaction policy and may honor or ignore the map — `from agent.context_compressor import resolve_model_threshold` to reuse the same resolution logic. For everything else, your engine should define its own config format if needed, reading from `config.yaml` during initialization.

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

## Thread safety

When `compression.context_timeout_seconds > 0` (the default), Hermes runs the
whole compression pass — including your engine's `compress()` and boundary
callbacks, and any memory provider's `on_pre_compress` /
`on_session_switch` — on a pooled daemon thread with a host-side timeout.
Your engine must therefore assume:

- Calls may arrive on an arbitrary pooled thread. Do not rely on thread
  affinity or `threading.local` state shared with the conversation thread.
- The message list you receive is a private deep snapshot; mutating it in
  place is allowed (legacy contract), but the mutation only becomes visible
  if the pass commits. After a host timeout your still-running work is
  discarded — never publish to external/durable state outside the commit.
- Passes for *different* sessions can run concurrently on pool siblings; a
  single engine/provider instance shared across sessions must be thread-safe.

## See also

- [Context Compression and Caching](./context-compression-and-caching.md) — how the built-in compressor works
- [Memory Provider Plugins](./memory-provider-plugin.md) — analogous single-select plugin system for memory
- [Plugins](../user-guide/features/plugins.md) — general plugin system overview
