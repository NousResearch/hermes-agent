---
title: "Middleware"
description: "Behavior-changing plugin middleware for LLM and tool calls: contract, execution order, examples"
---

# Hermes Middleware

Hermes middleware is the behavior-changing companion to observer hooks.
Observer hooks report what happened. Middleware can change what happens by
rewriting a request before execution or by wrapping the execution callback
itself.

This contract is intentionally backend-neutral. A plugin can use it for local
policy, request shaping, tracing, adaptive routing, cache control, sandbox
selection, or handoff to runtimes such as NeMo Relay without changing Hermes'
planner, model provider adapters, tool registry, memory, or CLI UX.

With middleware enabled, plugins can:

- Rewrite LLM provider request kwargs before Hermes calls the provider.
- Rewrite tool arguments before guardrails, approval checks, hooks, and tool
  execution see them.
- Wrap the actual LLM execution callback while preserving Hermes retry,
  streaming, interrupt, and hook behavior.
- Wrap the actual tool execution callback while preserving Hermes guardrails,
  approval, post-tool hooks, and tool-result transformation.

## Contract

Plugins register middleware from `register(ctx)`:

```python
def register(ctx):
    ctx.register_middleware("llm_request", on_llm_request)
    ctx.register_middleware("llm_execution", on_llm_execution)
    ctx.register_middleware("tool_request", on_tool_request)
    ctx.register_middleware("tool_execution", on_tool_execution)
```

Every middleware callback receives:

- `telemetry_schema_version`: currently `hermes.observer.v1`
- `middleware_schema_version`: currently `hermes.middleware.v1`
- Runtime context such as `session_id`, `task_id`, `turn_id`,
  `api_request_id`, `provider`, `model`, `api_mode`, `tool_name`, and
  `tool_call_id` when applicable.

Supported middleware kinds:

| Kind | Payload | Return shape | Purpose |
| --- | --- | --- | --- |
| `llm_request` | `request`, `original_request` | `{"request": {...}}` | Replace effective provider kwargs before provider execution. |
| `tool_request` | `tool_name`, `args`, `original_args` | `{"args": {...}}` | Replace effective tool args before hooks, guardrails, approvals, and execution. |
| `llm_execution` | `request`, `original_request`, `next_call` | Any provider response | Wrap or replace the actual provider call. |
| `tool_execution` | `tool_name`, `args`, `original_args`, `next_call` | Any tool result | Wrap or replace the actual tool call. |
| `authorized_tool_execution` | `tool_name`, approved `args`, `env_type`, zero-argument `next_call`, trusted runtime identity | Actual completed tool result, or a pre-execution refusal | Acquire/release resources immediately around an approved terminal command. |

Request middleware can return optional trace fields:

```python
return {
    "request": updated_request,
    "source": "my-plugin",
    "reason": "selected fallback model",
}
```

Hermes stores those trace entries in later observer hook payloads as
`middleware_trace`.

Execution middleware receives a `next_call` callback. Call it to continue the
chain:

```python
def on_tool_execution(**kwargs):
    result = kwargs["next_call"](kwargs["args"])
    return result
```

If multiple plugins register the same execution middleware kind, Hermes runs
them as a nested chain in registration order. The four original middleware kinds are fail-open:
Hermes logs a warning and continues with the next middleware or the base
runtime path. A callback that fails the same way on every call (typically a
signature naming a field the middleware does not send) is reported **once** at
WARNING — the message lists the fields it does provide — and identical repeats go
to DEBUG, so a mis-declared middleware cannot flood the log; a plugin reload
resets the report.

### Approved terminal execution

`authorized_tool_execution` is a separate, **fail-closed** synchronous contract.
Its first consumer is `terminal`, after command/workdir validation and all terminal
approval guards. The existing `tool_execution` wrapper runs outside those terminal
guards and is not an appropriate place to acquire a resource needed only by an
approved command. The new kind does not alter other tools or their approvals.

An enabled, trusted **in-process** plugin can use this seam for a bounded resource
handoff before a build or GPU test starts. A separate supervisor resource broker
is the concrete consumer; the engine protocol and policy belong to that plugin,
not Hermes core. Registration alone does not suppress normal terminal yielding:
the consumer explicitly protects only the calls it has admitted.

```python
from hermes_cli.authorized_tool_execution import hold_foreground_execution

def register(ctx):
    ctx.register_middleware("authorized_tool_execution", admitted_terminal)

def admitted_terminal(*, args, next_call, lineage_valid, **context):
    if not resource_policy.matches(args["command"]):
        return next_call()
    if not lineage_valid or args.get("background"):
        raise RuntimeError("This resource policy requires an owned foreground call")
    with resource_policy.acquire(context):
        with hold_foreground_execution():
            return next_call()
```

Here `resource_policy` is supplied by the plugin. Admission must be bounded and
honor `tools.interrupt.is_interrupted()`. The continuation checks cancellation
again immediately before dispatch. It takes no arguments, runs once on its owning
thread, and expires when the callback returns. Changing the callback's private
copy of `args` cannot rewrite the already-approved command. Raising before
execution blocks that call; Hermes does not fall through to execute it. Once the
command completes, its actual result is retained even if middleware cleanup
fails or returns a different result.

Within `hold_foreground_execution()`, a foreground terminal call cannot use
Hermes's automatic yield-to-background path. Explicit background calls and
commands promoted to background by the existing timeout policy are reported with
`args["background"] == True`; a foreground-only consumer must reject those before
acquiring resources. This scope does not prevent a shell command itself from
spawning detached children, nor does it change backend timeout/cleanup semantics.
The terminal's ordinary retry loop is disabled inside this scope: a backend error
may follow a real side effect, so the protected command is not silently replayed.
Use bounded commands whose process lifetime the terminal backend owns. Plugin-host
isolation is not supported by this synchronous in-process contract.

### Trusted tool identity

`agent.tool_execution_context.current_tool_execution_context()` returns an
immutable snapshot bound around actual agent tool dispatch, including direct
`invoke_tool` dispatch. The snapshot contains `session_id`, `root_session_id`,
`parent_session_id`, `delegate_depth`, `lineage_valid`, `profile_home`, `task_id`,
`tool_call_id`, `turn_id`, and `api_request_id`. Approved execution middleware
receives these same fields. `profile_home` is the current context-local profile,
so multiplexed A/B/A sessions do not inherit the first profile's configuration.

Lineage is derived from runtime parent references and delegation depth, not model
arguments. Missing, dead, inconsistent or cyclic ancestry has `lineage_valid=False`
and grants no root identity. Outside dispatch the snapshot is unbound and invalid.
A plugin supervisor-only tool can require a valid depth of zero before recording
a plan; worker terminal calls can consume that plan without supplying an engine
credential or choosing another root. Context is restored after nested dispatch.

This is a trusted extension API, not a sandbox against arbitrary Python or shell
code. No live agent object or credential is included, and the context must not be
copied into a model prompt. Consumers must retain ordinary approval enforcement
and their own profile-scoped configuration and secret handling.

### Private control credentials

An in-process plugin using a control credential can declare its environment key
with `ctx.register_private_env_keys(["EXAMPLE_CONTROL_TOKEN"])`. This registers
**names only**; the plugin continues to read values through normal secret sources.
The returned registration handle and the plugin unload ledger remove only that
owner's declaration. Multiple plugins can protect the same name independently.

Declarations are scoped to the active Hermes profile. The existing child-env
builders strip declared names from inherited values, extras, credential-inheriting
children, and skill passthrough additions. Name matching is case-insensitive,
including Windows environment blocks. `_HERMES_FORCE_...` extras cannot reintroduce
a private key; other existing FORCE behavior is unchanged. Snapshot the configured
key at registration and require plugin reload when it changes, so the credential
used by the control client always has a matching declaration.

This controls normal Hermes child-environment construction. It does not erase an
already-running shell's environment or prevent a trusted local tool from reading
the user's secret files. Disabling/unloading the plugin removes its declaration;
remove obsolete control credentials from the environment as part of disabling the
integration. Do not publish credential values or expose them as tool arguments.

## Execution Order

### LLM Calls

For each provider request, Hermes applies middleware in this order:

1. Build provider kwargs from the current conversation.
2. Apply `llm_request` middleware.
3. Emit `pre_api_request` observer hooks with the effective request.
4. Run provider execution through `llm_execution` middleware.
5. Emit `post_api_request` or `api_request_error` observer hooks.

Request middleware sees the full provider kwargs, including `messages` or
Responses API `input`, model settings, tool definitions, stream options, and
provider-specific options. Execution middleware receives the same effective
request plus `next_call`.

### Tool Calls

For each tool call, Hermes applies middleware in this order:

1. Parse and coerce model-provided tool arguments.
2. Apply `tool_request` middleware.
3. Run the normal Hermes pre-execution path against the effective arguments:
   tool availability checks, observer block directives, guardrails, and
   approval checks.
4. Run tool execution through `tool_execution` middleware.
5. Emit `post_tool_call` observer hooks.
6. Apply `transform_tool_result` hooks before the result is appended back into
   conversation context.

Tool request middleware runs before approval checks. Use it carefully: a
rewritten path, command, or URL is the value downstream policy will evaluate.

## Enablement

Middleware only runs for enabled plugins. For a bundled plugin:

```bash
hermes plugins enable <plugin-name>
```

For isolated local testing, use one `HERMES_HOME` for plugin enablement and the
agent run:

```bash
export HERMES_HOME=$HOME/.hermes/cache/scratch/hermes-middleware-test
mkdir -p "$HERMES_HOME"
hermes plugins enable <plugin-name>
hermes chat --query 'Reply exactly ok'
```

For source checkouts, use the [PM developer workflow](../reference/package-management.md#developer-workflow)
and a separate development home so the runtime sees plugins and middleware from
the working tree:

```bash
export HERMES_HOME="$HOME/hermes-middleware-test"
export HERMES_RUNTIME_DIR="$HERMES_HOME/tools"
source ./activate
python hermes plugins enable <plugin-name>
python hermes chat --query 'Reply exactly ok'
```

## Generic Plugin Examples

The examples below are intentionally small. They show the middleware contract
shape without depending on NeMo Relay.

### LLM Request Middleware

This plugin tags provider requests and records a middleware trace entry:

```python
def register(ctx):
    ctx.register_middleware("llm_request", tag_llm_request)


def tag_llm_request(**kwargs):
    request = dict(kwargs["request"])
    extra_body = dict(request.get("extra_body") or {})
    extra_body.setdefault("metadata", {})["hermes_middleware_demo"] = True
    request["extra_body"] = extra_body
    return {
        "request": request,
        "source": "middleware-demo",
        "reason": "tagged provider request",
    }
```

The effective request is passed to `pre_api_request`, provider execution, and
`post_api_request`.

### Tool Request Middleware

This plugin constrains `terminal` calls to a known working directory:

```python
from pathlib import Path


def register(ctx):
    ctx.register_middleware("tool_request", normalize_terminal_workdir)


def normalize_terminal_workdir(**kwargs):
    if kwargs.get("tool_name") != "terminal":
        return None
    args = dict(kwargs["args"])
    args.setdefault("workdir", str(Path.home() / ".hermes" / "cache" / "scratch" / "hermes-middleware-demo"))
    return {
        "args": args,
        "source": "middleware-demo",
        "reason": "defaulted terminal workdir",
    }
```

Because this runs before hooks and approvals, downstream telemetry and policy
observe the rewritten `workdir`.

### LLM Execution Middleware

This plugin wraps the provider call and preserves the raw provider response:

```python
import time


def register(ctx):
    ctx.register_middleware("llm_execution", time_llm_execution)


def time_llm_execution(**kwargs):
    started = time.monotonic()
    response = kwargs["next_call"](kwargs["request"])
    elapsed_ms = int((time.monotonic() - started) * 1000)
    print(f"llm_execution elapsed_ms={elapsed_ms}")
    return response
```

Return the same response shape Hermes expects from the provider adapter. Do not
wrap the response in a plugin-specific envelope unless the rest of the runtime
expects that envelope.

### Tool Execution Middleware

This plugin wraps tool execution while preserving the tool result:

```python
def register(ctx):
    ctx.register_middleware("tool_execution", annotate_tool_execution)


def annotate_tool_execution(**kwargs):
    result = kwargs["next_call"](kwargs["args"])
    # Metrics, logging, or external routing can happen here.
    return result
```

Execution middleware may call `next_call(modified_args)` to pass a changed
payload to later middleware and the base tool dispatcher.

Plugin-specific examples should live with the plugin that owns the behavior.
NeMo Relay execution middleware is installed through Relay's discovered user
and system configuration, or through an explicit `plugins.toml` selected with
`HERMES_NEMO_RELAY_PLUGINS_TOML`; see
[Relay shared metrics](relay-shared-metrics.md).

## Safety Notes

- Middleware should be deterministic for the same input unless it is explicitly
  routing to a dynamic external system.
- Request middleware should return complete replacement payloads, not partial
  patches.
- Execution middleware should call `next_call(...)` exactly once unless it is
  intentionally short-circuiting execution.
- If `llm_execution` or `tool_execution` raises before calling `next_call(...)`, Hermes treats
  that as middleware failure and continues with the remaining middleware chain
  and base execution.
- If `llm_execution` or `tool_execution` calls `next_call(...)` successfully and then raises
  during post-processing, Hermes preserves the downstream result and does not
  run the provider or tool a second time.
- If downstream provider or tool execution fails, middleware may let that error
  propagate or translate it deliberately. Hermes does not convert downstream
  failure into a successful `None` result.
- Tool request middleware runs before approvals. If it mutates file paths,
  commands, URLs, or arguments, the mutated values are what guardrails and
  approvals evaluate.
- Observer hooks remain the right place for read-only telemetry. Use middleware
  only when a plugin needs to alter or wrap behavior.
