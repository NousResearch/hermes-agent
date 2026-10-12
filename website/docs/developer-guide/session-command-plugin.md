---
title: Session command plugins
description: Host-authorized slash commands and durable boolean session policies
---

Use ordinary `ctx.register_command` for stateless actions. A command that reads
or changes the current session registers with `ctx.register_session_command`.
Hermes resolves the caller's canonical session and profile, keeps authorization
and control leases in the host, and gives the handler a short-lived capability.
Plugins never receive a database or live agent handle.

```python
from hermes_cli.session_commands import SessionCommandRequest

KEY = "rolling_override"

def prepare(raw):
    action = raw.strip().lower()
    if action not in {"on", "off", "inherit", "status"}:
        raise ValueError("Usage: /rolling on|off|inherit|status")
    return SessionCommandRequest(write=action != "status", data=action)

def apply(runtime, enabled):
    return runtime.set_context_option("micro_compact_enabled", enabled)

def handle(request, session):
    if request.data == "status":
        return str(session.get_override(KEY))
    supported = session.set_override(KEY, {"on": True, "off": False, "inherit": None}[request.data])
    return "Saved" if supported else "Saved; applies when a capable engine starts"

def register(ctx):
    ctx.register_session_policy(KEY, config_path="compression.micro_compact", apply=apply, default=False)
    ctx.register_session_command("rolling", handle, prepare=prepare,
                                 args_hint="on|off|inherit|status",
                                 subcommands=("on", "off", "inherit", "status"),
                                 advertise_in_gateway_menu=False)
```

`prepare` and `handle` are synchronous. Parsing runs before session routing,
lease acquisition, configuration reads or row creation. Invalid input raises a
`ValueError` with a user-facing usage message. A `SessionCommandRequest` is
immutable; `write=False` refuses mutation even if a handler calls a setter.

The handler's session exposes `session_id`, `profile_home`, and `live` (whether
an agent exists), plus `get_config()`, `get_override(key)` and
`set_override(key, bool_or_none)`. Config is a defensive copy of the active
profile's effective configuration. Methods are valid only while the handler is
running. Keys must be registered policies. Reads never create session rows.
Writes validate configuration, take the existing cross-process conversation
turn lease without waiting, ensure the exact canonical row, atomically update
one `model_config` key, and only then update the live engine. Other metadata
survives. A busy or foreign session fails before changing the policy.

Policy overrides accept literal booleans; `None` removes the key and inherits
the profile default. Legacy malformed override values inherit. The host hydrates
policies on agent creation and session resume, clears them at new-session
boundaries, and refreshes inherited defaults at turn completion. Explicit
choices survive resume and compression. Persistence-isolated forks never read
or apply canonical overrides. Engine options change no prompt, toolset or past
transcript, and the built-in rolling compactor preserves its cursor and summary.

The policy callback receives `SessionPolicyRuntime` with `checkpoint_required`
and `set_context_option(name, bool)`. Unknown engine capabilities return `False`.
The built-in engine supports `micro_compact_enabled`; the host always disables
that lossy option while a pre-compression checkpoint is required. An application
failure returns `False` after retaining the committed override; the command
should report that the saved choice could not be applied live.

Commands work through the CLI, messaging gateway, `slash.exec` and
`command.dispatch` (TUI/Desktop). TUI session commands bypass the slash worker
and run directly under the session transport's authority and the shared idle
control lease. Deferred compression re-anchor keeps that lease through I/O and
checks successful writes on the exact continuation; replacement sessions are
never mutated. Ordinary raw plugin handlers retain their original dispatch.
Session capability commands take precedence over quick-command fallthrough in
the TUI, matching `slash.exec`'s direct session-control path.

Both registration methods use the normal plugin registration ledger: failed
loads roll back and unload removes them. They currently require
`plugins.isolation: in_process`; host isolation refuses them because the live
capability and engine callbacks have no host-wire implementation. Do not switch
isolation modes implicitly or replace the API with direct SQLite access.
