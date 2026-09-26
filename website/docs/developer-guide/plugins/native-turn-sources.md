# Native turn source plugins

Native turn sources let an optional plugin offer a new user turn to an exact,
already-owned idle Hermes session. They are intended for durable external
systems that need normal host admission and lifecycle behavior. The interface
is generic: Hermes does not know the external system's records, verbs, paths,
or receipt schema.

## Register a source

Register during the plugin's `register(ctx)` callback:

```python
from hermes_cli.native_turn_sources import NativeTurnLease

class Source:
    name = "example-native-source"

    def poll(self, session):
        candidate = lookup_exact_candidate(session)
        if candidate is None:
            return None
        return NativeTurnLease(
            lease_id=candidate.id,
            prompt=candidate.prompt,
            commit=lambda exact_session: commit_candidate(candidate, exact_session),
            abort=lambda outcome, reason: abort_candidate(candidate, outcome, reason),
        )

    def on_user_boundary(self, session, reason):
        cancel_candidates_for_exact_session(session, reason)


def register(ctx):
    ctx.register_native_turn_source(Source(), surfaces=("cli", "tui", "gateway"))
```

The source object is structural; it must expose a stable non-empty `name`, a
synchronous `poll(session)` method, and a synchronous
`on_user_boundary(session, reason)` method. `poll` returns either `None` or a
`NativeTurnLease`.

`NativeSessionView` is frozen. It carries the canonical profile name and home,
current session id, surface, compression lineage, owner token, and immutable
host metadata. Sources must use the complete view for exact-target validation;
a matching session id in another profile or owner is not the same target.

## Admission contract

The host polls only a live idle session it owns and only when no human prompt
is pending. It reserves the normal host slot before invoking `commit`.

- `commit(session) -> True` admits the lease exactly once before model or
  transcript effects.
- `False` prevents the turn and causes a classified `not_sent` abort.
- A commit exception is an `unknown` outcome. The source must reconcile it
  durably; Hermes does not retry it blindly.
- A host cancellation before commit is classified as `canceled` or
  `not_sent`. Abort callbacks are at-most-once.

The admitted prompt enters the ordinary new-turn path. It is not injected into
a running tool/model loop. Existing cancellation, authorization, routing,
compression, prompt caching, author metadata, and role alternation remain in
force.

Before a stop, new session, session switch, close, or process exit releases or
rotates identity, Hermes calls `on_user_boundary` synchronously for the exact
old session. Callback failures are isolated and logged rather than changing the
host lifecycle operation.

## Discovery and lifecycle

Registrations belong to the `PluginManager` for one canonical profile home.
They are not visible to another profile, and disposing or unloading the plugin
removes them. With no registered source, polling is inert.

A plugin requiring this interface should call the registrar directly in
`register(ctx)`. `hermes -p PROFILE plugins doctor /path/to/plugin --ci` then
executes the same registration path and fails with the missing attribute on an
older host. Do not substitute version checks, configuration checks, or source
file checks for that registration test.

CLI commands that need built-in tools can use `ctx.dispatch_tool`. For
machine-wide read-only audits, `ctx.list_profile_homes()` returns canonical
profile homes and `ctx.dispatch_tool(..., profile_home=home)` runs the built-in
tool under that explicit profile scope.
