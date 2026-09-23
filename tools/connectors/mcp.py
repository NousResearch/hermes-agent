from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from tools.connectors.contract import Actor, SettleReason, TargetState
from tools.connectors.gateway.config import operation_session_key
from tools.connectors.operation import ConnectionOperation, DetachedOperation, IllegalTransition, Target
from tools.connectors.run import Kind, run_operation
from tools.connectors.targets import hosted_names, misrouted_to_mcp_error
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# One wait for every authorization URL of a call, not one per target: the flows are started
# together, and a provider that is slow to publish its URL must not delay the others.
PREPARE_WAIT_SECONDS = 30.0

NOTE = (
    "Settled once; do not re-ask on your own for any target the user skipped or that timed out, but "
    "a later request from the USER for that same app is not a re-ask — run it. Connected "
    "targets' tools are available now through tool_describe/tool_call and are named under "
    "tools_listing. A target with discovery_error is authorized but its tools are unavailable; "
    "retry discovery with manage_connections using that target's authorize or install action "
    "without asking for consent again."
)

NO_CARD_NOTE = (
    "No connection card is drawn for this turn. Show any connect_url to the user so they "
    "open it in a browser, then ask them to say when they are done. Connected targets' tools are "
    "available now through tool_describe/tool_call and are named under tools_listing. A target with "
    "discovery_error is authorized but its tools are unavailable; retry discovery with "
    "manage_connections using that target's authorize or install action without asking for consent "
    "again. Do not re-ask on your own for skipped or timed-out targets, but a later request from "
    "the USER for that same app is not a re-ask — run it."
)


def _catalog_names() -> List[str]:
    from hermes_cli.mcp_catalog import list_catalog

    return sorted(e.name for e in list_catalog())


def _configured_names() -> List[str]:
    from hermes_cli.mcp_catalog import installed_servers

    return sorted(installed_servers())


def validate_mcp_names(action: str, names: List[str]) -> Optional[str]:
    try:
        catalog = _catalog_names()
        configured = _configured_names()
    except Exception as exc:
        return f"could not read the MCP catalog: {exc}"
    allowed = set(catalog) if action == "install" else set(configured)
    unknown = [n for n in names if n not in allowed]
    if not unknown:
        return None
    foreign = [n for n in unknown if n not in catalog and n not in configured]
    if foreign:
        hosted = hosted_names() or set()
        misrouted = [n for n in foreign if n in hosted]
        if misrouted:
            return " ".join(misrouted_to_mcp_error(action, name) for name in misrouted)
    if action == "install":
        return (
            f"unknown MCP server(s) for install: {', '.join(unknown)}. Install works for "
            f"catalog entries only: {', '.join(catalog) or '(empty catalog)'}."
            + (f" Already configured (use enable/authorize): {', '.join(configured)}." if configured else "")
        )
    return (
        f"unknown MCP server(s) for {action}: {', '.join(unknown)}. {action} works for servers "
        f"already in mcp_servers: {', '.join(configured) or '(none configured)'}."
        + (f" Catalog entries you can install: {', '.join(catalog)}." if catalog else "")
    )


# ---------------------------------------------------------------------------
# the backend: the catalog, the installer, the OAuth flow
# ---------------------------------------------------------------------------


def _catalog_entry(name: str):
    from hermes_cli.mcp_catalog import get_entry

    entry = get_entry(name)
    if entry is None:
        raise ValueError(f"no catalog entry '{name}'")
    return entry


class _CatalogBackend:
    """The real work behind an MCP target. One object so a caller can pass another one in."""

    def required_env(self, name: str) -> List[Dict[str, Any]]:
        """The credentials the catalog entry declares that have no value yet."""
        from hermes_cli.config import get_env_value

        return [{"name": spec.name, "prompt": spec.prompt, "required": spec.required,
                 "secret": spec.secret, "default": "" if spec.secret else spec.default}
                for spec in (_catalog_entry(name).auth.env or []) if not get_env_value(spec.name)]

    def start_oauth(self, name: str) -> Any:
        from tools.connectors import mcp_oauth

        # The TUI/Desktop card path supplies its advertised client redirect in a later part.
        return mcp_oauth.start(name, client_redirect_uri=None)

    def installs_with_oauth(self, name: str) -> bool:
        """A catalog entry whose own OAuth the card must run. Provider-mediated OAuth is not one:
        its token comes from ``hermes auth <provider>``, so the plain probe covers it."""
        auth = _catalog_entry(name).auth
        return auth.type == "oauth" and not auth.provider

    def start_install_oauth(self, name: str, env: Dict[str, str]) -> Any:
        """Install an OAuth entry through the card's flow. The configuration is built in memory and
        lands, together with the setup values, only when ``initialize`` accepts the token."""
        from hermes_cli.mcp_catalog import card_install_config
        from tools.connectors import mcp_oauth

        entry = _catalog_entry(name)
        _check_declared(name, entry, env)
        return mcp_oauth.start(name, cfg=card_install_config(entry), env=env,
                               on_commit=lambda: _save_env(env))

    def install(self, name: str, env: Dict[str, str]) -> List[str]:
        """Probe the entry's in-memory configuration with ephemeral credentials; save both only
        after the server answered. A failure writes nothing, so a failed reinstall keeps the
        previous configuration."""
        from agent.secret_scope import current_secret_scope, reset_secret_scope, set_secret_scope
        from hermes_cli.mcp_catalog import _inline_non_secret_value, card_install_config
        from hermes_cli.mcp_config import _probe_single_server, _save_mcp_server

        entry = _catalog_entry(name)
        _check_declared(name, entry, env)
        cfg = card_install_config(entry)
        # `.env` is secrets-only: non-secret values (hostnames, client ids, workspace names) are
        # inlined into the server block, the same split `install_entry` makes for the terminal path.
        secret_names = {spec.name for spec in (entry.auth.env or []) if spec.secret}
        for key, value in env.items():
            if key not in secret_names and value:
                cfg = _inline_non_secret_value(cfg, key, value)
        token = set_secret_scope({**dict(current_secret_scope() or {}), **env})
        try:
            tools = [str(tool[0]) for tool in (_probe_single_server(name, cfg) or [])]
        finally:
            reset_secret_scope(token)
        from hermes_cli.mcp_config import _mcp_server_entry, _restore_mcp_server_entry

        previous_cfg = _mcp_server_entry(name)
        if not _save_mcp_server(name, cfg):
            raise RuntimeError(f"'{name}' was rejected: suspicious command/args configuration")
        try:
            _save_env({k: v for k, v in env.items() if k in secret_names})
        except Exception:
            _restore_mcp_server_entry(name, previous_cfg)
            raise
        return tools

    def enable(self, name: str) -> None:
        """Flip ``enabled`` under the scope and lock the dashboard's toggle route uses
        (``PUT /api/mcp/servers/{name}/enabled``): the two read-modify-write paths run in one
        process, so an unserialised write here drops whichever landed first."""
        from hermes_cli.config import load_config, save_config
        from hermes_cli.web_routers._common import config_write_scope

        with config_write_scope(None):
            config = load_config()
            servers = config.get("mcp_servers")
            if not isinstance(servers, dict) or not isinstance(servers.get(name), dict):
                raise ValueError(f"'{name}' is not a configured MCP server")
            servers[name]["enabled"] = True
            save_config(config)


def _check_declared(name: str, entry: Any, env: Dict[str, str]) -> None:
    """Configuring one MCP is not a general env-writing primitive: refuse the whole map before the
    first write if any key is undeclared or unwritable."""
    from hermes_cli.config import validate_env_var_name_for_write

    declared = {spec.name for spec in (entry.auth.env or [])}
    for key in env:
        if key not in declared:
            raise ValueError(f"'{name}' does not declare the environment variable {key}")
        validate_env_var_name_for_write(key)


def _save_env(env: Dict[str, str]) -> None:
    """Persist setup secrets atomically: a mid-write failure restores every value already
    written, so a failed install leaves the previous .env state rather than a half-written set."""
    from hermes_cli.config import get_env_value_prefer_dotenv, remove_env_value, save_env_value

    previous = {key: get_env_value_prefer_dotenv(key) for key, value in env.items() if value}
    written: List[str] = []
    try:
        for key, value in env.items():
            if value:
                save_env_value(key, value)
                written.append(key)
    except Exception:
        for key in written:
            try:
                if previous[key] is None:
                    remove_env_value(key)
                else:
                    save_env_value(key, previous[key])
            except Exception:
                logger.warning("MCP install rollback could not restore env %s", key, exc_info=True)
        raise


def _default_backend() -> Any:
    return _CatalogBackend()


# ---------------------------------------------------------------------------
# the runner: per-operation work, reachable from the RPC thread by op_id
# ---------------------------------------------------------------------------


@dataclass
class _Work:
    """One target's work in flight: an OAuth attempt the watcher polls, or a worker's outcome."""

    attempt: Any = None
    done: threading.Event = field(default_factory=threading.Event)
    tools: List[str] = field(default_factory=list)
    error: str = ""


class _Runner:
    """The backend plus the work for one operation's targets."""

    def __init__(self, action: str, backend: Any):
        self.action = action
        self.backend = backend
        self.op_id: Optional[str] = None
        self.work: Dict[str, _Work] = {}
        # The credentials the card approved, per target. Try again carries none (a failed row has
        # no fields), so the install that runs again is the one the user approved. Kept here and
        # not on the target: the values are secrets, and the runner is the one object whose life
        # is exactly the operation's.
        self.approved_env: Dict[str, Dict[str, str]] = {}

    def run(self, table: Dict[str, Callable], operation: ConnectionOperation, target: Target,
            env: Optional[Dict[str, str]] = None) -> None:
        table[self.action](self, operation, target, env or {})

    def spawn(self, operation: ConnectionOperation, target: Target, call: Callable[[], Any]) -> None:
        """Run one blocking backend call on a worker thread; ``observe`` reports its outcome."""
        if operation.settled:  # Continue landed between the state read and here
            logger.debug("mcp %s %s: not started, the operation settled first", self.action, target.name)
            return
        work = _Work()
        self.work[target.name] = work

        def body() -> None:
            tools: List[str] = []
            error = ""
            try:
                tools = [str(name) for name in (call() or [])]
            except Exception as exc:
                error = _detail(exc, self, target)
            if operation.settled:
                # The result froze while the work ran; there is no row left to report into.
                logger.debug("mcp %s %s: outcome dropped, the operation settled first",
                             self.action, target.name)
                return
            work.tools, work.error = tools, error
            work.done.set()
            operation.wake.set()

        # The worker runs in a copy of the calling thread's context: a named-profile turn binds its
        # home through a contextvar, and the install must write the credentials into that home.
        threading.Thread(target=contextvars.copy_context().run, args=(body,), daemon=True,
                         name=f"mcp-{self.action}-{target.name}").start()

    def prepare(self, operation: ConnectionOperation) -> None:
        _RUNNERS[operation.op_id] = self
        self.op_id = operation.op_id
        self.operation = operation
        if self.action == "authorize" and len(operation.targets) > 1:
            self._prepare_together(operation)
            return
        for target in operation.targets:
            self.run(_PREPARE, operation, target)

    def _prepare_together(self, operation: ConnectionOperation) -> None:
        """Start every OAuth flow at once and wait for the URLs once. Each flow blocks until its
        provider publishes an authorization URL, so a sequential prepare would keep the card empty
        for one wait per target.

        The wait bounds how long prepare blocks, not how long a provider may take: a row still
        pending afterwards is left to its own thread, which is the only writer of that row and
        ends with the URL or the flow's own failure. Failing it here as well would make two
        writers of one row, and a URL that arrives a moment later would have no row to land on.

        Each thread runs in its own copy of the calling thread's context: a named-profile turn
        binds its home through a contextvar, and the flow resolves ``mcp_servers`` and stores the
        token by that home."""
        threads = [threading.Thread(target=contextvars.copy_context().run,
                                    args=(self.run, _PREPARE, operation, target), daemon=True,
                                    name=f"mcp-prepare-{target.name}") for target in operation.targets]
        for thread in threads:
            thread.start()
        deadline = time.time() + PREPARE_WAIT_SECONDS
        for thread in threads:
            thread.join(max(0.0, deadline - time.time()))

    def observe(self, operation: ConnectionOperation) -> None:
        for target in operation.targets:
            # Only a live target can be advanced by a read; a failed one waits for Try again.
            if operation.settled or target.state not in (TargetState.pending, TargetState.initiated):
                continue
            _OBSERVE[self.action](self, operation, target)

    def close(self) -> None:
        """The operation is over. An attempt still waiting on the browser is stopped when the user
        ended the turn, and otherwise left to finish: the card closed, not the authorization. What
        it commits is picked up before the session's next turn (``adopt_late_connections``)."""
        if self.op_id is not None:
            _RUNNERS.pop(self.op_id, None)
        operation = getattr(self, "operation", None)
        for name, work in list(self.work.items()):
            if work.attempt is None or operation is None:
                continue
            if operation.settled_by == SettleReason.interrupt:
                from tools.connectors.mcp_oauth import cancel_attempt

                cancel_attempt(work.attempt.flow)
            else:
                _LATE_ATTEMPTS.setdefault(operation.session_key, {})[name] = work.attempt
        self.work.clear()


# session key -> {server: attempt} for OAuth attempts that outlived their card.
_LATE_ATTEMPTS: Dict[str, Dict[str, Any]] = {}


def adopt_late_connections(agent: Any) -> List[str]:
    """Register the servers whose authorization committed after their card had closed, and add
    them to the agent's toolset selection. Runs between turns, so the result that said "not
    connected" is followed by a turn in which the tools are there."""
    session_key = operation_session_key(getattr(agent, "session_id", None))
    attempts = _LATE_ATTEMPTS.get(session_key)
    if not attempts:
        return []
    adopted: List[str] = []
    for name, attempt in list(attempts.items()):
        snapshot = attempt.poll()
        if snapshot["status"] == "pending":
            continue
        attempts.pop(name, None)
        if snapshot["status"] != "approved" or snapshot.get("discovery_error"):
            continue
        try:
            from tools.mcp_tool_config import _load_mcp_config
            from tools.mcp_tool_discovery import register_mcp_servers

            config = _load_mcp_config().get(name)
            if isinstance(config, dict):
                register_mcp_servers({name: config})
                adopted.append(name)
        except Exception:
            logger.debug("late MCP connection %s was not adopted", name, exc_info=True)
    if not attempts:
        _LATE_ATTEMPTS.pop(session_key, None)
    enabled = getattr(agent, "enabled_toolsets", None)
    if adopted and enabled is not None and "no_mcp" not in enabled:
        agent.enabled_toolsets = [*enabled, *(n for n in adopted if n not in enabled)]
    return adopted


# op_id -> the runner driving it, so the card's answer and Try again (RPC thread) find the work.
_RUNNERS: Dict[str, _Runner] = {}


def open_runner(action: str, backend: Any = None) -> _Runner:
    """The runner for one MCP operation. ``prepare`` binds it to the operation, so the card's
    answer and its Try again — both of which arrive on another thread — find the same work."""
    return _Runner(action, backend or _default_backend())


# Errors that carry no message reach the user as their class name; say what happened instead.
_BARE_ERRORS = {
    "CancelledError": "tool discovery was interrupted; run the same action again to list the tools",
    "TimeoutError": "the server did not answer in time",
}


def _detail(exc: Any, runner: _Runner, target: Target) -> str:
    """The user-facing text of a failure. Every value the card submitted for this target is
    replaced by exact match before the pattern redactor runs: an opaque credential has no
    recognizable shape, so only the runner knows what to remove."""
    from agent.redact import redact_sensitive_text

    text = str(exc) or (_BARE_ERRORS.get(exc.__class__.__name__, exc.__class__.__name__)
                        if isinstance(exc, BaseException) else "error")
    text = _BARE_ERRORS.get(text, text)  # a worker hands over the class name of a message-less error
    for value in runner.approved_env.get(target.name, {}).values():
        if value:
            text = text.replace(value, "[REDACTED]")
    return redact_sensitive_text(text, force=True) or "error"


def _catalog_instructions(name: str) -> str:
    """The manifest's ``post_install`` text for a catalog name; a custom configured server has none."""
    from hermes_cli.mcp_catalog import get_entry

    entry = get_entry(name)
    return str(entry.post_install or "") if entry is not None else ""


def _move(operation: ConnectionOperation, target: Target, to: TargetState, actor: Actor, **fields: Any) -> bool:
    """Move one target from the prepare, worker-outcome or observe path.

    Continue on the RPC thread can settle the operation between any read of the target's state and
    this call. A settled operation has a frozen result, so the lost move is dropped rather than
    raised into the tool result; anything else is a real contract violation."""
    try:
        operation.transition(target.name, to, actor, **fields)
        return True
    except IllegalTransition:
        if not operation.settled:
            raise
        logger.debug("mcp target %s: %s dropped, the operation settled first", target.name, to.value)
        return False


def _fail(operation: ConnectionOperation, target: Target, detail: str) -> None:
    """Report a failure, whatever the row was doing: a repeated failure has no state change to
    emit, only newer text."""
    if operation.settled:
        logger.debug("mcp target %s: failure dropped, the operation settled first", target.name)
        return
    if target.state == TargetState.failed:
        operation.refresh(target.name, connect_url=None, detail=detail, actor=Actor.backend_watcher)
        return
    target.connect_url = None  # whatever link the row was offering is dead
    _move(operation, target, TargetState.failed, Actor.backend_watcher, detail=detail)


def _register_connected(runner: _Runner, target: Target, name: str) -> tuple[List[str], str]:
    """Register one committed server in the current profile scope and report its callable names."""
    try:
        from tools.mcp_tool_config import _load_mcp_config
        from tools.mcp_tool_discovery import register_mcp_servers

        config = _load_mcp_config().get(name)
        if not isinstance(config, dict):
            raise RuntimeError(f"no committed MCP configuration for '{name}'")
        register_mcp_servers({name: config})
        return _registered_tool_names(name), ""
    except Exception as exc:
        return [], _detail(exc, runner, target)


def _registered_tool_names(name: str, wait_seconds: float = 30.0) -> List[str]:
    """The server's callable names, read from the registry once its registration has finished.

    Registration is a no-op for a server the process already holds, and that includes one another
    task is still connecting: saving the configuration wakes the config watcher, which starts its
    own connect, so a large server (hundreds of tools) was reported with no tools three seconds
    before they were registered. A server that failed discovery earlier is parked with no tools
    and is woken once. A server that finished registering with no tools is a valid empty list."""
    from tools import mcp_tool as _core
    from tools.mcp_tool_loop import reconnect_mcp_server
    from tools.mcp_tool_scope import _resolve_server_key
    from tools.registry import registry

    key = _resolve_server_key(name)
    deadline = time.time() + wait_seconds
    woken = False
    while True:
        names = registry.get_tool_names_for_toolset(f"mcp-{name}")
        if names or time.time() >= deadline:
            return names
        if key not in _core._server_connecting:
            server = _core._servers.get(key)
            finished = server is not None and getattr(server, "session", None) is not None \
                and hasattr(server, "_registered_tool_names")
            if finished:
                return names
            if woken or server is None or not reconnect_mcp_server(name):
                return names
            woken = True
        time.sleep(0.25)


def _connect(operation: ConnectionOperation, target: Target, tools: List[str], discovery_error: str = "") -> None:
    extra: Dict[str, Any] = {"tools": tools}
    if discovery_error:
        extra["discovery_error"] = discovery_error
    _move(operation, target, TargetState.connected, Actor.backend_watcher, **extra)


def _actor(target: Target) -> Actor:
    """Try again is the user's move; a first attempt is the backend's."""
    return Actor.user if target.state == TargetState.failed else Actor.backend_watcher


def _start_oauth(runner: _Runner, operation: ConnectionOperation, target: Target, env: Dict[str, str]) -> None:
    actor = _actor(target)
    target.instructions = _catalog_instructions(target.name)
    try:
        attempt = runner.backend.start_oauth(target.name)
    except Exception as exc:
        _fail(operation, target, _detail(exc, runner, target))
        return
    runner.work[target.name] = _Work(attempt=attempt)
    _move(operation, target, TargetState.initiated, actor, connect_url=attempt.auth_url,
          detail=getattr(attempt, "detail", ""))


def _declare_env(runner: _Runner, operation: ConnectionOperation, target: Target, env: Dict[str, str]) -> None:
    """The install row waits pending; the card draws a field per credential it still needs."""
    target.instructions = _catalog_instructions(target.name)
    try:
        required = runner.backend.required_env(target.name)
    except Exception as exc:
        _fail(operation, target, _detail(exc, runner, target))
        return
    target.required_env = required


def _missing_required(runner: _Runner, target: Target, env: Dict[str, str]) -> List[Dict[str, Any]]:
    """The declared credentials that still have no value. The install runs on a worker thread,
    where ``install_entry``'s prompt for a missing credential would block on stdin forever."""
    declared = runner.backend.required_env(target.name)
    return [spec for spec in declared
            if spec.get("required", True) and not env.get(str(spec.get("name") or ""))]


def _start_install(runner: _Runner, operation: ConnectionOperation, target: Target, env: Dict[str, str]) -> None:
    approved = {**runner.approved_env.get(target.name, {}), **env}
    try:
        missing = _missing_required(runner, target, approved)
    except Exception as exc:
        _fail(operation, target, _detail(exc, runner, target))
        return
    if missing:
        # The row stays pending and the card draws a field per credential it still needs; the
        # refresh is what tells the renderer to ask again.
        target.required_env = missing
        operation.refresh(target.name, connect_url=target.connect_url, actor=Actor.backend_watcher,
                          detail=f"waiting for {', '.join(str(spec['name']) for spec in missing)}")
        return
    runner.approved_env[target.name] = approved
    actor = _actor(target)
    target.required_env = []  # the credentials are written by the install; the row stops asking
    if not _move(operation, target, TargetState.initiated, actor, detail=""):
        return
    if _installs_with_oauth(runner, target):
        _start_install_oauth(runner, operation, target, approved)
        return
    runner.spawn(operation, target, lambda: runner.backend.install(target.name, approved))


def _installs_with_oauth(runner: _Runner, target: Target) -> bool:
    try:
        return bool(runner.backend.installs_with_oauth(target.name))
    except Exception:
        return False  # the install itself reports a bad entry


def _start_install_oauth(runner: _Runner, operation: ConnectionOperation, target: Target,
                         env: Dict[str, str]) -> None:
    """The row is already ``initiated``; publish the authorization link onto it. The same
    ``initiated`` + ``connect_url`` pair is what every card reads as its URL step."""
    try:
        attempt = runner.backend.start_install_oauth(target.name, env)
    except Exception as exc:
        _fail_install(runner, operation, target, exc)
        return
    runner.work[target.name] = _Work(attempt=attempt)
    if operation.settled:
        return
    operation.refresh(target.name, connect_url=attempt.auth_url, actor=Actor.backend_watcher,
                      detail=getattr(attempt, "detail", ""))


def _fail_install(runner: _Runner, operation: ConnectionOperation, target: Target, error: Any) -> None:
    """A failed install asks for its fields again, so the card can reopen the form over the draft
    it kept. Nothing was saved, so every declared field is still missing."""
    with contextlib.suppress(Exception):
        target.required_env = runner.backend.required_env(target.name)
    _fail(operation, target, _detail(error, runner, target))


def _do_enable(runner: _Runner, operation: ConnectionOperation, target: Target, env: Dict[str, str]) -> None:
    actor = _actor(target)
    if not _move(operation, target, TargetState.initiated, actor, detail=""):
        return
    try:
        runner.backend.enable(target.name)
    except Exception as exc:
        _fail(operation, target, _detail(exc, runner, target))
        return
    tools, discovery_error = _register_connected(runner, target, target.name)
    _connect(operation, target, tools, discovery_error)


def _install_now(runner: _Runner, operation: ConnectionOperation, target: Target, env: Dict[str, str]) -> None:
    """Off the desktop nobody can fill a credential in, so a missing one is the answer."""
    try:
        missing = [spec["name"] for spec in runner.backend.required_env(target.name) if spec.get("required", True)]
    except Exception as exc:
        _fail(operation, target, _detail(exc, runner, target))
        return
    if missing:
        from hermes_constants import display_hermes_home

        _fail(operation, target, f"set {', '.join(missing)} in the environment or "
                                 f"{display_hermes_home()}/.env, then install again")
        return
    actor = _actor(target)
    if _installs_with_oauth(runner, target):
        # No card here, so the result carries the link; the flow's own worker commits the install.
        try:
            attempt = runner.backend.start_install_oauth(target.name, {})
        except Exception as exc:
            _fail(operation, target, _detail(exc, runner, target))
            return
        runner.work[target.name] = _Work(attempt=attempt)
        _move(operation, target, TargetState.initiated, actor, connect_url=attempt.auth_url,
              detail=getattr(attempt, "detail", ""))
        return
    _move(operation, target, TargetState.initiated, actor)
    try:
        runner.backend.install(target.name, {})
    except Exception as exc:
        _fail(operation, target, _detail(exc, runner, target))
        return
    tools, discovery_error = _register_connected(runner, target, target.name)
    _connect(operation, target, tools, discovery_error)


def _observe_oauth(runner: _Runner, operation: ConnectionOperation, target: Target) -> None:
    work = runner.work.get(target.name)
    if work is None or work.attempt is None:
        return
    snapshot = work.attempt.poll()
    status = snapshot.get("status")
    if status not in ("approved", "error"):
        return
    runner.work.pop(target.name, None)
    if status == "approved":
        discovery_error = snapshot.get("discovery_error") or ""
        if discovery_error:
            _connect(operation, target, [], _detail(discovery_error, runner, target))
        else:
            tools, registration_error = _register_connected(runner, target, target.name)
            _connect(operation, target, tools, registration_error)
        return
    error = snapshot.get("error") or "the authorization flow failed"
    if runner.action == "install":
        _fail_install(runner, operation, target, error)
        return
    _fail(operation, target, _detail(error, runner, target))


def _observe_install(runner: _Runner, operation: ConnectionOperation, target: Target) -> None:
    """An install is an OAuth attempt for an OAuth entry and a worker for every other one."""
    work = runner.work.get(target.name)
    observe = _observe_oauth if work is not None and work.attempt is not None else _observe_worker
    observe(runner, operation, target)


def _observe_worker(runner: _Runner, operation: ConnectionOperation, target: Target) -> None:
    work = runner.work.get(target.name)
    if work is None or not work.done.is_set():
        return
    runner.work.pop(target.name, None)
    if work.error:
        if runner.action == "install":
            _fail_install(runner, operation, target, work.error)
            return
        _fail(operation, target, _detail(work.error, runner, target))
        return
    tools, discovery_error = _register_connected(runner, target, target.name)
    _connect(operation, target, tools, discovery_error)


def _nothing(runner: _Runner, operation: ConnectionOperation, target: Target, env: Dict[str, str]) -> None:
    """Authorize needs no approval: the row's verb opens the link the flow already minted."""


_PREPARE = {"authorize": _start_oauth, "install": _declare_env, "enable": _nothing}
_APPROVE = {"authorize": _nothing, "install": _start_install, "enable": _do_enable}
_RETRY = {"authorize": _start_oauth, "install": _start_install, "enable": _do_enable}
_OBSERVE = {"authorize": _observe_oauth, "install": _observe_install, "enable": _observe_worker}
_NO_CARD = {"authorize": _start_oauth, "install": _install_now, "enable": _do_enable}


# ---------------------------------------------------------------------------
# the card's answer and its Try again
# ---------------------------------------------------------------------------


def _answer_env(entry: Dict[str, Any]) -> Dict[str, str]:
    raw = entry.get("env")
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def apply_answer(operation: ConnectionOperation, raw: str) -> None:
    """Fold the card's ``connection.respond`` payload into the operation: a skip, an approval that
    starts the backend's work, and Continue. The card never reports an outcome, so any other claim
    moves nothing."""
    try:
        answer = json.loads(raw)
    except (TypeError, ValueError):
        answer = {}
    if not isinstance(answer, dict):
        answer = {}
    runner = _RUNNERS.get(operation.op_id)
    for entry in answer.get("targets") or ():
        if not isinstance(entry, dict):
            continue
        target = operation.target(str(entry.get("name") or "").strip().lower())
        if target is None:
            continue
        status = str(entry.get("status") or "").lower()
        if status == "skipped":
            # A row the backend resolved before this move landed has nothing to move; the rest of
            # the answer still applies. A settled operation is frozen. The check and the move are
            # not one step, so the refusal itself is the witness, not a read taken before it.
            kept = _cancel_attempt(runner, target)
            try:
                operation.transition(target.name, TargetState.skipped, Actor.user,
                                     **({"detail": kept} if kept else {}))
            except IllegalTransition:
                if not target.resolved and not operation.settled:
                    raise
        elif status == "approved" and runner is not None:
            if target.state == TargetState.pending:
                runner.run(_APPROVE, operation, target, _answer_env(entry))
            elif target.state in (TargetState.failed, TargetState.expired):
                # Connect on the form a failed row reopened: the same attempt, with the new values.
                runner.run(_RETRY, operation, target, _answer_env(entry))
    if answer.get("settled_by") == SettleReason.continue_.value and not operation.all_resolved:
        operation.settle(SettleReason.continue_)


AUTHORIZATION_KEPT = ("the authorization had already completed when this was canceled, so it was "
                      "kept; run the same action again to list the tools")


def _cancel_attempt(runner: Optional[_Runner], target: Target) -> str:
    """Stop the target's OAuth attempt so a late reply cannot be adopted. Returns the note for a
    cancel that lost the race: the attempt had committed, and a completed authorization stays."""
    work = runner.work.pop(target.name, None) if runner is not None else None
    flow = getattr(getattr(work, "attempt", None), "flow", None)
    if flow is None:
        return ""
    from tools.connectors.mcp_oauth import cancel_attempt

    return AUTHORIZATION_KEPT if cancel_attempt(flow) else ""


def retry(operation: ConnectionOperation, names: List[str]) -> Optional[str]:
    """Re-run the named MCP targets on the open operation (the card's Try again): a fresh OAuth
    flow, a fresh install, a fresh enable. Returns an error message when the operation is not one
    this module is running."""
    runner = _RUNNERS.get(operation.op_id)
    if runner is None:
        return "this operation has no MCP work to re-run"
    if operation.settled:
        return "this operation has settled; its result is frozen"
    for name in names:
        target = operation.target(name)
        if target is not None:
            runner.run(_RETRY, operation, target)
    return None


# ---------------------------------------------------------------------------
# the tool entry point
# ---------------------------------------------------------------------------


def _no_card_result(runner: _Runner, names: List[str], action: str, session_key: str) -> str:
    operation = DetachedOperation([Target(n, "mcp", action) for n in names], session_key=session_key)
    for target in operation.targets:
        runner.run(_NO_CARD, operation, target)
    payload = operation.result(with_urls=True)
    payload["status"] = "initiated" if any(t.state == TargetState.initiated for t in operation.targets) else "settled"
    payload["note"] = NO_CARD_NOTE
    return json.dumps(payload, ensure_ascii=False)


def run_mcp_operation(
    names: List[str],
    action: str,
    *,
    connection_callback: Optional[Callable[[Dict[str, Any]], Optional[str]]],
    session_id: Optional[str],
    tool_call_id: Optional[str] = None,
    backend: Any = None,
) -> str:
    error = validate_mcp_names(action, names)
    if error:
        return tool_error(error)
    runner = open_runner(action, backend)
    session_key = operation_session_key(session_id)
    # Every interactive surface that renders the card attaches this callback. Registry dispatch and
    # messaging sessions attach none, so they receive the link instead of opening an unanswerable op.
    if connection_callback is None:
        return _no_card_result(runner, names, action, session_key)
    try:
        return run_operation(
            [Target(n, "mcp", action) for n in names],
            Kind(prepare=runner.prepare, observe=runner.observe, note=NOTE),
            session_key=session_key, tool_call_id=tool_call_id,
            connection_callback=connection_callback, with_urls_in_result=False,
        )
    finally:
        runner.close()
