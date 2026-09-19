"""MCP legs of ``manage_connections``: the backend installs, enables and authorizes; the card is
a projection of the operation and may only say approved, skipped or continue.

An MCP leg runs the same ``run.py`` lifecycle a managed connector runs. ``prepare`` starts an
OAuth flow, or records the credentials an install still needs; the card's approval starts the
install or the enable; ``observe`` reads the outcome on every tick. Off the desktop there is no
card, so every action runs at once and the result carries the authorization URL for the user.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from tools.connectors.contract import Actor, SettleReason, LegState
from tools.connectors.gateway.config import operation_session_key, session_platform
from tools.connectors.operation import ConnectionOperation, IllegalTransition, Leg
from tools.connectors.run import LegDriver, run_operation
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# One wait for every authorization URL of a call, not one per leg: the flows are started
# together, and a provider that is slow to publish its URL must not delay the others.
PREPARE_WAIT_SECONDS = 30.0

NOTE = (
    "Settled once; do not re-ask for any leg the user skipped or that timed out — continue "
    "without it or ask in chat. Tools of a newly installed or authorized server become available "
    "on your next turn."
)

OFF_DESKTOP_NOTE = (
    "There is no approval card in this session. Show any connect_url to the user so they open it "
    "in a browser. The authorization then finishes in the background, and the server's tools "
    "arrive on your next turn; ask the user to say when they are done. A failed leg's detail "
    "says what the user must do; do not retry it on your own."
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
    """The real work behind an MCP leg. One object so a caller can pass another one in."""

    def required_env(self, name: str) -> List[Dict[str, Any]]:
        """The credentials the catalog entry declares that have no value yet."""
        from hermes_cli.config import get_env_value

        return [{"name": spec.name, "prompt": spec.prompt, "required": spec.required,
                 "secret": spec.secret, "default": spec.default or None}
                for spec in (_catalog_entry(name).auth.env or []) if not get_env_value(spec.name)]

    def start_oauth(self, name: str) -> Any:
        from tools.connectors.legs import oauth

        return oauth.start(name)

    def install(self, name: str, env: Dict[str, str]) -> List[str]:
        """Write the declared credentials, install the entry, report the tools it offers."""
        from hermes_cli.config import save_env_value, validate_env_var_name_for_write
        from hermes_cli.mcp_catalog import install_entry

        entry = _catalog_entry(name)
        declared = {spec.name for spec in (entry.auth.env or [])}
        # Validate the whole map before the first write: configuring one MCP is not a general
        # env-writing primitive, and a mixed valid/invalid answer must persist nothing.
        for key in env:
            if key not in declared:
                raise ValueError(f"'{name}' does not declare the environment variable {key}")
            validate_env_var_name_for_write(key)
        from agent.secret_scope import current_secret_scope, reset_secret_scope, set_secret_scope

        # install_entry's prompt and the probe's ${VAR} interpolation both read get_secret, so the
        # values reach them through the scope and touch disk only after both succeed.
        token = set_secret_scope({**(current_secret_scope() or {}), **{k: v for k, v in env.items() if v}})
        try:
            install_entry(entry, enable=True)
            tools = _probe_tool_names(name)
        except Exception:
            from hermes_cli.mcp_config import _remove_mcp_server

            _remove_mcp_server(name)
            raise
        finally:
            reset_secret_scope(token)
        for key, value in env.items():
            if value:
                save_env_value(key, value)
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


def _probe_tool_names(name: str) -> List[str]:
    from hermes_cli.mcp_catalog import _probe_tools

    return [str(tool[0]) for tool in (_probe_tools(name) or [])]


def _default_backend() -> Any:
    return _CatalogBackend()


# ---------------------------------------------------------------------------
# the runner: per-operation work, reachable from the RPC thread by op_id
# ---------------------------------------------------------------------------


@dataclass
class _Work:
    """One leg's work in flight: an OAuth attempt the watcher polls, or a worker's outcome."""

    attempt: Any = None
    done: threading.Event = field(default_factory=threading.Event)
    tools: List[str] = field(default_factory=list)
    error: str = ""


class _Runner:
    """The backend plus the work for one operation's legs."""

    def __init__(self, action: str, backend: Any):
        self.action = action
        self.backend = backend
        self.op_id: Optional[str] = None
        self.work: Dict[str, _Work] = {}
        # Approved credentials per leg. Try again sends none, so a retried install reuses these.
        # Kept off the Leg: the values are secrets and the runner lives exactly as long as the operation.
        self.approved_env: Dict[str, Dict[str, str]] = {}

    def run(self, table: Dict[str, Callable], operation: ConnectionOperation, leg: Leg,
            env: Optional[Dict[str, str]] = None) -> None:
        table[self.action](self, operation, leg, env or {})

    def spawn(self, operation: ConnectionOperation, leg: Leg, call: Callable[[], Any]) -> None:
        """Run one blocking backend call on a worker thread; ``observe`` reports its outcome."""
        if operation.settled:  # Continue landed between the state read and here
            logger.debug("mcp %s %s: not started, the operation settled first", self.action, leg.name)
            return
        work = _Work()
        self.work[leg.name] = work

        def body() -> None:
            tools: List[str] = []
            error = ""
            try:
                tools = [str(name) for name in (call() or [])]
            except Exception as exc:
                error = _detail(exc)
            if operation.settled:
                # The result froze while the work ran; there is no row left to report into.
                logger.debug("mcp %s %s: outcome dropped, the operation settled first",
                             self.action, leg.name)
                return
            work.tools, work.error = tools, error
            work.done.set()
            operation.wake.set()

        # The worker runs in a copy of the calling thread's context: a named-profile turn binds its
        # home through a contextvar, and the install must write the credentials into that home.
        threading.Thread(target=contextvars.copy_context().run, args=(body,), daemon=True,
                         name=f"mcp-{self.action}-{leg.name}").start()

    def prepare(self, operation: ConnectionOperation) -> None:
        _RUNNERS[operation.op_id] = self
        self.op_id = operation.op_id
        if self.action == "authorize" and len(operation.legs) > 1:
            self._prepare_together(operation)
            return
        for leg in operation.legs:
            self.run(_PREPARE, operation, leg)

    def _prepare_together(self, operation: ConnectionOperation) -> None:
        """Start every OAuth flow at once and wait for the URLs once. Each flow blocks until its
        provider publishes an authorization URL, so a sequential prepare would keep the card empty
        for one wait per leg.

        The wait bounds how long prepare blocks, not how long a provider may take: a row still
        pending afterwards is left to its own thread, which is the only writer of that row and
        ends with the URL or the flow's own failure. Failing it here as well would make two
        writers of one row, and a URL that arrives a moment later would have no row to land on.

        Each thread runs in its own copy of the calling thread's context: a named-profile turn
        binds its home through a contextvar, and the flow resolves ``mcp_servers`` and stores the
        token by that home."""
        threads = [threading.Thread(target=contextvars.copy_context().run,
                                    args=(self.run, _PREPARE, operation, leg), daemon=True,
                                    name=f"mcp-prepare-{leg.name}") for leg in operation.legs]
        for thread in threads:
            thread.start()
        deadline = time.time() + PREPARE_WAIT_SECONDS
        for thread in threads:
            thread.join(max(0.0, deadline - time.time()))

    def observe(self, operation: ConnectionOperation) -> None:
        for leg in operation.legs:
            # Only a live leg can be advanced by a read; a failed one waits for Try again.
            if operation.settled or leg.state not in (LegState.pending, LegState.initiated):
                continue
            _OBSERVE[self.action](self, operation, leg)

    def close(self) -> None:
        if self.op_id is not None:
            _RUNNERS.pop(self.op_id, None)


# op_id -> the runner driving it, so the card's answer and Try again (RPC thread) find the work.
_RUNNERS: Dict[str, _Runner] = {}


def open_runner(action: str, backend: Any = None) -> _Runner:
    """The runner for one MCP operation. ``prepare`` binds it to the operation, so the card's
    answer and its Try again — both of which arrive on another thread — find the same work."""
    return _Runner(action, backend or _default_backend())


def _detail(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__


def _move(operation: ConnectionOperation, leg: Leg, to: LegState, actor: Actor, **fields: Any) -> bool:
    """Move one leg from the prepare, worker-outcome or observe path.

    Continue on the RPC thread can settle the operation between any read of the leg's state and
    this call. A settled operation has a frozen result, so the lost move is dropped rather than
    raised into the tool result; anything else is a real contract violation."""
    try:
        operation.transition(leg.name, to, actor, **fields)
        return True
    except IllegalTransition:
        if not operation.settled:
            raise
        logger.debug("mcp leg %s: %s dropped, the operation settled first", leg.name, to.value)
        return False


def _fail(operation: ConnectionOperation, leg: Leg, detail: str) -> None:
    """Report a failure, whatever the row was doing: a repeated failure has no state change to
    emit, only newer text."""
    if operation.settled:
        logger.debug("mcp leg %s: failure dropped, the operation settled first", leg.name)
        return
    if leg.state == LegState.failed:
        operation.refresh(leg.name, connect_url=None, detail=detail, actor=Actor.backend_watcher)
        return
    leg.connect_url = None  # whatever link the row was offering is dead
    _move(operation, leg, LegState.failed, Actor.backend_watcher, detail=detail)


def _connect(operation: ConnectionOperation, leg: Leg, tools: List[str]) -> None:
    extra = {"tools": tools} if tools else {}
    _move(operation, leg, LegState.connected, Actor.backend_watcher, **extra)


def _actor(leg: Leg) -> Actor:
    """Try again is the user's move; a first attempt is the backend's."""
    return Actor.user if leg.state == LegState.failed else Actor.backend_watcher


def _start_oauth(runner: _Runner, operation: ConnectionOperation, leg: Leg, env: Dict[str, str]) -> None:
    actor = _actor(leg)
    try:
        attempt = runner.backend.start_oauth(leg.name)
    except Exception as exc:
        _fail(operation, leg, _detail(exc))
        return
    runner.work[leg.name] = _Work(attempt=attempt)
    _move(operation, leg, LegState.initiated, actor, connect_url=attempt.auth_url, detail="")


def _declare_env(runner: _Runner, operation: ConnectionOperation, leg: Leg, env: Dict[str, str]) -> None:
    """The install row waits pending; the card draws a field per credential it still needs."""
    try:
        required = runner.backend.required_env(leg.name)
    except Exception as exc:
        _fail(operation, leg, _detail(exc))
        return
    leg.required_env = required


def _missing_required(runner: _Runner, leg: Leg, env: Dict[str, str]) -> List[Dict[str, Any]]:
    """The declared credentials that still have no value. The install runs on a worker thread,
    where ``install_entry``'s prompt for a missing credential would block on stdin forever."""
    declared = runner.backend.required_env(leg.name)
    return [spec for spec in declared
            if spec.get("required", True) and not env.get(str(spec.get("name") or ""))]


def _start_install(runner: _Runner, operation: ConnectionOperation, leg: Leg, env: Dict[str, str]) -> None:
    approved = {**runner.approved_env.get(leg.name, {}), **env}
    try:
        missing = _missing_required(runner, leg, approved)
    except Exception as exc:
        _fail(operation, leg, _detail(exc))
        return
    if missing:
        # The row stays pending and the card draws a field per credential it still needs; the
        # refresh is what tells the renderer to ask again.
        leg.required_env = missing
        operation.refresh(leg.name, connect_url=leg.connect_url, actor=Actor.backend_watcher,
                          detail=f"waiting for {', '.join(str(spec['name']) for spec in missing)}")
        return
    runner.approved_env[leg.name] = approved
    actor = _actor(leg)
    leg.required_env = []  # the credentials are written by the install; the row stops asking
    if not _move(operation, leg, LegState.initiated, actor, detail=""):
        return
    runner.spawn(operation, leg, lambda: runner.backend.install(leg.name, approved))


def _do_enable(runner: _Runner, operation: ConnectionOperation, leg: Leg, env: Dict[str, str]) -> None:
    actor = _actor(leg)
    if not _move(operation, leg, LegState.initiated, actor, detail=""):
        return
    try:
        runner.backend.enable(leg.name)
    except Exception as exc:
        _fail(operation, leg, _detail(exc))
        return
    _connect(operation, leg, [])


def _install_now(runner: _Runner, operation: ConnectionOperation, leg: Leg, env: Dict[str, str]) -> None:
    """Off the desktop nobody can fill a credential in, so a missing one is the answer."""
    try:
        missing = [spec["name"] for spec in runner.backend.required_env(leg.name) if spec.get("required", True)]
    except Exception as exc:
        _fail(operation, leg, _detail(exc))
        return
    if missing:
        from hermes_constants import display_hermes_home

        _fail(operation, leg, f"set {', '.join(missing)} in the environment or "
                                 f"{display_hermes_home()}/.env, then install again")
        return
    actor = _actor(leg)
    _move(operation, leg, LegState.initiated, actor)
    try:
        tools = [str(name) for name in (runner.backend.install(leg.name, {}) or [])]
    except Exception as exc:
        _fail(operation, leg, _detail(exc))
        return
    _connect(operation, leg, tools)


def _observe_oauth(runner: _Runner, operation: ConnectionOperation, leg: Leg) -> None:
    work = runner.work.get(leg.name)
    if work is None or work.attempt is None:
        return
    snapshot = work.attempt.poll()
    status = snapshot.get("status")
    if status not in ("approved", "error"):
        return
    runner.work.pop(leg.name, None)
    if status == "approved":
        _connect(operation, leg, list(snapshot.get("tools") or []))
        return
    _fail(operation, leg, snapshot.get("error") or "the authorization flow failed")


def _observe_worker(runner: _Runner, operation: ConnectionOperation, leg: Leg) -> None:
    work = runner.work.get(leg.name)
    if work is None or not work.done.is_set():
        return
    runner.work.pop(leg.name, None)
    if work.error:
        _fail(operation, leg, work.error)
        return
    _connect(operation, leg, work.tools)


def _nothing(runner: _Runner, operation: ConnectionOperation, leg: Leg, env: Dict[str, str]) -> None:
    """Authorize needs no approval: the row's verb opens the link the flow already minted."""


_PREPARE = {"authorize": _start_oauth, "install": _declare_env, "enable": _nothing}
_APPROVE = {"authorize": _nothing, "install": _start_install, "enable": _do_enable}
_RETRY = {"authorize": _start_oauth, "install": _start_install, "enable": _do_enable}
_OBSERVE = {"authorize": _observe_oauth, "install": _observe_worker, "enable": _observe_worker}
_OFF_DESKTOP = {"authorize": _start_oauth, "install": _install_now, "enable": _do_enable}


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
    for entry in answer.get("legs") or ():
        if not isinstance(entry, dict):
            continue
        leg = operation.leg(str(entry.get("name") or "").strip().lower())
        if leg is None:
            continue
        status = str(entry.get("status") or "").lower()
        if status == "skipped":
            # A row the backend resolved before this move landed has nothing to move; the rest of
            # the answer still applies. A settled operation is frozen. The check and the move are
            # not one step, so the refusal itself is the witness, not a read taken before it.
            try:
                operation.transition(leg.name, LegState.skipped, Actor.user)
            except IllegalTransition:
                if not leg.resolved and not operation.settled:
                    raise
        elif status == "approved" and runner is not None and leg.state == LegState.pending:
            runner.run(_APPROVE, operation, leg, _answer_env(entry))
    if answer.get("settled_by") == SettleReason.continue_.value and not operation.all_resolved:
        operation.settle(SettleReason.continue_)


def retry(operation: ConnectionOperation, names: List[str]) -> Optional[str]:
    """Re-run the named MCP legs on the open operation (the card's Try again): a fresh OAuth
    flow, a fresh install, a fresh enable. Returns an error message when the operation is not one
    this module is running."""
    runner = _RUNNERS.get(operation.op_id)
    if runner is None:
        return "this operation has no MCP work to re-run"
    if operation.settled:
        return "this operation has settled; its result is frozen"
    for name in names:
        leg = operation.leg(name)
        if leg is not None:
            runner.run(_RETRY, operation, leg)
    return None


# ---------------------------------------------------------------------------
# the tool entry point
# ---------------------------------------------------------------------------


class _DetachedOperation(ConnectionOperation):
    """The operation behind an off-desktop call. It is never registered in ``live`` and no card
    renders it, so it publishes no ``connection.update``: a frame would reach a session whose
    renderer knows nothing about the operation."""

    on_change = None


def _off_desktop_result(runner: _Runner, names: List[str], action: str, session_key: str) -> str:
    operation = _DetachedOperation([Leg(n, "mcp", action) for n in names], session_key=session_key)
    for leg in operation.legs:
        runner.run(_OFF_DESKTOP, operation, leg)
    payload = operation.result(with_urls=True)
    payload["status"] = "initiated" if any(t.state == LegState.initiated for t in operation.legs) else "settled"
    payload["note"] = OFF_DESKTOP_NOTE
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
    # The card exists only where the desktop renders it AND the callback can emit it (the rule
    # ``managed.run_managed_action`` uses). The Ink TUI has the callback but no card; a desktop call
    # that arrives without the callback (registry dispatch, e.g. from execute_code) would open an
    # operation nobody renders and block the tool for its whole deadline. Both get the link instead.
    if session_platform() != "desktop" or connection_callback is None:
        return _off_desktop_result(runner, names, action, session_key)
    try:
        return run_operation(
            [Leg(n, "mcp", action) for n in names],
            LegDriver(prepare=runner.prepare, observe=runner.observe, note=NOTE),
            session_key=session_key, tool_call_id=tool_call_id,
            connection_callback=connection_callback, with_urls_in_result=False,
        )
    finally:
        runner.close()
