"""Managed connectors (Nous tool gateway) on the connection operation.

``connect`` mints a link for every target up front and stores it on the target; ``reconnect``
reads status first and reinitiates only what is not connected (``force`` always reinitiates).
On a desktop session the call blocks until the operation settles and the result carries no URL;
the card owns the links. Off the desktop the result carries the URLs and returns at once, until
PR3 delivers them as their own message. The watcher hook reads one route per pending target:
that target's own account row, at 1 Hz."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.connectors.contract import Actor, TargetState, allowed
from tools.connectors.gateway.config import operation_session_key, session_platform
from tools.connectors.gateway.errors import RateLimited
from tools.connectors.operation import ConnectionOperation, IllegalTransition, Target
from tools.connectors.run import Kind, run_operation
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# The account route carries its own 180/min budget, so one read per pending target per second stays
# inside it and still flips the card within a second of the user finishing at the vendor.
WATCH_TICK_SECONDS = 1.0

# A read never outlives the operation, never asks for less than one second, and never holds the
# loop for more than ten: Continue must be able to return the tool while a gateway hangs.
_MIN_READ_SECONDS = 1.0
_MAX_READ_SECONDS = 10.0

# The six-state account vocabulary -> the state that read ends the attempt in, and who caused it.
# `pending` is not here: it is the attempt still running, and moves nothing.
_ACCOUNT_OUTCOME: Dict[str, Tuple[TargetState, Actor]] = {
    "active": (TargetState.connected, Actor.backend_watcher),
    "failed": (TargetState.failed, Actor.backend_watcher),
    "revoked": (TargetState.failed, Actor.backend_watcher),
    "inactive": (TargetState.failed, Actor.backend_watcher),
    # The link's TTL ran out; the gateway reports it, the clock caused it.
    "expired": (TargetState.expired, Actor.clock),
}

NOTE = (
    "Settled once. connected → use the app now; skipped → the user chose Not now, do not connect it "
    "or route around it; not_connected → ask the user what to do, never re-mint on your own."
)


def managed_client():
    """The managed connector client created inside the caller's active profile scope."""
    from tools.connectors.gateway.client import ConnectorClient

    return ConnectorClient()


def managed_kind(client: Any, action: str, force: bool) -> Kind:
    """The managed operation hooks shared by model- and account-originated connections."""
    return Kind(prepare=_prepare(client, action, force), observe=lambda operation: _observe(client, operation), note=NOTE)


def _status_by_slug(client: Any) -> Dict[str, Dict[str, Any]]:
    """The toolkit list, by slug. Only the reconnect repair check reads it: it answers "is this app
    already connected" before any account exists for the watcher to read."""
    return {str(i.get("connector", "")).lower(): i for i in client.list_connectors() if isinstance(i, dict)}


def mint(client: Any, operation: ConnectionOperation, names: List[str], *, reinitiate: bool, actor: Actor) -> None:
    """Mint links for ``names`` and apply the gateway's per-app answer to the operation. ``actor`` is
    the watcher on the first mint and the user on Try again. The operation id rides along so the
    vendor's done page can name it on the way back to the desktop."""
    from tools.connectors.gateway.client import return_to_args

    if not names:
        return
    response = client.connections(names, reinitiate=reinitiate, **return_to_args(op=operation.op_id))
    for entry in response.get("results", []):
        name = str(entry.get("connector") or "").lower()
        target = operation.target(name)
        if target is None:
            continue
        status = str(entry.get("status") or "")
        detail = str(entry.get("status_reason") or entry.get("statusReason") or "")
        connection_id = entry.get("connection_id") or entry.get("connectionId")
        if status == "active":
            operation.transition(name, TargetState.initiated, actor)
            operation.transition(name, TargetState.connected, Actor.backend_watcher, connection_id=connection_id)
        elif status == "initiated":
            if not connection_id:
                logger.warning("connector %s: the mint named no account, so the watcher cannot read it; "
                               "only the card or the deadline can end the row", name)
            operation.transition(
                name, TargetState.initiated, actor,
                connect_url=entry.get("connect_url"), connection_id=connection_id, attempt=entry.get("attempt"),
                detail=detail,
            )
        elif target.state == TargetState.failed:
            # Failed again: no state change to emit, but the old link is dead and the vendor's text is new.
            operation.refresh(name, connect_url=None, detail=detail)
        elif target.state == TargetState.expired:
            # The table has no expired → failed; the re-mint attempt is the user's, so step through initiated.
            operation.transition(name, TargetState.initiated, actor)
            operation.transition(name, TargetState.failed, Actor.backend_watcher, detail=detail)
            operation.refresh(name, connect_url=None, detail=detail)
        else:
            # `detail` is the vendor's text or empty; the state itself is never written into it (the card prints it).
            operation.transition(name, TargetState.failed, Actor.backend_watcher, detail=detail)


def _status_for(client: Any, target: Target, *, timeout: float) -> Optional[Dict[str, Any]]:
    """The one route the watcher reads: that target's own account row. ``None`` means "nothing to
    apply this tick" — no account to read, a rate-limit still in force, an account the gateway does
    not know yet (404 until the deadline), or a read that failed. A 429 is raised to the tick: its
    budget is the principal's, so it is not this one target's to wait out."""
    if not target.connection_id or time.time() < target.next_read_at:
        return None
    try:
        return client.account_status(target.connection_id, timeout=timeout)
    except RateLimited:
        raise
    except Exception as exc:
        logger.debug("connector account read failed for %s: %s", target.name, exc)
        return None


def _apply_read(operation: ConnectionOperation, target: Target, status: str, reason: str) -> None:
    """Apply one account read to one target. The RPC thread can move the row while the read is in
    flight — a Skip resolves it, a Continue freezes the whole result — and the read then has no
    live row to move: it is dropped, not raised into the tool result (that would end the watch
    with the operation still open and no card to answer it). Any other refusal is a real
    contract violation."""
    outcome = _ACCOUNT_OUTCOME.get(status)
    if outcome is None:
        return
    to, actor = outcome
    try:
        if allowed(target.kind, target.state, to) is None:
            # No edge from pending: the read is itself the witness that the attempt started.
            operation.transition(target.name, TargetState.initiated, Actor.backend_watcher)
        operation.transition(target.name, to, actor, detail=reason or target.detail)
    except IllegalTransition:
        if not operation.settled and _live(target):
            raise
        logger.debug("connector %s: %s read dropped, the row is %s", target.name, status, target.state.value)


def _live(target: Target) -> bool:
    """Only a live attempt (pending, initiated) can be advanced by a gateway read; a failed or
    expired link waits for the user, and a resolved row is done."""
    return target.state in (TargetState.pending, TargetState.initiated)


def _park(operation: ConnectionOperation, until: float) -> None:
    """A 429 is per principal, not per account: every live target waits out the same Retry-After."""
    for target in operation.targets:
        if _live(target):
            target.next_read_at = until


def _observe(client: Any, operation: ConnectionOperation) -> None:
    """One account read per live target per tick, sequential: this is the only thread reading them.
    A 429 ends the tick: the next read would spend the same refused budget."""
    for target in operation.targets:
        # A settled op is frozen; a row that is not live waits for the user or is done.
        if operation.settled or not _live(target):
            continue
        timeout = min(_MAX_READ_SECONDS, max(_MIN_READ_SECONDS, operation.remaining_seconds()))
        try:
            row = _status_for(client, target, timeout=timeout)
        except RateLimited as exc:
            _park(operation, time.time() + exc.retry_after)
            return
        if row is None:
            continue
        _apply_read(operation, target, str(row.get("status") or "").lower(), str(row.get("statusReason") or ""))


def _prepare(client: Any, action: str, force: bool) -> Callable[[ConnectionOperation], None]:
    def prepare(operation: ConnectionOperation) -> None:
        names = [t.name for t in operation.targets]
        if action == "connect":
            mint(client, operation, names, reinitiate=False, actor=Actor.backend_watcher)
            return
        if force:
            # The re-mint names a new account; the watcher reads that one, never the old row.
            mint(client, operation, names, reinitiate=True, actor=Actor.backend_watcher)
            return
        status = _status_by_slug(client)
        repair = []
        for name in names:
            if status.get(name, {}).get("connected"):
                operation.transition(name, TargetState.initiated, Actor.backend_watcher)
                operation.transition(name, TargetState.connected, Actor.backend_watcher)
            else:
                repair.append(name)
        mint(client, operation, repair, reinitiate=True, actor=Actor.backend_watcher)

    return prepare


def _off_desktop_result(client: Any, action: str, names: List[str], force: bool, session_id: str) -> str:
    operation = ConnectionOperation([Target(n, "connector", action) for n in names], session_key=session_id)
    _prepare(client, action, force)(operation)
    payload = operation.result(with_urls=True)
    payload["status"] = "initiated" if any(t.state == TargetState.initiated for t in operation.targets) else "settled"
    payload["note"] = (
        "Show each connect_url to the user; they open it in a browser to authorize. Ask them to tell you "
        "when they are done, then check with action 'status'. Do not call connect again for the same app."
    )
    return json.dumps(payload, ensure_ascii=False)


def run_managed_action(
    action: str,
    connectors: List[str],
    args: Dict[str, Any],
    *,
    client_factory: Optional[Callable[[], Any]] = None,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    connection_callback: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    connectors_available: Optional[Callable[[], bool]] = None,
) -> str:
    if connectors_available is not None and not connectors_available():
        return tool_error("Connectors are not available in this session.")
    try:
        client = (client_factory or managed_client)()
        if action == "status":
            items = client.list_connectors()
            if connectors:
                wanted = set(connectors)
                items = [i for i in items if str(i.get("connector", "")).lower() in wanted]
            return json.dumps({"connectors": items, "hint": (
                "connected=false means calls to that connector will return CONNECTION_REQUIRED. "
                "Use action 'connect' to start an authorization.")}, ensure_ascii=False)
        if not connectors:
            return tool_error(
                f"'{action}' requires 'connectors': the connector slugs to authorize (e.g. [\"gmail\"]). "
                "Use action 'status' to list them."
            )
        force = bool(args.get("force", False))
        session_key = operation_session_key(session_id)
        if session_platform() != "desktop" or connection_callback is None:
            return _off_desktop_result(client, action, connectors, force, session_key)
        return run_operation(
            [Target(n, "connector", action) for n in connectors],
            managed_kind(client, action, force),
            session_key=session_key, tool_call_id=tool_call_id, tick_seconds=WATCH_TICK_SECONDS,
            connection_callback=connection_callback, with_urls_in_result=False,
        )
    except Exception as exc:
        logger.debug("manage_connections %s failed: %s", action, exc)
        return tool_error(
            f"The connector gateway request failed: {exc}. "
            "If this persists, the user can manage connections in the Nous Portal."
        )
