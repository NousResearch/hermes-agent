"""Managed connectors (Nous tool gateway) on the connection operation.

``connect`` mints a link for every target up front and stores it on the target; ``reconnect``
reads status first and reinitiates only what is not connected (``force`` always reinitiates).
On a desktop session the call blocks until the operation settles and the result carries no URL;
the card owns the links. Off the desktop the result carries the URLs and returns at once, until
PR3 delivers them as their own message. The watcher hook reads the gateway list once per tick
(the exact-status route replaces this call when the gateway ships it)."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from tools.connectors.contract import Actor, TargetState
from tools.connectors.gateway.config import operation_session_key, session_platform
from tools.connectors.operation import ConnectionOperation, Target
from tools.connectors.run import Kind, run_operation
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Statuses on the gateway list that end an attempt. The list's statusReason is generic copy, so the
# detail recorded at mint time is kept; only a missing detail is filled from the list.
_TERMINAL_LIST_STATUS = {"failed": TargetState.failed, "expired": TargetState.expired, "revoked": TargetState.failed}

NOTE = (
    "Settled once. connected → use the app now; skipped → the user chose Not now, do not connect it "
    "or route around it; not_connected → ask the user what to do, never re-mint on your own."
)


def _default_client():
    from tools.connectors.gateway.client import ConnectorClient

    return ConnectorClient()


def _status_by_slug(client: Any) -> Dict[str, Dict[str, Any]]:
    return {str(i.get("connector", "")).lower(): i for i in client.list_connectors() if isinstance(i, dict)}


def mint(client: Any, operation: ConnectionOperation, names: List[str], *, reinitiate: bool, actor: Actor) -> None:
    """Mint links for ``names`` and apply the gateway's per-app answer to the operation. ``actor`` is
    the watcher on the first mint and the user on Try again."""
    if not names:
        return
    response = client.connections(names, reinitiate=reinitiate)
    for entry in response.get("results", []):
        name = str(entry.get("connector") or "").lower()
        target = operation.target(name)
        if target is None:
            continue
        status = str(entry.get("status") or "")
        detail = str(entry.get("status_reason") or entry.get("statusReason") or "")
        if status == "active":
            operation.transition(name, TargetState.initiated, actor)
            operation.transition(name, TargetState.connected, Actor.backend_watcher)
        elif status == "initiated":
            operation.transition(
                name, TargetState.initiated, actor,
                connect_url=entry.get("connect_url"), attempt=entry.get("attempt"), detail=detail,
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


def _observe(client: Any, operation: ConnectionOperation) -> None:
    try:
        status = _status_by_slug(client)
    except Exception as exc:
        logger.debug("connector watch poll failed: %s", exc)
        return
    for target in operation.targets:
        # Only a live attempt (pending, initiated) can be advanced by a gateway read; a failed or expired
        # link waits for the user, and a settled op is frozen.
        if operation.settled or target.state not in (TargetState.pending, TargetState.initiated):
            continue
        row = status.get(target.name)
        if row is None:
            continue
        row_status = str(row.get("connectionStatus") or "").lower()
        if target.awaiting_new_attempt:
            if row.get("connected") or row_status == "active":
                continue
            target.awaiting_new_attempt = False
        if row.get("connected"):
            if target.state == TargetState.pending:
                operation.transition(target.name, TargetState.initiated, Actor.backend_watcher)
            operation.transition(target.name, TargetState.connected, Actor.backend_watcher)
            continue
        terminal = _TERMINAL_LIST_STATUS.get(row_status)
        if terminal is not None and target.state == TargetState.initiated:
            # `expired` is the link TTL running out; the gateway reports it, the clock caused it.
            actor = Actor.clock if terminal == TargetState.expired else Actor.backend_watcher
            operation.transition(target.name, terminal, actor, detail=target.detail or str(row.get("statusReason") or ""))


def _prepare(client: Any, action: str, force: bool) -> Callable[[ConnectionOperation], None]:
    def prepare(operation: ConnectionOperation) -> None:
        names = [t.name for t in operation.targets]
        if action == "connect":
            mint(client, operation, names, reinitiate=False, actor=Actor.backend_watcher)
            return
        if force:
            mint(client, operation, names, reinitiate=True, actor=Actor.backend_watcher)
            for target in operation.targets:
                if target.state == TargetState.initiated:
                    target.awaiting_new_attempt = True
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
        client = (client_factory or _default_client)()
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
            Kind(prepare=_prepare(client, action, force), observe=lambda op: _observe(client, op), note=NOTE),
            session_key=session_key, tool_call_id=tool_call_id,
            connection_callback=connection_callback, with_urls_in_result=False,
        )
    except Exception as exc:
        logger.debug("manage_connections %s failed: %s", action, exc)
        return tool_error(
            f"The connector gateway request failed: {exc}. "
            "If this persists, the user can manage connections in the Nous Portal."
        )
