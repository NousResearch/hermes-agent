from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.connectors.contract import Actor, SettleReason, TargetState, allowed
from tools.connectors.gateway.config import operation_session_key
from tools.connectors.gateway.errors import RateLimited, ToolGatewayError
from tools.connectors.operation import ConnectionOperation, DetachedOperation, IllegalTransition, Target
from tools.connectors.run import Kind, run_operation
from tools.connectors.targets import HostedTarget, catalog_names, hosted_names, misrouted_to_hosted_error
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
_ACCOUNT_OUTCOME: dict[str, tuple[TargetState, Actor]] = {
    "active": (TargetState.connected, Actor.backend_watcher),
    "failed": (TargetState.failed, Actor.backend_watcher),
    "revoked": (TargetState.failed, Actor.backend_watcher),
    "inactive": (TargetState.failed, Actor.backend_watcher),
    # The link's TTL ran out; the gateway reports it, the clock caused it.
    "expired": (TargetState.expired, Actor.clock),
}

NOTE = (
    "Settled once. connected → use the app now; skipped → the user chose Not now, do not connect it "
    "or route around it; not_connected → ask the user what to do, never re-mint on your own. "
    "A later request from the USER for that same app is not a re-ask — run it. "
    "If you named an account, tell the user its name and that they can rename it."
)


def managed_client():
    from tools.connectors.gateway.client import ConnectorClient

    return ConnectorClient()


def managed_kind(client: Any, action: str, force: bool) -> Kind:
    return Kind(prepare=_prepare(client, action, force), observe=lambda operation: _observe(client, operation), note=NOTE)


def _status_by_slug(client: Any) -> dict[str, dict[str, Any]]:
    """The toolkit list, by slug. Only the reconnect repair check reads it: it answers "is this app
    already connected" before any account exists for the watcher to read."""
    return {str(i.get("connector", "")).lower(): i for i in client.list_connectors() if isinstance(i, dict)}


def mint(client: Any, operation: ConnectionOperation, names: list[str], *, reinitiate: bool, actor: Actor,
         connection_id: Optional[str] = None) -> None:
    """Mint links for ``names`` and apply the gateway's per-app answer to the operation. ``actor`` is
    the watcher on the first mint and the user on Try again. The operation id rides along so the
    vendor's done page can name it on the way back to the desktop. ``connection_id`` names the one
    existing account a repair restarts; the gateway then needs no alias to find it. A repair without
    one (Try again) reuses the account the named target already minted, so a rename in between
    cannot point the retry at another login."""
    from tools.connectors.gateway.client import return_to_args

    if not names:
        return
    # One aliased target per call, so the first named target is the request's.
    named = next((t for t in operation.targets if t.name in names and t.alias), None)
    if reinitiate and connection_id is None:
        repaired = next((t.repair_id for t in operation.targets if t.name in names and t.repair_id), None)
        connection_id = repaired or (named.connection_id if named is not None else None)
    alias = None if connection_id or named is None else named.alias
    response = client.connections(names, reinitiate=reinitiate, alias=alias, connection_id=connection_id,
                                  **return_to_args(op=operation.op_id))
    for entry in response.get("results", []):
        name = str(entry.get("connector") or "").lower()
        target = operation.target(name)
        if target is None:
            continue
        status = str(entry.get("status") or "")
        detail = str(entry.get("status_reason") or "")
        connection_id = entry.get("connection_id")
        if status == "active":
            operation.transition(name, TargetState.initiated, actor)
            operation.transition(name, TargetState.connected, Actor.backend_watcher, connection_id=connection_id)
        elif status == "initiated":
            if not connection_id:
                logger.warning("connector %s: the mint named no account, so the watcher cannot read it; "
                               "only the card or the deadline can end the row", name)
            operation.transition(
                name, TargetState.initiated, actor,
                connect_url=entry.get("connect_url"), connection_id=connection_id,
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


def _status_for(client: Any, target: Target, *, timeout: float) -> Optional[dict[str, Any]]:
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


def _mark_misrouted(operation: ConnectionOperation) -> None:
    unminted = [t for t in operation.targets
                if t.state == TargetState.failed and not t.connection_id]
    if not unminted:
        return
    catalog = catalog_names()
    candidates = [t for t in unminted if t.name in catalog]
    if not candidates:
        return
    hosted = hosted_names()
    if hosted is None:
        return
    misrouted = [t for t in candidates if t.name not in hosted]
    for target in misrouted:
        operation.refresh(target.name, connect_url=None, actor=Actor.backend_watcher,
                          detail=misrouted_to_hosted_error(target.name))
    if misrouted and len(misrouted) == len(operation.targets):
        operation.settle(SettleReason.all_resolved)


def _prepare(client: Any, action: str, force: bool) -> Callable[[ConnectionOperation], None]:
    def prepare(operation: ConnectionOperation) -> None:
        names = [t.name for t in operation.targets]
        if action == "connect":
            mint(client, operation, names, reinitiate=False, actor=Actor.backend_watcher)
            _mark_misrouted(operation)
            return
        # A named account is checked on its own row, read before force so a label resolves to its
        # account either way: the connector-wide flag reports any account.
        named = _alias_status(operation.targets)
        if force:
            # The re-mint names a new account; the watcher reads that one, never the old row.
            account = next((s["connection_id"] for s in (named or {}).values() if s.get("connection_id")), None)
            mint(client, operation, names, reinitiate=True, actor=Actor.backend_watcher, connection_id=account)
            _mark_misrouted(operation)
            return
        status = named or _status_by_slug(client)
        repair = []
        for name in names:
            target = operation.target(name)
            if target is not None and target.repair_id and name not in (named or {}):
                # A surface asked to repair one account by id: its Reconnect is the user's choice, and
                # the connector's other accounts say nothing about this one.
                repair.append(name)
            elif status.get(name, {}).get("connected"):
                operation.transition(name, TargetState.initiated, Actor.backend_watcher)
                operation.transition(name, TargetState.connected, Actor.backend_watcher)
            else:
                repair.append(name)
        # A named target is alone in its call, so at most one account id applies.
        account = next((status[n].get("connection_id") for n in repair if status.get(n, {}).get("connection_id")), None)
        mint(client, operation, repair, reinitiate=True, actor=Actor.backend_watcher, connection_id=account)
        _mark_misrouted(operation)

    return prepare


def portal_accounts() -> list[dict[str, Any]]:
    from tools.connectors.portal.client import PortalConnectorClient

    return PortalConnectorClient().list_accounts()


def _alias_status(targets: list[Target]) -> Optional[dict[str, dict[str, Any]]]:
    """``{slug: {"connected": bool, "connection_id": str | None}}`` for an aliased target, read from
    that account's own row (``connection_id`` is None when no account has that name yet); ``None``
    when no target is aliased. A failed read raises: guessing "not connected" would send the user
    through a new login for an account that may be healthy."""
    aliased = [t for t in targets if t.alias]
    if not aliased:
        return None
    rows = portal_accounts()
    status: dict[str, dict[str, Any]] = {}
    for t in aliased:
        mine = [r for r in rows if r.get("connector") == t.name and not r.get("disabled")]
        row = next((r for r in mine if r.get("alias") == t.alias), None)
        if row is None:
            # An unnamed account is addressed by its label, as rename does; it stays unnamed.
            row = next((r for r in mine if not r.get("alias") and r.get("label") == t.alias), None)
            if row is not None:
                t.alias, t.repair_id = None, row.get("connectionId")
        status[t.name] = {"connected": bool(row) and row.get("status") == "active",
                          "connection_id": row.get("connectionId") if row else None}
    return status


def _targets(targets: list[HostedTarget], action: str) -> list[Target]:
    return [Target(t.name, "connector", action, alias=t.alias) for t in targets]


def _no_card_result(client: Any, action: str, targets: list[HostedTarget], force: bool, session_id: str) -> str:
    operation = DetachedOperation(_targets(targets, action), session_key=session_id)
    _prepare(client, action, force)(operation)
    payload = operation.result(with_urls=True)
    payload["status"] = "initiated" if any(t.state == TargetState.initiated for t in operation.targets) else "settled"
    payload["note"] = (
        "Show each connect_url to the user; they open it in a browser to authorize. Ask them to tell you "
        "when they are done, then check with action 'status'. Do not call connect again for the same app."
    )
    if any(t.alias for t in targets):
        payload["note"] += " You named this account: tell the user its name and that they can rename it."
    return json.dumps(payload, ensure_ascii=False)


def _status_result(client: Any, connectors: list[str]) -> str:
    items = client.list_connectors()
    if connectors:
        wanted = set(connectors)
        items = [i for i in items if str(i.get("connector", "")).lower() in wanted]
    try:
        rows = portal_accounts()
    except ToolGatewayError as exc:
        # The account list is extra detail; the connector list alone still answers status.
        logger.debug("connector accounts for status failed: %s", exc)
    else:
        by_slug: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            entry = {"alias": row.get("alias"), "label": row.get("label"), "status": row.get("status"),
                     "active": row.get("active")}
            if row.get("disabled"):
                entry["retired"] = True
            by_slug.setdefault(str(row.get("connector", "")).lower(), []).append(entry)
        items = [dict(i, accounts=by_slug.get(str(i.get("connector", "")).lower(), [])) for i in items]
    return json.dumps({"connectors": items, "hint": (
        "connected=false means calls to that connector will return CONNECTION_REQUIRED. "
        "Use action 'connect' to start an authorization. 'accounts' lists each account by its name "
        "(alias) and the vendor's label; address an unnamed account by its label. A retired account was "
        "replaced by a reconnect and is kept only for the record.")}, ensure_ascii=False)


def _rename_result(target: HostedTarget) -> str:
    """Rename the one account of ``target.name`` called ``target.alias`` (or, when unnamed, labelled
    so) to ``target.to``. Synchronous: no operation and no card."""
    from tools.connectors.gateway.errors import GatewayUnavailable, IdempotencyConflict
    from tools.connectors.portal.client import PortalConnectorClient

    rows = [r for r in portal_accounts() if str(r.get("connector", "")).lower() == target.name]
    matches = [r for r in rows if r.get("alias") == target.alias] or [
        r for r in rows if not r.get("alias") and r.get("label") == target.alias]
    if len(matches) != 1:
        names = sorted({str(r.get("alias") or r.get("label")) for r in rows})
        found = f"Its accounts are: {', '.join(names)}." if names else f"The user has no {target.name} account."
        what = "No" if not matches else "More than one"
        return tool_error(f"{what} {target.name} account is called {target.alias!r}. {found}")
    try:
        account = PortalConnectorClient().rename_account(matches[0]["connectionId"], target.to)
    except IdempotencyConflict:
        return tool_error(f"The name {target.to!r} is already used by another {target.name} account. Pick another name.")
    except GatewayUnavailable as exc:
        if exc.code == "connection_not_found":
            return tool_error(f"That {target.name} account no longer exists. Use action status to list accounts.")
        return tool_error("This Nous Portal cannot rename connector accounts yet.")
    except ToolGatewayError as exc:
        if exc.status == 405:
            return tool_error("This Nous Portal cannot rename connector accounts yet.")
        raise
    return json.dumps({
        "renamed": {"connector": account["connector"], "alias": account.get("alias"), "label": account["label"]},
        "notice": f"Tell the user their {target.name} account {target.alias!r} is now called {target.to!r}.",
    }, ensure_ascii=False)


def run_managed_action(
    action: str,
    targets: list[HostedTarget],
    args: dict[str, Any],
    *,
    client_factory: Optional[Callable[[], Any]] = None,
    session_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    connection_callback: Optional[Callable[[dict[str, Any]], Optional[str]]] = None,
    connectors_available: Optional[Callable[[], bool]] = None,
) -> str:
    if connectors_available is not None and not connectors_available():
        return tool_error("Connectors are not available in this session.")
    try:
        if action == "rename":
            return _rename_result(targets[0])
        client = (client_factory or managed_client)()
        if action == "status":
            return _status_result(client, [t.name for t in targets])
        if not targets:
            return tool_error(
                f"'{action}' requires 'connectors': the connector slugs to authorize (e.g. [\"gmail\"]). "
                "Use action 'status' to list them."
            )
        force = bool(args.get("force", False))
        session_key = operation_session_key(session_id)
        if connection_callback is None:
            return _no_card_result(client, action, targets, force, session_key)
        return run_operation(
            _targets(targets, action),
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
