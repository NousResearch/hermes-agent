"""Serve -> owner cutover for one typed prompt (RT-37, #140385).

A legacy ``hermes serve`` backend (no session authority) refuses a turn while the gateway's durable
ledger holds work for the same stored session (``session_lifecycle._canonical_ledger_refusal``): a
legacy turn would run beside the canonical FIFO and write the transcript twice. A Bot Chat the
gateway also owns hits that on every gateway turn, so the user's own Desktop prompt was refused
``SESSION_NOT_OWNED`` and the card offered "Start new session".

The #106742 contract is one owner and one FIFO, with clients that submit. So the prompt goes to the
owner instead: this backend dials the ready gateway serving the session's profile home, attaches to
the session and admits the prompt into its durable FIFO (``prompt.submit`` with a fresh submission
id). The owner runs it after the current turn, exactly once; nothing runs here. The owner's frames
for THAT admission are relayed onto the Desktop's live session, so the client sees an ordinary
queued send that later streams and completes. If the relay loses the owner, an empty
``message.complete`` settles the bubble and the client re-reads the stored transcript the owner
writes. An ``unknown`` admission is never forwarded: only Discard resolves it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Ledger states a forwarded prompt may wait behind: they end on their own.
FORWARDABLE_LEDGER_STATES = frozenset({"started", "queued", "registered", "running"})
#: How long ``prompt.submit`` waits for the owner's admission receipt.
ADMIT_TIMEOUT_S = 30.0
#: Ceiling on one relay; the owner keeps the work either way (it is durable).
RELAY_TIMEOUT_S = 6 * 3600.0
#: How often an idle relay re-checks that its Desktop session still exists: a closed session
#: releases the thread and the owner socket without waiting for the owner's next frame.
RELAY_FINALIZE_POLL_S = 1.0
#: Owner turn frames relayed for the forwarded admission. Not its deltas: a Desktop turn that streamed
#: nothing re-reads the stored transcript at ``message.complete``, which brings the user's own row
#: (written by the owner, not by this backend) and the reply in their committed order.
_RELAYED = frozenset({"message.start", "message.complete"})
#: The owner's ``prompt.submit`` refusals raised before its durable write (dispatcher, ``authorize``,
#: ``submission``, the admission gate, and the ledger's own checks inside the rolled-back write): only
#: these prove nothing will run. ``storage_unavailable`` and an internal error can also come from the
#: pending publication AFTER the commit (``SessionAuthority._admitted``), so they are never listed.
#: ``not_found`` can follow the commit too, when the session is deleted meanwhile (``_handle``), but a
#: session row is deleted only with its admissions (``ON DELETE RESTRICT``), so that one never runs.
_REFUSED_BEFORE_COMMIT = frozenset({
    "invalid_params", "runtime_draining", "runtime_retired", "profile_mismatch", "permission_denied",
    "not_found", "stale_epoch", "admission_conflict", "queue_full"})


class OwnerUnavailable(RuntimeError):
    """The owner could not take the prompt; ``str(exc)`` names why for the refusal."""


class OwnerNotRunning(OwnerUnavailable):
    """No ready gateway serves the home: nothing will end the ledger's work on its own."""


class OwnerUnconfirmed(OwnerUnavailable):
    """The owner's admission receipt did not arrive in time. The relay keeps going, so the prompt
    may still be (or already be) queued there: never "nothing was sent", never a retry."""


def discover_owner(home: Path):
    """The ready gateway endpoint serving ``home``. Discovery only: never starts one."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    discovery = discover_gateway_endpoint(home, timeout=5)
    if discovery.state != "ready" or discovery.endpoint is None:
        raise OwnerNotRunning(f"the gateway that owns it is {discovery.state}")
    return discovery.endpoint


@contextlib.asynccontextmanager
async def connect_owner(home: Path, endpoint, *, scope=None, max_size: int = 2 ** 20):
    """A ``GatewayClient`` on ``endpoint``, the ready gateway serving ``home``; ``scope``: the
    ticket's profile scope (``"host"``: the one native Desktop holds); ``max_size``: the largest
    answer frame accepted (websockets' 1 MiB default)."""
    from hermes_cli.gateway_client import GATEWAY_WS_PROTOCOL, GatewayClient, _session_ticket, gateway_ws_target
    from websockets.asyncio.client import connect

    ticket = await asyncio.to_thread(_session_ticket, home, endpoint, scope=scope)
    url, protocols = gateway_ws_target(endpoint, ticket)
    # Loopback owner dial: never through HTTP(S)_PROXY (websockets>=14 honours it by default).
    async with connect(url, subprotocols=protocols, open_timeout=10, proxy=None, max_size=max_size) as ws:
        if ws.subprotocol != GATEWAY_WS_PROTOCOL:
            raise OwnerUnavailable("the gateway answered with another protocol")
        async with GatewayClient(ws) as client:
            yield client


def forwardable(refusal: Any, session: dict, text: Any, params: dict) -> bool:
    """Whether a ledger refusal of this ``prompt.submit`` can go to the owner instead."""
    if getattr(refusal, "ledger_status", None) not in FORWARDABLE_LEDGER_STATES:
        return False
    return (isinstance(text, str) and bool(text.strip()) and not session.get("attached_images")
            and not any(params.get(key) is not None for key in (
                "truncate_before_user_ordinal", "truncate_before_row_id", "truncate_before_message_id")))


def _payload_for(event_type: str, payload: Any) -> dict | None:
    """The owner payload projected onto this wire's contract for ``event_type`` (None: drop it)."""
    from tui_gateway.contracts.registry import EVENTS
    contract = EVENTS.get(event_type)
    if contract is None or contract.payload is None or not isinstance(payload, dict):
        return None
    shaped = {key: value for key, value in payload.items() if key in contract.payload.model_fields}
    from pydantic import ValidationError
    try:
        contract.payload.model_validate(shaped)
    except ValidationError:
        if event_type != "message.complete":
            return None
        shaped = {"text": payload.get("text") if isinstance(payload.get("text"), str) else ""}
    return shaped


class _Relay:
    def __init__(self, sid: str, session: dict, key: str, home: Path, endpoint, text: str, emit):
        self.sid, self.session, self.key, self.home, self.text, self.emit = sid, session, key, home, text, emit
        self.endpoint, self.listed = endpoint, None
        self.admitted: Future = Future()
        self.request_id = "desktop-forward-" + uuid.uuid4().hex

    def run(self) -> None:
        try:
            asyncio.run(self._run())
        except BaseException as exc:  # health: allow BLE001 -- thread boundary: handed to the waiting submit or logged with traceback
            if not self.admitted.done():
                self.admitted.set_exception(exc)
            else:
                logger.warning("owner relay for %s ended: %s", self.key, exc, exc_info=True)

    async def _run(self) -> None:
        async with connect_owner(self.home, self.endpoint) as client:
            attached = await client.rpc("session.resume", session_id=self.key)
            target = str(attached.get("session_id") or self.key)
            receipt = await self._submit(client, target)
            if receipt.get("voice_stopped") is True:
                # TUI-26: the owner took a typed stop phrase as the end of its LOCAL voice chat and
                # admitted nothing: there is no turn to relay.
                self.admitted.set_result(receipt)
                return
            admission_id = receipt.get("admission_id")
            if not isinstance(admission_id, str) or receipt.get("status") not in ("queued", "started", "terminal"):
                raise OwnerUnavailable("the gateway did not admit it")
            self.admitted.set_result(receipt)
            logger.info("forwarded Desktop prompt for %s to the owner FIFO: admission=%s",
                        self.key, admission_id)
            await self._relay(client, admission_id)

    async def _submit(self, client, target: str) -> dict:
        """``prompt.submit`` on the owner. Once the request is handed to the socket, only a received
        refusal the owner raises before its write (``_REFUSED_BEFORE_COMMIT``) proves nothing was
        queued: a lost connection, a missing answer or any other error may follow the owner's durable
        commit (a gateway restart mid-forward, a storage error publishing the pending list), so it is
        unconfirmed, never "nothing was sent" (a resend would run a second copy)."""
        from hermes_cli.gateway_client import GatewayRPCError
        try:
            return await client.rpc("prompt.submit", session_id=target, text=self.text,
                                    submission_id=self.request_id)
        except Exception as exc:
            # The owner names its refusal in ``data.reason``; ``message`` may be a display string.
            reason = exc.data.get("reason", str(exc)) if isinstance(exc, GatewayRPCError) else None
            if isinstance(reason, str) and reason in _REFUSED_BEFORE_COMMIT:
                raise
            raise OwnerUnconfirmed("the gateway did not confirm it") from exc

    def _pending(self, params: dict, admission_id: str) -> None:
        """The owner's pending list as this wire's ``pending_submissions``, this admission only: the
        Desktop's queued-row card while it waits, retired once the owner starts it."""
        rows = (params.get("payload") or {}).get("pending")
        if not isinstance(rows, list):
            return
        listed = [{"admission_id": admission_id, "status": "queued", "user": self.text, "input_id": self.request_id}
                  for row in rows if isinstance(row, dict) and row.get("admission_id") == admission_id
                  and row.get("status") == "queued"]
        if listed != self.listed:
            self.listed = listed
            self.emit("session.info", self.sid, {"stored_session_id": self.key, "pending_submissions": listed})

    async def _relay(self, client, admission_id: str) -> None:
        settled = False
        try:
            async with asyncio.timeout(RELAY_TIMEOUT_S):
                while not settled and not self.session.get("_finalized"):
                    try:
                        async with asyncio.timeout(RELAY_FINALIZE_POLL_S):
                            frame = await client.events.get()
                    except TimeoutError:
                        continue
                    if isinstance(frame, BaseException):
                        return
                    params = frame.get("params") if isinstance(frame, dict) else None
                    if not isinstance(params, dict):
                        continue
                    if params.get("type") == "session.info":
                        self._pending(params, admission_id)
                        continue
                    if params.get("admission_id") != admission_id:
                        continue
                    event_type = params.get("type")
                    if event_type not in _RELAYED:
                        continue
                    payload = _payload_for(event_type, params.get("payload"))
                    if payload is None:
                        continue
                    self.emit(event_type, self.sid, payload)
                    settled = event_type == "message.complete"
        finally:
            if not settled and not self.session.get("_finalized"):
                # Lost the owner mid-relay: the work is durable and still runs there. An empty
                # completion settles the bubble; the client re-reads the stored transcript.
                self.emit("message.complete", self.sid, {"text": ""})


def forward_rpc(home: Path, endpoint, method: str, params: dict) -> Any:
    """One request/answer RPC on the owner serving ``home`` (Group Chat calls, WA-19), on the
    host-scoped grant native Desktop itself holds, so a room may seat the served sibling profiles."""
    from gateway.hosted_room_attachments import MAX_ATTACHMENT_BYTES
    # A groups.attachment.download answer carries a whole file as base64: room for the largest
    # one the room store accepts, plus its JSON-RPC envelope.
    frame = (MAX_ATTACHMENT_BYTES + 2) // 3 * 4 + 2 ** 20

    async def call():
        async with connect_owner(home, endpoint, scope="host", max_size=frame) as client:
            return await client.rpc(method, **params)
    return asyncio.run(call())


def forward_prompt(sid: str, session: dict, text: str, *, emit=None, home: Path | None = None) -> dict:
    """Admit ``text`` into the owner's FIFO for this session; the owner's receipt.

    Raises ``OwnerNotRunning`` when no gateway serves the home, ``OwnerUnavailable`` (or the
    client's refusal) when the owner did not take it, and ``OwnerUnconfirmed`` when the prompt may
    be queued there: the receipt timed out, or the owner was lost or failed after the request was sent."""
    from tui_gateway import server
    key = str(session.get("session_key") or "")
    if not key:
        raise OwnerUnavailable("the chat has no stored session yet")
    home = Path(home or session.get("profile_home") or server._hermes_home).resolve()
    relay = _Relay(sid, session, key, home, discover_owner(home), text, emit or server._emit)
    from agent.memory_provider import spawn_context_thread
    spawn_context_thread(relay.run, name=f"owner-forward-{key[:24]}").start()
    try:
        return relay.admitted.result(timeout=ADMIT_TIMEOUT_S)
    except FutureTimeout as exc:
        raise OwnerUnconfirmed("the gateway did not confirm it in time") from exc
