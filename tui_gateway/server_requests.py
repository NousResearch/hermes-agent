"""Server→client JSON-RPC requests: the backend asks the renderer a question and waits for the
response frame carrying the same ``id``.

JSON-RPC is peer-to-peer; this is the backend's half. Every "ask the renderer" bridge (clarify,
approval, sudo, secret, vault prompts, desktop GUI reads, MCP setup consent, the tour) is one
:func:`send` (blocking) or :func:`send_async` (queue-backed approvals) and one response frame from
the client — no paired ``*.request`` notification / ``*.respond`` method, no per-kind ``*.expire``.

Ids are ``srq-<12 hex>``: strings never collide with client-minted integer ids, and the random
part keeps a compute-host child's requests distinct from the parent's when both reach one socket.
A request that times out or is cancelled (interrupt, session close, shutdown) emits ONE
``request.cancel {id, method, reason}`` notification so every renderer tears the card down the
same way. A response for an id that is no longer open is dropped — the wait already returned.

Reconnect: unanswered requests are returned as ``open_requests`` by ``session.resume`` /
``session.activate`` / ``session.events.since`` (:func:`open_requests`); the shared TypeScript
channel re-delivers them as if they had just arrived, so the notification replay ring never
has to carry "a question still waiting for an answer".

Batch clarify keeps per-question locks (``clarify.lock`` → :func:`lock_answer`): answers stay
editable until every question is locked, locked answers survive a timeout, and the last lock
resolves the request with the full answer set.

Capability: a client says once per connection that it answers server→client requests
(``client.capabilities {server_requests: true}`` → :func:`advertise`). A WebSocket client that never
did is a build older than this half of the protocol — it drops the frame silently and the agent
would wait the full deadline (clarify's 300s) for nothing — so :func:`send` / :func:`send_async`
return the same ``None`` an error response produces without writing the frame (#112548).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any, Callable

logger = logging.getLogger(__name__)

_WINDOW_OWNED_REQUESTS = frozenset(("preview.act", "preview.read", "terminal.read", "window.read", "tour"))
_REQUEST_NOT_OWNER = -32004
_NO_WINDOW_OWNER_VALUE = json.dumps({
    "success": False,
    "error": "This chat is not displayed in any Hermes Desktop window. Bring it to the front and retry.",
})


class ServerRequest:
    __slots__ = ("id", "sid", "method", "params", "event", "result", "answered", "created_at",
                 "qids", "locked", "on_result", "owner_candidates", "owner_rejections")

    def __init__(self, sid: str, method: str, params: dict, *, qids: list[str] | None = None,
                 on_result: Callable[[dict | None], None] | None = None) -> None:
        self.id = f"srq-{uuid.uuid4().hex[:12]}"
        self.sid = sid
        self.method = method
        self.params = dict(params)
        self.event = threading.Event()
        self.result: dict | None = None
        self.answered = False
        self.created_at = time.time()
        # Batch clarify: question ids still to lock, and the answers locked so far.
        self.qids = list(qids) if qids else None
        self.locked: dict[str, str] = {}
        self.on_result = on_result
        self.owner_candidates: set[Any] = set()
        self.owner_rejections: set[Any] = set()

    def frame(self) -> dict:
        owner_quorum = {"owner_quorum": True} if self.owner_candidates else {}
        return {"jsonrpc": "2.0", "id": self.id, "method": self.method,
                "params": {"session_id": self.sid, **owner_quorum, **self.params}}

    def snapshot(self) -> dict:
        """``open_requests`` entry: the request as sent, plus the batch answers locked so far so a
        reconnecting client restores its ✓ state."""
        owner_quorum = {"owner_quorum": True} if self.owner_candidates else {}
        params = {"session_id": self.sid, **owner_quorum, **self.params}
        if self.locked:
            params["answers"] = dict(self.locked)
        return {"id": self.id, "method": self.method, "params": params}


_lock = threading.Lock()
_open: dict[str, ServerRequest] = {}

# Frame sinks, bound by ``bind_sinks`` from server.py at import time (like the method_ctx split
# modules): importing server back from here would pick a different module object under the test
# fixtures that patch ``sys.modules`` around the server import.
_write: Callable[[dict], Any] = lambda frame: None  # noqa: E731
_emit: Callable[[str, str, dict], Any] = lambda event, sid, payload: None  # noqa: E731
# ``answerable(sid)``: False only when every client attached to the session is a build that never
# advertised handling server→client requests (session_transports.py::_session_client_answers_requests).
_answerable: Callable[[str], bool] = lambda sid: True  # noqa: E731
_request_clients: Callable[[str], set] = lambda sid: set()  # noqa: E731

# Client transports that sent ``client.capabilities {server_requests: true}`` (identity set: StdioTransport
# has __slots__ and cannot be weak-referenced; ws.py forgets a peer on disconnect).
_answering_clients: set = set()


def bind_sinks(write_json: Callable[[dict], Any], emit: Callable[[str, str, dict], Any],
               answerable: Callable[[str], bool], request_clients: Callable[[str], set]) -> None:
    global _write, _emit, _answerable, _request_clients
    _write, _emit, _answerable, _request_clients = write_json, emit, answerable, request_clients


def advertise(transport: Any, server_requests: bool) -> None:
    """Record whether *transport*'s client answers server→client requests (``client.capabilities``)."""
    with _lock:
        if server_requests:
            _answering_clients.add(transport)
        else:
            _answering_clients.discard(transport)


def forget(transport: Any) -> None:
    """Drop a disconnected transport's advertisement."""
    with _lock:
        _answering_clients.discard(transport)


def answers_requests(transport: Any) -> bool:
    with _lock:
        return transport in _answering_clients


def _unanswerable(method: str, sid: str) -> bool:
    if _answerable(sid):
        return False
    logger.info("server request %s for %s not sent: the attached client predates server→client requests "
                "(update the Hermes app)", method, sid)
    return True


def _emit_cancel(req: ServerRequest, reason: str) -> None:
    _emit("request.cancel", req.sid, {"id": req.id, "method": req.method, "reason": reason})


def _register(req: ServerRequest) -> None:
    from tui_gateway.contracts import registry as contracts

    contract = contracts.SERVER_REQUESTS.get(req.method)
    if contract is None:
        raise RuntimeError(f"server request {req.method!r} has no contract in tui_gateway/contracts")
    _, problem = contracts.validate_params(contract, {"session_id": req.sid, **req.params})
    if problem is not None:
        raise ValueError(problem)  # a key the renderer's typed handler would never read: our bug
    if req.method in _WINDOW_OWNED_REQUESTS:
        req.owner_candidates = set(_request_clients(req.sid))
    with _lock:
        _open[req.id] = req
    _write(req.frame())


def send(method: str, sid: str, params: dict, *, timeout: float | None,
         qids: list[str] | None = None) -> dict | None:
    """Send one request and block for the response ``result`` (a dict).

    Returns ``None`` when the renderer never answered (timeout, cancel, or an error response — e.g.
    a client without a handler for ``method``). ``timeout`` semantics: None → wait until answered or
    cancelled, 0 → return immediately, > 0 → bounded wait. A batch (``qids``) that times out
    returns ``{"answers": <locked so far>, "timed_out": True}`` instead of None.
    """
    if _unanswerable(method, sid):
        return None
    req = ServerRequest(sid, method, params, qids=qids)
    _register(req)
    try:
        req.event.wait(timeout)
    except BaseException:
        # The wait itself died (KeyboardInterrupt, SystemExit, injected error): withdraw the request
        # or it stays in _open forever — replayed to every reconnecting client and reported by
        # pending_kind() as a human still being waited on.
        with _lock:
            still_open = _open.pop(req.id, None) is req
        if still_open:
            _emit_cancel(req, "interrupted")
        raise
    with _lock:
        # The verdict is the state committed under the lock, never wait()'s return value: a
        # response frame can land after the deadline expires and before this removal, and
        # settlement (resolve_response / lock_answer / cancel) already popped it (#112548).
        timed_out = _open.pop(req.id, None) is req
        answered, result, locked = req.answered, req.result, dict(req.locked)
    if answered:
        return result
    if timed_out:
        _emit_cancel(req, "timeout")
        if req.qids is not None:
            return {"answers": locked, "timed_out": True}
    return None


def send_async(method: str, sid: str, params: dict, on_result: Callable[[dict | None], None]) -> Callable[[str], None]:
    """Send one request whose wait is owned elsewhere (the approval queue's own timeout). ``on_result``
    runs on the dispatching thread when the response lands. Returns ``settle(reason)``: call it when
    the underlying wait ends; if the request is still open it is withdrawn with ``request.cancel``."""
    if _unanswerable(method, sid):
        on_result(None)
        return lambda reason: None
    req = ServerRequest(sid, method, params, on_result=on_result)
    _register(req)

    def settle(reason: str) -> None:
        with _lock:
            still_open = _open.pop(req.id, None) is not None
        if still_open:
            _emit_cancel(req, reason)

    return settle


def resolve_response(frame: dict, transport: Any = None) -> bool:
    """Route one client response frame to its open request. False when nothing is waiting for that id
    (already timed out / cancelled, or owned by another process — see the compute-host bridge)."""
    rid = frame.get("id")
    if not isinstance(rid, str):
        return False
    with _lock:
        req = _open.get(rid)
        if req is None:
            # Already settled (timed out, cancelled, answered from another surface) or owned by
            # another process; say so — a dropped answer used to vanish without a trace.
            logger.debug("server request %s: response dropped, request no longer open", rid)
            return False
        error = frame.get("error")
        if (isinstance(error, dict) and error.get("code") == _REQUEST_NOT_OWNER
                and transport in req.owner_candidates):
            req.owner_rejections.add(transport)
            if not req.owner_candidates.issubset(req.owner_rejections):
                return True
            _open.pop(rid, None)
            req.result, req.answered = {"value": _NO_WINDOW_OWNER_VALUE}, True
        else:
            # Removing the request and committing its outcome are one settlement.
            # ``cancel()`` also settles under this lock, so the first side to get
            # here wins instead of a later cancellation overwriting a response.
            _open.pop(rid, None)
            if "error" in frame:
                logger.debug("server request %s (%s) answered with error: %s", rid, req.method, frame.get("error"))
                req.result, req.answered = None, False
            else:
                result = frame.get("result")
                req.result = result if isinstance(result, dict) else {}
                if req.qids and "answers" in req.result:
                    # Batch clarify: answers locked early via clarify.lock belong to the final set even when
                    # the closing response only carries the tail the user answered last.
                    answers = req.result.get("answers")
                    merged = dict(req.locked)
                    if isinstance(answers, dict):
                        merged.update(answers)
                    req.result = {**req.result, "answers": merged}
                req.answered = True
    if req.on_result is not None:
        req.on_result(req.result)
    req.event.set()
    return True


def lock_answer(request_id: str, question_id: str, answer: str) -> list[str] | None:
    """Lock one batch-clarify answer (update-in-place). Returns the question ids still unanswered;
    the last lock resolves the request with the full ``{"answers"}`` set. ``None`` when no open
    batch has that id (expired or foreign); ``ValueError`` for an unknown question id."""
    with _lock:
        req = _open.get(request_id)
        if req is None or req.qids is None:
            return None
        if question_id not in req.qids:
            raise ValueError(f"unknown question_id {question_id!r}")
        req.locked[question_id] = answer
        remaining = [qid for qid in req.qids if qid not in req.locked]
        if not remaining:
            req.result, req.answered = {"answers": dict(req.locked)}, True
            _open.pop(request_id, None)
    if not remaining:
        req.event.set()
    return remaining


def cancel(sid: str | None = None, reason: str = "interrupted") -> int:
    """Withdraw open requests — only *sid*'s (session.interrupt must not touch other sessions'), or
    every one when *sid* is None (shutdown). Blocked waits return None; queue-backed requests run
    ``on_result(None)`` so their owner can settle. Returns the number withdrawn."""
    with _lock:
        targets = [req for req in _open.values() if sid is None or req.sid == sid]
        for req in targets:
            _open.pop(req.id, None)
            req.result, req.answered = None, False
    for req in targets:
        if req.on_result is not None:
            req.on_result(None)
        req.event.set()
        _emit_cancel(req, reason)
    return len(targets)


def open_requests(sid: str) -> list[dict]:
    """Unanswered requests for *sid*, oldest first."""
    with _lock:
        reqs = sorted((req for req in _open.values() if req.sid == sid), key=lambda r: r.created_at)
    return [req.snapshot() for req in reqs]


def open_request_count() -> int:
    """Unanswered server→client requests across every session: the process is waiting on a
    human (clarify, approval, sudo, secret, ...) and must not be treated as idle."""
    with _lock:
        return len(_open)


def pending_kind(sid: str) -> str:
    """Method of the oldest open request for *sid* ("" when none) — the session is waiting on a human."""
    with _lock:
        reqs = [req for req in _open.values() if req.sid == sid]
    return min(reqs, key=lambda r: r.created_at).method if reqs else ""


def is_response_frame(obj: Any) -> bool:
    """A client response: has an ``id`` and a ``result``/``error`` member but no ``method``."""
    return isinstance(obj, dict) and "method" not in obj and "id" in obj and ("result" in obj or "error" in obj)


def reset_for_tests() -> None:
    with _lock:
        _open.clear()
        _answering_clients.clear()
