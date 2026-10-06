"""Child-owned conditional reservations. The private pipe attests parent authority,
not client claims; identity locks never migrate between threads or processes."""
from __future__ import annotations

import concurrent.futures
import contextlib
import threading
import time
import uuid

from pydantic import ValidationError

from .contracts.sessions import SessionActivateBoundParams, SessionCreationBinding, SessionInvokeBoundParams
from .session_creation_binding import CreationBinding, EngineCreationBinding
from .transport import bind_transport, reset_transport

RESERVATION_SECONDS = 5.0


class HostBoundPeer:
    """Logical subscriber; delivery remains on the host's existing relay exactly once."""
    def __init__(self, owner):
        provider, _, user = owner.partition(":")
        self.auth_identity = {"provider": provider, "user_id": user}
        self._closed = False

    def write(self, _obj):
        return not self._closed

    def close(self):
        self._closed = True


class HostConditionalProtocol:
    def __init__(self, host):
        self.host = host
        self._origins = {}
        self._members = {}
        self._pending = {}
        self._admissions = {}
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(2)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="host-conditional")

    def capture(self, server, session, frame):
        """Only the first actual child record can capture the parent's original origin."""
        from pathlib import Path

        sid = frame["sid"]
        with self._lock:
            if sid in self._origins:
                return
            self._origins[sid] = None  # failed/unknown first identity is never restamped
        try:
            wire = SessionCreationBinding.model_validate(frame.get("conditional_origin")).model_dump()
        except ValidationError:
            return
        if (frame.get("conditional_boot") != self.host._boot_id or wire["session_id"] != sid
                or wire["stored_session_id"] != session.get("session_key")
                or wire["authenticated_owner"] != session.get("auth_user_id")):
            return
        store = str((Path(session.get("profile_home") or server._hermes_home) / "state.db").resolve())
        origin = CreationBinding(sid, wire["stored_session_id"], wire["authenticated_owner"], store,
                                 wire["runtime_incarnation"], wire["profile_store_scope"], session)
        witness = EngineCreationBinding.capture(origin, session["agent"])
        if witness is None:
            return
        session["creation_binding"], session["creation_engine"] = origin, witness
        session["agent_ready"] = threading.Event()
        session["agent_ready"].set()
        with self._lock:
            self._origins[sid] = session

    def handle(self, frame):
        if frame.get("boot_id") != self.host._boot_id or self.host._closed.is_set():
            self._reply(frame, error=4007)
            return
        handler = {"prepare": self._prepare, "commit": self._commit, "admitted": self._admitted,
                   "abort": self._abort, "release": self._release}.get(frame.get("action"))
        if handler is None:
            self._reply(frame, error=4000)
        else:
            handler(frame)

    def _prepare(self, frame):
        if not self._slots.acquire(blocking=False):
            self._reply(frame, error=4009)
            return
        self._executor.submit(self._reserve, dict(frame))

    def _commit(self, frame):
        with self._lock:
            pending = self._pending.get(frame.get("reservation"))
            if pending is not None:
                pending["commit"] = dict(frame)
                pending["event"].set()
            else:
                self._reply(frame, error=4007)

    def _admitted(self, frame):
        with self._lock:
            admission = self._admissions.get(frame.get("turn_id"))
            if admission is not None:
                admission["allowed"] = frame.get("allowed") is True
                admission["event"].set()

    def _abort(self, frame):
        with self._lock:
            pending = self._pending.get(frame.get("reservation"))
            if pending is not None:
                pending["event"].set()

    def _release(self, frame):
        with self._lock:
            member = self._members.pop(frame.get("subscription"), None)
            for pending in self._pending.values():
                if pending.get("subscription") == frame.get("subscription"):
                    pending["peer"].close()
                    pending["event"].set()
        if member is not None:
            from . import server
            member[0].close()
            self._executor.submit(server._detach_session_transport, member[1], member[0])

    def _reply(self, frame, **payload):
        self.host.emit({"type": "conditional.ack", "request_id": frame.get("request_id"),
                        "boot_id": self.host._boot_id, **payload})

    @contextlib.contextmanager
    def _cut(self, server, session, params, peer):
        from pathlib import Path
        from sqlite3 import Error as SQLiteError

        sid = params["session_id"]
        with server._session_resume_lock, session["history_lock"], server._sessions_lock, server._session_transport_lock:
            with server._activation_engine_guard(session) as proven:
                home = server._profile_home(params.get("profile"))
                store = str((Path(home or server._hermes_home) / "state.db").resolve())
                receipt = server._bound_activation_receipt(session, sid, server._transport_auth_user_id(peer),
                                                           store, params["expected_binding"])
                if not proven or receipt is None or server._sessions.get(sid) is not session or peer._closed:
                    yield False
                    return
                try:
                    tip = session["creation_engine"].database.resolve_resume_session_id(
                        params["expected_binding"]["stored_session_id"], strict=True)
                except (OSError, SQLiteError):
                    tip = None
                yield tip == params["expected_binding"]["stored_session_id"]

    def _resolve(self, server, frame):
        params = frame.get("params")
        operation = isinstance(params, dict) and "operation" in params
        model = SessionInvokeBoundParams if operation else SessionActivateBoundParams
        try:
            params = model.model_validate(params).model_dump(exclude_unset=True)
        except ValidationError:
            return None
        if frame.get("boot_id") != self.host._boot_id:
            return None
        sid = params["session_id"]
        with self._lock:
            session = self._origins.get(sid)
            member = self._members.get(frame.get("subscription"))
            if operation:
                if member is None or member[1] is not session:
                    return None
                peer = member[0]
                subscription = frame["subscription"]
            elif member is not None and member[1] is session:
                peer, subscription = member[0], frame["subscription"]
            else:
                if len(self._members) >= 64:
                    return None
                peer = HostBoundPeer(params["expected_binding"]["authenticated_owner"])
                subscription = uuid.uuid4().hex
        if session is None:
            return None
        return session, params, peer, subscription, operation

    def _reserve(self, frame):
        from . import server

        reservation = uuid.uuid4().hex
        pending = {"event": threading.Event(), "commit": None}
        try:
            resolved = self._resolve(server, frame)
            if resolved is None:
                self._reply(frame, error=4007)
                return
            session, params, peer, subscription, operation = resolved
            pending.update(peer=peer, subscription=subscription)
            token = bind_transport(peer)
            try:
                with self._cut(server, session, params, peer) as valid:
                    if not valid:
                        self._reply(frame, error=4007)
                        return
                    with self._lock:
                        if not operation and subscription not in self._members and len(self._members) >= 64:
                            self._reply(frame, error=4007)
                            return
                    if (operation and params["operation"]["method"] == "prompt.submit"
                            and (session.get("running") or session.get("attached_images"))):
                        self._reply(frame, error=4009)
                        return
                    with self._lock:
                        self._pending[reservation] = pending
                    deadline = time.monotonic() + RESERVATION_SECONDS
                    self._reply(frame, reservation=reservation, subscription=subscription)
                    pending["event"].wait(RESERVATION_SECONDS)
                    commit = pending["commit"]
                    if commit is None:
                        return  # abandoned reservation changes no subscription or execution
                    if (time.monotonic() >= deadline or commit.get("boot_id") != self.host._boot_id
                            or peer._closed or self.host._closed.is_set()):
                        self._reply(commit, error=4007)
                        return
                    if operation and params["operation"]["method"] == "prompt.submit":
                        peer.conditional_turn_admission = self._admission_check(params, frame["turn_id"])
                    method = "session.invoke_bound" if operation else "session.activate_bound"
                    response = server._methods[method](commit["request_id"], params)
                    if not operation and "result" in response:
                        with self._lock:
                            if not peer._closed:
                                self._members[subscription] = (peer, session)
                    self._reply(commit, response=response, subscription=subscription)
                    if operation and params["operation"]["method"] == "prompt.submit" and "result" in response:
                        future = self.host._executor.submit(self._watch_turn, session, params["session_id"], frame["turn_id"])
                        self.host._track_turn_future(future, params["session_id"])
            finally:
                reset_transport(token)
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Conditional host reservation failed")
            self._reply(pending["commit"] or frame, error=5019 if pending["commit"] else 4007)
        finally:
            with self._lock:
                self._pending.pop(reservation, None)
            self._slots.release()

    def _admission_check(self, params, turn_id):
        checked = None
        def verify():
            nonlocal checked
            if checked is not None:
                return checked  # both tip checks are in the same engine transaction
            admission = {"event": threading.Event(), "allowed": False}
            with self._lock:
                self._admissions[turn_id] = admission
            try:
                self.host.emit({"type": "conditional.admit", "sid": params["session_id"],
                                "turn_id": turn_id, "boot_id": self.host._boot_id})
                admission["event"].wait(6.0)
                checked = admission["allowed"] and not self.host._closed.is_set()
                return checked
            finally:
                with self._lock:
                    self._admissions.pop(turn_id, None)
        return verify

    def _watch_turn(self, session, sid, turn_id):
        from .compute_host import _history_meta

        # prompt.submit can replace its first worker with the production turn worker.
        while True:
            worker = session.get("_run_thread")
            if worker is not None:
                worker.join()
            if session.get("_run_thread") is worker:
                break
        with session["history_lock"]:
            meta = _history_meta(session)
        self.host._reply("turn.end", sid, turn_id, **meta, interrupted=bool(session.get("_turn_cancel_requested")))

    def close(self):
        with self._lock:
            for pending in self._pending.values():
                pending["event"].set()
            for admission in self._admissions.values():
                admission["event"].set()
            for peer, _session in self._members.values():
                peer.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
