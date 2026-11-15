"""Cross-surface sibling-serve prompt forwarding & live event relay (CF-68).

When a user switches between Mobile (``ai.hermes.serve`` on ``:9119``, ``ROLE_SERVE``)
and Desktop (``hermes serve --isolated``, ``ROLE_DESKTOP_SERVE``) on the same host
while a turn is actively running on the other surface, ``_ensure_active_session_slot``
previously failed with ``SESSION_NOT_OWNED`` ("This chat is open in another Hermes
window/terminal. Use it there, or start a new chat here.").

Instead of refusing the prompt when the active lease is held by a live sibling ``serve``
process on the same host:
1. We dial the sibling ``serve`` owner over loopback WebSocket (``/api/ws?token=...``)
   using the credentials published in ``gateway.host_rendezvous`` (``ROLE_SERVE`` /
   ``ROLE_DESKTOP_SERVE`` or ``serve-peer-<pid>.json``).
2. We call ``session.resume(session_id=key, omit_messages=True)`` on the sibling owner:
   - If the sibling's session is already idle (``running == False`` and ``not queued``),
     we reclaim the cross-surface lease in ``active_sessions.json``, sync out-of-band
     turns from ``state.db``, and return ``SIBLING_LEASE_RECLAIMED`` so the caller runs
     the new turn natively in this process.
   - If the sibling's session is actively mid-turn (``running == True``), we forward
     ``prompt.submit`` over that loopback WebSocket so the sibling's ``_handle_busy_submit``
     redirects/steers/queues the user's message into the live turn without killing
     in-flight work, and we relay the sibling's stream events (``message.delta``,
     ``tool.start``, ``tool.complete``, ``tool.progress``, ``todo.update``, ``status.update``,
     ``session.info``, ``message.complete``, ``error``) onto the local ``sid`` until the
     turn settles, then sync ``session["history"]`` via ``_adopt_out_of_band_turns``.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SIBLING_LEASE_RECLAIMED = object()
_ADMIT_TIMEOUT_S = 15.0
_RELAY_IDLE_TIMEOUT_S = 1800.0
_SETTLE_CHECK_DELAY_S = 0.15


def _peer_record_path(pid: int) -> Path | None:
    try:
        from gateway import host_rendezvous as hr
        return hr.host_state_dir() / f"serve-peer-{int(pid)}.json"
    except Exception:
        return None


def register_local_serve_peer(*, host: str, port: int, token: str) -> None:
    """Publish a 0600 per-PID peer file so any sibling ``serve`` can dial this process."""
    if not port or not token:
        return
    try:
        from gateway import host_rendezvous as hr
        from utils import atomic_json_write
        state_dir = hr.ensure_host_state_dir()
        path = state_dir / f"serve-peer-{os.getpid()}.json"
        dial_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else str(host or "127.0.0.1")
        atomic_json_write(path, {"pid": os.getpid(), "host": dial_host, "port": int(port), "token": str(token)})
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)

        def _cleanup() -> None:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)

        atexit.register(_cleanup)
    except Exception:
        logger.debug("Failed to publish serve-peer record for pid=%s", os.getpid(), exc_info=True)


def discover_sibling_serve_endpoint(owner_pid: int) -> tuple[str, int, str] | None:
    """Return ``(host, port, token)`` for a live sibling ``serve`` process ``owner_pid`` on this host."""
    if not isinstance(owner_pid, int) or owner_pid <= 0 or owner_pid == os.getpid():
        return None
    try:
        from gateway import host_rendezvous as hr
        for role in (hr.ROLE_SERVE, hr.ROLE_DESKTOP_SERVE):
            rec = hr.read_record(role)
            if rec is not None and rec.pid == owner_pid and not hr.record_is_stale(rec) and rec.port:
                tok = hr.read_token(role)
                if tok:
                    host = "127.0.0.1" if rec.host in ("0.0.0.0", "::", "") else str(rec.host or "127.0.0.1")
                    return (host, int(rec.port), tok)
    except Exception:
        logger.debug("host_rendezvous lookup failed for pid=%s", owner_pid, exc_info=True)

    peer_path = _peer_record_path(owner_pid)
    if peer_path is not None and peer_path.is_file():
        try:
            st = peer_path.stat()
            if hasattr(os, "getuid") and st.st_uid != os.getuid():
                return None
            data = json.loads(peer_path.read_text(encoding="utf-8"))
            port = int(data.get("port") or 0)
            tok = str(data.get("token") or "").strip()
            host = str(data.get("host") or "127.0.0.1").strip() or "127.0.0.1"
            if 0 < port <= 65535 and tok:
                return (host, port, tok)
        except Exception:
            logger.debug("serve-peer lookup failed for pid=%s", owner_pid, exc_info=True)
    return None


def _find_lease_owner_entry(key: str, profile_home: str | Path | None) -> dict | None:
    if not key:
        return None
    try:
        from hermes_cli.active_sessions import active_session_registry_snapshot
        snap = active_session_registry_snapshot(registry_home=profile_home)
        entries = snap.get("entries") if isinstance(snap, dict) else snap
        for entry in entries or ():
            if isinstance(entry, dict) and str(entry.get("session_id") or "") == key:
                return entry
    except Exception:
        logger.debug("active_session_registry_snapshot failed for %s", key, exc_info=True)
    return None


def force_take_over_cross_surface_lease(
    server, sid: str, session: dict, key: str, *, surface: str, expected_owner_pid: int,
) -> bool:
    """Take over a cross-surface lease once the sibling owner confirmed ``running == False``."""
    if not key:
        return False
    try:
        from hermes_cli.active_sessions import (
            ActiveSessionLease, _FileLock, _lease_entry, _lease_paths, _read_live_entries, _write_entries,
        )
        profile_home = session.get("profile_home")
        state_path, lock_path = _lease_paths(registry_home=profile_home)
        track_liveness = str(surface or "").strip().lower() in {"desktop", "mobile"}
        with _FileLock(lock_path):
            loaded = _read_live_entries(
                state_path, track_liveness=track_liveness,
                warn="Active-session registry unavailable during sibling lease handover",
                target_session_id=key,
            )
            if loaded is None:
                return False
            entries = loaded[1]
            idx = next((i for i, e in enumerate(entries) if str(e.get("session_id") or "") == key), None)
            existing = entries[idx] if idx is not None else None
            if existing is not None and int(existing.get("pid") or 0) not in (0, expected_owner_pid, os.getpid()):
                return False
            lease_id = uuid.uuid4().hex
            new_meta = server._lease_metadata(sid, turn_active=True)
            new_entry = _lease_entry(
                lease_id=lease_id, session_id=key, surface=surface, metadata=new_meta,
                track_liveness=track_liveness,
            )
            if idx is not None:
                entries[idx] = new_entry
            else:
                entries.append(new_entry)
            _write_entries(state_path, entries)
            lease = ActiveSessionLease(
                lease_id=lease_id, session_id=key, surface=str(surface), state_path=state_path,
                lock_path=lock_path, track_liveness=track_liveness,
            )
            server._attach_lease(session, lease)
        logger.info(
            "Session %s (%s) reclaimed cross-surface lease for %s after sibling pid=%s confirmed idle",
            sid, surface, key, expected_owner_pid,
        )
        return True
    except Exception:
        logger.debug("force_take_over_cross_surface_lease failed for %s", key, exc_info=True)
        return False


class _SiblingServeRelay:
    """Loopback WebSocket bridge that submits a prompt to a running sibling ``serve`` and relays its stream."""

    def __init__(
        self,
        server,
        *,
        sid: str,
        session: dict,
        key: str,
        owner_pid: int,
        host: str,
        port: int,
        token: str,
        text: str,
        params: dict,
        display_kind: str | None,
    ) -> None:
        self.server = server
        self.sid = sid
        self.session = session
        self.key = key
        self.owner_pid = owner_pid
        self.host = host
        self.port = port
        self.token = token
        self.text = text
        self.params = dict(params or {})
        self.display_kind = display_kind
        self.owner_sid = ""
        self.admitted: concurrent.futures.Future = concurrent.futures.Future()
        self._control_queue: list[tuple[str, dict, concurrent.futures.Future | None]] = []
        self._pending_control_rpcs: dict[int, concurrent.futures.Future] = {}
        self._control_lock = threading.Lock()
        self._next_rpc_id = 10

    def start(self) -> None:
        threading.Thread(
            target=self._thread_main,
            name=f"sibling-serve-relay-{self.sid[:8]}",
            daemon=True,
        ).start()

    def request_interrupt(self) -> bool:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        with self._control_lock:
            self._control_queue.append(("session.interrupt", {}, fut))
        try:
            return bool(fut.result(timeout=3.0))
        except Exception:
            return True

    def request_correction(self, method_name: str, text: str) -> bool:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        with self._control_lock:
            self._control_queue.append((method_name, {"text": text}, fut))
        try:
            return bool(fut.result(timeout=3.0))
        except Exception:
            return False

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            if not self.admitted.done():
                self.admitted.set_result(None)
            logger.debug("Sibling serve relay exited for %s: %s", self.key, exc, exc_info=True)

    async def _flush_control_queue(self, ws) -> None:
        with self._control_lock:
            pending = list(self._control_queue)
            self._control_queue.clear()
        for method_name, extra_params, fut in pending:
            try:
                self._next_rpc_id += 1
                rpc_id = self._next_rpc_id
                req = {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "method": method_name,
                    "params": {"session_id": self.owner_sid, **extra_params},
                }
                if fut is not None:
                    self._pending_control_rpcs[rpc_id] = fut
                await ws.send(json.dumps(req) + "\n")
            except Exception:
                if fut is not None and not fut.done():
                    fut.set_result(False)

    def _resolve_control_rpc(self, msg: dict) -> bool:
        rpc_id = msg.get("id")
        if not isinstance(rpc_id, int) or rpc_id not in self._pending_control_rpcs:
            return False
        fut = self._pending_control_rpcs.pop(rpc_id, None)
        if fut is not None and not fut.done():
            if "error" in msg:
                fut.set_result(False)
            else:
                res = msg.get("result") or {}
                ok = isinstance(res, dict) and str(res.get("status") or "") != "rejected"
                fut.set_result(ok)
        return True

    async def _run(self) -> None:
        from websockets.asyncio.client import connect

        url = f"ws://{self.host}:{self.port}/api/ws?token={self.token}"
        profile_name = str(self.session.get("profile_name") or "").strip()
        admitted_ok = False
        final_complete_payload: dict | None = None
        seen_target_turn = True
        buffered_events: list[dict] = []

        try:
            async with connect(url, open_timeout=5.0, proxy=None, max_size=None) as ws:
                await asyncio.wait_for(ws.recv(), timeout=5.0)
                resume_params: dict[str, Any] = {"session_id": self.key, "omit_messages": True}
                if profile_name:
                    resume_params["profile"] = profile_name
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session.resume", "params": resume_params}) + "\n")
                resumed: dict[str, Any] | None = None
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), timeout=max(0.5, deadline - time.monotonic()))
                    msg = json.loads(raw)
                    if msg.get("id") == 1:
                        if "error" in msg:
                            self.admitted.set_result(None)
                            return
                        resumed = msg.get("result") or {}
                        break
                if not isinstance(resumed, dict):
                    self.admitted.set_result(None)
                    return
                self.owner_sid = str(resumed.get("session_id") or "")
                owner_running = bool(resumed.get("running"))
                owner_queued = bool(resumed.get("queued"))
                if not self.owner_sid:
                    self.admitted.set_result(None)
                    return
                if not owner_running and not owner_queued:
                    self.admitted.set_result({"reclaimed": True, "owner_pid": self.owner_pid})
                    return

                submit_params: dict[str, Any] = {"session_id": self.owner_sid, "text": self.text}
                if self.params.get("queued"):
                    submit_params["queued"] = True
                if self.display_kind == "hidden":
                    submit_params["display_kind"] = "hidden"
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "prompt.submit", "params": submit_params}) + "\n")

                submit_res: dict[str, Any] | None = None
                deadline = time.monotonic() + _ADMIT_TIMEOUT_S
                while time.monotonic() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), timeout=max(0.5, deadline - time.monotonic()))
                    msg = json.loads(raw)
                    if msg.get("id") == 2:
                        if "error" in msg:
                            err_obj = msg.get("error") or {}
                            self.admitted.set_result({"error": err_obj})
                            return
                        submit_res = msg.get("result") or {}
                        break
                    if msg.get("method") == "event":
                        buffered_events.append(msg)

                if not isinstance(submit_res, dict):
                    self.admitted.set_result(None)
                    return

                submit_status = str(submit_res.get("status") or "streaming")
                seen_target_turn = submit_status != "queued"
                with self.session["history_lock"]:
                    self.session["running"] = True
                    self.session["_sibling_serve_relay"] = self
                    self.session["last_active"] = time.time()
                admitted_ok = True
                self.admitted.set_result(submit_res)

                self.server._emit("message.start", self.sid)
                if not seen_target_turn:
                    self.server._emit(
                        "status.update",
                        self.sid,
                        {"kind": "process", "text": "Queued on active turn in sibling Hermes window…"},
                    )

                def _forward_event(ev_msg: dict) -> dict | None:
                    nonlocal seen_target_turn
                    p = ev_msg.get("params") or {}
                    if str(p.get("session_id") or "") != self.owner_sid:
                        return None
                    ev_type = str(p.get("type") or "")
                    payload = {k: v for k, v in p.items() if k not in ("type", "session_id")}
                    if ev_type == "message.start":
                        seen_target_turn = True
                        return None
                    if ev_type == "message.complete":
                        return payload
                    if ev_type == "message.delta":
                        if seen_target_turn:
                            self.server._emit("message.delta", self.sid, payload)
                        return None
                    if ev_type in {
                        "tool.start", "tool.complete", "tool.progress",
                        "todo.update", "status.update", "session.info", "error",
                    }:
                        self.server._emit(ev_type, self.sid, payload)
                    return None

                for buffered in buffered_events:
                    if (done_payload := _forward_event(buffered)) is not None:
                        final_complete_payload = done_payload
                buffered_events.clear()

                relay_deadline = time.monotonic() + _RELAY_IDLE_TIMEOUT_S
                while time.monotonic() < relay_deadline:
                    await self._flush_control_queue(ws)
                    if final_complete_payload is not None:
                        await asyncio.sleep(_SETTLE_CHECK_DELAY_S)
                        self._next_rpc_id += 1
                        chk_id = self._next_rpc_id
                        await ws.send(
                            json.dumps({"jsonrpc": "2.0", "id": chk_id, "method": "session.resume", "params": resume_params}) + "\n"
                        )
                        chk_res: dict[str, Any] | None = None
                        chk_deadline = time.monotonic() + 5.0
                        while time.monotonic() < chk_deadline:
                            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.5, chk_deadline - time.monotonic()))
                            msg = json.loads(raw)
                            if self._resolve_control_rpc(msg):
                                continue
                            if msg.get("id") == chk_id:
                                chk_res = msg.get("result") or {}
                                break
                            if msg.get("method") == "event":
                                if (done2 := _forward_event(msg)) is not None:
                                    final_complete_payload = done2
                        if isinstance(chk_res, dict) and (chk_res.get("running") or chk_res.get("queued")):
                            seen_target_turn = True
                            final_complete_payload = None
                            continue
                        break

                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.25)
                    except TimeoutError:
                        continue
                    msg = json.loads(raw)
                    if self._resolve_control_rpc(msg):
                        continue
                    if msg.get("method") == "event":
                        if (done_payload := _forward_event(msg)) is not None:
                            final_complete_payload = done_payload
        finally:
            for fut in list(self._pending_control_rpcs.values()):
                if not fut.done():
                    fut.set_result(False)
            self._pending_control_rpcs.clear()
            if not self.admitted.done():
                self.admitted.set_result(None)
            if admitted_ok:
                with contextlib.suppress(Exception):
                    self.server._adopt_out_of_band_turns(self.session)
                with self.session["history_lock"]:
                    if self.session.get("_sibling_serve_relay") is self:
                        self.session.pop("_sibling_serve_relay", None)
                        self.session["running"] = False
                    self.session["last_active"] = time.time()
                self.server._emit(
                    "message.complete",
                    self.sid,
                    final_complete_payload if isinstance(final_complete_payload, dict) else {"text": ""},
                )
                with contextlib.suppress(Exception):
                    self.server._emit_session_info(self.sid, self.session)


def try_submit_to_sibling_serve_owner(
    server, rid, sid: str, session: dict, text: Any, params: dict, display_kind: str | None = None,
) -> Any:
    """Forward a cross-surface ``prompt.submit`` to the sibling ``serve`` process holding the lease.

    Returns:
    - ``SIBLING_LEASE_RECLAIMED`` when the sibling owner confirmed its turn already finished and
      this session took over the lease in ``active_sessions.json`` (caller should proceed with
      local turn execution).
    - A JSON-RPC response dict (``_ok(rid, {"status": "streaming"})``) when the sibling owner
      accepted the prompt into its live turn and the background relay is streaming events onto ``sid``.
    - ``None`` when no reachable sibling ``serve`` owner exists (caller should return normal refusal).
    """
    if not isinstance(text, str) or not text.strip():
        return None
    if session.get("attached_images"):
        return None
    truncation_keys = getattr(server, "_TRUNCATION_PARAMS", ())
    if any(params.get(k) is not None for k in truncation_keys):
        return None

    with session["history_lock"]:
        existing_relay = session.get("_sibling_serve_relay")
    if isinstance(existing_relay, _SiblingServeRelay):
        if existing_relay.request_correction("session.redirect", text.strip()):
            return server._ok(rid, {"status": "streaming"})

    key = str(session.get("session_key") or "")
    profile_home = session.get("profile_home")
    surface = "mobile" if server._is_mobile_session(session) else server._session_source(session)

    owner_entry = _find_lease_owner_entry(key, profile_home)
    if owner_entry is None:
        if server._ensure_active_session_slot(sid, session) is None:
            with contextlib.suppress(Exception):
                server._adopt_out_of_band_turns(session)
            return SIBLING_LEASE_RECLAIMED
        return None

    owner_pid = int(owner_entry.get("pid") or 0)
    endpoint = discover_sibling_serve_endpoint(owner_pid)
    if endpoint is None:
        return None
    host, port, token = endpoint

    with server._session_resume_lock:
        if (t := server.current_transport()) is not None:
            server._rebind_live_transport(sid, session, t)

    relay = _SiblingServeRelay(
        server,
        sid=sid,
        session=session,
        key=key,
        owner_pid=owner_pid,
        host=host,
        port=port,
        token=token,
        text=text,
        params=params,
        display_kind=display_kind,
    )
    relay.start()
    try:
        outcome = relay.admitted.result(timeout=_ADMIT_TIMEOUT_S + 2.0)
    except Exception:
        logger.debug("Timed out waiting for sibling serve admission on %s (pid=%s)", key, owner_pid, exc_info=True)
        return None

    if not isinstance(outcome, dict):
        return None
    if outcome.get("reclaimed"):
        if force_take_over_cross_surface_lease(
            server, sid, session, key, surface=surface, expected_owner_pid=owner_pid,
        ):
            with contextlib.suppress(Exception):
                server._adopt_out_of_band_turns(session)
            return SIBLING_LEASE_RECLAIMED
        return None
    if "error" in outcome:
        return None

    logger.info(
        "Forwarded cross-surface prompt.submit for %s from %s (%s) to sibling serve pid=%s (:%s) -> status=%s",
        key, sid, surface, owner_pid, port, outcome.get("status"),
    )
    return server._ok(rid, {"status": "streaming"})


def try_interrupt_sibling_serve_owner(server, session: dict) -> bool:
    """Forward ``session.interrupt`` to a sibling ``serve`` process holding an active lease for ``session``."""
    with session["history_lock"]:
        relay = session.get("_sibling_serve_relay")
    if isinstance(relay, _SiblingServeRelay):
        return relay.request_interrupt()

    key = str(session.get("session_key") or "")
    owner_entry = _find_lease_owner_entry(key, session.get("profile_home"))
    if not isinstance(owner_entry, dict):
        return False
    meta = owner_entry.get("metadata") or {}
    if meta.get("turn_active") is not True:
        return False
    owner_pid = int(owner_entry.get("pid") or 0)
    endpoint = discover_sibling_serve_endpoint(owner_pid)
    if endpoint is None:
        return False
    host, port, token = endpoint
    profile_name = str(session.get("profile_name") or "").strip()

    async def _send_interrupt() -> bool:
        from websockets.asyncio.client import connect
        url = f"ws://{host}:{port}/api/ws?token={token}"
        async with connect(url, open_timeout=3.0, proxy=None) as ws:
            await asyncio.wait_for(ws.recv(), timeout=3.0)
            resume_params: dict[str, Any] = {"session_id": key, "omit_messages": True}
            if profile_name:
                resume_params["profile"] = profile_name
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session.resume", "params": resume_params}) + "\n")
            owner_sid = ""
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                if msg.get("id") == 1:
                    res = msg.get("result") or {}
                    owner_sid = str(res.get("session_id") or "")
                    break
            if not owner_sid:
                return False
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "session.interrupt", "params": {"session_id": owner_sid}}) + "\n")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                if msg.get("id") == 2:
                    return "result" in msg
        return False

    try:
        return asyncio.run(_send_interrupt())
    except Exception:
        logger.debug("try_interrupt_sibling_serve_owner failed for %s (pid=%s)", key, owner_pid, exc_info=True)
        return False


def try_correct_sibling_serve_owner(server, session: dict, method_name: str, text: str) -> bool:
    """Forward ``session.steer`` or ``session.redirect`` to a sibling ``serve`` holding an active lease."""
    with session["history_lock"]:
        relay = session.get("_sibling_serve_relay")
    if isinstance(relay, _SiblingServeRelay):
        return relay.request_correction(method_name, text)

    key = str(session.get("session_key") or "")
    owner_entry = _find_lease_owner_entry(key, session.get("profile_home"))
    if not isinstance(owner_entry, dict):
        return False
    meta = owner_entry.get("metadata") or {}
    if meta.get("turn_active") is not True:
        return False
    owner_pid = int(owner_entry.get("pid") or 0)
    endpoint = discover_sibling_serve_endpoint(owner_pid)
    if endpoint is None:
        return False
    host, port, token = endpoint
    profile_name = str(session.get("profile_name") or "").strip()

    async def _send_correction() -> bool:
        from websockets.asyncio.client import connect
        url = f"ws://{host}:{port}/api/ws?token={token}"
        async with connect(url, open_timeout=3.0, proxy=None) as ws:
            await asyncio.wait_for(ws.recv(), timeout=3.0)
            resume_params: dict[str, Any] = {"session_id": key, "omit_messages": True}
            if profile_name:
                resume_params["profile"] = profile_name
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session.resume", "params": resume_params}) + "\n")
            owner_sid = ""
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                if msg.get("id") == 1:
                    res = msg.get("result") or {}
                    owner_sid = str(res.get("session_id") or "")
                    break
            if not owner_sid:
                return False
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": method_name, "params": {"session_id": owner_sid, "text": text}}) + "\n")
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                if msg.get("id") == 2:
                    if "error" in msg:
                        return False
                    res = msg.get("result") or {}
                    return isinstance(res, dict) and str(res.get("status") or "") != "rejected"
        return False

    try:
        return asyncio.run(_send_correction())
    except Exception:
        logger.debug("try_correct_sibling_serve_owner failed for %s (pid=%s)", key, owner_pid, exc_info=True)
        return False
