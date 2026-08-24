"""Runtime wiring for the Android turn-recovery v2 JSON-RPC contract.

The journal stays isolated in :mod:`tui_gateway.turn_recovery`; this module is
the narrow adapter around the existing session/prompt handlers. Legacy clients
continue through the original methods unchanged.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from .turn_recovery import (
    PROMPT_SUBMIT_VERSION,
    TurnJournal,
    TurnLimits,
    TurnRecoveryError,
    build_protocol_frame,
    build_turn_recovery_capability,
)


def _canonical_uuid(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError):
        return None
    canonical = str(parsed)
    if canonical != value:
        return None
    return canonical


def _canonical_v4_uuid(value: Any) -> Optional[str]:
    canonical = _canonical_uuid(value)
    if canonical is None or uuid.UUID(canonical).version != 4:
        return None
    return canonical


def _default_bindings_path() -> Path:
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
    except Exception:
        home = Path.home() / ".hermes"
    return home / "state" / "turn_recovery_bindings.json"


class TurnRecoveryRuntime:
    """Own bindings, method wrappers, and live-event journal decoration."""

    def __init__(
        self,
        server,
        *,
        bindings_path: Optional[Path] = None,
        limits: Optional[TurnLimits] = None,
    ) -> None:
        self.server = server
        self.limits = limits or TurnLimits()
        self.journal = TurnJournal(limits=self.limits)
        self.bindings_path = Path(bindings_path or _default_bindings_path())
        self._lock = threading.RLock()
        self._bindings = self._load_bindings()
        self._runtime_to_stored: dict[str, str] = {}
        self._active: dict[str, dict[str, str]] = {}
        self._original_prompt_submit = None
        self._original_session_interrupt = None

    def _load_bindings(self) -> dict[str, dict[str, Any]]:
        try:
            raw = json.loads(self.bindings_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        if not isinstance(raw, dict):
            return {}
        valid: dict[str, dict[str, Any]] = {}
        for mobile_id, entry in raw.items():
            if (
                _canonical_uuid(mobile_id)
                and isinstance(entry, dict)
                and isinstance(entry.get("stored_session_id"), str)
                and entry["stored_session_id"]
                and isinstance(entry.get("binding_version"), int)
                and not isinstance(entry["binding_version"], bool)
                and entry["binding_version"] > 0
            ):
                valid[mobile_id] = {
                    "stored_session_id": entry["stored_session_id"],
                    "binding_version": entry["binding_version"],
                }
        return valid

    def _save_bindings(self) -> None:
        self.bindings_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.bindings_path.with_suffix(self.bindings_path.suffix + ".tmp")
        payload = json.dumps(self._bindings, sort_keys=True, indent=2) + "\n"
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.bindings_path)

    def install(self) -> None:
        if self._original_prompt_submit is not None:
            return
        self._original_prompt_submit = self.server._methods["prompt.submit"]
        self._original_session_interrupt = self.server._methods["session.interrupt"]
        self.server._methods["session.open"] = self.session_open
        self.server._methods["prompt.submit"] = self.prompt_submit
        self.server._methods["turn.reconcile"] = self.turn_reconcile
        self.server._methods["turn.interrupt"] = self.turn_interrupt

    def ready_payload(self, base: dict[str, Any]) -> dict[str, Any]:
        payload = dict(base)
        payload["protocol"] = build_protocol_frame()
        capabilities = dict(payload.get("capabilities") or {})
        capabilities["turn_recovery"] = build_turn_recovery_capability(
            limits=self.limits,
            attachments_supported=False,
        )
        payload["capabilities"] = capabilities
        return payload

    def _error(self, rid, message: str, *, reason: str = "invalid_params") -> dict:
        return self.server._err(rid, -32602, message, data={"reason": reason})

    def session_open(self, rid, params: dict) -> dict:
        mobile_id = _canonical_uuid(params.get("mobile_session_id"))
        if mobile_id is None:
            return self._error(rid, "mobile_session_id must be a canonical lowercase UUID")

        with self._lock:
            previous = self._bindings.get(mobile_id)
            if previous is None:
                response = self.server._methods["session.create"](
                    rid,
                    {
                        "source": "android",
                        "close_on_disconnect": False,
                    },
                )
            else:
                response = self.server._methods["session.resume"](
                    rid,
                    {
                        "session_id": previous["stored_session_id"],
                        "source": "android",
                    },
                )
            if not isinstance(response, dict) or "error" in response:
                return response
            result = response.get("result") or {}
            runtime_id = result.get("session_id")
            # session.create reports the durable identity as ``stored_session_id``;
            # session.resume reports it as ``session_key`` (and may point at a
            # rotated compression-continuation tip rather than the id we asked
            # for). Accept either spelling, then fall back to the live session
            # record, so a reconnect can never be rejected as "invalid identity".
            stored_id = result.get("stored_session_id") or result.get("session_key")
            if not isinstance(stored_id, str) or not stored_id:
                live = self.server._sessions.get(runtime_id) if isinstance(runtime_id, str) else None
                if isinstance(live, dict):
                    candidate = live.get("session_key")
                    if isinstance(candidate, str) and candidate:
                        stored_id = candidate
            if not isinstance(runtime_id, str) or not runtime_id or not isinstance(stored_id, str) or not stored_id:
                return self.server._err(rid, -32603, "session binding returned an invalid identity")

            version = 1 if previous is None else int(previous["binding_version"]) + 1
            session = self.server._sessions.get(runtime_id)
            if previous is None and isinstance(session, dict):
                ensure_persisted = getattr(self.server, "_ensure_session_db_row", None)
                if callable(ensure_persisted):
                    try:
                        ensure_persisted(session)
                    except Exception as exc:
                        return self.server._err(
                            rid,
                            -32603,
                            f"failed to persist mobile session binding: {exc}",
                        )
            self._bindings[mobile_id] = {
                "stored_session_id": stored_id,
                "binding_version": version,
            }
            self._save_bindings()
            self._runtime_to_stored[runtime_id] = stored_id
            if isinstance(session, dict):
                session["_turn_recovery_mobile_session_id"] = mobile_id

            return self.server._ok(
                rid,
                {
                    "turn_recovery": True,
                    "automatic_resubmit": False,
                    "runtime_session_id": runtime_id,
                    "stored_session_id": stored_id,
                    "mobile_session_id": mobile_id,
                    "binding_version": version,
                    "capabilities": {
                        "turn_recovery": build_turn_recovery_capability(
                            limits=self.limits,
                            attachments_supported=False,
                        )
                    },
                },
            )

    @staticmethod
    def _ack(handle) -> dict[str, Any]:
        return {
            "accepted": True,
            "automatic_resubmit": False,
            "client_turn_id": handle.client_turn_id,
            "turn_id": handle.turn_id,
            "status": handle.status,
            "last_seq": handle.last_seq,
            "created": handle.created,
        }

    def prompt_submit(self, rid, params: dict) -> dict:
        if params.get("version") != PROMPT_SUBMIT_VERSION:
            return self._original_prompt_submit(rid, params)

        sid = params.get("session_id")
        client_turn_id = _canonical_v4_uuid(params.get("client_turn_id"))
        text = params.get("text")
        stored_id = self._runtime_to_stored.get(sid) if isinstance(sid, str) else None
        if stored_id is None or client_turn_id is None:
            return self._error(rid, "v2 submit requires an open runtime session and canonical client_turn_id")
        if not isinstance(text, str):
            return self._error(rid, "text must be a string")
        if len(text.encode("utf-8")) > self.limits.max_prompt_bytes:
            return self._error(rid, "prompt exceeds max_prompt_bytes")
        if params.get("attachments"):
            return self._error(rid, "v2 attachments are not enabled on this route")

        with self._lock:
            active = self._active.get(sid)
        if active is not None and active["client_turn_id"] != client_turn_id:
            return self.server._err(
                rid,
                4091,
                "another recoverable turn is active for this session",
                data={"reason": "turn_active"},
            )

        handle = self.journal.open_turn(
            session_id=stored_id,
            client_turn_id=client_turn_id,
        )
        if not handle.created:
            return self.server._ok(rid, self._ack(handle))

        active = {
            "stored_session_id": stored_id,
            "client_turn_id": client_turn_id,
            "turn_id": handle.turn_id,
            "message_id": str(uuid.uuid4()),
        }
        with self._lock:
            self._active[sid] = active
        session = self.server._sessions.get(sid)
        if isinstance(session, dict):
            session["_turn_recovery_active"] = active

        response = self._original_prompt_submit(rid, params)
        if not isinstance(response, dict) or "error" in response:
            self.journal.discard_turn(stored_id, client_turn_id)
            with self._lock:
                self._active.pop(sid, None)
            if isinstance(session, dict):
                session.pop("_turn_recovery_active", None)
            return response
        return self.server._ok(rid, self._ack(handle))

    def _resolve(self, params: dict):
        sid = params.get("session_id")
        stored_id = self._runtime_to_stored.get(sid) if isinstance(sid, str) else None
        if stored_id is None:
            raise TurnRecoveryError("runtime session has no mobile binding")
        client_turn_id = params.get("client_turn_id")
        turn_id = params.get("turn_id")
        if client_turn_id is not None:
            client_turn_id = _canonical_v4_uuid(client_turn_id)
            if client_turn_id is None:
                raise TurnRecoveryError("client_turn_id must be a canonical UUIDv4")
        turn = self.journal.resolve_turn(
            stored_id,
            client_turn_id=client_turn_id,
            turn_id=turn_id,
        )
        return sid, stored_id, turn

    def turn_reconcile(self, rid, params: dict) -> dict:
        try:
            _sid, stored_id, turn = self._resolve(params)
            page = self.journal.reconcile(
                stored_id,
                turn.client_turn_id,
                after_seq=params.get("after_seq"),
            )
            return self.server._ok(rid, page)
        except TurnRecoveryError as exc:
            reason = "turn_unknown" if "unknown turn" in str(exc) else "turn_replay_pruned"
            return self.server._err(rid, 4041, str(exc), data={"reason": reason})

    def turn_interrupt(self, rid, params: dict) -> dict:
        try:
            sid, stored_id, turn = self._resolve(params)
        except TurnRecoveryError as exc:
            return self.server._err(rid, 4041, str(exc), data={"reason": "turn_unknown"})

        legacy = self._original_session_interrupt(rid, {"session_id": sid})
        if not isinstance(legacy, dict) or "error" in legacy:
            return legacy
        result = self.journal.interrupt(stored_id, turn.client_turn_id)
        with self._lock:
            self._active.pop(sid, None)
        session = self.server._sessions.get(sid)
        if isinstance(session, dict):
            session.pop("_turn_recovery_active", None)
        return self.server._ok(
            rid,
            {
                "automatic_resubmit": False,
                "client_turn_id": turn.client_turn_id,
                "turn_id": result.turn_id,
                "status": result.status,
                "last_seq": result.last_seq,
            },
        )

    def decorate_event(self, event: str, sid: str, payload: Optional[dict]) -> dict:
        """Return event params, journal-enriched only for an active v2 turn."""

        base: dict[str, Any] = {"type": event, "session_id": sid}
        if payload is not None:
            base["payload"] = payload
        with self._lock:
            active = self._active.get(sid)
        if active is None:
            return base

        event_type = event
        source = payload if isinstance(payload, dict) else {}
        if event == "message.start":
            recovery_payload: dict[str, Any] = {}
        elif event == "message.delta":
            recovery_payload = {"text": str(source.get("text") or "")}
        elif event == "message.complete":
            status = {
                "complete": "completed",
                "completed": "completed",
                "error": "failed",
                "failed": "failed",
                "interrupted": "interrupted",
            }.get(source.get("status"), "failed")
            recovery_payload = {"text": str(source.get("text") or ""), "status": status}
        elif event == "error":
            event_type = "message.complete"
            message = str(source.get("message") or "turn failed")
            recovery_payload = {"text": f"Error: {message}", "status": "failed"}
        else:
            return base

        try:
            handle = self.journal.append_event(
                active["stored_session_id"],
                active["client_turn_id"],
                event_type,
                active["message_id"],
                recovery_payload,
            )
            if event_type == "message.start":
                self.journal.set_status(
                    active["stored_session_id"], active["client_turn_id"], "running"
                )
            elif event_type == "message.complete":
                self.journal.set_status(
                    active["stored_session_id"],
                    active["client_turn_id"],
                    recovery_payload["status"],
                )
                with self._lock:
                    self._active.pop(sid, None)
                session = self.server._sessions.get(sid)
                if isinstance(session, dict):
                    session.pop("_turn_recovery_active", None)
        except TurnRecoveryError:
            return base

        return {
            "type": event_type,
            "session_id": sid,
            "turn_id": handle.turn_id,
            "seq": handle.seq,
            "message_id": active["message_id"],
            "payload": recovery_payload,
        }


_runtime: Optional[TurnRecoveryRuntime] = None


def install(server) -> TurnRecoveryRuntime:
    global _runtime
    if _runtime is None or _runtime.server is not server:
        _runtime = TurnRecoveryRuntime(server)
        _runtime.install()
    return _runtime


def get_runtime() -> Optional[TurnRecoveryRuntime]:
    return _runtime