"""Owner-session RPC and runtime binding for classic producer custody."""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from contextvars import ContextVar

from gateway.classic_output_exports import ClassicExports
from gateway.hosted_room_artifacts import RoomArtifactError
from gateway.hosted_rooms import local_authority_gateway_id


@dataclass
class Admission:
    store: ClassicExports
    row: dict
    session: dict | None = None


_active: ContextVar[Admission | None] = ContextVar("classic_export_admission", default=None)


def bind(session, admission):
    return _active.set(Admission(admission.store, admission.row, session) if isinstance(admission, Admission) else None)


def reset(token):
    _active.reset(token)


def plumbing(session):
    if session.get("source") == "bot_room":
        return False
    if session.get("room_plumbing") is True:
        return True
    from tui_gateway import server
    with server._session_db(session) as db:
        row = db.get_session(session["session_key"]) if db else None
    config = (row or {}).get("model_config") or {}
    if isinstance(config, str):
        config = json.loads(config)
    return isinstance(config, dict) and config.get("room_plumbing") is True


def install_schema(session):
    agent = session.get("agent")
    if agent is None or getattr(agent, "_classic_export_schema_checked", False):
        return
    agent._classic_export_schema_checked = True
    try:
        eligible = plumbing(session)
    except Exception:
        eligible = False  # Missing metadata disables export, not ordinary chat construction.
    if eligible:
        from tools.hosted_room_artifact import ensure_share_group_file_tool
        agent._classic_export_enabled = ensure_share_group_file_tool(agent, force=True)


def owned(session_id):
    from tui_gateway import server
    transport, session = server._current_session_steer_authority(session_id)
    if transport is None or session is None:
        raise RoomArtifactError("Classic exports require the current session owner")
    return session


def store_for(session):
    from tui_gateway import server
    return ClassicExports(server._session_home(session))


def preflight(sid, session, request, text):
    if owned(sid) is not session or not plumbing(session):
        raise RoomArtifactError("Classic exports require an owned group-plumbing session")
    agent = session.get("agent")
    if agent is not None and not getattr(agent, "_classic_export_enabled", False):
        raise RoomArtifactError("Reopen this group session on the updated backend to enable file sharing")
    if not isinstance(request, dict):
        raise RoomArtifactError("Invalid classic export request")
    store = store_for(session)
    previous = store.prior(session["session_key"], request.get("request_id"))
    if previous:
        row, _ = store.admit(session["session_key"], request, text)
        return {"status": "accepted", "classic_export": store.status(row["export_id"])}
    return None


def admit(session, request, text):
    store = store_for(session)
    row, fresh = store.admit(session["session_key"], request, text)
    if fresh:
        session["_classic_export_admission"] = Admission(store, row)
    return fresh, store.status(row["export_id"])


def active_scope():
    from hermes_constants import get_hermes_home
    admission = _active.get()
    if (admission is not None and not (admission.session or {}).get("_turn_cancel_requested")
            and str(get_hermes_home().resolve()) == admission.store.home):
        return admission.store.scope(admission.row)
    return None


def settle(session, text, success):
    admission = session.get("_classic_export_admission")
    active = _active.get()
    if isinstance(admission, Admission) and active is not None and active.row["export_id"] == admission.row["export_id"]:
        admission.store.settle(admission.row["export_id"], text,
                               success and not session.get("_turn_cancel_requested"))


def finish(session):
    admission = session.get("_classic_export_admission")
    if isinstance(admission, Admission):
        if admission.store.lookup(admission.row["export_id"])["state"] == "running":
            admission.store.retire(admission.row["export_id"])
        session.pop("_classic_export_admission", None)


def abort_before_run(session):
    try:
        finish(session)
    except Exception:
        logging.getLogger(__name__).exception("Classic start failed; durable cleanup remains pending, never replay the request")


def register(server):
    def read(rid, params):
        try:
            session = owned(params.get("session_id"))
            if params.get("installation") != local_authority_gateway_id():
                raise RoomArtifactError("Classic export installation changed")
            store = store_for(session)
            export_id = params.get("export_id")
            if not export_id:
                # Exact lost-response recovery, scoped to this owned durable session.
                row = store.prior(session["session_key"], params.get("request_id"))
                if not row:
                    with store.outbox._connect() as conn:
                        candidates = conn.execute("SELECT * FROM classic_output_exports WHERE profile_home=? AND request_id=? LIMIT 2",
                                                  (store.home, params.get("request_id"))).fetchall()
                    with server._session_db(session) as db:
                        matches = [dict(item) for item in candidates if db and
                                   db.get_compression_tip(item["session_key"]) == session["session_key"]]
                    row = matches[0] if len(matches) == 1 else None
                if not row:
                    raise RoomArtifactError("Classic export request is unknown; do not assume it ran")
                export_id = row["export_id"]
            result = store.status(export_id)
            if params.get("group_id") != result["group_id"]:
                raise RoomArtifactError("Classic export group changed")
            if params.get("artifact_id"):
                metadata, data = store.read(export_id, params["artifact_id"])
                if owned(params.get("session_id")) is not session:
                    raise RoomArtifactError("Classic export owner changed")
                result.update(item=metadata, content_base64=base64.b64encode(data).decode("ascii"))
            return server._ok(rid, result)
        except (RoomArtifactError, ValueError) as exc:
            return server._err(rid, 4150, str(exc))

    def discard(rid, params):
        try:
            store = store_for(owned(params.get("session_id")))
            if params.get("installation") != local_authority_gateway_id():
                raise RoomArtifactError("Classic export installation changed")
            if params.get("export_id"):
                if store.status(params["export_id"])["group_id"] != params.get("group_id"):
                    raise RoomArtifactError("Classic export group changed")
                store.retire(params["export_id"])
            else:
                store.retire_group(params.get("group_id"))
            return server._ok(rid, {"retired": True})
        except (RoomArtifactError, ValueError) as exc:
            return server._err(rid, 4150, str(exc))

    server._methods.update({"session.export.read": read, "session.export.discard": discard})
    server._LONG_HANDLERS |= {"session.export.read", "session.export.discard"}
