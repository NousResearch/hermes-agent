"""Observer-only HTTP/SSE source router.

The router intentionally has its own bearer audience and imports only the
observer plane.  It is not a JSON-RPC adapter and has no controller methods.
"""
from __future__ import annotations

import hmac
import json
import os
import queue
from .observer import ObserverPlane


def _source_token() -> str:
    return str(os.environ.get("HERMES_OBSERVER_SOURCE_TOKEN") or "").strip()


def observer_token_is_valid(value: str | None) -> bool:
    expected = _source_token()
    return bool(expected and value and hmac.compare_digest(str(value), expected))


def create_observer_router(plane: ObserverPlane):
    """Build a FastAPI router for the Hermes-side observer source routes."""
    try:
        from fastapi import APIRouter, Header, HTTPException, Query
        from fastapi.responses import StreamingResponse
    except ImportError as exc:  # pragma: no cover - FastAPI is a server dependency
        raise RuntimeError("FastAPI is required for the observer router") from exc

    router = APIRouter(prefix="/observer/v1", tags=["observer"])

    def authorize(authorization: str | None) -> None:
        scheme, _, token = str(authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not observer_token_is_valid(token):
            raise HTTPException(status_code=401, detail="observer authorization required")

    @router.get("/me")
    def me(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return {"schema": "iris.observe.me.v1", "audience": "hermes-observer", "scope": "read"}

    @router.get("/sessions")
    def sessions(limit: int = Query(default=20, ge=1, le=20), authorization: str | None = Header(default=None)):
        authorize(authorization)
        return plane.list_sessions(limit)

    @router.get("/sessions/{logical_session_id}/snapshot")
    def snapshot(logical_session_id: str, authorization: str | None = Header(default=None)):
        authorize(authorization)
        result = plane.snapshot(logical_session_id)
        if result is None:
            raise HTTPException(status_code=404, detail="observer session not found")
        return result

    @router.get("/sessions/{logical_session_id}/events")
    def events(logical_session_id: str, stream_id: str = "", after_seq: int = Query(default=0, ge=0), authorization: str | None = Header(default=None)):
        authorize(authorization)
        q, replay, gap = plane.subscribe(logical_session_id, after_seq, stream_id=stream_id or None)

        def body():
            try:
                if gap:
                    yield "event: observed\ndata: " + json.dumps({"schema": "iris.observe.event.v1", "kind": "gap", "payload": {"reason": gap, "rehydrate_required": True}}) + "\n\n"
                    return
                for record in replay:
                    yield "event: observed\ndata: " + json.dumps(record.as_dict(), ensure_ascii=False) + "\n\n"
                    if record.kind == "gap":
                        return
                while True:
                    try:
                        record = q.get(timeout=30)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    yield "event: observed\ndata: " + json.dumps(record.as_dict(), ensure_ascii=False) + "\n\n"
                    if record.kind == "gap":
                        return
            finally:
                plane.unsubscribe(logical_session_id, q)

        return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return router


__all__ = ["create_observer_router", "observer_token_is_valid"]