"""JSON-RPC admission and worker dispatch. Published onto the server namespace (``srv.dispatch``)."""

from __future__ import annotations

import contextvars
import logging
from .method_ctx import bind_module
from .transport import bind_transport, reset_transport

logger = logging.getLogger("tui_gateway.server")  # siblings log as the gateway facade (operators and caplog filter on it)


def handle_request(req: dict) -> dict | None:
    from hermes_cli.backend_retirement import retirement

    with retirement.work() as admitted:
        if not admitted:
            return srv._err(req.get("id"), 5035, "backend is retiring; reconnect to continue")
        return srv._handle_admitted_request(req)


def _handle_admitted_request(req: dict) -> dict | None:
    normalized = srv._normalize_request(req)
    if isinstance(normalized, dict):
        return normalized
    rid, method, params = normalized
    if not (fn := srv._methods.get(method)):
        return srv._err(rid, -32601, f"unknown method: {method} — the client and the Hermes backend are out of sync "
                    "(different versions); run `hermes update` and restart both")
    token = srv._current_rpc_method.set(method)
    try:
        return fn(rid, params)
    except srv.ProfileUnavailableError as exc:
        return srv._err(rid, 4064, str(exc))
    finally:
        srv._current_rpc_method.reset(token)


def _handler_error(req: dict, exc: Exception) -> dict:
    """One error frame for a handler crash, inline or pooled. The model boundary raises ``TypeError`` on a
    contract slip (a handler returning a dict, an emit handing a dict to a typed event); on stdio the
    entry loop writes whatever ``dispatch`` returns and has no catch of its own, so an unwound inline
    handler would end ``hermes --tui``. ``handle_request`` itself still raises: the pool worker and the
    tests want the exception, not a frame."""
    logger.exception("RPC handler crashed method=%r id=%r", req.get("method"), req.get("id"), exc_info=exc)
    return srv._err(req.get("id"), -32000, f"handler error: {exc}")


def dispatch(req: dict, transport: srv.Transport | None = None) -> dict | None:
    """Route inbound RPCs — long handlers to the pool (returns None; the worker writes its own
    response via the bound transport), everything else inline (returns the response dict).
    *transport* pins every write of this request — events included — to that transport;
    omitted → the module stdio transport (``tui_gateway.entry`` behaviour)."""
    t = transport or srv._stdio_transport
    token = bind_transport(t)
    try:
        from tui_gateway import server_requests
        if server_requests.is_response_frame(req):
            # The renderer answering one of OUR requests (clarify, approval, …): no response frame goes back.
            if not server_requests.resolve_response(req) and not srv._relay_compute_host_response(req):
                logger.debug("dropping response for unknown server request id=%r", req.get("id"))
            return None
        normalized = srv._normalize_request(req)
        if isinstance(normalized, dict):
            return normalized
        if normalized[1] not in srv._LONG_HANDLERS:
            try:
                return srv.handle_request(req)
            except Exception as exc:
                return srv._handler_error(req, exc)
        from hermes_cli.backend_retirement import retirement

        # Reserve BEFORE enqueueing: a queued handler has accepted work even though no worker runs yet.
        if not retirement.acquire():
            return srv._err(req.get("id"), 5035, "backend is retiring; reconnect to continue")
        try:
            ctx = contextvars.copy_context()  # the pool worker must see the bound transport
            if normalized[1] in srv._CONNECTOR_RPC_METHODS:
                ctx.run(srv._capture_connector_rpc_owner, normalized[2])

            def run():
                try:
                    resp = srv._handle_admitted_request(req)
                except Exception as exc:
                    resp = srv._handler_error(req, exc)
                if resp is not None:
                    t.write(resp)
            future = srv._pool.submit(lambda: ctx.run(run))
        except BaseException:
            retirement.release()
            raise
        # Also releases cancelled queued futures; the worker's own finally would never execute.
        future.add_done_callback(lambda _: retirement.release())
        return None
    finally:
        reset_transport(token)


def register(server):
    bind_module(globals(), server)

# Bound last, after every definition, so importing this module first (tests, the gateway process)
# lets server.py's own tail import see a complete module — the same tail-import idiom server.py uses.
from tui_gateway import server as srv  # noqa: E402
