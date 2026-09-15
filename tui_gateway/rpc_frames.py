"""JSON-RPC 2.0 response frames. Pure functions with no server state, so modules the gateway
process imports on its own (``methods_groups``' hosted-room lifecycle) can build error replies
without importing ``tui_gateway.server``. server.py binds them as ``_ok`` / ``_err``."""

from .contracts.base import Result


def ok_frame(rid, result: Result) -> dict:
    if not isinstance(result, Result):
        raise TypeError("RPC results must be Result instances")
    return {"jsonrpc": "2.0", "id": rid, "result": result.model_dump(mode="json")}


def err_frame(rid, code: int, msg: str, data=None) -> dict:
    error = {"code": code, "message": msg, **({"data": data} if data is not None else {})}
    return {"jsonrpc": "2.0", "id": rid, "error": error}
