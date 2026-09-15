"""Wire-boundary invariants the closed-model gateway relies on.

Handlers receive Params models and return Result models; ``_emit`` requires the declared Payload;
error frames carry the caller's request id even when produced through an in-process ``invoke``.
"""

import pytest

from tui_gateway import server
from tui_gateway.contracts import registry
from tui_gateway.contracts.base import MethodParams, Params, Result
from tui_gateway.contracts.common import SessionLiveInfo
from tui_gateway.contracts.events import SessionInfoPayload


def test_session_info_payload_of_accepts_the_result_model_and_a_mapping():
    """Pydantic refuses a base-class instance where the subclass is required; ``of`` is the one conversion."""
    info = SessionLiveInfo(model="m", provider="p")
    frame = server._event_frame("session.info", "sid", SessionInfoPayload.of(info, config_warning="w"))
    assert frame["params"]["payload"]["model"] == "m" and frame["params"]["payload"]["config_warning"] == "w"
    lazy = SessionInfoPayload.of({"cwd": "/tmp", "lazy": True})
    assert lazy.cwd == "/tmp" and lazy.lazy is True
    with pytest.raises(TypeError):
        server._event_frame("session.info", "sid", info)


def test_invoke_stamps_the_caller_rid_on_relayed_error_frames(monkeypatch):
    class _P(MethodParams):
        pass

    class _R(Result):
        ok: bool = True

    monkeypatch.setitem(registry.METHODS, "test.invoke", registry.MethodContract("test.invoke", _P, _R, ""))
    monkeypatch.setitem(server._methods, "test.invoke", None)  # restored on teardown
    server.register_method("test.invoke", lambda rid, params: server._err(rid, 4999, "nope"))
    assert server.invoke("test.invoke", _P(), rid="req-9")["id"] == "req-9"
    assert server.invoke("test.invoke", _P())["id"] is None


def test_server_request_params_never_carry_the_client_profile_key():
    class _Inbound(MethodParams):
        pass

    class _Outbound(Params):
        pass

    with pytest.raises(TypeError):
        registry.server_request("test.sr", params=_Inbound, result=Result)
    registry.SERVER_REQUESTS.pop("test.sr", None)
    assert "profile" not in _Outbound.model_fields
