"""Model-boundary behavior for gateway RPC dispatch and server requests."""

from __future__ import annotations

from contextlib import contextmanager
from typing import cast

import pytest

from tui_gateway import server, server_requests
from tui_gateway.contracts import registry as contracts
from tui_gateway.contracts.base import MethodParams, Payload, Result
from tui_gateway.contracts.server_requests import EmptyRequestParams


class _DispatchParams(MethodParams):
    count: int


class _DispatchResult(Result):
    count: int


@contextmanager
def _registered(handler):
    name = "test.contract.dispatch"
    contracts.method(name, params=_DispatchParams, result=_DispatchResult)
    server.register_method(name, handler)
    try:
        yield name
    finally:
        server._methods.pop(name, None)
        contracts.METHODS.pop(name, None)


def test_dispatch_validates_models_and_serializes_result_once():
    seen = []

    def handler(rid, params):
        seen.append(params)
        return _DispatchResult(count=params.count)

    with _registered(handler) as name:
        response = server.handle_request({"id": "rpc-1", "method": name, "params": {"count": 7}})

    assert isinstance(seen[0], _DispatchParams)
    assert response is not None
    assert response["result"] == _DispatchResult(count=7).model_dump(mode="json")


def test_dispatch_rejects_unknown_params_without_calling_handler():
    with _registered(lambda rid, params: _DispatchResult(count=params.count)) as name:
        response = server.handle_request({"id": "rpc-2", "method": name, "params": {"count": 7, "extra": True}})

    assert response is not None
    assert response["error"]["code"] == 4000
    assert response["error"]["data"][0]["loc"] == ["extra"]


def test_dispatch_redacts_invalid_param_input_from_error_frame():
    secret = "not-for-the-wire"
    with _registered(lambda rid, params: _DispatchResult(count=params.count)) as name:
        response = server.handle_request({"id": "rpc-3", "method": name, "params": {"count": secret}})

    assert response is not None
    assert response["error"]["code"] == 4000
    assert secret not in repr(response)


def test_dispatch_rejects_non_error_dict_handler_result():
    with _registered(lambda rid, params: {"count": params.count}) as name:
        with pytest.raises(TypeError, match="dict that is not an error frame"):
            server.handle_request({"id": "rpc-4", "method": name, "params": {"count": 7}})


def test_event_frame_rejects_dict_payload():
    with pytest.raises(TypeError, match="payload must be"):
        server._event_frame("error", "sid", cast(Payload, {"message": "not a model"}))


def test_server_request_invalid_result_does_not_answer_request():
    request = server_requests.ServerRequest("sid", "sudo", EmptyRequestParams(session_id="sid"))
    server_requests._open[request.id] = request
    try:
        assert server_requests.resolve_response({"id": request.id, "result": {"wrong": "shape"}})
        assert request.answered is False
        assert request.result is None
    finally:
        server_requests._open.pop(request.id, None)
