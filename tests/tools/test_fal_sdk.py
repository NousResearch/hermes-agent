"""Managed FAL adapter against the installed SDK; only HTTP leaves are replaced."""
import json
from types import SimpleNamespace

import fal_client
import httpx
import pytest

from tools.fal_common import (
    _ManagedFalSyncClient, _extract_http_status, _managed_fal_billing_error,
    _normalize_fal_queue_url_format, import_fal_client,
)


@pytest.fixture
def transport(monkeypatch):
    requests, responses, clients = [], [], []
    original = httpx.Client.__init__

    def handle(request):
        assert request.url.host == "queue.test", "no external FAL requests allowed"
        requests.append(request)
        reply = responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, body = reply
        return httpx.Response(status, json=body)

    def initialize(client, *args, **kwargs):
        kwargs.update(transport=httpx.MockTransport(handle), trust_env=False)
        original(client, *args, **kwargs)
        clients.append(client)

    monkeypatch.setattr(httpx.Client, "__init__", initialize)
    # Keep the SDK retry algorithm, but do not spend wall time backing off.
    monkeypatch.setattr(fal_client.client.time, "sleep", lambda delay: None)
    yield requests, responses
    for client in clients:
        client.close()


def accepted():
    return 200, {
        "request_id": "r1", "response_url": "https://queue.test/result",
        "status_url": "https://queue.test/status", "cancel_url": "https://queue.test/cancel",
    }


@pytest.mark.parametrize("path", ["sub/path", "/sub/path"])
def test_sdk_submit_preserves_wire_contract(transport, path):
    requests, replies = transport
    replies.append(accepted())
    client = _ManagedFalSyncClient(fal_client, key="test-token", queue_run_origin="  https://queue.test///  ")
    assert isinstance(client._sync_client, fal_client.SyncClient)
    object.__setattr__(client._sync_client, "default_timeout", 37)
    headers = {"X-Custom": "kept", "X-Idempotency-Key": "one-operation"}
    handle = client.submit("app", {"prompt": "local fixture"}, path=path, hint="runner",
                           webhook_url="https://hook.test/cb?a=1", priority="low",
                           start_timeout=60, headers=headers)
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://queue.test/app/sub/path?fal_webhook=https%3A%2F%2Fhook.test%2Fcb%3Fa%3D1"
    assert json.loads(request.content) == {"prompt": "local fixture"}
    for key, value in {
        "authorization": "Key test-token", "x-custom": "kept", "x-idempotency-key": "one-operation",
        "x-fal-runner-hint": "runner", "x-fal-queue-priority": "low", "x-fal-request-timeout": "60.0",
    }.items():
        assert request.headers[key] == value
    assert request.extensions["timeout"]["read"] == 37
    assert headers == {"X-Custom": "kept", "X-Idempotency-Key": "one-operation"}
    assert isinstance(handle, fal_client.client.SyncRequestHandle)
    assert (handle.request_id, handle.response_url, handle.status_url, handle.cancel_url) == (
        "r1", "https://queue.test/result", "https://queue.test/status", "https://queue.test/cancel")
    assert handle.client is client._http_client


@pytest.mark.parametrize("status", [402, 409, 429, 503, "transport"])
@pytest.mark.parametrize("keyed", [True, False])
def test_keyed_errors_survive_without_duplicate_submission(transport, status, keyed):
    requests, replies = transport
    body = {"error": {"code": "BILLING_ERROR", "message": "meter disabled",
                       "details": {"upstreamPayload": {"code": "meter_missing", "error": "not configured"}}}}
    replies.extend([httpx.ConnectError("connection lost") if status == "transport" else (status, body), accepted()])
    client = _ManagedFalSyncClient(fal_client, key="token", queue_run_origin="https://queue.test")
    headers = {"x-IDEMPOTENCY-key": "operation"} if keyed else None
    # A JSON 503 is an application error, not the SDK's retryable ingress 503.
    if keyed or status in (402, 503):
        error = httpx.ConnectError if status == "transport" else fal_client.client.FalClientHTTPError
        with pytest.raises(error) as caught:
            client.submit("app", {}, headers=headers)
        assert len(requests) == 1
        if status != "transport":
            assert _extract_http_status(caught.value) == status
            assert caught.value.response.json() == body
            assert "meter_missing" in _managed_fal_billing_error(caught.value, "model")
    else:
        assert client.submit("app", {}, headers=headers).request_id == "r1"
        assert len(requests) == 2
    assert str(requests[0].url) == "https://queue.test/app"
    assert "x-custom" not in requests[0].headers


@pytest.mark.parametrize("origin", ["", None, "   "])
def test_empty_origin_refused(origin):
    with pytest.raises(ValueError, match="origin is required"):
        _normalize_fal_queue_url_format(origin)


@pytest.mark.parametrize("response_status,status,expected", [
    (404, None, 404), (None, 500, 500), (None, None, None),
    ("bad", None, None), (None, "bad", None), (200, 500, 200), ("bad", 503, 503),
])
def test_exception_status_shapes(response_status, status, expected):
    exc = SimpleNamespace(response=SimpleNamespace(status_code=response_status), status_code=status)
    assert _extract_http_status(exc) == expected
    assert _extract_http_status(Exception("plain")) is None


@pytest.mark.parametrize("missing,message", [
    ("SyncClient", "fal_client.SyncClient"), ("client", "fal_client.client"),
    ("_client", "SyncClient._client"), ("_maybe_retry_request", "request helpers"),
    ("_raise_for_status", "request helpers"), ("SyncRequestHandle", "SyncRequestHandle"),
])
def test_missing_sdk_surface_is_actionable(transport, monkeypatch, missing, message):
    owner = fal_client if missing in {"SyncClient", "client"} else fal_client.client
    if missing == "_client":
        owner = fal_client.SyncClient
    monkeypatch.setattr(owner, missing, None)
    with pytest.raises(RuntimeError, match=message):
        _ManagedFalSyncClient(fal_client, key="token", queue_run_origin="https://queue.test")


@pytest.mark.parametrize("helper,options,message", [
    ("add_priority_header", {"priority": "low"}, "add_priority_header"),
    ("add_timeout_header", {"start_timeout": 60}, "add_timeout_header"),
])
def test_optional_sdk_surface_refused_before_http(transport, monkeypatch, helper, options, message):
    monkeypatch.setattr(fal_client.client, helper, None)
    client = _ManagedFalSyncClient(fal_client, key="token", queue_run_origin="https://queue.test")
    with pytest.raises(RuntimeError, match=message):
        client.submit("app", {}, **options)
    assert transport[0] == []


@pytest.mark.parametrize("failure", [None, ImportError("external SDK"), RuntimeError("PM acquisition failed")])
def test_import_adapter_keeps_pm_failure_policy(monkeypatch, failure):
    import pm
    calls = []

    def ensure(feature):
        calls.append(feature)
        if failure:
            raise failure

    monkeypatch.setattr(pm, "ensure_import", ensure)
    if isinstance(failure, RuntimeError):
        with pytest.raises(ImportError, match="PM acquisition failed"):
            import_fal_client()
    else:
        assert import_fal_client() is fal_client
    assert calls == ["fal"]


def test_import_adapter_runs_real_pm_availability_without_install(monkeypatch):
    import pm.client
    monkeypatch.setattr(pm.client, "sync_venv", lambda *a, **kw: pytest.fail("available SDK must not install"))
    assert import_fal_client() is fal_client


def test_external_sdk_remains_importable_without_pm(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pm", None)
    assert import_fal_client() is fal_client


@pytest.mark.parametrize("timeout,expected", [(None, 120.0), (300.0, 300.0)])
def test_older_sdk_timeout_default_and_optional_hint(transport, monkeypatch, timeout, expected):
    monkeypatch.setattr(fal_client.client, "add_hint_header", None)
    client = _ManagedFalSyncClient(fal_client, key="token", queue_run_origin="https://queue.test")
    # Keep its real HTTP client and request helpers, vary only older SDK metadata.
    client._sync_client = SimpleNamespace(**({} if timeout is None else {"default_timeout": timeout}))
    transport[1].append(accepted())
    assert client.submit("app", {}, hint="ignored-on-old-sdk").request_id == "r1"
    assert transport[0][0].extensions["timeout"]["read"] == expected
    assert "x-fal-runner-hint" not in transport[0][0].headers
