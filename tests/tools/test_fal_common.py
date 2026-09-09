"""Tests for tools/fal_common.py — shared FAL.ai SDK plumbing.

Covers: import_fal_client, _normalize_fal_queue_url_format,
_extract_http_status, _ManagedFalSyncClient (init + submit).
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from tools.fal_common import (
    _ManagedFalSyncClient,
    _extract_http_status,
    _normalize_fal_queue_url_format,
    import_fal_client,
)


# ---------------------------------------------------------------------------
# import_fal_client
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_fal_client(monkeypatch):
    module = types.ModuleType("fal_client")
    monkeypatch.setitem(sys.modules, "fal_client", module)
    return module


class TestImportFalClient:
    def test_returns_fal_client_module(self, monkeypatch, fake_fal_client):
        """import_fal_client returns the fal_client module reference."""
        ensure = MagicMock()
        monkeypatch.setattr("tools.lazy_deps.ensure", ensure)

        result = import_fal_client()
        assert result is fake_fal_client
        ensure.assert_called_once_with("image.fal", prompt=False)

    def test_lazy_ensure_import_error_is_swallowed(self, monkeypatch, fake_fal_client):
        """If lazy_deps.ensure raises ImportError, it's swallowed (fal_client still imported)."""
        monkeypatch.setattr(
            "tools.lazy_deps.ensure",
            MagicMock(side_effect=ImportError("no lazy_deps")),
        )

        assert import_fal_client() is fake_fal_client

    def test_lazy_ensure_other_exception_raises_import_error(self):
        """If lazy_deps.ensure raises a non-ImportError, it's re-raised as ImportError."""
        with patch("tools.lazy_deps.ensure", side_effect=RuntimeError("install hint")):
            with pytest.raises(ImportError, match="install hint"):
                import_fal_client()

    def test_lazy_ensure_module_missing_is_swallowed(self, fake_fal_client):
        """If tools.lazy_deps itself can't be imported, ImportError is swallowed."""
        import builtins

        original_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "tools.lazy_deps":
                raise ImportError("no module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            result = import_fal_client()
            assert result is fake_fal_client


# ---------------------------------------------------------------------------
# _normalize_fal_queue_url_format
# ---------------------------------------------------------------------------


class TestNormalizeFalQueueUrlFormat:
    def test_adds_trailing_slash(self):
        assert (
            _normalize_fal_queue_url_format("https://queue.example.com")
            == "https://queue.example.com/"
        )

    def test_strips_trailing_slashes_then_adds_one(self):
        assert (
            _normalize_fal_queue_url_format("https://queue.example.com///")
            == "https://queue.example.com/"
        )

    def test_strips_whitespace(self):
        assert (
            _normalize_fal_queue_url_format("  https://queue.example.com  ")
            == "https://queue.example.com/"
        )

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Managed FAL queue origin is required"):
            _normalize_fal_queue_url_format("")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Managed FAL queue origin is required"):
            _normalize_fal_queue_url_format(None)  # type: ignore[arg-type]

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="Managed FAL queue origin is required"):
            _normalize_fal_queue_url_format("   ")


# ---------------------------------------------------------------------------
# _extract_http_status
# ---------------------------------------------------------------------------


class TestExtractHttpStatus:
    def test_returns_status_from_response_attribute(self):
        """httpx.HTTPStatusError exposes .response.status_code."""
        exc = MagicMock()
        exc.response = MagicMock()
        exc.response.status_code = 404
        assert _extract_http_status(exc) == 404

    def test_returns_status_from_exc_status_code(self):
        """fal_client wrappers expose .status_code directly."""
        exc = MagicMock()
        exc.response = None
        exc.status_code = 500
        assert _extract_http_status(exc) == 500

    def test_returns_none_when_no_response_and_no_status_code(self):
        exc = Exception("plain error")
        assert _extract_http_status(exc) is None

    def test_returns_none_when_response_is_none(self):
        exc = MagicMock()
        exc.response = None
        del exc.status_code
        assert _extract_http_status(exc) is None

    def test_returns_none_when_response_status_code_not_int(self):
        exc = MagicMock()
        exc.response = MagicMock()
        exc.response.status_code = "not-int"
        del exc.status_code
        assert _extract_http_status(exc) is None

    def test_returns_none_when_status_code_not_int(self):
        exc = MagicMock()
        exc.response = None
        exc.status_code = "not-int"
        assert _extract_http_status(exc) is None

    def test_response_status_takes_precedence_over_exc_status(self):
        exc = MagicMock()
        exc.response = MagicMock()
        exc.response.status_code = 200
        exc.status_code = 500
        assert _extract_http_status(exc) == 200

    def test_falls_back_to_exc_status_when_response_status_not_int(self):
        exc = MagicMock()
        exc.response = MagicMock()
        exc.response.status_code = "bad"
        exc.status_code = 503
        assert _extract_http_status(exc) == 503


# ---------------------------------------------------------------------------
# _ManagedFalSyncClient — __init__
# ---------------------------------------------------------------------------


def _make_fal_client_mock(
    *,
    sync_client_class=None,
    client_module=None,
    http_client=None,
    default_timeout=120.0,
):
    """Build a mock fal_client module with all required attributes."""
    if sync_client_class is None:
        sync_client_class = MagicMock()
    if client_module is None:
        client_module = MagicMock()

    sync_client_instance = MagicMock()
    sync_client_instance._client = (
        http_client if http_client is not None else MagicMock()
    )
    sync_client_instance.default_timeout = default_timeout
    sync_client_class.return_value = sync_client_instance

    fal_client = MagicMock()
    fal_client.SyncClient = sync_client_class
    fal_client.client = client_module
    return fal_client, sync_client_instance


class TestManagedFalSyncClientInit:
    def test_init_succeeds_with_all_attributes(self):
        fal_client, _ = _make_fal_client_mock()
        client = _ManagedFalSyncClient(
            fal_client, key="test-key", queue_run_origin="https://queue.example.com"
        )
        assert client._queue_url_format == "https://queue.example.com/"
        assert client._http_client is not None
        assert client._maybe_retry_request is not None
        assert client._raise_for_status is not None
        assert client._request_handle_class is not None

    def test_init_raises_when_sync_client_missing(self):
        fal_client = MagicMock()
        fal_client.SyncClient = None
        fal_client.client = MagicMock()
        with pytest.raises(RuntimeError, match="fal_client.SyncClient is required"):
            _ManagedFalSyncClient(
                fal_client, key="k", queue_run_origin="https://q.example.com"
            )

    def test_init_raises_when_client_module_missing(self):
        fal_client = MagicMock()
        fal_client.SyncClient = MagicMock()
        fal_client.client = None
        with pytest.raises(RuntimeError, match="fal_client.client is required"):
            _ManagedFalSyncClient(
                fal_client, key="k", queue_run_origin="https://q.example.com"
            )

    def test_init_raises_when_http_client_missing(self):
        """SyncClient._client is None → RuntimeError."""
        fal_client = MagicMock()
        sync_instance = MagicMock()
        sync_instance._client = None
        fal_client.SyncClient.return_value = sync_instance
        fal_client.client = MagicMock()
        with pytest.raises(RuntimeError, match="SyncClient._client is required"):
            _ManagedFalSyncClient(
                fal_client, key="k", queue_run_origin="https://q.example.com"
            )

    def test_init_raises_when_retry_request_missing(self):
        fal_client, _ = _make_fal_client_mock()
        fal_client.client._maybe_retry_request = None
        with pytest.raises(RuntimeError, match="request helpers are required"):
            _ManagedFalSyncClient(
                fal_client, key="k", queue_run_origin="https://q.example.com"
            )

    def test_init_raises_when_raise_for_status_missing(self):
        fal_client, _ = _make_fal_client_mock()
        fal_client.client._raise_for_status = None
        with pytest.raises(RuntimeError, match="request helpers are required"):
            _ManagedFalSyncClient(
                fal_client, key="k", queue_run_origin="https://q.example.com"
            )

    def test_init_raises_when_request_handle_class_missing(self):
        fal_client, _ = _make_fal_client_mock()
        fal_client.client.SyncRequestHandle = None
        with pytest.raises(RuntimeError, match="SyncRequestHandle is required"):
            _ManagedFalSyncClient(
                fal_client, key="k", queue_run_origin="https://q.example.com"
            )

    def test_init_passes_key_to_sync_client(self):
        fal_client, sync_instance = _make_fal_client_mock()
        _ManagedFalSyncClient(
            fal_client, key="my-secret-key", queue_run_origin="https://q.example.com"
        )
        fal_client.SyncClient.assert_called_once_with(key="my-secret-key")

    def test_init_normalizes_queue_url(self):
        fal_client, _ = _make_fal_client_mock()
        client = _ManagedFalSyncClient(
            fal_client, key="k", queue_run_origin="https://q.example.com///"
        )
        assert client._queue_url_format == "https://q.example.com/"


# ---------------------------------------------------------------------------
# _ManagedFalSyncClient — submit
# ---------------------------------------------------------------------------


class TestManagedFalSyncClientSubmit:
    def _make_client(self, **kwargs):
        fal_client, sync_instance = _make_fal_client_mock(**kwargs)
        client = _ManagedFalSyncClient(
            fal_client, key="k", queue_run_origin="https://queue.example.com"
        )
        return client, sync_instance, fal_client

    def test_submit_basic(self):
        client, sync_instance, fal_client = self._make_client()
        response = MagicMock()
        response.json.return_value = {
            "request_id": "req-1",
            "response_url": "https://q.example.com/resp",
            "status_url": "https://q.example.com/status",
            "cancel_url": "https://q.example.com/cancel",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        result = client.submit("my-app", {"prompt": "hello"})

        client._http_client.request.assert_called_once()
        call_args = client._http_client.request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "https://queue.example.com/my-app"
        assert call_args[1]["json"] == {"prompt": "hello"}
        assert call_args[1]["timeout"] == 120.0
        client._raise_for_status.assert_called_once_with(response)
        client._request_handle_class.assert_called_once()
        assert result is client._request_handle_class.return_value

    def test_submit_with_path(self):
        client, _, _ = self._make_client()
        response = MagicMock()
        response.json.return_value = {
            "request_id": "r",
            "response_url": "",
            "status_url": "",
            "cancel_url": "",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        client.submit("my-app", {}, path="sub/path")

        url = client._http_client.request.call_args[0][1]
        assert url == "https://queue.example.com/my-app/sub/path"

    def test_submit_with_path_strips_leading_slash(self):
        client, _, _ = self._make_client()
        response = MagicMock()
        response.json.return_value = {
            "request_id": "r",
            "response_url": "",
            "status_url": "",
            "cancel_url": "",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        client.submit("my-app", {}, path="/leading/slash")

        url = client._http_client.request.call_args[0][1]
        assert url == "https://queue.example.com/my-app/leading/slash"

    def test_submit_with_webhook_url(self):
        client, _, _ = self._make_client()
        response = MagicMock()
        response.json.return_value = {
            "request_id": "r",
            "response_url": "",
            "status_url": "",
            "cancel_url": "",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        client.submit("my-app", {}, webhook_url="https://hook.example.com/cb")

        url = client._http_client.request.call_args[0][1]
        assert "fal_webhook=https%3A%2F%2Fhook.example.com%2Fcb" in url

    def test_submit_with_hint(self):
        client, _, _ = self._make_client()
        client._add_hint_header = MagicMock()
        response = MagicMock()
        response.json.return_value = {
            "request_id": "r",
            "response_url": "",
            "status_url": "",
            "cancel_url": "",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        client.submit("my-app", {}, hint="my-hint")

        client._add_hint_header.assert_called_once()
        args = client._add_hint_header.call_args[0]
        assert args[0] == "my-hint"

    def test_submit_with_priority(self):
        client, _, _ = self._make_client()
        client._add_priority_header = MagicMock()
        response = MagicMock()
        response.json.return_value = {
            "request_id": "r",
            "response_url": "",
            "status_url": "",
            "cancel_url": "",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        client.submit("my-app", {}, priority=5)

        client._add_priority_header.assert_called_once()
        args = client._add_priority_header.call_args[0]
        assert args[0] == 5

    def test_submit_with_priority_raises_when_header_fn_missing(self):
        client, _, _ = self._make_client()
        client._add_priority_header = None
        response = MagicMock()
        response.json.return_value = {
            "request_id": "r",
            "response_url": "",
            "status_url": "",
            "cancel_url": "",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        with pytest.raises(RuntimeError, match="add_priority_header is required"):
            client.submit("my-app", {}, priority=5)

    def test_submit_with_start_timeout(self):
        client, _, _ = self._make_client()
        client._add_timeout_header = MagicMock()
        response = MagicMock()
        response.json.return_value = {
            "request_id": "r",
            "response_url": "",
            "status_url": "",
            "cancel_url": "",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        client.submit("my-app", {}, start_timeout=30)

        client._add_timeout_header.assert_called_once()
        args = client._add_timeout_header.call_args[0]
        assert args[0] == 30

    def test_submit_with_start_timeout_raises_when_header_fn_missing(self):
        client, _, _ = self._make_client()
        client._add_timeout_header = None
        response = MagicMock()
        response.json.return_value = {
            "request_id": "r",
            "response_url": "",
            "status_url": "",
            "cancel_url": "",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        with pytest.raises(RuntimeError, match="add_timeout_header is required"):
            client.submit("my-app", {}, start_timeout=30)

    def test_submit_with_custom_headers(self):
        client, _, _ = self._make_client()
        response = MagicMock()
        response.json.return_value = {
            "request_id": "r",
            "response_url": "",
            "status_url": "",
            "cancel_url": "",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        client.submit("my-app", {}, headers={"X-Custom": "value"})

        headers = client._http_client.request.call_args[1]["headers"]
        assert headers["X-Custom"] == "value"

    def test_submit_with_none_headers_defaults_to_empty_dict(self):
        client, _, _ = self._make_client()
        response = MagicMock()
        response.json.return_value = {
            "request_id": "r",
            "response_url": "",
            "status_url": "",
            "cancel_url": "",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        client.submit("my-app", {}, headers=None)

        headers = client._http_client.request.call_args[1]["headers"]
        assert headers == {}

    def test_submit_uses_custom_default_timeout(self):
        """SyncClient.default_timeout is used if present."""
        fal_client, sync_instance = _make_fal_client_mock(default_timeout=300.0)
        client = _ManagedFalSyncClient(
            fal_client, key="k", queue_run_origin="https://q.example.com"
        )
        response = MagicMock()
        response.json.return_value = {
            "request_id": "r",
            "response_url": "",
            "status_url": "",
            "cancel_url": "",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        client.submit("app", {})

        assert client._http_client.request.call_args[1]["timeout"] == 300.0

    def test_submit_falls_back_to_120_when_no_default_timeout(self):
        """SyncClient without default_timeout attr → 120.0 fallback."""
        fal_client, sync_instance = _make_fal_client_mock()
        # Remove default_timeout
        del sync_instance.default_timeout
        client = _ManagedFalSyncClient(
            fal_client, key="k", queue_run_origin="https://q.example.com"
        )
        response = MagicMock()
        response.json.return_value = {
            "request_id": "r",
            "response_url": "",
            "status_url": "",
            "cancel_url": "",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        client.submit("app", {})

        assert client._http_client.request.call_args[1]["timeout"] == 120.0

    def test_submit_passes_request_handle_kwargs(self):
        client, _, _ = self._make_client()
        response = MagicMock()
        response.json.return_value = {
            "request_id": "req-123",
            "response_url": "https://q.example.com/resp",
            "status_url": "https://q.example.com/status",
            "cancel_url": "https://q.example.com/cancel",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        client.submit("app", {})

        kwargs = client._request_handle_class.call_args[1]
        assert kwargs["request_id"] == "req-123"
        assert kwargs["response_url"] == "https://q.example.com/resp"
        assert kwargs["status_url"] == "https://q.example.com/status"
        assert kwargs["cancel_url"] == "https://q.example.com/cancel"
        assert kwargs["client"] is client._http_client

    def test_submit_with_all_optional_params(self):
        """All optional params set at once."""
        client, _, _ = self._make_client()
        client._add_hint_header = MagicMock()
        client._add_priority_header = MagicMock()
        client._add_timeout_header = MagicMock()
        response = MagicMock()
        response.json.return_value = {
            "request_id": "r",
            "response_url": "",
            "status_url": "",
            "cancel_url": "",
        }
        client._http_client.request = MagicMock(return_value=response)
        client._raise_for_status = MagicMock()
        client._request_handle_class = MagicMock()

        client.submit(
            "app",
            {"prompt": "test"},
            path="sub",
            hint="hint-val",
            webhook_url="https://hook.example.com",
            priority=10,
            headers={"X-Custom": "val"},
            start_timeout=60,
        )

        url = client._http_client.request.call_args[0][1]
        assert "app/sub?" in url
        assert "fal_webhook=" in url
        client._add_hint_header.assert_called_once()
        client._add_priority_header.assert_called_once()
        client._add_timeout_header.assert_called_once()
        headers = client._http_client.request.call_args[1]["headers"]
        assert headers["X-Custom"] == "val"


# ---------------------------------------------------------------------------
# Managed queue submit must not retry deterministic 409s (billing / idempotency)
# ---------------------------------------------------------------------------

# Managed queue submit must not retry deterministic 409s (billing / idempotency).
# The tests wire fal_client's retry helper so a regression that calls
# _maybe_retry_request again would POST more than once with the same
# x-idempotency-key and fail call_count == 1.
_FAL_RETRY_CODES = (408, 409, 429)
_FAL_MAX_ATTEMPTS = 10
_BILLING_409_DETAIL = "BILLING_ERROR: unsupported_pricing_meter"
_IDEMPOTENCY_409_DETAIL = (
    "idempotency key is already bound to a different request "
    "without a reusable request handle"
)


class _FakeFalResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)
        self.headers = {}

    def json(self):
        return self._payload


class _FakeFalHTTPError(Exception):
    """Stand-in for fal_client.client.FalClientHTTPError / httpx.HTTPStatusError."""

    def __init__(self, message, response):
        super().__init__(message)
        self.response = response
        self.status_code = response.status_code


def _fal_style_raise_for_status(response):
    if getattr(response, "status_code", 200) < 400:
        return
    payload = {}
    try:
        payload = response.json()
    except Exception:
        payload = {}
    detail = payload.get("detail", getattr(response, "text", "")) if isinstance(payload, dict) else getattr(response, "text", "")
    raise _FakeFalHTTPError(str(detail), response)


def _fal_style_maybe_retry_request(
    client, method, url, json=None, timeout=None, headers=None, extra_retry_codes=None, **kwargs
):
    extra = extra_retry_codes or []
    last_exc = None
    for attempt in range(1, _FAL_MAX_ATTEMPTS + 1):
        try:
            response = client.request(
                method, url, json=json, timeout=timeout, headers=headers, **kwargs
            )
            _fal_style_raise_for_status(response)
            return response
        except _FakeFalHTTPError as exc:
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in _FAL_RETRY_CODES or status in extra:
                if attempt < _FAL_MAX_ATTEMPTS:
                    continue
            raise
    raise last_exc


def _wire_production_retry_client(http_client):
    """_ManagedFalSyncClient whose submit goes through the real retry helper path."""
    fal_client, _ = _make_fal_client_mock(http_client=http_client)
    fal_client.client._maybe_retry_request = _fal_style_maybe_retry_request
    fal_client.client._raise_for_status = _fal_style_raise_for_status
    fal_client.client.SyncRequestHandle = MagicMock()
    client = _ManagedFalSyncClient(
        fal_client, key="k", queue_run_origin="https://queue.example.com"
    )
    return client, fal_client.client.SyncRequestHandle


class TestManagedFalSubmitNoRetryBilling409:
    def test_billing_409_does_not_retry_same_idempotency_key(self):
        http_client = MagicMock()
        billing = _FakeFalResponse(
            409,
            {"error": "BILLING_ERROR", "detail": _BILLING_409_DETAIL},
        )
        idempotency_conflict = _FakeFalResponse(409, {"detail": _IDEMPOTENCY_409_DETAIL})

        def fake_request(*_args, **_kwargs):
            # First POST is the billing 409; subsequent POSTs (the production bug)
            # replay the same idempotency key and get a bound-key conflict.
            if http_client.request.call_count <= 1:
                return billing
            return idempotency_conflict

        http_client.request.side_effect = fake_request

        client, _ = _wire_production_retry_client(http_client)

        with pytest.raises(_FakeFalHTTPError) as exc_info:
            client.submit(
                "openai/gpt-image-2.5/flare/text-to-image",
                {"prompt": "a cat"},
                headers={"x-idempotency-key": "fixed-key"},
            )

        assert http_client.request.call_count == 1
        err = str(exc_info.value)
        assert "unsupported_pricing_meter" in err
        assert "BILLING_ERROR" in err
        assert "already bound" not in err
        headers = http_client.request.call_args.kwargs["headers"]
        assert headers["x-idempotency-key"] == "fixed-key"

    def test_idempotency_bound_409_without_reusable_handle_is_not_retried(self):
        http_client = MagicMock()
        http_client.request.return_value = _FakeFalResponse(
            409, {"detail": _IDEMPOTENCY_409_DETAIL}
        )
        client, _ = _wire_production_retry_client(http_client)

        with pytest.raises(_FakeFalHTTPError) as exc_info:
            client.submit(
                "fal-ai/flux-2-pro",
                {"prompt": "x"},
                headers={"x-idempotency-key": "fixed-key"},
            )

        assert http_client.request.call_count == 1
        assert "already bound" in str(exc_info.value)
        assert "reusable request handle" in str(exc_info.value)

    def test_success_200_parses_handle_with_single_post(self):
        http_client = MagicMock()
        payload = {
            "request_id": "req-ok",
            "response_url": "https://q.example.com/resp",
            "status_url": "https://q.example.com/status",
            "cancel_url": "https://q.example.com/cancel",
        }
        http_client.request.return_value = _FakeFalResponse(200, payload)
        client, handle_cls = _wire_production_retry_client(http_client)

        result = client.submit("my-app", {"prompt": "hello"})

        assert http_client.request.call_count == 1
        request_call = http_client.request.call_args
        assert request_call.args[0] == "POST"
        assert request_call.args[1] == "https://queue.example.com/my-app"
        assert request_call.kwargs["json"] == {"prompt": "hello"}
        handle_cls.assert_called_once()
        kwargs = handle_cls.call_args.kwargs
        assert kwargs["request_id"] == "req-ok"
        assert kwargs["response_url"] == payload["response_url"]
        assert kwargs["status_url"] == payload["status_url"]
        assert kwargs["cancel_url"] == payload["cancel_url"]
        assert kwargs["client"] is http_client
        assert result is handle_cls.return_value

    def test_transport_error_is_not_treated_as_billing(self):
        class TransportError(Exception):
            pass

        http_client = MagicMock()
        http_client.request.side_effect = TransportError("connection reset")
        client, _ = _wire_production_retry_client(http_client)

        with pytest.raises(TransportError, match="connection reset") as exc_info:
            client.submit("my-app", {"prompt": "hello"})

        err = str(exc_info.value)
        assert "BILLING_ERROR" not in err
        assert "unsupported_pricing_meter" not in err
