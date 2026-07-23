"""Tests for the HTTP Request tool.

Fully hermetic — every test mocks the network boundary
(``tools.http_request_tool._open``); nothing talks to a live host and no
test converts an assertion failure into a skip.

Covers:
- Basic GET request shape (status, headers, body)
- POST with body and custom headers
- HTTPError handling (4xx, 5xx)
- URLError handling (network failure)
- SSRF / private-IP blocking, on the initial URL and on redirect targets
- URL scheme validation (file:// rejected), on redirects too
- Secret exfiltration guard (URLs with embedded tokens blocked)
- Method validation
- Response size truncation and server-side clamping of model-provided limits
"""

import json
import urllib.error
import urllib.request

import pytest

from tools.http_request_tool import (
    http_request_tool,
    _check_http_request_requirements,
    _SafeRedirectHandler,
    _open,
    _MAX_RESPONSE_SIZE_CAP,
    _MAX_RESPONSE_SIZE_DEFAULT,
    _TIMEOUT_CAP,
    _TIMEOUT_DEFAULT,
    HTTP_REQUEST_SCHEMA,
)


class _FakeResponse:
    """Mock urllib response."""

    def __init__(self, body_bytes=b"", status=200, reason="OK", headers=None):
        self._body = body_bytes
        self.code = status
        self.reason = reason
        self._headers = headers or {}

    def getcode(self):
        return self.code

    def getheaders(self):
        return list(self._headers.items())

    def read(self, size=-1):
        if size >= 0:
            return self._body[:size]
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _allow_all_urls(monkeypatch):
    monkeypatch.setattr("tools.http_request_tool.is_safe_url", lambda url: True)


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------

def test_get_request_success(monkeypatch):
    """Basic GET returns structured JSON with all expected fields."""
    _allow_all_urls(monkeypatch)

    captured = {}

    def _fake_open(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.method
        captured["headers"] = dict(req.headers)
        return _FakeResponse(
            body_bytes=b'{"data": "hello"}',
            status=200,
            reason="OK",
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr("tools.http_request_tool._open", _fake_open)

    result = json.loads(http_request_tool("https://api.example.com/v1/items"))

    assert result["success"] is True
    assert result["status_code"] == 200
    assert result["status_text"] == "OK"
    assert result["headers"]["Content-Type"] == "application/json"
    assert result["body"] == '{"data": "hello"}'
    assert "elapsed_ms" in result
    assert result["elapsed_ms"] >= 0


def test_post_with_body_and_headers(monkeypatch):
    """POST with JSON body and custom headers forwarded correctly."""
    _allow_all_urls(monkeypatch)

    captured = {}

    def _fake_open(req, timeout=None):
        captured["method"] = req.method
        captured["body"] = req.data
        captured["headers"] = dict(req.headers)
        return _FakeResponse(
            body_bytes=b'{"id": 42}',
            status=201,
            reason="Created",
            headers={"Content-Type": "application/json"},
        )

    monkeypatch.setattr("tools.http_request_tool._open", _fake_open)

    result = json.loads(
        http_request_tool(
            "https://api.example.com/v1/items",
            method="POST",
            headers={"Authorization": "Bearer tok", "Content-Type": "application/json"},
            body='{"name": "foo"}',
        )
    )

    assert result["success"] is True
    assert result["status_code"] == 201
    assert captured["method"] == "POST"
    assert captured["body"] == b'{"name": "foo"}'
    assert captured["headers"]["Authorization"] == "Bearer tok"


def test_method_case_insensitive(monkeypatch):
    """Method is normalised to uppercase."""
    _allow_all_urls(monkeypatch)

    captured = {}

    def _fake_open(req, timeout=None):
        captured["method"] = req.method
        return _FakeResponse(body_bytes=b"ok")

    monkeypatch.setattr("tools.http_request_tool._open", _fake_open)

    result = json.loads(http_request_tool("https://example.com", method="post"))

    assert result["success"] is True
    assert captured["method"] == "POST"


def test_head_request_no_body(monkeypatch):
    """HEAD requests succeed with an empty body."""
    _allow_all_urls(monkeypatch)

    monkeypatch.setattr(
        "tools.http_request_tool._open",
        lambda req, timeout=None: _FakeResponse(body_bytes=b"", headers={"Content-Length": "123"}),
    )

    result = json.loads(http_request_tool("https://example.com", method="HEAD"))

    assert result["success"] is True
    assert result["body"] == ""
    assert result["headers"]["Content-Length"] == "123"


def test_unicode_body_and_response(monkeypatch):
    """Non-ASCII request/response bodies round-trip cleanly."""
    _allow_all_urls(monkeypatch)

    captured = {}

    def _fake_open(req, timeout=None):
        captured["body"] = req.data
        return _FakeResponse(body_bytes="şehir: Köln ✓".encode("utf-8"))

    monkeypatch.setattr("tools.http_request_tool._open", _fake_open)

    result = json.loads(
        http_request_tool("https://example.com", method="POST", body="merhaba dünya")
    )

    assert result["success"] is True
    assert captured["body"] == "merhaba dünya".encode("utf-8")
    assert result["body"] == "şehir: Köln ✓"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def _make_http_error(code=404, reason="Not Found", body=b'{"error": "not found"}'):
    class _FakeHTTPError(urllib.error.HTTPError):
        def __init__(self):
            # HTTPError expects (url, code, msg, hdrs, fp)
            super().__init__(
                "https://api.example.com/missing",
                code,
                reason,
                {"Content-Type": "text/plain"},
                None,
            )
            self._body = body

        def read(self, n=-1):
            return self._body[:n] if n >= 0 else self._body

    return _FakeHTTPError()


def test_http_error_4xx(monkeypatch):
    """4xx responses are returned as structured error, not exceptions."""
    _allow_all_urls(monkeypatch)

    err = _make_http_error(404, "Not Found")

    def _raise(req, timeout=None):
        raise err

    monkeypatch.setattr("tools.http_request_tool._open", _raise)

    result = json.loads(http_request_tool("https://api.example.com/missing"))

    assert result["success"] is False
    assert result["status_code"] == 404
    assert result["status_text"] == "Not Found"
    assert result["error"] == "HTTP 404: Not Found"
    assert "body" in result


def test_http_error_5xx(monkeypatch):
    """5xx responses surface the status and any body."""
    _allow_all_urls(monkeypatch)

    err = _make_http_error(503, "Service Unavailable", b"try later")

    def _raise(req, timeout=None):
        raise err

    monkeypatch.setattr("tools.http_request_tool._open", _raise)

    result = json.loads(http_request_tool("https://api.example.com/flaky"))

    assert result["success"] is False
    assert result["status_code"] == 503
    assert result["body"] == "try later"


def test_url_error_network_failure(monkeypatch):
    """Network failures return structured error."""
    _allow_all_urls(monkeypatch)

    def _raise(req, timeout=None):
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr("tools.http_request_tool._open", _raise)

    result = json.loads(http_request_tool("https://dead-host.example"))

    assert result["success"] is False
    assert "Name or service not known" in result["error"]


# ---------------------------------------------------------------------------
# Security / validation
# ---------------------------------------------------------------------------

def test_rejects_non_http_scheme():
    """file:// and ftp:// are rejected outright."""
    result = json.loads(http_request_tool("file:///etc/passwd"))
    assert result["success"] is False
    assert "http:// or https://" in result["error"]


def test_rejects_unsafe_url(monkeypatch):
    """SSRF protection blocks private/internal URLs."""
    monkeypatch.setattr("tools.http_request_tool.is_safe_url", lambda url: False)

    result = json.loads(http_request_tool("http://192.168.1.1/admin"))
    assert result["success"] is False
    assert "private" in result["error"].lower()


def test_rejects_url_with_embedded_secret(monkeypatch):
    """URLs containing API keys in query params are blocked."""
    _allow_all_urls(monkeypatch)

    result = json.loads(
        http_request_tool("https://api.example.com?api_key=sk-12345678901234567890")
    )
    assert result["success"] is False
    assert "API key" in result["error"]


def test_rejects_unsupported_method():
    """Methods outside the allowed enum are rejected."""
    result = json.loads(http_request_tool("https://example.com", method="FOOBAR"))
    assert result["success"] is False
    assert "Unsupported" in result["error"]


def test_empty_url_rejected():
    """Empty or missing URL returns an error, not an exception."""
    result = json.loads(http_request_tool(""))
    assert result["success"] is False

    result = json.loads(http_request_tool(None))
    assert result["success"] is False


# ---------------------------------------------------------------------------
# Redirect revalidation (SSRF via redirect)
# ---------------------------------------------------------------------------

class TestRedirectGuard:
    def _redirect(self, monkeypatch, newurl, safe):
        monkeypatch.setattr("tools.http_request_tool.is_safe_url", lambda url: safe)
        handler = _SafeRedirectHandler()
        req = urllib.request.Request("https://public.example.com/start")
        return handler.redirect_request(req, None, 302, "Found", {}, newurl)

    def test_private_redirect_target_blocked(self, monkeypatch):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            self._redirect(monkeypatch, "http://169.254.169.254/latest/meta-data/", safe=False)
        assert "Blocked unsafe redirect target" in str(excinfo.value.reason)

    def test_non_http_redirect_target_blocked(self, monkeypatch):
        # Scheme gate fires before is_safe_url, so even a "safe" verdict
        # cannot let a file:// redirect through.
        with pytest.raises(urllib.error.HTTPError):
            self._redirect(monkeypatch, "file:///etc/passwd", safe=True)

    def test_redirect_target_with_embedded_secret_blocked(self, monkeypatch):
        with pytest.raises(urllib.error.HTTPError):
            self._redirect(
                monkeypatch,
                "https://evil.example.com/?api_key=sk-12345678901234567890",
                safe=True,
            )

    def test_safe_redirect_target_allowed(self, monkeypatch):
        new_req = self._redirect(monkeypatch, "https://cdn.example.com/next", safe=True)
        assert isinstance(new_req, urllib.request.Request)
        assert new_req.full_url == "https://cdn.example.com/next"

    def test_open_uses_safe_redirect_handler(self, monkeypatch):
        """_open must build its opener with the revalidating handler."""
        captured = {}

        class _FakeOpener:
            def open(self, req, timeout=None):
                captured["timeout"] = timeout
                return _FakeResponse(body_bytes=b"ok")

        def _fake_build_opener(*handlers):
            captured["handlers"] = handlers
            return _FakeOpener()

        monkeypatch.setattr(
            "tools.http_request_tool.urllib.request.build_opener", _fake_build_opener
        )

        req = urllib.request.Request("https://example.com")
        resp = _open(req, timeout=7)

        assert captured["handlers"] == (_SafeRedirectHandler,)
        assert captured["timeout"] == 7
        assert resp.getcode() == 200

    def test_tool_surfaces_blocked_redirect_as_error(self, monkeypatch):
        """A redirect blocked mid-flight comes back as a structured error."""
        _allow_all_urls(monkeypatch)

        err = urllib.error.HTTPError(
            "https://public.example.com/start",
            302,
            "Blocked unsafe redirect target: URL targets a private, internal, "
            "or cloud-metadata address.",
            {},
            None,
        )

        def _raise(req, timeout=None):
            raise err

        monkeypatch.setattr("tools.http_request_tool._open", _raise)

        result = json.loads(http_request_tool("https://public.example.com/start"))

        assert result["success"] is False
        assert "Blocked unsafe redirect target" in result["error"]


# ---------------------------------------------------------------------------
# Response size cap and server-side clamping
# ---------------------------------------------------------------------------

def test_response_truncation(monkeypatch):
    """Responses larger than max_response_size are truncated."""
    _allow_all_urls(monkeypatch)

    monkeypatch.setattr(
        "tools.http_request_tool._open",
        lambda req, timeout=None: _FakeResponse(body_bytes=b"x" * 200),
    )

    result = json.loads(http_request_tool("https://example.com/big", max_response_size=50))

    assert result["success"] is True
    assert result["body"] == "x" * 50
    assert result["truncated"] is True
    assert "truncated" in result["note"].lower()


class TestServerSideClamps:
    def test_max_response_size_clamped_to_cap(self, monkeypatch):
        """A huge model-provided max_response_size cannot force a larger read."""
        _allow_all_urls(monkeypatch)

        reads = {}

        class _CapturingResponse(_FakeResponse):
            def read(self, size=-1):
                reads["requested"] = size
                return super().read(size)

        monkeypatch.setattr(
            "tools.http_request_tool._open",
            lambda req, timeout=None: _CapturingResponse(
                body_bytes=b"y" * (_MAX_RESPONSE_SIZE_CAP + 100)
            ),
        )

        result = json.loads(
            http_request_tool("https://example.com/huge", max_response_size=10**9)
        )

        assert reads["requested"] == _MAX_RESPONSE_SIZE_CAP + 1
        assert result["truncated"] is True
        assert len(result["body"]) == _MAX_RESPONSE_SIZE_CAP

    def test_timeout_clamped_to_cap(self, monkeypatch):
        _allow_all_urls(monkeypatch)

        captured = {}

        def _fake_open(req, timeout=None):
            captured["timeout"] = timeout
            return _FakeResponse(body_bytes=b"ok")

        monkeypatch.setattr("tools.http_request_tool._open", _fake_open)

        result = json.loads(http_request_tool("https://example.com", timeout=99999))

        assert result["success"] is True
        assert captured["timeout"] == _TIMEOUT_CAP

    def test_non_numeric_limits_fall_back_to_defaults(self, monkeypatch):
        _allow_all_urls(monkeypatch)

        captured = {}

        def _fake_open(req, timeout=None):
            captured["timeout"] = timeout
            return _FakeResponse(body_bytes=b"ok")

        monkeypatch.setattr("tools.http_request_tool._open", _fake_open)

        result = json.loads(
            http_request_tool(
                "https://example.com", timeout="soon", max_response_size="lots"
            )
        )

        assert result["success"] is True
        assert captured["timeout"] == _TIMEOUT_DEFAULT

    def test_schema_declares_bounds(self):
        props = HTTP_REQUEST_SCHEMA["parameters"]["properties"]
        assert props["max_response_size"]["maximum"] == _MAX_RESPONSE_SIZE_CAP
        assert props["max_response_size"]["minimum"] == 1
        assert props["timeout"]["maximum"] == _TIMEOUT_CAP
        assert props["timeout"]["minimum"] == 1


# ---------------------------------------------------------------------------
# Requirement check
# ---------------------------------------------------------------------------

def test_check_requirements_always_true():
    """http_request has no external dependency gate."""
    assert _check_http_request_requirements() is True


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_registry_has_http_request():
    """http_request is discoverable in the tool registry."""
    from tools.registry import discover_builtin_tools, registry

    discover_builtin_tools()
    entry = registry.get_entry("http_request")
    assert entry is not None
    assert entry.name == "http_request"
    assert entry.toolset == "web"
    assert entry.emoji == "🌐"


def test_toolset_includes_http_request():
    """http_request appears in the web toolset definitions."""
    from tools.registry import discover_builtin_tools

    from toolsets import resolve_toolset

    discover_builtin_tools()
    web_tools = resolve_toolset("web")
    assert "http_request" in web_tools


def test_registry_dispatch(monkeypatch):
    """Registry.dispatch() correctly invokes the http_request handler."""
    from tools.registry import discover_builtin_tools, registry

    discover_builtin_tools()
    _allow_all_urls(monkeypatch)

    monkeypatch.setattr(
        "tools.http_request_tool._open",
        lambda req, timeout=None: _FakeResponse(
            body_bytes=b'{"ok": true}',
            headers={"Content-Type": "application/json"},
        ),
    )

    result = json.loads(
        registry.dispatch("http_request", {"url": "https://api.example.com/test"})
    )

    assert result["success"] is True
    assert result["status_code"] == 200
    assert result["body"] == '{"ok": true}'
