"""``kanban_attach_url`` must fetch through the connect-time SSRF-safe client.

``is_safe_url`` resolves the host once and then a bare ``httpx.stream`` resolves it again when it
connects, so a rebinding host answers public to the check and loopback (or the cloud metadata
address) to the request. ``tools.url_safety.create_ssrf_safe_client`` re-checks the destination it
actually connects to, and ``trust_env=False`` keeps a proxy mount (which does not run that check)
from carrying the request instead.
"""

from __future__ import annotations

import pytest


class _FakeResponse:
    def __init__(self, body: bytes = b"", *, location: str | None = None):
        self._body = body
        self.headers = {"content-type": "application/pdf"}
        self.is_redirect = location is not None
        if location is not None:
            self.headers["location"] = location

    def raise_for_status(self):
        return None

    def iter_bytes(self, size):
        for start in range(0, len(self._body), size):
            yield self._body[start:start + size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, responses, calls):
        self._responses = responses
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream(self, method, url, **kwargs):
        self._calls.append((method, url, kwargs))
        return self._responses.pop(0)


@pytest.fixture()
def safe_client_factory(monkeypatch):
    """Replace the SSRF-safe client factory and record how kanban calls it."""
    from tools import kanban_tools, url_safety

    factory_calls = []
    responses = []
    stream_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return _FakeClient(responses, stream_calls)

    def bare_httpx_stream(*args, **kwargs):  # pragma: no cover - asserted against
        raise AssertionError("kanban_attach_url used httpx.stream, not the SSRF-safe client")

    monkeypatch.setattr(url_safety, "create_ssrf_safe_client", factory)
    monkeypatch.setattr(kanban_tools, "is_safe_url", lambda url: True, raising=False)
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
    monkeypatch.setattr("httpx.stream", bare_httpx_stream)

    return {"factory": factory_calls, "responses": responses, "streams": stream_calls}


def test_download_uses_the_ssrf_safe_client(safe_client_factory):
    from tools import kanban_tools

    safe_client_factory["responses"].append(_FakeResponse(b"%PDF-1.4 body"))

    data, content_type = kanban_tools._download_url_with_cap(
        "http://files.example.com/docs/spec.pdf", max_bytes=1024)

    assert data == b"%PDF-1.4 body"
    assert content_type == "application/pdf"
    assert safe_client_factory["factory"] == [{"trust_env": False}]


def test_a_redirected_hop_also_uses_the_ssrf_safe_client(safe_client_factory):
    from tools import kanban_tools

    safe_client_factory["responses"].append(
        _FakeResponse(location="http://cdn.example.net/spec.pdf"))
    safe_client_factory["responses"].append(_FakeResponse(b"%PDF-1.4 moved"))

    data, _ = kanban_tools._download_url_with_cap(
        "http://files.example.com/docs/spec.pdf", max_bytes=1024)

    assert data == b"%PDF-1.4 moved"
    assert len(safe_client_factory["factory"]) == 2
    assert all(kwargs == {"trust_env": False} for kwargs in safe_client_factory["factory"])
    assert [call[1] for call in safe_client_factory["streams"]] == [
        "http://files.example.com/docs/spec.pdf",
        "http://cdn.example.net/spec.pdf",
    ]
