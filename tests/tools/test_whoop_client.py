from __future__ import annotations

import json

import pytest

from plugins.whoop import client as whoop_mod


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, *, text: str = "", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers = headers or {"content-type": "application/json"}
        self.content = self.text.encode("utf-8") if self.text else b""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


_RUNTIME = {
    "access_token": "token-1",
    "base_url": "https://api.prod.whoop.com/developer/v2",
}


def test_whoop_client_uses_bearer_auth_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(
        whoop_mod, "resolve_whoop_runtime_credentials", lambda **kw: {**_RUNTIME}
    )

    def fake_request(method, url, headers=None, params=None, **kw):
        seen.update({"method": method, "url": url, "headers": headers, "params": params})
        return _FakeResponse(200, {"records": []})

    monkeypatch.setattr(whoop_mod.httpx, "request", fake_request)

    client = whoop_mod.WHOOPClient()
    payload = client.request("GET", "/activity/sleep", params={"start": "2026-01-01T00:00:00Z"})

    assert payload == {"records": []}
    assert seen["method"] == "GET"
    assert seen["url"] == "https://api.prod.whoop.com/developer/v2/activity/sleep"
    assert seen["headers"]["Authorization"] == "Bearer token-1"
    assert seen["params"] == {"start": "2026-01-01T00:00:00Z"}


def test_whoop_client_retries_once_after_401(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    tokens = iter([{**_RUNTIME}, {**_RUNTIME, "access_token": "token-2"}])
    monkeypatch.setattr(
        whoop_mod, "resolve_whoop_runtime_credentials", lambda **kw: next(tokens)
    )

    def fake_request(method, url, headers=None, **kw):
        calls.append(headers["Authorization"])
        if len(calls) == 1:
            return _FakeResponse(401, {"error": "expired"})
        return _FakeResponse(200, {"user_id": 42})

    monkeypatch.setattr(whoop_mod.httpx, "request", fake_request)

    client = whoop_mod.WHOOPClient()
    payload = client.request("GET", "/user/profile/basic")
    assert payload["user_id"] == 42
    assert calls == ["Bearer token-1", "Bearer token-2"]


def test_whoop_client_raises_after_double_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        whoop_mod, "resolve_whoop_runtime_credentials", lambda **kw: {**_RUNTIME}
    )
    monkeypatch.setattr(
        whoop_mod.httpx, "request", lambda *a, **kw: _FakeResponse(401, {"error": "denied"})
    )

    client = whoop_mod.WHOOPClient()
    with pytest.raises(whoop_mod.WHOOPAPIError):
        client.request("GET", "/user/profile/basic")


def test_whoop_client_429_error_includes_retry_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        whoop_mod, "resolve_whoop_runtime_credentials", lambda **kw: {**_RUNTIME}
    )
    monkeypatch.setattr(
        whoop_mod.httpx,
        "request",
        lambda *a, **kw: _FakeResponse(
            429,
            {"message": "slow down"},
            headers={"content-type": "application/json", "Retry-After": "30"},
        ),
    )

    client = whoop_mod.WHOOPClient()
    with pytest.raises(whoop_mod.WHOOPAPIError) as exc:
        client.request("GET", "/recovery")

    message = str(exc.value)
    assert "rate limited" in message
    assert "retry after 30s" in message
    assert "slow down" in message


def test_whoop_client_paginates_records_with_next_token(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_params: list[dict] = []
    pages = iter([
        _FakeResponse(200, {"records": [{"id": 1}], "next_token": "page-2"}),
        _FakeResponse(200, {"records": [{"id": 2}], "next_token": None}),
    ])
    monkeypatch.setattr(
        whoop_mod, "resolve_whoop_runtime_credentials", lambda **kw: {**_RUNTIME}
    )

    def fake_request(method, url, headers=None, params=None, **kw):
        seen_params.append(dict(params or {}))
        return next(pages)

    monkeypatch.setattr(whoop_mod.httpx, "request", fake_request)

    client = whoop_mod.WHOOPClient()
    payload = client.list_sleep(start="2026-01-01T00:00:00Z", max_pages=5)

    assert payload["records"] == [{"id": 1}, {"id": 2}]
    assert seen_params[0] == {"start": "2026-01-01T00:00:00Z"}
    assert seen_params[1]["nextToken"] == "page-2"


def test_whoop_client_max_pages_stops_runaway_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    monkeypatch.setattr(
        whoop_mod, "resolve_whoop_runtime_credentials", lambda **kw: {**_RUNTIME}
    )

    def fake_request(method, url, headers=None, params=None, **kw):
        nonlocal calls
        calls += 1
        return _FakeResponse(200, {"records": [{"page": calls}], "next_token": "more"})

    monkeypatch.setattr(whoop_mod.httpx, "request", fake_request)

    client = whoop_mod.WHOOPClient()
    payload = client.list_cycles(max_pages=2)

    assert calls == 2
    assert payload["records"] == [{"page": 1}, {"page": 2}]
    assert payload["next_token"] == "more"

def test_whoop_client_get_recovery_uses_cycle_recovery_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(
        whoop_mod, "resolve_whoop_runtime_credentials", lambda **kw: {**_RUNTIME}
    )

    def fake_request(method, url, headers=None, params=None, **kw):
        seen["url"] = url
        return _FakeResponse(200, {"cycle_id": "cycle-123", "score": {"recovery_score": 91}})

    monkeypatch.setattr(whoop_mod.httpx, "request", fake_request)

    client = whoop_mod.WHOOPClient()
    payload = client.get_recovery("cycle-123")

    assert payload["cycle_id"] == "cycle-123"
    assert seen["url"].endswith("/cycle/cycle-123/recovery")
