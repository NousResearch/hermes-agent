"""Zendesk OAuth (client_credentials) for the zendesk toolsets: credentials via
get_secret only, least-privilege scopes, a per-profile token cache with expiry
refresh, one retry on 401, and no token or secret in any error. HTTP is mocked
at _token_request / _zendesk_request, or at aiohttp.ClientSession for the
transport tests."""
import json

import pytest

import tools.zendesk_tool as zt
from model_tools import _run_async

BASE = "https://everafterai.zendesk.com"
SECRET = "sec-SUPERSECRET-123"


@pytest.fixture(autouse=True)
def _fresh_cache():
    zt._TOKEN_CACHE.clear()
    yield
    zt._TOKEN_CACHE.clear()


@pytest.fixture
def secrets(monkeypatch):
    import agent.secret_scope as ss
    vals = {"ZENDESK_SUBDOMAIN": "everafterai", "ZENDESK_OAUTH_CLIENT_ID": "cid",
            "ZENDESK_OAUTH_CLIENT_SECRET": SECRET}
    monkeypatch.setattr(ss, "get_secret", lambda n, d=None: vals.get(n, d))
    return vals


class TokenServer:
    def __init__(self, monkeypatch, expires_in=None):
        self.calls = []
        self.n = 0
        self.expires_in = expires_in
        self.status = 201
        self.body = None
        monkeypatch.setattr(zt, "_token_request", self.handle)

    async def handle(self, base, payload):
        self.calls.append((base, dict(payload)))
        if self.body is not None:
            return {"status": self.status, "text": self.body}
        self.n += 1
        tok = {"access_token": f"AT{self.n}", "token_type": "bearer", "scope": payload["scope"]}
        if self.expires_in is not None:
            tok["expires_in"] = self.expires_in
        return {"status": self.status, "text": json.dumps(tok)}


class Api:
    def __init__(self, monkeypatch, statuses=(200,)):
        self.calls = []
        self.statuses = list(statuses)
        monkeypatch.setattr(zt, "_zendesk_request", self.handle)

    async def handle(self, base, token, path, method="GET", json_body=None):
        self.calls.append((token, method, path))
        status = self.statuses.pop(0) if self.statuses else 200
        return {"status": status, "text": '{"ok": 1}' if status < 300 else '{"error": "x"}'}


def _read(path="api/v2/help_center/categories.json"):
    return json.loads(_run_async(zt._zendesk_api_handler({"path": path})))


# ── credentials ──────────────────────────────────────────────────────────

def test_creds_are_oauth_client_only(monkeypatch, secrets):
    assert zt._zendesk_creds() == (BASE, "cid", SECRET)
    assert zt._check_zendesk() is True


def test_api_token_vars_are_ignored(monkeypatch):
    import agent.secret_scope as ss
    legacy = {"ZENDESK_SUBDOMAIN": "everafterai", "ZENDESK_EMAIL": "bot@ex.com",
              "ZENDESK_API_TOKEN": "legacy-token"}
    monkeypatch.setattr(ss, "get_secret", lambda n, d=None: legacy.get(n, d))
    monkeypatch.setenv("ZENDESK_OAUTH_CLIENT_ID", "from-environ")
    monkeypatch.setenv("ZENDESK_OAUTH_CLIENT_SECRET", "from-environ")
    assert zt._zendesk_creds() == ("", "", "")
    assert zt._check_zendesk() is False
    out = _read()
    assert "ZENDESK_OAUTH_CLIENT_ID" in out["error"]


def test_unscoped_secret_is_not_configured(monkeypatch):
    import agent.secret_scope as ss

    def unscoped(n, d=None):
        raise ss.UnscopedSecretError(n)

    monkeypatch.setattr(ss, "get_secret", unscoped)
    assert zt._zendesk_creds() == ("", "", "")


# ── token fetch, scopes, cache ───────────────────────────────────────────

def test_token_request_payload_and_scopes(monkeypatch, secrets):
    srv = TokenServer(monkeypatch)
    assert _run_async(zt._access_token(zt._READ_SCOPE)) == "AT1"
    assert _run_async(zt._access_token(zt._WRITE_SCOPE)) == "AT2"
    (b1, p1), (b2, p2) = srv.calls
    assert b1 == b2 == BASE
    assert p1 == {"grant_type": "client_credentials", "client_id": "cid",
                  "client_secret": SECRET, "scope": "hc:read", "expires_in": 1800}
    assert p2["scope"] == "hc:read hc:write"
    for _b, p in srv.calls:
        scopes = set(p["scope"].split())
        assert scopes <= {"hc:read", "hc:write"}  # never global read/write or tickets


def test_read_tool_uses_bearer_with_read_only_scope(monkeypatch, secrets):
    srv = TokenServer(monkeypatch)
    api = Api(monkeypatch)
    assert _read()["ok"] is True
    assert api.calls == [("AT1", "GET", "api/v2/help_center/categories.json")]
    assert [p["scope"] for _b, p in srv.calls] == ["hc:read"]


def test_write_uses_write_scope(monkeypatch, secrets):
    srv = TokenServer(monkeypatch)
    api = Api(monkeypatch, statuses=[201])
    body = {"article": {"title": "T", "locale": "en-us", "body": "<p>x</p>", "draft": True,
                        "permission_group_id": 1, "user_segment_id": None},
            "notify_subscribers": False}
    _run_async(zt._zendesk_write("POST", "api/v2/help_center/en-us/sections/1/articles.json", body))
    assert [p["scope"] for _b, p in srv.calls] == ["hc:read hc:write"]
    assert api.calls[0][:2] == ("AT1", "POST")


def test_token_cached_until_expiry_then_refreshed(monkeypatch, secrets):
    clock = {"t": 1000.0}
    monkeypatch.setattr(zt, "_monotonic", lambda: clock["t"])
    srv = TokenServer(monkeypatch)  # no expires_in in the response → requested 1800 applies
    api = Api(monkeypatch)
    _read()
    _read()
    assert len(srv.calls) == 1
    clock["t"] += 1800 - zt._TOKEN_SKEW_SECONDS - 1
    _read()
    assert len(srv.calls) == 1
    clock["t"] += 2  # inside the skew window: refresh before Zendesk expires it
    _read()
    assert len(srv.calls) == 2
    assert [c[0] for c in api.calls] == ["AT1", "AT1", "AT1", "AT2"]


def test_expires_in_from_response_wins(monkeypatch, secrets):
    clock = {"t": 0.0}
    monkeypatch.setattr(zt, "_monotonic", lambda: clock["t"])
    srv = TokenServer(monkeypatch, expires_in=400)
    Api(monkeypatch)
    _read()
    clock["t"] += 400 - zt._TOKEN_SKEW_SECONDS + 1
    _read()
    assert len(srv.calls) == 2


def test_401_drops_token_and_retries_once(monkeypatch, secrets):
    srv = TokenServer(monkeypatch)
    api = Api(monkeypatch, statuses=[401, 200])
    assert _read()["ok"] is True
    assert [c[0] for c in api.calls] == ["AT1", "AT2"]
    assert len(srv.calls) == 2
    # A second 401 is returned as an error, not retried forever.
    api.statuses = [401, 401, 401]
    out = _read()
    assert "401" in out["error"]
    assert len(api.calls) == 4  # one try + exactly one retry


def test_cache_is_per_profile(monkeypatch, secrets, tmp_path):
    srv = TokenServer(monkeypatch)
    api = Api(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    _read()
    _read()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "b"))
    _read()
    assert len(srv.calls) == 2
    assert [c[0] for c in api.calls] == ["AT1", "AT1", "AT2"]


def test_cache_is_per_client(monkeypatch, secrets):
    srv = TokenServer(monkeypatch)
    Api(monkeypatch)
    _read()
    secrets["ZENDESK_OAUTH_CLIENT_ID"] = "other-client"
    _read()
    assert [p["client_id"] for _b, p in srv.calls] == ["cid", "other-client"]


# ── nothing secret ever surfaces ─────────────────────────────────────────

def test_token_endpoint_error_never_leaks_secret(monkeypatch, secrets):
    srv = TokenServer(monkeypatch)
    srv.status = 401
    srv.body = json.dumps({"error": "invalid_client",
                           "error_description": f"bad client secret {SECRET}"})
    out = _read()
    assert "invalid_client" in out["error"]
    assert SECRET not in json.dumps(out)
    srv.status = 400
    srv.body = json.dumps({"error": "invalid_scope"})
    assert "invalid_scope" in _read()["error"]


def test_api_error_never_leaks_token(monkeypatch, secrets):
    TokenServer(monkeypatch)

    async def echo(base, token, path, method="GET", json_body=None):
        return {"status": 500, "text": f"upstream said Bearer {token} and {SECRET}"}

    monkeypatch.setattr(zt, "_zendesk_request", echo)
    out = json.dumps(_read())
    assert "AT1" not in out and SECRET not in out and "500" in out


def test_transport_exception_never_leaks(monkeypatch, secrets):
    TokenServer(monkeypatch)

    async def boom(base, token, path, method="GET", json_body=None):
        raise RuntimeError(f"connection reset while sending Authorization: Bearer {token}")

    monkeypatch.setattr(zt, "_zendesk_request", boom)
    out = json.dumps(_read())
    assert "AT1" not in out and "failed" in out


# ── transport: fixed host, no redirects, Bearer ──────────────────────────

class _Resp:
    def __init__(self, status, text):
        self.status, self._text = status, text

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session:
    seen = []

    def __init__(self, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def request(self, method, url, **kw):
        _Session.seen.append((method, url, kw))
        return _Resp(201, '{"access_token": "AT", "token_type": "bearer"}')

    def post(self, url, **kw):
        return self.request("POST", url, **kw)


def test_transport_fixed_host_no_redirects_bearer(monkeypatch):
    import aiohttp
    _Session.seen = []
    monkeypatch.setattr(aiohttp, "ClientSession", _Session)
    _run_async(zt._token_request(BASE, {"grant_type": "client_credentials"}))
    _run_async(zt._zendesk_request(BASE, "AT", "api/v2/help_center/categories.json"))
    (m1, u1, k1), (m2, u2, k2) = _Session.seen
    assert (m1, u1) == ("POST", f"{BASE}/oauth/tokens")
    assert k1["allow_redirects"] is False and k1["json"] == {"grant_type": "client_credentials"}
    assert "Authorization" not in (k1.get("headers") or {})
    assert (m2, u2) == ("GET", f"{BASE}/api/v2/help_center/categories.json")
    assert k2["allow_redirects"] is False
    assert k2["headers"]["Authorization"] == "Bearer AT"
