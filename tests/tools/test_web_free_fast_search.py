"""Free managed Perplexity fast search for a REGISTERED Nous Portal identity with no usable credits
and no tool pool. The gateway serves ``POST /search`` + ``search_type: "fast"`` without funding
checks, so the local paid-tool gate must not block this one route while every other managed vendor
keeps it. The anonymous guest tier is excluded — it has no Portal account behind it, so it keeps the
keyless ring. Real config, real auth store, real selection-to-provider dispatch over local HTTP."""

import base64
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

# Anonymous guest: NAS tier claim says "anonymous" — no Portal account behind the identity.
ANON = {"sub": "anon-1", "account_tier": "anonymous", "paid_access": False}
# Registered Portal account, signed in, no credits and no tool pool.
EXHAUSTED = {"sub": "user-1", "paid_access": False, "tool_access": {"enabled": False, "coverage": {}}}


def _nous_state(claims: dict, auth_method: str) -> dict:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    token = f"{seg({'alg': 'none'})}.{seg({**claims, 'exp': int(time.time()) + 3600})}.sig"
    return {"auth_method": auth_method, "access_token": token, "expires_at": "2099-01-01T00:00:00Z"}


@pytest.fixture
def gateway_server(monkeypatch):
    class Log(list):
        base = ""

    requests = Log()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, dict(self.headers), body))
            ok = self.path in ("/perplexity/search", "/direct/search")
            content = json.dumps({"results": [{"title": "t", "url": "https://example.test", "snippet": "s"}]}
                                 if ok else {"error": "payment required"}).encode()
            self.send_response(200 if ok else 402)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    requests.base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("PERPLEXITY_GATEWAY_URL", requests.base + "/perplexity")
    monkeypatch.setenv("PERPLEXITY_BASE_URL", requests.base + "/direct")
    monkeypatch.setenv("FIRECRAWL_GATEWAY_URL", requests.base)
    monkeypatch.setattr("tools.web_tools._firecrawl_client", None, raising=False)
    try:
        yield requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _write_home(home, monkeypatch, *, nous_state=None, config=None):
    from hermes_cli.config import atomic_config_write
    from hermes_cli.nous_account import reset_nous_portal_account_info_cache

    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("TOOL_GATEWAY_USER_TOKEN", raising=False)
    atomic_config_write(home / "config.yaml", {"web": {"keyless_rescue": False}, **(config or {})})
    if nous_state is not None:
        (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {"nous": nous_state}}))
    reset_nous_portal_account_info_cache()


def _search():
    from tools import web_tools
    from tests.tools.conftest import register_all_web_providers

    register_all_web_providers()
    return json.loads(web_tools.web_search_tool("free fast search", limit=3))


def _calls(log):
    return [(path, headers["Authorization"], body.get("search_type")) for path, headers, body in log]


def test_zero_credit_portal_identity_autodetects_free_fast_search(monkeypatch, tmp_path, gateway_server):
    from tools import web_tools
    from tools.tool_backend_helpers import managed_nous_tools_enabled

    state = _nous_state(EXHAUSTED, "oauth")
    _write_home(tmp_path / "home", monkeypatch, nous_state=state)
    # Premise: the paid-tool gate is closed for this identity and stays closed for extract.
    assert managed_nous_tools_enabled() is False
    assert web_tools.check_firecrawl_api_key() is False
    assert web_tools._get_extract_backend() != "perplexity"

    result = _search()

    assert result["success"] is True, result
    assert _calls(gateway_server) == [("/perplexity/search", f"Bearer {state['access_token']}", "fast")]


def test_anonymous_guest_keeps_the_keyless_ring(monkeypatch, tmp_path, gateway_server):
    """The guest tier holds a usable Nous token, so the exclusion must come from the tier itself."""
    from tools import web_tools
    from tools.managed_tool_gateway import peek_nous_access_token, resolve_free_search_gateway
    from tools.tool_backend_helpers import fast_search_entitled

    _write_home(tmp_path / "home", monkeypatch, nous_state=_nous_state(ANON, "anonymous"))
    # Premise: a token IS present — anonymous is refused by entitlement, not by a missing credential.
    assert peek_nous_access_token()
    assert fast_search_entitled() is False
    assert resolve_free_search_gateway(token_reader=peek_nous_access_token) is None

    assert web_tools._managed_web_search() is False
    assert web_tools._get_search_backend() != "perplexity"


def test_free_fast_search_failure_skips_paid_firecrawl_but_keeps_keyless_rescue(monkeypatch, tmp_path, gateway_server):
    from plugins.web import keyless_mcp

    state = _nous_state(EXHAUSTED, "oauth")
    _write_home(tmp_path / "home", monkeypatch, nous_state=state, config={"web": {"keyless_rescue": True}})
    monkeypatch.setenv("PERPLEXITY_GATEWAY_URL", gateway_server.base + "/down")
    rescued = {"success": True, "data": {"web": [{"title": "ring", "url": "https://ring.test", "description": "", "position": 1}]}}
    monkeypatch.setattr(keyless_mcp, "search_with_failover", lambda *a, **kw: dict(rescued, data=dict(rescued["data"])))

    result = _search()

    assert result["data"]["rescued_from"] == "perplexity"
    assert _calls(gateway_server) == [("/down/search", f"Bearer {state['access_token']}", "fast")]


@pytest.mark.parametrize("state,config", [
    (_nous_state(ANON, "anonymous"), {"nous": {"guest": False}}),
    (None, {}),
    (_nous_state(EXHAUSTED, "oauth"), {"web": {"search_backend": "exa"}}),
])
def test_free_fast_search_needs_a_usable_identity_and_no_explicit_selection(monkeypatch, tmp_path, gateway_server, state, config):
    from tools import web_tools

    _write_home(tmp_path / "home", monkeypatch, nous_state=state, config=config)

    assert web_tools._managed_web_search() is False
    assert web_tools._get_search_backend() != "perplexity"


def test_direct_perplexity_key_beats_free_fast_search(monkeypatch, tmp_path, gateway_server):
    _write_home(tmp_path / "home", monkeypatch, nous_state=_nous_state(EXHAUSTED, "oauth"))
    monkeypatch.setenv("PERPLEXITY_API_KEY", "direct-key")

    assert _search()["success"] is True
    assert _calls(gateway_server) == [("/direct/search", "Bearer direct-key", None)]


def test_free_fast_search_eligibility_follows_the_served_profile(monkeypatch, tmp_path, gateway_server):
    """Multiplex A→B→A: profile A holds a zero-credit Portal identity, B none; each answers from its own store."""
    from hermes_cli.nous_account import reset_nous_portal_account_info_cache
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from tools import web_tools

    root = tmp_path / ".hermes"
    profile_a, profile_b = root / "profiles" / "a", root / "profiles" / "b"
    _write_home(profile_a, monkeypatch, nous_state=_nous_state(EXHAUSTED, "oauth"), config={"web": {"keyless_fallback": False}})
    _write_home(profile_b, monkeypatch, config={"web": {"keyless_fallback": False}})
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "config.yaml").write_text("")
    set_multiplex_active(True)
    try:
        seen = []
        for home in (profile_a, profile_b, profile_a):
            reset_nous_portal_account_info_cache()
            with _profile_runtime_scope(home):
                seen.append(web_tools._get_search_backend())
    finally:
        set_multiplex_active(False)
    assert seen[0] == seen[2] == "perplexity"
    assert seen[1] != "perplexity"