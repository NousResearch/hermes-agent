"""Exercise usage RPCs through real config/auth resolution and loopback HTTP."""

import json
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest


@pytest.fixture
def usage_runtime(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    profile = tmp_path / "profile"
    launch.mkdir()
    profile.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    # Distinct credentials catch accidentally consulting the launch profile.
    for home, token in ((launch, "wrong-launch-token"), (profile, "profile-pool-token")):
        (home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {"openai-codex": [{
                "source": "device_code", "access_token": token,
                "refresh_token": "unused-test-refresh", "last_status": "ok", "auth_type": "oauth",
            }]},
        }))
    (profile / "config.yaml").write_text("model:\n  provider: openai-codex\n")
    requests = []
    entered = threading.Event()
    release = threading.Event()
    release.set()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            entered.set()
            release.wait(5)
            payload = json.dumps({
                "plan_type": "plus",
                "rate_limit": {
                    "primary_window": {"used_percent": 7},
                    "secondary_window": {"used_percent": 20},
                },
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{http.server_port}"
    monkeypatch.setenv("HERMES_CODEX_BASE_URL", base_url)
    stdout = sys.stdout
    from tui_gateway import server
    sys.stdout = stdout
    session = {
        "session_key": "usage-integration", "agent": None,
        "history": [], "history_lock": threading.Lock(), "profile_home": str(profile),
    }
    server._sessions["usage-integration"] = session
    try:
        yield SimpleNamespace(server=server, session=session, requests=requests,
                              base_url=base_url, entered=entered, release=release)
    finally:
        release.set()
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)
        server._close_session_by_id("usage-integration", end_reason="test_cleanup")


@pytest.mark.parametrize("multiplex", [False, True])
def test_quota_worker_binds_and_resets_selected_profile_secrets(tmp_path, monkeypatch, multiplex):
    from concurrent.futures import ThreadPoolExecutor
    from agent import account_usage, secret_scope
    from tui_gateway import usage_provider

    profile = tmp_path / "quota-profile"
    profile.mkdir()
    (profile / ".env").write_text("OPENROUTER_API_KEY=profile-only-key\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "wrong-launch-key")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", multiplex)
    seen = []

    def fetch(*args, **kwargs):
        seen.append(secret_scope.get_secret("OPENROUTER_API_KEY"))
        raise RuntimeError("quota endpoint unavailable")

    monkeypatch.setattr(account_usage, "fetch_account_usage", fetch)
    session = {"profile_home": str(profile), "agent": SimpleNamespace(
        provider="openrouter", base_url="https://openrouter.ai/api/v1", api_key=None)}
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(usage_provider, "_account_usage_pool", pool)
        assert usage_provider._usage_provider_lines(session) == ([], [])
        assert seen == ["profile-only-key"]
        assert pool.submit(secret_scope.current_secret_scope).result(timeout=5) is None


@pytest.mark.parametrize("identity", ["live", "mirror", "config"])
def test_usage_windows_resolve_real_profile_credentials(usage_runtime, identity):
    runtime = usage_runtime
    if identity == "live":
        runtime.session["agent"] = SimpleNamespace(
            provider="openai-codex", base_url=runtime.base_url, api_key="live-test-token")
    elif identity == "mirror":
        # No usage snapshot yet: provider limits must still be visible.
        runtime.session["_metadata_mirror"] = {"provider": "openai-codex"}
    params = {"session_id": "usage-integration"}
    desktop = runtime.server.handle_request({
        "id": 1, "method": "slash.exec", "params": {**params, "command": "usage"}})
    tui = runtime.server.handle_request({"id": 2, "method": "session.usage", "params": params})
    text = desktop["result"]["output"]
    lines = "\n".join(tui["result"]["account_lines"])
    for rendered in (text, lines):
        assert "Session: 93% remaining" in rendered
        assert "Weekly: 80% remaining" in rendered
        assert all(token not in rendered for token in (
            "live-test-token", "profile-pool-token", "wrong-launch-token"))
    expected_token = "live-test-token" if identity == "live" else "profile-pool-token"
    assert runtime.requests == [("/api/codex/usage", f"Bearer {expected_token}")] * 2
    assert "api_key" not in runtime.session
    from hermes_constants import get_hermes_home_override
    assert get_hermes_home_override() is None


@pytest.mark.parametrize("method", ["session.usage", "slash.exec"])
def test_slow_usage_endpoint_does_not_block_rpc_reader(usage_runtime, monkeypatch, method):
    from tui_gateway import usage_provider

    runtime = usage_runtime
    runtime.release.clear()
    monkeypatch.setattr(usage_provider, "_ACCOUNT_USAGE_TIMEOUT_SECONDS", 0.2)
    frames = queue.Queue()

    class Transport:
        def write(self, frame):
            frames.put(frame)

    params = {"session_id": "usage-integration", "command": "usage"}
    try:
        assert runtime.server.dispatch({"id": 1, "method": method, "params": params}, Transport()) is None
        assert runtime.entered.wait(5), "real account request never started"
        ping = runtime.server.dispatch({"id": 2, "method": "ping", "params": {}}, Transport())
        assert ping["id"] == 2 and "result" in ping
        response = frames.get(timeout=5)
        assert response["id"] == 1 and "result" in response
        assert "account_lines" not in response["result"]
        assert not runtime.release.is_set(), "usage should time out before HTTP is released"
    finally:
        runtime.release.set()
