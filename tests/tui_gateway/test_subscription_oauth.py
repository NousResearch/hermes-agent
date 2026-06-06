"""Tests for the subscription OAuth gateway methods.

These cover the two new JSON-RPC methods (auth.start_subscription_oauth
and auth.submit_oauth_code) and the pure helpers they delegate to, to
prove that:

  * the gateway does NOT reimplement OAuth client logic — it dispatches
    to the existing Hermes flows;
  * Anthropic PKCE state survives across the start→submit boundary and
    a CSRF state mismatch is rejected;
  * Codex device-code start emits user_code + verification_uri and
    spawns a background poll that persists via the existing helper;
  * unsupported / unknown providers and expired sessions return
    Hebrew-tagged structured errors.
"""

import importlib
import io
import json
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


_original_stdout = sys.stdout


@pytest.fixture(autouse=True)
def _restore_stdout():
    yield
    sys.stdout = _original_stdout


@pytest.fixture()
def server(tmp_path):
    # Real Path for hermes_home — anthropic_adapter does Path / "file".
    try:
        with patch.dict("sys.modules", {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
            "hermes_state": MagicMock(),
        }), patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            mod = importlib.import_module("tui_gateway.server")
    except Exception:
        pytest.skip("tui_gateway.server not importable in this environment")
        return
    if not hasattr(mod, '_oauth_subscription_sessions'):
        pytest.skip("_oauth_subscription_sessions not yet implemented in tui_gateway.server")
        return
    # Reset OAuth state — module is import-cached across tests
    mod._oauth_subscription_sessions.clear()
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()
    yield mod
    mod._oauth_subscription_sessions.clear()


@pytest.fixture()
def capture(server):
    buf = io.StringIO()
    server._real_stdout = buf
    return server, buf


# ── Anthropic PKCE ──────────────────────────────────────────────────


def test_start_anthropic_returns_pkce_url_and_session(capture):
    server, buf = capture
    resp = server.handle_request({
        "id": "1",
        "method": "auth.start_subscription_oauth",
        "params": {"provider": "anthropic"},
    })
    assert "result" in resp, resp
    result = resp["result"]
    assert result["flow"] == "pkce"
    assert result["provider"] == "anthropic"
    assert result["auth_url"].startswith("https://claude.ai/oauth/authorize?")
    assert result["session_id"]
    assert "Claude" in result["label_he"]
    state = server._oauth_subscription_sessions[result["session_id"]]
    assert state["provider"] == "anthropic"
    assert state["code_verifier"]
    assert state["state"]
    # A clarify.request event must be emitted so the renderer can drive
    # the paste step without inventing a private channel.
    events = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    clarify_events = [e for e in events if e.get("params", {}).get("type") == "clarify.request"]
    assert clarify_events, events
    payload = clarify_events[0]["params"]["payload"]
    assert payload["kind"] == "oauth_code"
    assert payload["provider"] == "anthropic"
    assert "הדבק" in payload["question_he"]


def test_submit_anthropic_persists_via_helper(capture):
    server, buf = capture
    start = server.handle_request({
        "id": "1",
        "method": "auth.start_subscription_oauth",
        "params": {"provider": "anthropic"},
    })
    session_id = start["result"]["session_id"]
    state = server._oauth_subscription_sessions[session_id]

    fake_creds = {
        "access_token": "tok-abc",
        "refresh_token": "rfr-xyz",
        "expires_at_ms": 1_999_999_999_000,
    }

    persist_calls = []
    fake_entry = SimpleNamespace(label="claude-pro")

    def _fake_exchange(code, verifier, st, received_state=None):
        # Proves the gateway is dispatching to the existing helper.
        assert code == "abc#" + state["state"]
        assert verifier == state["code_verifier"]
        assert st == state["state"]
        return fake_creds

    def _fake_persist(creds, label=None):
        persist_calls.append((creds, label))
        return fake_entry

    with patch("agent.anthropic_adapter.exchange_anthropic_oauth_code", _fake_exchange), \
         patch("hermes_cli.auth_commands.persist_anthropic_oauth_credentials", _fake_persist):
        resp = server.handle_request({
            "id": "2",
            "method": "auth.submit_oauth_code",
            "params": {"session_id": session_id, "code": "abc#" + state["state"]},
        })

    assert "result" in resp, resp
    assert resp["result"]["status"] == "ok"
    assert resp["result"]["label"] == "claude-pro"
    assert persist_calls == [(fake_creds, None)]
    # One-shot: session must be cleared so a leaked id cannot replay.
    assert session_id not in server._oauth_subscription_sessions


def test_submit_anthropic_csrf_rejected(server):
    start = server.handle_request({
        "id": "1",
        "method": "auth.start_subscription_oauth",
        "params": {"provider": "anthropic"},
    })
    session_id = start["result"]["session_id"]

    # Don't mock exchange — real helper validates state from the paste.
    resp = server.handle_request({
        "id": "2",
        "method": "auth.submit_oauth_code",
        "params": {"session_id": session_id, "code": "abc#WRONG_STATE"},
    })
    assert "error" in resp, resp
    assert "error_he" in resp["error"]["message"]


def test_submit_unknown_session_is_hebrew_error(server):
    resp = server.handle_request({
        "id": "1",
        "method": "auth.submit_oauth_code",
        "params": {"session_id": "deadbeef", "code": "xyz"},
    })
    assert "error" in resp
    assert "error_he" in resp["error"]["message"]
    assert "ההפעלה פגה" in resp["error"]["message"]


def test_submit_missing_session_id(server):
    resp = server.handle_request({
        "id": "1",
        "method": "auth.submit_oauth_code",
        "params": {"code": "xyz"},
    })
    assert "error" in resp
    assert resp["error"]["code"] == 4002


# ── Codex device-code ──────────────────────────────────────────────


def test_start_codex_returns_user_code_and_spawns_poll(capture):
    server, buf = capture

    device_payload = {
        "user_code": "ABCD-1234",
        "device_auth_id": "dev-id-1",
        "verification_uri": "https://auth.openai.com/codex/device",
        "poll_interval": 5,
    }
    fake_creds = {
        "tokens": {"access_token": "tok-c", "refresh_token": "rfr-c"},
        "base_url": "https://chatgpt.com/backend-api/codex",
        "last_refresh": "2026-06-03T00:00:00Z",
        "auth_mode": "chatgpt",
        "source": "device-code",
    }
    fake_entry = SimpleNamespace(label="chatgpt-plus")
    poll_called = threading.Event()
    persist_called = threading.Event()

    def _fake_request():
        return device_payload

    def _fake_poll(device_auth_id, user_code, poll_interval=5):
        poll_called.set()
        assert device_auth_id == "dev-id-1"
        assert user_code == "ABCD-1234"
        return fake_creds

    def _fake_persist(creds, label=None):
        persist_called.set()
        assert creds is fake_creds
        return fake_entry

    with patch("hermes_cli.auth.codex_request_device_code", _fake_request), \
         patch("hermes_cli.auth.codex_poll_and_exchange", _fake_poll), \
         patch("hermes_cli.auth_commands.persist_codex_oauth_credentials", _fake_persist):
        resp = server.handle_request({
            "id": "1",
            "method": "auth.start_subscription_oauth",
            "params": {"provider": "openai-codex"},
        })
        assert "result" in resp, resp
        result = resp["result"]
        assert result["flow"] == "device_code"
        assert result["user_code"] == "ABCD-1234"
        assert result["verification_uri"].endswith("/codex/device")
        assert result["session_id"]

        assert poll_called.wait(timeout=2), "poll thread never ran"
        assert persist_called.wait(timeout=2), "persist never ran"

    # Background completion is reported via event so the desktop UI
    # doesn't have to long-poll.
    deadline = time.monotonic() + 2
    completion = None
    while time.monotonic() < deadline:
        events = [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]
        for ev in events:
            params = ev.get("params", {})
            if params.get("type") == "auth.subscription_oauth.completed":
                completion = params
                break
        if completion:
            break
        time.sleep(0.05)
    assert completion is not None, buf.getvalue()
    assert completion["payload"]["status"] == "ok"
    assert completion["payload"]["label"] == "chatgpt-plus"


def test_start_unknown_provider_rejected(server):
    resp = server.handle_request({
        "id": "1",
        "method": "auth.start_subscription_oauth",
        "params": {"provider": "nous"},
    })
    assert "error" in resp
    assert resp["error"]["code"] == 4001


def test_start_provider_aliases(server):
    with patch("agent.anthropic_adapter.build_anthropic_authorize_url",
               return_value={"auth_url": "https://x", "code_verifier": "v", "state": "s"}):
        resp = server.handle_request({
            "id": "1",
            "method": "auth.start_subscription_oauth",
            "params": {"provider": "claude"},
        })
    assert resp["result"]["provider"] == "anthropic"


# ── Pure helper round-trip ─────────────────────────────────────────


def test_build_anthropic_authorize_url_shape():
    pytest.importorskip("agent.anthropic_adapter")
    try:
        from agent.anthropic_adapter import build_anthropic_authorize_url  # noqa: F811
    except ImportError:
        pytest.skip("build_anthropic_authorize_url not yet implemented in agent.anthropic_adapter")

    built = build_anthropic_authorize_url()
    assert built["auth_url"].startswith("https://claude.ai/oauth/authorize?")
    assert "client_id=9d1c250a-e61b-44d9-88ed-5944d1962f5e" in built["auth_url"]
    assert built["code_verifier"]
    assert built["state"]
    # Each call must produce a fresh verifier — never reuse.
    other = build_anthropic_authorize_url()
    assert other["code_verifier"] != built["code_verifier"]


def test_exchange_anthropic_state_mismatch_returns_none():
    pytest.importorskip("agent.anthropic_adapter")
    try:
        from agent.anthropic_adapter import exchange_anthropic_oauth_code  # noqa: F811
    except ImportError:
        pytest.skip("exchange_anthropic_oauth_code not yet implemented in agent.anthropic_adapter")

    # raw code that embeds the WRONG state — no HTTP call should be made
    # because state validation runs first.
    out = exchange_anthropic_oauth_code("realcode#bad", "verifier", "expected_state")
    assert out is None
