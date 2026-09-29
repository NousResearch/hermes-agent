"""Synthetic remote Desktop transport canaries for profile-scoped vault previews.

These tests keep the production vault, preview evaluator, gateway callback, and
server-request code real. Only the remote Desktop frame sink is synthetic; no
provider, SSH host, or user credential is contacted.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def gateway(monkeypatch):
    import tui_gateway.server as server
    from tui_gateway import server_requests

    server_requests.reset_for_tests()
    yield server
    server_requests.reset_for_tests()
    server._sessions.pop("remote-vault-canary", None)


def _seed_login(home, password: str) -> str:
    from agent.vault_store import get_vault_store

    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "vault:\n  onepassword:\n    enabled: false\n  bitwarden:\n    enabled: false\n",
        encoding="utf-8",
    )
    return get_vault_store().add_item(
        "login", "Synthetic remote preview login",
        {"identifier": "canary-user", "identifier_type": "username", "password": password},
        origin="https://login.example.test",
    ).id


class _RemotePreview:
    """Small deterministic renderer at the far end of the real JSON-RPC callback."""

    def __init__(self, binding: str, password: str):
        self.binding = binding
        self.password = password
        self.calls: list[dict] = []
        self.target = f"{binding}:guest-page"
        self.evaluate_count = 0

    def answer(self, frame: dict) -> dict:
        params = frame["params"]
        assert params["action"] == "vault"
        request = params["vault"]
        self.calls.append(dict(request))
        if request["operation"] == "open":
            self.evaluate_count = 0
            return {"success": True, "target": self.target}
        assert request["target"] == self.target
        if request["operation"] == "close":
            return {"success": True}

        self.evaluate_count += 1
        expression = request["expression"]
        if self.evaluate_count == 1:
            return {"success": True, "result": "https://login.example.test/login"}
        if self.evaluate_count == 2:
            return {"success": True, "result": [{
                "index": 0, "formIndex": 0, "type": "password",
                "autocomplete": "current-password", "name": "password",
            }]}
        assert self.evaluate_count == 3
        assert json.dumps(self.password) in expression
        return {"success": True, "result": {"filled": 1}}


def test_remote_preview_dispatch_routes_a_default_b_default_a_default(gateway, monkeypatch, tmp_path):
    """The same session/task labels still resolve and fill from each selected profile home."""
    from agent.redact import clear_vault_redaction_values
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway import server_requests

    sid = "remote-vault-canary"
    session = {"history_lock": threading.Lock(), "history": [], "profile_home": None}
    gateway._sessions[sid] = session
    homes = {name: tmp_path / name for name in ("a", "b")}
    passwords = {
        "a": 'synthetic-A-quote"-slash\\-secret',
        "b": "synthetic-B-distinct-secret",
    }
    handles: dict[str, str] = {}
    for name, home in homes.items():
        token = set_hermes_home_override(home)
        try:
            handles[name] = _seed_login(home, passwords[name])
        finally:
            reset_hermes_home_override(token)

    session = {"history_lock": threading.Lock(), "profile_home": str(homes["a"])}
    gateway._sessions[sid] = session
    active_binding = {"key": "connection-a/default"}
    remote = {
        "connection-a/default": _RemotePreview("connection-a/default", passwords["a"]),
        "connection-b/default": _RemotePreview("connection-b/default", passwords["b"]),
    }
    wire_frames: list[dict] = []
    delivered_bindings: list[str] = []

    def desktop_transport(frame):
        wire_frames.append(frame)
        binding = active_binding["key"]
        delivered_bindings.append(binding)
        answer = remote[binding].answer(frame)
        assert server_requests.resolve_response({
            "id": frame["id"], "result": {"value": json.dumps(answer)},
        })
        return True

    monkeypatch.setattr(server_requests, "_write", desktop_transport)

    try:
        for profile_name in ("a", "b", "a"):
            binding = "connection-a/default" if profile_name == "a" else "connection-b/default"
            active_binding["key"] = binding
            session["profile_home"] = str(homes[profile_name])
            profile_token = set_hermes_home_override(homes[profile_name])
            try:
                agent = SimpleNamespace(**gateway._agent_cbs(sid), valid_tool_names={"drive_preview"})
                raw = INLINE_TOOL_EXECUTORS["browser_vault_fill"](
                    agent,
                    {"handle": handles[profile_name], "target": "preview"},
                    InlineToolContext(effective_task_id="same-task-label"),
                )
                result = json.loads(raw)
                assert result["success"] is True, (
                    result,
                    [(call["operation"], len(call.get("expression", "")))
                     for call in remote[binding].calls],
                )
                assert result["filled_fields"] == 1
                assert result["origin"] == "https://login.example.test"
            finally:
                clear_vault_redaction_values()
                reset_hermes_home_override(profile_token)

        expected_bindings = [
            "connection-a/default", "connection-b/default", "connection-a/default",
        ]
        assert delivered_bindings == [binding for binding in expected_bindings for _ in range(5)]
        assert [frame["params"]["session_id"] for frame in wire_frames] == [sid] * 15
        assert [frame["params"]["vault"]["operation"] for frame in wire_frames] == (
            ["open", "evaluate", "evaluate", "evaluate", "close"] * 3
        )
        assert len(remote["connection-a/default"].calls) == 10
        assert remote["connection-a/default"].evaluate_count == 3
        assert remote["connection-b/default"].evaluate_count == 3
        assert passwords["a"] not in json.dumps([remote["connection-b/default"].calls], ensure_ascii=False)
        assert passwords["b"] not in json.dumps([remote["connection-a/default"].calls], ensure_ascii=False)
        assert gateway._open_requests(sid) == []
        assert server_requests.open_request_count() == 0
    finally:
        gateway._sessions.pop(sid, None)


def test_cancelled_remote_vault_binding_is_not_replayed_or_applied(gateway, monkeypatch):
    """A profile/connection switch withdraws the live secret request; a late answer is ignored."""
    from tui_gateway import server_requests

    sid = "remote-vault-canary"
    session = {"history_lock": threading.Lock(), "history": [], "profile_home": None}
    gateway._sessions[sid] = session
    emitted = []
    sent = threading.Event()
    late_responses = []
    expression = '(() => "synthetic-stale-binding-secret")()'
    monkeypatch.setattr(server_requests, "_answerable", lambda _sid: True)
    monkeypatch.setattr(server_requests, "_emit", lambda event, session_id, payload: emitted.append(
        (event, session_id, payload)))

    def disconnected_desktop(frame):
        assert frame["method"] == "preview.act"
        assert frame["params"]["vault"]["expression"] == expression
        # Pending state has already discarded the secret-bearing payload.
        assert server_requests._open[frame["id"]].params == {}
        sent.set()
        late_responses.append({"id": frame["id"], "result": {"value": "stale"}})
        return True

    monkeypatch.setattr(server_requests, "_write", disconnected_desktop)
    result: dict = {}
    worker = threading.Thread(target=lambda: result.setdefault("value", gateway._ask(
        "preview.act", sid,
        {"action": "vault", "vault": {"operation": "evaluate", "target": "old-guest",
                                         "expression": expression}},
        timeout=None,
    )), daemon=True)
    worker.start()
    assert sent.wait(2), "preview request did not reach the synthetic remote Desktop"
    assert gateway._open_requests(sid) == []
    assert expression not in json.dumps(gateway._open_requests(sid))
    reconnect = gateway._live_session_payload(sid, session, omit_messages=True)
    assert reconnect.get("open_requests", []) == []
    assert expression not in json.dumps(reconnect)

    # The currently focused profile/connection has changed; tear down the old session's ask.
    assert server_requests.cancel(sid, reason="interrupted") == 1
    worker.join(2)
    assert not worker.is_alive()
    assert result["value"] == ""
    assert server_requests.resolve_response(late_responses[0]) is False
    assert gateway._open_requests(sid) == []
    assert [(event, session_id, payload["reason"]) for event, session_id, payload in emitted] == [
        ("request.cancel", sid, "interrupted"),
    ]
