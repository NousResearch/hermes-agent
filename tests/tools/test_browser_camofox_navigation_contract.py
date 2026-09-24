"""Camofox wire-contract regressions for PR #93249 and 0xble's feedback.

Uses a real loopback HTTP fixture, not a Camofox engine. Tab creation and
navigation helpers are deliberately not mocked: request bodies and request
counts must prove that creation cannot consume a one-time navigation URL.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest
import requests

from tools import browser_camofox as cf
from tools import browser_tool as bt
from tools import browser_tool_eval_policy as policy

_REAL_PAGE_GUARD = cf._camofox_private_page_block


class _State:
    def __init__(self):
        self.calls = []
        self.tabs = {}
        self.loads = []
        self.failures = {}
        self.created = 0


@pytest.fixture
def wire(monkeypatch, tmp_path):
    state = _State()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("CAMOFOX_ADOPT_EXISTING_TAB", raising=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, payload):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def handle_request(self):
            path = urlsplit(self.path).path
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size)) if size else {}
            state.calls.append((self.command, path, body, self.headers.get("Authorization")))
            failures = state.failures.get(path, [])
            if failures:
                status, payload = failures.pop(0)
                return self.reply(status, payload)
            if path == "/tabs" and self.command == "GET":
                return self.reply(200, {"tabs": [
                    {"tabId": key, "listItemId": value["listItemId"]}
                    for key, value in state.tabs.items()
                ]})
            if path == "/tabs" and self.command == "POST":
                state.created += 1
                tab_id = f"fresh-{state.created}"
                state.tabs[tab_id] = {**body, "url": "about:blank"}
                # Model the review's allocate-then-validate failure. The test
                # proves the client never sends this bad request at all.
                if "url" in body:
                    if urlsplit(body["url"]).scheme not in {"http", "https"}:
                        return self.reply(400, {"error": "Unsupported URL scheme"})
                    state.tabs[tab_id]["url"] = body["url"]
                    state.loads.append((tab_id, body["url"]))
                return self.reply(200, {"tabId": tab_id})
            parts = path.strip("/").split("/")
            if len(parts) != 3 or parts[0] != "tabs":
                return self.reply(404, {"error": "route not found"})
            tab_id, operation = parts[1:]
            if tab_id not in state.tabs:
                return self.reply(404, {"code": "tab_not_found"})
            if operation == "navigate":
                state.tabs[tab_id]["url"] = body["url"]
                state.loads.append((tab_id, body["url"]))
            return self.reply(200, {
                "ok": True, "url": state.tabs[tab_id]["url"],
                "snapshot": "- heading Contract fixture", "refsCount": 0,
                "result": "2",
            })

        do_GET = handle_request
        do_POST = handle_request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr(cf, "get_camofox_url", lambda: f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(cf, "_auth_headers", lambda: {"Authorization": "Bearer fixture-only"})
    monkeypatch.setattr(cf, "_get_command_timeout", lambda: 5)
    monkeypatch.setattr(cf, "get_vnc_url", lambda: None)
    monkeypatch.setattr(cf, "_get_camofox_config", lambda: {"adopt_existing_tab": False})
    monkeypatch.setattr(cf, "_camofox_identity_override", lambda *_: {
        "user_id": "fixture-user", "session_key": "fixture-session",
    })
    monkeypatch.setattr(cf, "_rewrite_loopback_url_for_camofox", lambda url: (url, None))
    monkeypatch.setattr(cf, "_camofox_private_page_block", lambda *_: None)
    monkeypatch.setattr(policy, "_eval_ssrf_guard_active", lambda *_: False)
    with cf._sessions_lock:
        cf._sessions.clear()
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        with cf._sessions_lock:
            cf._sessions.clear()
        assert not thread.is_alive()


def _seed(state, tab_id="existing"):
    session = cf._get_session("contract")
    session["tab_id"] = tab_id
    state.tabs[tab_id] = {"url": "https://old.example/", "listItemId": session["session_key"]}
    return session


def _posts(state):
    return [(path, body) for method, path, body, _ in state.calls if method == "POST"]


@pytest.mark.parametrize("arguments", [(), (None,), ("about:blank",)])
def test_blank_creation_omits_url(wire, arguments):
    session = cf._ensure_tab("contract", *arguments)
    assert session["tab_id"] == "fresh-1"
    assert _posts(wire) == [("/tabs", {"userId": "fixture-user", "listItemId": "fixture-session"})]
    assert wire.loads == []
    assert len(wire.tabs) == 1


def test_explicit_creation_url_remains_supported(wire):
    session = cf._ensure_tab("contract", "https://example.test/direct")
    assert wire.loads == [(session["tab_id"], "https://example.test/direct")]
    assert len(_posts(wire)) == 1


def test_fresh_navigation_loads_target_exactly_once(wire):
    target = "https://example.test/one-time?token=fixture-token"
    result = json.loads(cf.camofox_navigate(target, task_id="contract"))
    assert result["success"] is True and result["url"] == target
    assert wire.loads == [("fresh-1", target)]
    assert _posts(wire) == [
        ("/tabs", {"userId": "fixture-user", "listItemId": "fixture-session"}),
        ("/tabs/fresh-1/navigate", {"userId": "fixture-user", "url": target}),
    ]
    assert all(auth == "Bearer fixture-only" for _, _, _, auth in wire.calls)


@pytest.mark.parametrize("adopt", [False, True])
def test_reuse_or_adoption_still_navigates(wire, monkeypatch, adopt):
    session = _seed(wire)
    if adopt:
        with cf._sessions_lock:
            cf._sessions.clear()
        monkeypatch.setattr(cf, "_get_camofox_config", lambda: {"adopt_existing_tab": True})
    target = "https://example.test/new-target"
    result = json.loads(cf.camofox_navigate(target, task_id="contract"))
    assert result["success"] is True
    assert wire.loads == [("existing", target)]
    assert _posts(wire) == [("/tabs/existing/navigate", {"userId": session["user_id"], "url": target})]
    assert wire.created == 0


@pytest.mark.parametrize("status,payload", [(410, {}), (404, {"code": "tab_not_found"})])
def test_recovery_creates_blank_and_loads_target_once(wire, status, payload):
    session = _seed(wire)
    identity = session["user_id"], session["session_key"]
    wire.failures["/tabs/existing/navigate"] = [(status, payload)]
    result = json.loads(cf.camofox_navigate("https://example.test/one-time", task_id="contract"))
    assert result["success"] is True
    assert wire.loads == [("fresh-1", "https://example.test/one-time")]
    assert [path for path, _ in _posts(wire)] == [
        "/tabs/existing/navigate", "/tabs", "/tabs/fresh-1/navigate",
    ]
    assert "url" not in _posts(wire)[1][1]
    assert session["tab_id"] == "fresh-1"
    assert (session["user_id"], session["session_key"]) == identity


@pytest.mark.parametrize("status,payload,cleared", [
    (410, {}, True), (404, {"error": "tab not found"}, True),
    (404, {"error": "route not found"}, False), (500, {}, False), (503, {}, False),
])
def test_second_navigation_failure_is_bounded(wire, status, payload, cleared):
    session = _seed(wire)
    wire.failures["/tabs/existing/navigate"] = [(410, {})]
    wire.failures["/tabs/fresh-1/navigate"] = [(status, payload)]
    result = json.loads(cf.camofox_navigate("https://example.test/one-time", task_id="contract"))
    assert result["success"] is False
    assert session["tab_id"] == (None if cleared else "fresh-1")
    if cleared:
        assert "Call browser_navigate" in result["error"]
    assert wire.created == 1
    assert len(_posts(wire)) == 3
    assert wire.loads == []


@pytest.mark.parametrize("status,payload", [
    (404, {}), (404, {"error": "route not found"}), (401, {}),
    (405, {}), (500, {"code": "tab_not_found"}), (503, {}),
])
def test_non_stale_navigation_failure_keeps_identity_and_never_retries(wire, status, payload):
    session = _seed(wire)
    before = session.copy()
    wire.failures["/tabs/existing/navigate"] = [(status, payload)]
    result = json.loads(cf.camofox_navigate("https://example.test/target", task_id="contract"))
    assert result["success"] is False
    assert session == before
    assert len(_posts(wire)) == 1
    assert wire.created == 0
    assert wire.loads == []


@pytest.mark.parametrize("exception", [requests.Timeout("fixture timeout"), requests.ConnectionError("fixture disconnect")])
def test_ambiguous_transport_failure_never_replays(wire, monkeypatch, exception):
    session = _seed(wire)
    calls = []

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise exception

    monkeypatch.setattr(cf.requests, "post", fail)
    result = json.loads(cf.camofox_navigate("https://example.test/target", task_id="contract"))
    assert result["success"] is False
    assert session["tab_id"] == "existing"
    assert len(calls) == 1
    assert wire.created == 0


_OPERATIONS = [
    ("click", lambda: cf.camofox_click("@e1", task_id="contract")),
    ("type", lambda: cf.camofox_type("@e1", "fixture text", task_id="contract")),
    ("press", lambda: cf.camofox_press("Enter", task_id="contract")),
    ("scroll", lambda: cf.camofox_scroll("down", task_id="contract")),
    ("back", lambda: cf.camofox_back(task_id="contract")),
    ("evaluate", lambda: bt._camofox_eval("1+1", task_id="contract")),
]


@pytest.mark.parametrize("operation,invoke", _OPERATIONS, ids=[name for name, _ in _OPERATIONS])
@pytest.mark.parametrize("status,payload,cleared", [
    (410, {}, True), (404, {"error": "tab_not_found"}, True),
    (404, {}, False), (405, {}, False), (501, {}, False),
    (500, {"code": "tab_not_found"}, False), (503, {}, False),
])
def test_action_failure_never_replays_or_creates(wire, operation, invoke, status, payload, cleared):
    session = _seed(wire)
    wire.failures[f"/tabs/existing/{operation}"] = [(status, payload)]
    result = json.loads(invoke())
    assert result["success"] is False
    assert session["tab_id"] == (None if cleared else "existing")
    assert wire.created == 0 and wire.loads == []
    assert len(_posts(wire)) == 1
    if cleared:
        assert "Call browser_navigate" in result["error"]
    if operation == "evaluate" and status in (404, 405, 501) and not cleared:
        assert "not supported" in result["error"]


@pytest.mark.parametrize("status,payload,cleared", [
    (410, {}, True), (404, {"code": "tab_not_found"}, True),
    (404, {}, False), (500, {}, False),
])
def test_bonus_snapshot_failure_does_not_replay_successful_navigation(wire, status, payload, cleared):
    session = _seed(wire)
    wire.failures["/tabs/existing/snapshot"] = [(status, payload)]
    result = json.loads(cf.camofox_navigate("https://example.test/target", task_id="contract"))
    assert result["success"] is True
    assert session["tab_id"] == (None if cleared else "existing")
    assert wire.loads == [("existing", "https://example.test/target")]
    assert len(_posts(wire)) == 1


@pytest.mark.parametrize("endpoint", ["mandatory", "evaluate"])
@pytest.mark.parametrize("body,missing", [
    ({"code": "tab_not_found"}, True), ({"error": "tab_not_found"}, True),
    ({"message": "Tab not found"}, True), ({"recovery": "create_new_tab"}, True),
    ({"code": "tab_timeout"}, True), ("tab_not_found", True),
    ("Tab not found", True), ("<html>Not Found</html>", False),
    ({}, False), (["route not found"], False),
])
def test_404_classifier_uses_response_evidence(endpoint, body, missing):
    response = requests.Response()
    response.status_code = 404
    response._content = (body if isinstance(body, str) else json.dumps(body)).encode("utf-8")
    error = requests.HTTPError(response=response)
    expected = "stale" if missing else ("capability" if endpoint == "evaluate" else "other")
    assert cf.classify_camofox_http_error(error, endpoint=endpoint) == expected


@pytest.mark.parametrize("operation,invoke", [
    ("snapshot", lambda: cf.camofox_snapshot(task_id="contract")),
    ("snapshot", lambda: cf.camofox_get_images(task_id="contract")),
    ("screenshot", lambda: cf.camofox_vision("fixture", task_id="contract")),
])
@pytest.mark.parametrize("status,payload,cleared", [
    (410, {}, True), (404, {"code": "tab_not_found"}, True),
    (404, {}, False), (500, {}, False),
])
def test_read_failure_clears_only_a_proven_stale_tab(wire, operation, invoke, status, payload, cleared):
    session = _seed(wire)
    wire.failures[f"/tabs/existing/{operation}"] = [(status, payload)]
    result = json.loads(invoke())
    assert result["success"] is False
    assert session["tab_id"] == (None if cleared else "existing")
    assert len(wire.calls) == 1
    assert wire.created == 0 and wire.loads == []


@pytest.mark.parametrize("invoke", [
    lambda: cf.camofox_snapshot(task_id="contract"),
    lambda: cf.camofox_get_images(task_id="contract"),
    lambda: cf.camofox_vision("fixture", task_id="contract"),
    lambda: cf.camofox_click("@e1", task_id="contract"),
    lambda: cf.camofox_type("@e1", "fixture text", task_id="contract"),
    lambda: cf.camofox_press("Enter", task_id="contract"),
])
@pytest.mark.parametrize("status,payload", [(410, {}), (404, {"error": "Tab not found"})])
def test_stale_preflight_probe_prevents_the_action(wire, monkeypatch, invoke, status, payload):
    session = _seed(wire)
    monkeypatch.setattr(cf, "_camofox_private_page_block", _REAL_PAGE_GUARD)
    monkeypatch.setattr(policy, "_eval_ssrf_guard_active", lambda *_: True)
    wire.failures["/tabs/existing/evaluate"] = [(status, payload)]
    result = json.loads(invoke())
    assert result["success"] is False
    assert "Call browser_navigate" in result["error"]
    assert session["tab_id"] is None
    assert [path for _, path, _, _ in wire.calls] == ["/tabs/existing/evaluate"]
    assert wire.created == 0 and wire.loads == []


@pytest.mark.parametrize("status,payload", [(410, {}), (404, {}), (500, {})])
def test_public_scroll_stops_on_first_failure(wire, monkeypatch, status, payload):
    _seed(wire)
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: True)
    wire.failures["/tabs/existing/scroll"] = [(status, payload)]
    result = json.loads(bt.browser_scroll("down", task_id="contract"))
    assert result["success"] is False
    assert len(_posts(wire)) == 1
    assert wire.created == 0


def test_public_scroll_preserves_successful_travel(wire, monkeypatch):
    _seed(wire)
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: True)
    result = json.loads(bt.browser_scroll("down", task_id="contract"))
    assert result["success"] is True
    assert [path for path, _ in _posts(wire)] == ["/tabs/existing/scroll"] * 5


@pytest.mark.parametrize("status", [404, 405, 501])
def test_creation_error_is_not_evaluate_capability(wire, status):
    wire.failures["/tabs"] = [(status, {})]
    result = json.loads(bt._camofox_eval("1+1", task_id="contract"))
    assert result["success"] is False
    assert "not supported" not in result["error"]
    assert [path for path, _ in _posts(wire)] == ["/tabs"]


def test_error_message_digits_are_not_http_status(wire, monkeypatch):
    _seed(wire)

    def fail(*_args, **_kwargs):
        raise RuntimeError("expression 404 + 405 + 501 failed")

    monkeypatch.setattr(cf, "_post", fail)
    result = json.loads(bt._camofox_eval("1+1", task_id="contract"))
    assert result["success"] is False
    assert "not supported" not in result["error"]
    assert cf._get_session("contract")["tab_id"] == "existing"
