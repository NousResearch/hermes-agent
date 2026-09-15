from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools import browser_workstation as bw


class _Handler(BaseHTTPRequestHandler):
    token = "test-token"

    def log_message(self, *_args):
        return

    def _auth(self):
        return self.headers.get("Authorization") == f"Bearer {self.token}"

    def _json(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802
        if not self._auth():
            self._json(401, {"success": False})
            return
        if self.path == "/health":
            self._json(200, {"success": True, "state": {"ready": True}})
            return
        self._json(404, {"success": False})

    def do_POST(self):  # noqa: N802
        if not self._auth():
            self._json(401, {"success": False})
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self._json(
            200,
            {
                "success": True,
                "result": {
                    "runtime": "electron-chromium",
                    "action": payload.get("action"),
                    "task_id": payload.get("task_id"),
                    "arguments": payload.get("arguments"),
                    "kanban_card_id": payload.get("kanban_card_id"),
                    "run_id": payload.get("run_id"),
                },
            },
        )


@pytest.fixture()
def controller(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    control = tmp_path / "browser-control.json"
    control.write_text(
        json.dumps(
            {
                "version": 1,
                "url": f"http://127.0.0.1:{server.server_address[1]}",
                "token": _Handler.token,
            }
        )
    )
    monkeypatch.setenv("HERMES_WORKSTATION_BROWSER_CONTROL_FILE", str(control))
    monkeypatch.setenv("HERMES_WORKSTATION_BROWSER", "1")
    monkeypatch.setenv("HERMES_WORKSTATION_BROWSER_ROUTING", "1")
    bw._LAST_HEALTH_AT = 0.0
    bw._LAST_HEALTH_VALUE = False
    bw.clear_workstation_task_binding("task-a")
    yield server, control
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    bw._LAST_HEALTH_AT = 0.0
    bw._LAST_HEALTH_VALUE = False
    bw.clear_workstation_task_binding("task-a")


def test_controller_health_and_dispatch(controller):
    assert bw.workstation_controller_available(force=True)
    result = bw.workstation_routed_browser_handler(
        "browser_navigate",
        {"url": "https://example.com"},
        fallback=lambda: pytest.fail("legacy fallback must not run"),
        task_id="task-a",
    )
    decoded = json.loads(result)
    assert decoded["runtime"] == "electron-chromium"
    assert decoded["action"] == "browser_navigate"
    assert decoded["task_id"] == "task-a"


def test_bound_task_fails_closed_after_controller_loss(controller, monkeypatch):
    server, control = controller
    bw.workstation_routed_browser_handler(
        "browser_navigate",
        {"url": "https://example.com"},
        fallback=lambda: "legacy",
        task_id="task-a",
    )
    control.unlink()
    bw._LAST_HEALTH_AT = 0.0
    called = False

    def legacy():
        nonlocal called
        called = True
        return "legacy"

    with pytest.raises(bw.WorkstationBrowserUnavailable):
        bw.workstation_routed_browser_handler(
            "browser_snapshot", {}, fallback=legacy, task_id="task-a"
        )
    assert not called


def test_unbound_task_can_use_legacy_when_routing_enabled(tmp_path, monkeypatch):
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("HERMES_WORKSTATION_BROWSER_CONTROL_FILE", str(missing))
    monkeypatch.setenv("HERMES_WORKSTATION_BROWSER", "1")
    monkeypatch.setenv("HERMES_WORKSTATION_BROWSER_ROUTING", "1")
    bw._LAST_HEALTH_AT = 0.0
    bw.clear_workstation_task_binding("new-task")
    assert bw.workstation_routed_browser_handler(
        "browser_snapshot", {}, fallback=lambda: "legacy-ok", task_id="new-task"
    ) == "legacy-ok"


def test_internal_only_mode_fails_closed_when_controller_missing(tmp_path, monkeypatch):
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("HERMES_WORKSTATION_BROWSER_CONTROL_FILE", str(missing))
    monkeypatch.setenv("HERMES_WORKSTATION_BROWSER", "1")
    monkeypatch.setenv("HERMES_WORKSTATION_BROWSER_ROUTING", "0")
    bw._LAST_HEALTH_AT = 0.0
    bw.clear_workstation_task_binding("new-task")
    with pytest.raises(bw.WorkstationBrowserUnavailable):
        bw.workstation_routed_browser_handler(
            "browser_snapshot", {}, fallback=lambda: "legacy", task_id="new-task"
        )


def test_unbound_task_defaults_to_fail_closed_when_routing_env_unset(tmp_path, monkeypatch):
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("HERMES_WORKSTATION_BROWSER_CONTROL_FILE", str(missing))
    monkeypatch.setenv("HERMES_WORKSTATION_BROWSER", "1")
    monkeypatch.delenv("HERMES_WORKSTATION_BROWSER_ROUTING", raising=False)
    bw._LAST_HEALTH_AT = 0.0
    bw.clear_workstation_task_binding("new-task")
    with pytest.raises(bw.WorkstationBrowserUnavailable):
        bw.workstation_routed_browser_handler(
            "browser_snapshot", {}, fallback=lambda: "legacy", task_id="new-task"
        )



def test_any_successful_internal_action_binds_task(controller):
    _server, control = controller
    bw.workstation_routed_browser_handler(
        "browser_snapshot",
        {},
        fallback=lambda: pytest.fail("legacy fallback must not run"),
        task_id="task-a",
    )
    control.unlink()
    bw._LAST_HEALTH_AT = 0.0
    with pytest.raises(bw.WorkstationBrowserUnavailable):
        bw.workstation_routed_browser_handler(
            "browser_snapshot", {}, fallback=lambda: "legacy", task_id="task-a"
        )


def test_navigation_blocks_recognizable_secret_before_search_egress():
    args = {"url": "search this sk-abcdefghijk please"}
    with pytest.raises(bw.WorkstationBrowserError, match="API key or token"):
        bw._validate_navigation(args)


def test_navigation_blocks_cloud_metadata_floor():
    args = {"url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/"}
    with pytest.raises(bw.WorkstationBrowserError, match="cloud metadata"):
        bw._validate_navigation(args)


def test_navigation_allows_and_normalizes_localhost():
    args = {"url": "localhost:3000/health"}
    bw._validate_navigation(args)
    assert args["url"] == "http://localhost:3000/health"


def test_dispatch_forwards_kanban_and_run_identities(controller, monkeypatch):
    result = bw.workstation_routed_browser_handler(
        "browser_navigate",
        {"url": "https://example.com"},
        fallback=lambda: pytest.fail("legacy fallback must not run"),
        task_id="task-kanban",
        kanban_card_id="card-123",
        run_id="run-456",
    )
    decoded = json.loads(result)
    assert decoded["kanban_card_id"] == "card-123"
    assert decoded["run_id"] == "run-456"

    monkeypatch.setenv("HERMES_KANBAN_TASK", "card-env-789")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "run-env-012")
    result_env = bw.workstation_routed_browser_handler(
        "browser_navigate",
        {"url": "https://example.com"},
        fallback=lambda: pytest.fail("legacy fallback must not run"),
        task_id="task-kanban-env",
    )
    decoded_env = json.loads(result_env)
    assert decoded_env["kanban_card_id"] == "card-env-789"
    assert decoded_env["run_id"] == "run-env-012"


def test_browser_type_dispatches_with_clear_and_append(controller):
    result = bw.workstation_routed_browser_handler(
        "browser_type",
        {"ref": "@e5", "text": "new query", "clear": False, "append": True},
        fallback=lambda: pytest.fail("legacy fallback must not run"),
        task_id="task-type",
    )
    decoded = json.loads(result)
    assert decoded["runtime"] == "electron-chromium"
    assert decoded["action"] == "browser_type"
    assert decoded["arguments"]["ref"] == "@e5"
    assert decoded["arguments"]["text"] == "new query"
    assert decoded["arguments"]["clear"] is False
    assert decoded["arguments"]["append"] is True


def test_browser_extract_items_dispatches_with_selector_and_limit(controller):
    result = bw.workstation_routed_browser_handler(
        "browser_extract_items",
        {"selector": "div.product-card", "limit": 15},
        fallback=lambda: pytest.fail("legacy fallback must not run"),
        task_id="task-extract",
    )
    decoded = json.loads(result)
    assert decoded["runtime"] == "electron-chromium"
    assert decoded["action"] == "browser_extract_items"
    assert decoded["arguments"]["selector"] == "div.product-card"
    assert decoded["arguments"]["limit"] == 15


def test_workstation_schema_tools_includes_extract_items():
    assert "browser_extract_items" in bw._WORKSTATION_SCHEMA_TOOLS
