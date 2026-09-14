from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

from workstation.client import get_workstation_resources


class _ResourceController:
    """Small authenticated loopback controller used by the surface test.

    The test deliberately uses a real HTTP boundary instead of monkeypatching
    ``workstation_controller_resources``.  That catches descriptor, bearer
    auth and JSON transport regressions while keeping the test provider- and
    Electron-independent.
    """

    token = "surface-test-token"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.event_payload = {
            "schema_version": 1,
            "runtime": "electron-chromium",
            "generated_at": "2026-09-11T12:00:00+00:00",
            "task_id": "task-surface",
            "events": [
                {
                    "event_id": "event-surface-1",
                    "kind": "task_started",
                    "task_id": "task-surface",
                    "session_id": "session-surface",
                    "message": "started",
                    "timestamp": "2026-09-11T12:00:00+00:00",
                    "metadata": {"source": "controller"},
                },
                {
                    "event_id": "event-surface-2",
                    "kind": "progress",
                    "task_id": "task-surface",
                    "session_id": "session-surface",
                    "message": "progress",
                    "timestamp": "2026-09-11T12:00:01+00:00",
                },
            ],
        }
        self.request_count = 0
        self.event_request_count = 0
        self.server: ThreadingHTTPServer | None = None
        self.thread: Thread | None = None

    def __enter__(self) -> "_ResourceController":
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
                if self.headers.get("Authorization") != f"Bearer {owner.token}":
                    self.send_response(401)
                    self.end_headers()
                    return
                if self.path == "/health":
                    body = {"success": True, "runtime": "electron-chromium"}
                elif self.path == "/resources":
                    owner.request_count += 1
                    body = {"success": True, **owner.payload}
                elif self.path.startswith("/events"):
                    owner.event_request_count += 1
                    body = {"success": True, **owner.event_payload}
                else:
                    self.send_response(404)
                    self.end_headers()
                    return

                raw = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    @property
    def descriptor(self) -> dict[str, Any]:
        assert self.server is not None
        port = self.server.server_address[1]
        return {"version": 1, "url": f"http://127.0.0.1:{port}", "token": self.token}

    def __exit__(self, *_args: object) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2)


def _surface_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runtime": "electron-chromium",
        "generated_at": "2026-09-11T12:00:00+00:00",
        "resources": [
            {
                "resource_type": "browser",
                "resource_id": "browser:electron-chromium",
                "permissions": ["read", "agent-control"],
                "state": {"ready": True, "task_count": 1, "tab_count": 1},
                "updated_at": "2026-09-11T12:00:00+00:00",
            },
            {
                "resource_type": "browser_task",
                "resource_id": "browser-task:task-surface",
                "task_id": "task-surface",
                "session_id": "session-surface",
                "permissions": ["read", "agent-control"],
                "state": {
                    "execution_status": "running",
                    "lineage": {
                        "task_id": "task-surface",
                        "session_id": "session-surface",
                        "kanban_card_id": "card-surface",
                        "run_id": "run-surface",
                    },
                },
                "updated_at": "2026-09-11T12:00:00+00:00",
            },
            {
                "resource_type": "execution_journal",
                "resource_id": "execution-journal:task-surface",
                "task_id": "task-surface",
                "session_id": "session-surface",
                "permissions": ["read"],
                "state": {"event_count": 1, "visible_event_count": 1},
                "updated_at": "2026-09-11T12:00:00+00:00",
            },
        ],
    }


def test_resource_client_normalizes_controller_snapshot(monkeypatch):
    monkeypatch.setattr(
        "tools.browser_workstation.workstation_controller_resources",
        lambda: {
            "success": True,
            "schema_version": 1,
            "runtime": "electron-chromium",
            "generated_at": "2026-09-11T12:00:00+00:00",
            "resources": [
                {
                    "resource_type": "browser_task",
                    "resource_id": "browser-task:task-1",
                    "task_id": "task-1",
                    "session_id": "session-1",
                    "permissions": ["read", "agent-control", 99],
                    "state": {"execution_status": "running"},
                    "updated_at": "2026-09-11T12:00:00+00:00",
                },
                {"resource_type": "invalid"},
            ],
        },
    )

    result = get_workstation_resources()

    assert result["available"] is True
    assert result["runtime"] == "electron-chromium"
    assert len(result["resources"]) == 1
    assert result["resources"][0]["permissions"] == ["read", "agent-control"]


def test_resource_client_returns_degraded_snapshot_when_controller_is_down(monkeypatch):
    monkeypatch.setattr(
        "tools.browser_workstation.workstation_controller_resources",
        lambda: (_ for _ in ()).throw(RuntimeError("controller unavailable")),
    )

    result = get_workstation_resources()

    assert result["available"] is False
    assert result["resources"] == []
    assert "controller unavailable" in result["error"]


def test_resource_client_rejects_protocol_mismatch(monkeypatch):
    monkeypatch.setattr(
        "tools.browser_workstation.workstation_controller_resources",
        lambda: {
            "success": True,
            "schema_version": 999,
            "runtime": "electron-chromium",
            "generated_at": "2026-09-11T12:00:00+00:00",
            "resources": [],
        },
    )

    result = get_workstation_resources()

    assert result["available"] is False
    assert result["error"] == "Workstation resource protocol mismatch"


def test_dashboard_and_tui_surfaces_share_one_live_controller_projection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Dashboard and TUI adapters resolve identical resources for one task.

    This is the two-client acceptance boundary: both high-level adapters read
    the same authenticated controller, and neither adapter owns persistence or
    synthesizes a second task/journal projection.
    """

    with _ResourceController(_surface_payload()) as controller:
        descriptor_path = tmp_path / "Runtime" / "browser-control.json"
        descriptor_path.parent.mkdir(parents=True)
        descriptor_path.write_text(json.dumps(controller.descriptor), encoding="utf-8")
        monkeypatch.setenv("HERMES_WORKSTATION_BROWSER_CONTROL_FILE", str(descriptor_path))

        from hermes_cli.web_server import get_workstation_resources as dashboard_resources
        from tui_gateway import server as tui_server

        dashboard = asyncio.run(dashboard_resources())
        tui_response = tui_server.handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "workstation.resources", "params": {}}
        )

    assert tui_response is not None
    assert "error" not in tui_response
    assert tui_response["result"] == dashboard
    assert controller.request_count == 2

    identities = [
        (resource["resource_type"], resource["resource_id"], resource.get("task_id"), resource.get("session_id"))
        for resource in dashboard["resources"]
    ]
    assert identities == [
        ("browser", "browser:electron-chromium", None, None),
        ("browser_task", "browser-task:task-surface", "task-surface", "session-surface"),
        ("execution_journal", "execution-journal:task-surface", "task-surface", "session-surface"),
    ]


def test_dashboard_and_tui_surfaces_share_bounded_event_projection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The event adapters preserve one task lineage and the controller limit."""

    with _ResourceController(_surface_payload()) as controller:
        descriptor_path = tmp_path / "Runtime" / "browser-control.json"
        descriptor_path.parent.mkdir(parents=True)
        descriptor_path.write_text(json.dumps(controller.descriptor), encoding="utf-8")
        monkeypatch.setenv("HERMES_WORKSTATION_BROWSER_CONTROL_FILE", str(descriptor_path))

        from hermes_cli.web_server import get_workstation_events as dashboard_events
        from tui_gateway import server as tui_server

        dashboard = asyncio.run(dashboard_events(task_id=" task-surface ", limit=1))
        tui_response = tui_server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "workstation.events",
                "params": {"task_id": "task-surface", "limit": 1},
            }
        )

    assert tui_response is not None
    assert "error" not in tui_response
    assert tui_response["result"] == dashboard
    assert controller.event_request_count == 2
    assert dashboard["task_id"] == "task-surface"
    assert [event["event_id"] for event in dashboard["events"]] == ["event-surface-1"]
