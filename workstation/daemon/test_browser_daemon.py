from __future__ import annotations

import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest

from workstation.daemon.browser_daemon import HermesBrowserDaemon


@pytest.fixture
def daemon(tmp_path):
    desc_dir = tmp_path / "workstation"
    d = HermesBrowserDaemon(port=0, descriptor_dir=desc_dir)
    d.start()
    yield d
    d.stop()


def _request_daemon(daemon: HermesBrowserDaemon, method: str, path: str, body=None, token=None):
    url = f"{daemon.url}{path}"
    headers = {"Content-Type": "application/json"}
    auth_token = token if token is not None else daemon.token
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_daemon_startup_and_health(daemon):
    """Scenario 1: Daemon starts on free loopback port and responds to /health without auth."""
    res = _request_daemon(daemon, "GET", "/health", token="")
    assert res["success"] is True
    assert res["status"] == "ok"
    assert res["daemon"] == "hermes-browser-daemon"
    assert res["tabs_count"] == 0
    assert daemon.descriptor_path.exists()

    # Verify descriptor content
    desc = json.loads(daemon.descriptor_path.read_text(encoding="utf-8"))
    assert desc["url"] == daemon.url
    assert desc["token"] == daemon.token


def test_daemon_auth_protection(daemon):
    """Scenario 2: Protected endpoints reject requests with missing or invalid bearer token."""
    with pytest.raises(HTTPError) as exc_info:
        _request_daemon(daemon, "GET", "/v1/resources", token="invalid_token")
    assert exc_info.value.code == 401


def test_daemon_action_dispatch_and_tab_persistence(daemon):
    """Scenario 3: Daemon executes actions and retains tabs across client requests independently of UI."""
    task_id = "test_daemon_task_42"

    # 1. Navigate
    nav_res = _request_daemon(
        daemon,
        "POST",
        "/v1/action",
        body={"action": "browser_navigate", "arguments": {"url": "https://example.org/dashboard"}, "task_id": task_id},
    )
    assert nav_res["success"] is True
    tab_id = nav_res["result"]["tab_id"]
    assert nav_res["result"]["url"] == "https://example.org/dashboard"

    # 2. Simulate client process termination and reconnect:
    # Next call from another "client" with same task_id reuses existing tab without resetting state
    extract_res = _request_daemon(
        daemon,
        "POST",
        "/v1/action",
        body={"action": "browser_extract_items", "arguments": {"limit": 5}, "task_id": task_id},
    )
    assert extract_res["success"] is True
    assert extract_res["result"]["tab_id"] == tab_id
    assert extract_res["result"]["count"] == 5

    # 3. Verify resources projection
    res = _request_daemon(daemon, "GET", "/v1/resources")
    assert res["success"] is True
    assert res["result"]["active_tab_id"] == tab_id
    assert len(res["result"]["tabs"]) == 1
    assert task_id in res["result"]["tasks"]


def test_daemon_viewport_attachment_lifecycle(daemon):
    """Scenario 4: Viewport attach and detach for external UI viewer without destroying the tab."""
    task_id = "test_viewport_task"

    nav_res = _request_daemon(
        daemon,
        "POST",
        "/v1/action",
        body={"action": "browser_navigate", "arguments": {"url": "https://example.org/feed"}, "task_id": task_id},
    )
    tab_id = nav_res["result"]["tab_id"]

    # Attach UI viewport
    attach_res = _request_daemon(
        daemon,
        "POST",
        "/v1/viewport/attach",
        body={"tab_id": tab_id, "client_id": "electron_window_1"},
    )
    assert attach_res["success"] is True
    assert attach_res["token"].startswith("vptoken_")
    assert daemon.tabs[tab_id].viewport_attached is True

    # Detach UI viewport (e.g. user minimized or closed Electron window)
    detach_res = _request_daemon(
        daemon,
        "POST",
        "/v1/viewport/detach",
        body={"tab_id": tab_id},
    )
    assert detach_res["success"] is True
    assert daemon.tabs[tab_id].viewport_attached is False

    # Tab and session still alive in background daemon!
    assert daemon.tabs[tab_id].url == "https://example.org/feed"
