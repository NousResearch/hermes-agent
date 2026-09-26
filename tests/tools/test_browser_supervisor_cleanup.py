"""Browser Use session teardown must close its supervisors before the cloud endpoint.

Real routing, session cache, registry and CDP sockets; only the cloud provider and
browser-use executable are substitutes. No cloud credentials or Chrome required.
"""

import json
import sys
import threading
from contextlib import contextmanager

import pytest
from websockets.sync.server import serve

from agent.browser_provider import BrowserProvider
from agent.browser_registry import register_provider, restore_registration, snapshot_registration
from agent.secret_scope import (
    build_profile_secret_scope, is_multiplex_active, reset_secret_scope,
    set_multiplex_active, set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import browser_supervisor as bs
from tools import browser_tool as bt
from tools import browser_tool_lifecycle as lifecycle
from tools import browser_use_cli as bu


@pytest.fixture
def cloud_browser(tmp_path, monkeypatch):
    connections = {}

    def handle(ws):
        closed = threading.Event()
        connections.setdefault(ws.request.path, []).append(closed)
        try:
            for raw in ws:
                message = json.loads(raw)
                result = {
                    "Target.getTargets": {"targetInfos": [{"type": "page", "targetId": "page"}]},
                    "Target.attachToTarget": {"sessionId": "page-session"},
                }.get(message["method"], {})
                ws.send(json.dumps({"id": message["id"], "result": result}))
        finally:
            closed.set()

    class CloudBrowser(BrowserProvider):
        name = "cleanup-test"

        def __init__(self, port):
            self.port = port
            self.created = []
            self.closed = []

        def is_available(self):
            return True

        def create_session(self, task_id):
            session_id = f"session-{len(self.created)}"
            self.created.append(task_id)
            return {"bb_session_id": session_id, "session_name": session_id,
                    "cdp_url": f"ws://127.0.0.1:{self.port}/devtools/browser/{session_id}"}

        def close_session(self, session_id):
            closed = connections[f"/devtools/browser/{session_id}"]
            self.closed.append((session_id, all(event.wait(2) for event in closed)))
            return True

        def emergency_cleanup(self, session_id):
            pass

    homes = [tmp_path / "a", tmp_path / "b"]
    monkeypatch.setattr(bs, "SUPERVISOR_REGISTRY", bs._SupervisorRegistry())
    monkeypatch.setattr(bu, "_find_cli", lambda: [sys.executable, "-c",
        "import os, sys; sys.stdin.read(); "
        "assert not any(k.startswith('_HERMES_BU_') for k in os.environ); print('ok')"])
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    @contextmanager
    def scope(home, *, served=True):
        multiplex = is_multiplex_active()
        set_multiplex_active(served)
        home_token = set_hermes_home_override(home if served else None)
        secret_token = set_secret_scope(build_profile_secret_scope(home))
        try:
            yield
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
            set_multiplex_active(multiplex)

    with serve(handle, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        provider = CloudBrowser(server.socket.getsockname()[1])
        for home in homes:
            home.mkdir()
            (home / "config.yaml").write_text(
                "browser:\n  backend: browser-use\n  cloud_provider: cleanup-test\n", encoding="utf-8")
            register_provider(provider, scope=str(home))
        try:
            yield provider, homes, scope
        finally:
            lifecycle._stop_browser_cleanup_thread()
            bs.SUPERVISOR_REGISTRY.stop_all()
            for key in list(bt._active_sessions):
                lifecycle._forget_session_tracking(key, session=True)
            for home in homes:
                registration = snapshot_registration(provider.name, scope=str(home))
                restore_registration(provider.name, registration, None, scope=str(home))
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("served", [False, True], ids=["default", "profile"])
@pytest.mark.parametrize("session", ["", "research"], ids=["unnamed", "named"])
def test_session_cleanup_closes_task_supervisors_before_cloud(cloud_browser, served, session):
    provider, homes, scope = cloud_browser
    with scope(homes[0], served=served):
        result = json.loads(bu.browser_exec("print('ok')", task_id="task", session=session))
        assert result["success"], result
        supervisor = bs.SUPERVISOR_REGISTRY.get("task")
        assert supervisor is not None and supervisor.snapshot().active

        session_key = provider.created[-1]
        # An expired cloud endpoint cannot accept agent-browser's close command.
        bt._active_sessions[session_key]["expires_at"] = 0
        lifecycle.cleanup_browser(session_key)

        assert bs.SUPERVISOR_REGISTRY.get("task") is None
        assert not supervisor.snapshot().active
        assert provider.closed == [("session-0", True)]


@pytest.mark.parametrize("served", [False, True], ids=["default", "profile"])
def test_cleanup_preserves_rebound_tasks_and_other_profiles(cloud_browser, served):
    provider, homes, scope = cloud_browser

    def run(task, session=""):
        result = json.loads(bu.browser_exec("print('ok')", task_id=task, session=session))
        assert result["success"], result
        supervisor = bs.SUPERVISOR_REGISTRY.get(task)
        assert supervisor is not None and supervisor.snapshot().active
        return supervisor

    def expire_and_close(key):
        bt._active_sessions[key]["expires_at"] = 0
        lifecycle.cleanup_browser(key)

    with scope(homes[0], served=served):
        run("task")
        old_key = provider.created[-1]
    with scope(homes[1]):
        other = run("other-task", "research")
        other_key = provider.created[-1]
    with scope(homes[0], served=served):
        current = run("task", "research")
        new_key = provider.created[-1]
        follower = run("follower", "research")
        expire_and_close(old_key)
        assert bs.SUPERVISOR_REGISTRY.get("task") is current
        assert current.snapshot().active and follower.snapshot().active
        assert other.snapshot().active

        expire_and_close(new_key)
        assert bs.SUPERVISOR_REGISTRY.get("task") is None
        assert bs.SUPERVISOR_REGISTRY.get("follower") is None
        assert not current.snapshot().active and not follower.snapshot().active
        assert other.snapshot().active
    with scope(homes[1]):
        expire_and_close(other_key)
        assert not other.snapshot().active
    assert len(provider.closed) == 3
    assert all(sockets_closed for _, sockets_closed in provider.closed)
