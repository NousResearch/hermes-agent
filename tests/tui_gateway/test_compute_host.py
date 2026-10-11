import json
import pytest
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


def _stdout_queue(proc: subprocess.Popen) -> queue.Queue[dict]:
    out: queue.Queue[dict] = queue.Queue()
    assert proc.stdout is not None

    def drain() -> None:
        for line in proc.stdout or []:
            out.put(json.loads(line))

    threading.Thread(target=drain, daemon=True).start()
    return out


def _read_json_line(out: queue.Queue[dict], timeout: float = 2.0) -> dict:
    try:
        return out.get(timeout=timeout)
    except queue.Empty as exc:
        raise AssertionError("timed out waiting for compute host JSON") from exc


@pytest.mark.platforms("linux")
def test_compute_host_line_json_hello_and_shutdown():
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, "-m", "tui_gateway.compute_host"],
        cwd=str(repo),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    out = _stdout_queue(proc)
    try:
        hello = _read_json_line(out)
        assert hello["type"] == "hello"
        assert hello["host_pid"] == proc.pid

        proc.stdin.write(json.dumps({"type": "bogus", "request_id": "b"}) + "\n")
        proc.stdin.flush()
        error = _read_json_line(out)
        assert error["type"] == "error"
        assert error["message"] == "unknown frame type: bogus"

        proc.stdin.write(json.dumps({"type": "shutdown", "request_id": "stop"}) + "\n")
        proc.stdin.flush()
        assert _read_json_line(out)["type"] == "shutdown.ack"
        proc.wait(timeout=2)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_compute_host_profile_is_set_before_notification_poller(monkeypatch, tmp_path):
    """Real session initialization exposes the routed home before services start."""
    import io
    from types import SimpleNamespace
    from hermes_state_registry import release_or_close
    from tui_gateway import server
    from tui_gateway.compute_host import ComputeHost

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launch_home = tmp_path / "launch"
    launch_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    agent = SimpleNamespace()
    seen = []
    def make_agent(*_args, **kwargs):
        agent._session_db = kwargs["session_db"]
        return agent

    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_hydrate_session_cwd", lambda *_args: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_args: None)
    monkeypatch.setattr(server, "_wire_session_agent", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_emit", lambda *_args: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *_args: None)
    monkeypatch.setattr(server, "_start_session_services",
                        lambda _sid, _key, session: seen.append(session.get("profile_home")))
    host = ComputeHost(stdout=io.StringIO(), heartbeat_secs=0, max_workers=1)
    sid = "profile-before-poller"
    try:
        expected = []
        for profile in ("a", "b", "a"):
            profile_home = tmp_path / profile
            profile_home.mkdir(exist_ok=True)
            try:
                session = host._ensure_server_session(server, {
                    "sid": sid, "session_key": sid, "profile_home": str(profile_home)})
                expected.append(str(profile_home))
                assert seen == expected
                assert session["profile_home"] == str(profile_home)
            finally:
                server._sessions.pop(sid, None)
                if getattr(agent, "_owns_session_db", False):
                    release_or_close(agent._session_db)
                    agent._owns_session_db = False
    finally:
        # No tasks were submitted; avoid closing unrelated process-registry fixtures.
        host._executor.shutdown(wait=True)
