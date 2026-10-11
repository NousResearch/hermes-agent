"""Real v0.1.13 IPC/CLI/daemon shutdown, with a fixture-owned fake CDP browser."""
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from tools import browser_use_cli as bu
from tools import browser_use_cli_lifecycle as leases
from tools import browser_tool as bt
from tools import browser_tool_lifecycle as lifecycle


_DAEMON = '''
import asyncio, json, os
from pathlib import Path
from browser_harness import daemon, _ipc
state = Path(os.environ["PROBE_STATE"])
class Browser:
    def __init__(self):
        self.data = {"url": "about:blank", "closed": False, "busy": False}
        self.save()
    def save(self):
        staged = state.with_suffix(".tmp")
        staged.write_text(json.dumps(self.data))
        staged.replace(state)
    async def send_raw(self, method, params=None, session_id=None):
        params = params or {}
        if method == "Target.getTargets":
            return {"targetInfos": [{"targetId": "tab", "type": "page", "url": self.data["url"]}]}
        if method == "Target.closeTarget":
            self.data["closed"] = True
        elif method == "Probe.navigate":
            self.data["url"] = params["url"]
        elif method == "Probe.wait":
            self.data["busy"] = True
            self.save()
            await asyncio.sleep(params["seconds"])
            self.data["busy"] = False
        self.save()
        return self.data
async def main():
    d = daemon.Daemon()
    d.stop = asyncio.Event()
    d.cdp = Browser()
    d.session = "tab-session"
    d.dedicated_target_id = "tab" if daemon.NAME != "default" else None
    _ipc.pid_path(daemon.NAME).write_text(str(os.getpid()))
    try:
        await daemon.serve(d)
    finally:
        _ipc.pid_path(daemon.NAME).unlink(missing_ok=True)
asyncio.run(main())
'''


def _wait(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("fixture daemon did not reach expected state")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    pytest.importorskip("browser_harness")
    # Short AF_UNIX paths; every endpoint belongs to this fixture, never the host browser.
    runtime = tempfile.TemporaryDirectory(prefix="bh-", dir="/tmp" if os.name != "nt" else None)
    monkeypatch.setattr(leases, "_daemons", {})
    monkeypatch.setattr(leases, "_runtimes", {})
    monkeypatch.setattr(bu, "_find_cli", bu._find_cli_unpatched)
    env = {"PATH": os.environ.get("PATH", ""), "BH_HOME": str(tmp_path),
           "BH_RUNTIME_DIR": runtime.name, "BH_RUNTIME_DIR_SHARED": "1",
           "BH_REQUIRE_EXISTING_DAEMON": "1", "ANONYMIZED_TELEMETRY": "false"}
    monkeypatch.setattr(bu, "_base_subprocess_env", lambda: dict(env))
    monkeypatch.setattr(bu, "_blocked_url_in_code", lambda code: None)
    monkeypatch.setattr(bu, "_read_browser_cfg", lambda: {"backend": "browser-use"})
    monkeypatch.setattr(bu, "_attach_vault_supervisor", lambda *args: None)
    def route(child_env, *args):
        child_env[bu._PRIVATE_BROWSER_SENTINEL] = True
    monkeypatch.setattr(bu, "_route_backend", route)
    monkeypatch.setattr(bt, "_active_sessions", {})
    monkeypatch.setattr(bt, "_cleanup_done", False)
    monkeypatch.setattr(lifecycle, "_reap_orphaned_browser_sessions", lambda: None)
    monkeypatch.setattr(lifecycle, "_stop_all_lightpanda", lambda: None)
    monkeypatch.setattr(lifecycle._real_profile, "_terminate_real_profile_chrome", lambda: None)
    script = tmp_path / "daemon.py"
    script.write_text(_DAEMON)
    processes = []
    def start(name="default", overrides=None, owned=True):
        state = tmp_path / f"{name}-{len(processes)}.json"
        child_env = {**env, **(overrides or {}), "PROBE_STATE": str(state)}
        if name != "default":
            child_env["BU_NAME"] = name
        if owned:
            leases.prepare_runtime(child_env, bu.get_hermes_home())
            env["BH_RUNTIME_DIR"] = child_env["BH_RUNTIME_DIR"]
        child_env["BU_NAME"] = name
        proc = subprocess.Popen([sys.executable, str(script)], env=child_env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        processes.append(proc)
        threading.Thread(target=proc.wait, daemon=True).start()
        stem = f"bu-{name}"
        endpoint = Path(child_env["BH_RUNTIME_DIR"]) / (stem + (".port" if os.name == "nt" else ".sock"))
        _wait(lambda: endpoint.exists() or proc.poll() is not None)
        assert proc.poll() is None, proc.stderr.read()
        return proc, state, endpoint
    yield env, start
    # Only direct fixture children are signaled on test failure.
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=10)
        proc.stderr.close()
    leases._daemons.clear()
    leases.cleanup_runtime_dirs()
    runtime.cleanup()


@pytest.mark.platforms("posix", "windows")
@pytest.mark.parametrize("session", ["research", ""])
def test_calls_preserve_tab_workspace_and_stop_at_task_end(harness, session):
    env, start = harness
    proc, state, endpoint = start(session or "default")
    first = json.loads(bu.browser_exec('print(cdp("Probe.navigate", url="about:blank?first"))',
                                     session=session, task_id="worker"))
    assert first["success"], first
    second = json.loads(bu.browser_exec('print(cdp("Probe.read"))', session=session, task_id="worker"))
    assert second["success"], second
    assert "about:blank?first" in second["output"]
    assert first["workspace"] == second["workspace"]
    assert first.get("session") == second.get("session")
    assert not json.loads(state.read_text())["closed"]
    lifecycle.cleanup_browser("worker")  # no active browser-cache entry on direct CDP routes
    proc.wait(timeout=10)
    assert not endpoint.exists()
    assert not endpoint.with_suffix(".pid").exists()
    # Default attaches an existing tab; named daemons own and close their dedicated tab.
    assert json.loads(state.read_text())["closed"] is bool(session)


@pytest.mark.platforms("posix", "windows")
def test_shared_name_concurrency_and_other_session_survive(harness):
    _, start = harness
    shared, state, _ = start("shared")
    other, other_state, _ = start("other")
    assert json.loads(bu.browser_exec('print(cdp("Probe.read"))', session="shared", task_id="fast"))["success"]
    assert json.loads(bu.browser_exec('print(cdp("Probe.read"))', session="other", task_id="other-owner"))["success"]
    with concurrent.futures.ThreadPoolExecutor() as pool:
        pending = pool.submit(bu.browser_exec, 'print(cdp("Probe.wait", seconds=0.5))', "shared", 10, "slow")
        _wait(lambda: json.loads(state.read_text())["busy"])
        lifecycle.cleanup_browser("fast")
        assert shared.poll() is None and other.poll() is None
        lifecycle.cleanup_browser("slow")  # in-flight call delays the last-owner stop
        assert shared.poll() is None
        result = json.loads(pending.result(timeout=10))
        assert result["success"], result
    shared.wait(timeout=10)
    assert json.loads(state.read_text())["closed"]
    assert other.poll() is None and not json.loads(other_state.read_text())["closed"]
    lifecycle.cleanup_browser("other-owner")
    other.wait(timeout=10)


@pytest.mark.platforms("posix", "windows")
def test_exit_with_empty_active_sessions(harness):
    _, start = harness
    proc, state, _ = start("exit")
    assert json.loads(bu.browser_exec('print(cdp("Probe.read"))', session="exit", task_id="worker"))["success"]
    assert not bt._active_sessions
    lifecycle._emergency_cleanup_all_sessions()
    proc.wait(timeout=10)
    assert json.loads(state.read_text())["closed"]


@pytest.mark.platforms("posix", "windows")
def test_failed_and_timed_out_calls_keep_reusable_daemon(harness, monkeypatch):
    _, start = harness
    proc, state, _ = start("retry")
    failed = json.loads(bu.browser_exec('raise ValueError("probe failure")', session="retry", task_id="worker"))
    assert not failed["success"] and "probe failure" in failed["stderr"]
    run = bu._run_cli_killing_process_group
    def timeout(*args):
        raise subprocess.TimeoutExpired("fixture CLI", 5)
    monkeypatch.setattr(bu, "_run_cli_killing_process_group", timeout)
    result = json.loads(bu.browser_exec('print(1)', session="retry", task_id="worker"))
    assert "timed out" in result["error"]
    assert proc.poll() is None and not json.loads(state.read_text())["closed"]
    monkeypatch.setattr(bu, "_run_cli_killing_process_group", run)
    assert json.loads(bu.browser_exec('print(cdp("Probe.read"))', session="retry", task_id="worker"))["success"]
    lifecycle.cleanup_browser("worker")
    proc.wait(timeout=10)


@pytest.mark.platforms("posix", "windows")
def test_session_idle_teardown_uses_recorded_profile_environment(harness, monkeypatch):
    env, start = harness
    proc, state, _ = start("idle")
    bu.browser_exec('print(cdp("Probe.read"))', session="idle", task_id="worker")
    # A janitor/exit thread must not resolve a different profile's IPC environment.
    monkeypatch.setattr(bu, "_base_subprocess_env", lambda: {"BH_HOME": "/wrong-profile"})
    lifecycle._release_session_resources(bu._backend_cache_key("worker", "idle"), {"bb_session_id": None})
    proc.wait(timeout=10)
    assert json.loads(state.read_text())["closed"]
    assert leases._endpoint_key(env) != leases._endpoint_key({"BH_HOME": "/wrong-profile"})


@pytest.mark.platforms("posix", "windows")
def test_real_cli_timeout_keeps_daemon_and_workspace(harness):
    _, start = harness
    proc, state, _ = start("timeout")
    result = json.loads(bu.browser_exec('print(cdp("Probe.wait", seconds=6))',
                                      session="timeout", timeout_s=5, task_id="worker"))
    assert "timed out after 5s" in result["error"]
    assert proc.poll() is None and not json.loads(state.read_text())["closed"]
    _wait(lambda: not json.loads(state.read_text())["busy"])
    retry = json.loads(bu.browser_exec('print(cdp("Probe.read"))', session="timeout", task_id="worker"))
    assert retry["success"], retry
    assert Path(retry["workspace"]).is_dir()
    lifecycle.cleanup_browser("worker")
    proc.wait(timeout=10)


@pytest.mark.platforms("posix", "windows")
def test_default_tasks_on_different_browsers_do_not_share_daemon(harness):
    env, start = harness
    env["BU_CDP_URL"] = "http://127.0.0.1:9400"
    first, _, _ = start()
    bu.browser_exec('print(cdp("Probe.navigate", url="about:blank?A"))', task_id="A")
    env["BU_CDP_URL"] = "http://127.0.0.1:9401"
    second, state, _ = start()
    bu.browser_exec('print(cdp("Probe.navigate", url="about:blank?B"))', task_id="B")
    lifecycle.cleanup_browser("A")
    first.wait(timeout=10)
    result = json.loads(bu.browser_exec('print(cdp("Probe.read"))', task_id="B"))
    assert result["success"] and "about:blank?B" in result["output"]
    assert second.poll() is None and not json.loads(state.read_text())["closed"]
    lifecycle.cleanup_browser("B")
    second.wait(timeout=10)


@pytest.mark.platforms("posix", "windows")
def test_profiles_with_same_name_and_task_are_isolated(harness, tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    _, start = harness
    profile_a = set_hermes_home_override(tmp_path / "A")
    try:
        first, state_a, _ = start("same")
        bu.browser_exec('print(cdp("Probe.navigate", url="about:blank?A"))', session="same", task_id="worker")
        profile_b = set_hermes_home_override(tmp_path / "B")
        try:
            second, _, _ = start("same")
            bu.browser_exec('print(cdp("Probe.navigate", url="about:blank?B"))', session="same", task_id="worker")
            lifecycle.cleanup_browser("worker")
            second.wait(timeout=10)
        finally:
            reset_hermes_home_override(profile_b)
        result = json.loads(bu.browser_exec('print(cdp("Probe.read"))', session="same", task_id="worker"))
        assert result["success"] and "about:blank?A" in result["output"]
        assert first.poll() is None and not json.loads(state_a.read_text())["closed"]
        lifecycle.cleanup_browser("worker")
        first.wait(timeout=10)
    finally:
        reset_hermes_home_override(profile_a)


@pytest.mark.platforms("posix", "windows")
@pytest.mark.parametrize("failure", ["already-exited", "endpoint-missing"])
def test_unavailable_daemon_cleanup_is_best_effort(harness, failure):
    _, start = harness
    proc, state, endpoint = start("gone")
    result = json.loads(bu.browser_exec('print(cdp("Probe.read"))', session="gone", task_id="worker"))
    assert result["success"]
    if failure == "already-exited":
        proc.terminate()
        proc.wait(timeout=10)
    else:
        endpoint.unlink()
    lifecycle.cleanup_browser("worker")
    assert result["success"]  # teardown cannot overwrite an already returned tool result
    if failure == "endpoint-missing":
        # Upstream reload cannot identify an unreachable daemon. Independent reaping is still needed.
        assert proc.poll() is None and not json.loads(state.read_text())["closed"]


@pytest.mark.platforms("windows")
def test_windows_port_token_rejects_unauthenticated_shutdown(harness):
    import socket
    _, start = harness
    proc, state, endpoint = start("token")
    port_data = json.loads(endpoint.read_text())
    with socket.create_connection(("127.0.0.1", port_data["port"]), timeout=1) as client:
        client.sendall(b'{"meta":"shutdown","token":"incorrect"}\n')
        response = json.loads(client.recv(65536))
    assert "error" in response
    assert proc.poll() is None and not json.loads(state.read_text())["closed"]
    assert json.loads(bu.browser_exec('print(cdp("Probe.read"))', session="token", task_id="worker"))["success"]
    lifecycle.cleanup_browser("worker")
    proc.wait(timeout=10)


@pytest.mark.platforms("posix", "windows")
def test_global_cleanup_does_not_touch_foreign_namespace(harness):
    _, start = harness
    with tempfile.TemporaryDirectory(prefix="foreign-bh-", dir="/tmp" if os.name != "nt" else None) as runtime:
        foreign, foreign_state, _ = start("research", {"BH_RUNTIME_DIR": runtime}, owned=False)
        owned, _, _ = start("research")
        assert json.loads(bu.browser_exec('print(cdp("Probe.read"))', session="research", task_id="worker"))["success"]
        lifecycle.cleanup_all_browsers()
        owned.wait(timeout=10)
        assert foreign.poll() is None and not json.loads(foreign_state.read_text())["closed"]
        foreign.terminate()
        foreign.wait(timeout=10)


@pytest.mark.platforms("posix", "windows")
def test_janitor_preserves_active_harness_call(harness, monkeypatch):
    _, start = harness
    proc, state, _ = start("busy")
    key = bu._backend_cache_key("worker", "busy")
    monkeypatch.setattr(bt, "_session_last_activity", {key: 0})
    monkeypatch.setattr(bt, "BROWSER_SESSION_INACTIVITY_TIMEOUT", 0)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        pending = pool.submit(bu.browser_exec, 'print(cdp("Probe.wait", seconds=0.5))', "busy", 10, "worker")
        _wait(lambda: json.loads(state.read_text())["busy"])
        lifecycle._cleanup_inactive_browser_sessions()
        assert bt._session_last_activity[key] > 0
        assert proc.poll() is None and not json.loads(state.read_text())["closed"]
        assert json.loads(pending.result(timeout=10))["success"]
    lifecycle.cleanup_browser("worker")
    proc.wait(timeout=10)


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("contents", ['not JSON', '{"port":"bad","token":"x"}', '{"port":1234}'])
def test_windows_malformed_port_file_is_unavailable(harness, monkeypatch, contents):
    from browser_harness import _ipc
    _, start = harness
    proc, _, endpoint = start("malformed")
    original = endpoint.read_text()
    monkeypatch.setattr(_ipc, "port_path", lambda name: endpoint)
    endpoint.write_text(contents)
    assert _ipc._read_port_file("malformed") == (None, None)
    endpoint.write_text(original)
    assert json.loads(bu.browser_exec('print(cdp("Probe.read"))', session="malformed", task_id="worker"))["success"]
    lifecycle.cleanup_browser("worker")
    proc.wait(timeout=10)


@pytest.mark.platforms("posix", "windows")
@pytest.mark.parametrize("failure", ["cannot-connect", "timeout", "nonzero"])
def test_stop_failure_preserves_result_and_retries_owned_daemon(harness, monkeypatch, failure):
    _, start = harness
    proc, _, _ = start("retry-stop")
    result = json.loads(bu.browser_exec('print(cdp("Probe.read"))', session="retry-stop", task_id="worker"))
    run = bu._run_cli_killing_process_group
    def failed_stop(cmd, code, env, timeout):
        assert cmd[-1] == "--reload"
        if failure == "cannot-connect":
            raise OSError("fixture connection failure")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, timeout)
        return subprocess.CompletedProcess(cmd, 1, "", "fixture failure")
    monkeypatch.setattr(bu, "_run_cli_killing_process_group", failed_stop)
    lifecycle.cleanup_browser("worker")
    assert result["success"] and proc.poll() is None
    assert len(leases._daemons) == 1
    monkeypatch.setattr(bu, "_run_cli_killing_process_group", run)
    bu.stop_harness_daemons()
    proc.wait(timeout=10)
    assert not leases._daemons
