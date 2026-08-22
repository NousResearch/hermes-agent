import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tui_gateway.compute_host import ComputeHost, HostSession


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


def test_compute_host_line_json_seed_turn_interrupt():
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

        proc.stdin.write(json.dumps({"type": "session.seed", "sid": "s1", "request_id": "seed"}) + "\n")
        proc.stdin.flush()
        assert _read_json_line(out)["type"] == "session.seeded"

        proc.stdin.write(
            json.dumps(
                {
                    "type": "turn.start",
                    "sid": "s1",
                    "request_id": "turn",
                    "prompt": "hello",
                    "delta_count": 3,
                    "delay_s": 0,
                }
            )
            + "\n"
        )
        proc.stdin.flush()

        seen = []
        while True:
            frame = _read_json_line(out)
            seen.append(frame["type"])
            if frame["type"] == "turn.end":
                assert frame["history_version"] == 1
                assert frame["message_count"] == 2
                break
        assert seen.count("delta") == 3

        proc.stdin.write(json.dumps({"type": "shutdown", "request_id": "stop"}) + "\n")
        proc.stdin.flush()
        assert _read_json_line(out)["type"] == "shutdown.ack"
        proc.wait(timeout=2)
    finally:
        if proc.poll() is None:
            proc.kill()


@pytest.mark.parametrize("kind", ["legacy", "hard-only", "dynamic-getattr"])
def test_compute_host_interrupt_uses_explicit_stop_compatibility(kind):
    calls = []

    class _Legacy:
        def interrupt(self):
            calls.append("legacy")

    class _HardOnly:
        def hard_interrupt(self):
            calls.append("hard")

    class _Dynamic:
        def interrupt(self):
            calls.append("legacy")

        def __getattr__(self, name):
            if name == "hard_interrupt":
                return lambda: calls.append("fabricated-hard")
            raise AttributeError(name)

    agent = {
        "legacy": _Legacy(),
        "hard-only": _HardOnly(),
        "dynamic-getattr": _Dynamic(),
    }[kind]
    host = ComputeHost(heartbeat_secs=0)
    host._sessions["s1"] = HostSession(sid="s1", agent=agent)
    emitted = []
    host.emit = emitted.append
    try:
        host._handle_interrupt({"sid": "s1", "request_id": "stop"})
    finally:
        host.close()

    assert calls == ["hard" if kind == "hard-only" else "legacy"]
    assert emitted[-1]["applied"] is True


def test_compute_host_spawn_env_excludes_tier1_secrets(monkeypatch, tmp_path):
    """#77463: the compute-host child env must come from the sanitized
    hermes_subprocess_env, NOT a post-scrub env.update(os.environ) which
    re-added every Tier-1 secret (gateway tokens, remote-compute auth).

    E2E with a REAL child: seed Tier-1 secrets in the parent, build the env
    exactly as the fixed _spawn_locked does (hermes_subprocess_env +
    heartbeat/PYTHONPATH additions), spawn a real Python child that reports
    which keys it can see in ITS OWN environment, and assert the secrets are
    absent while the legitimate additions survive.
    """
    import json as _json
    import subprocess as _sp

    from tools.environments.local import hermes_subprocess_env

    monkeypatch.setenv("GATEWAY_RELAY_SECRET", "«redacted:tier1-secret»")
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "«redacted:session»")

    # Build the env exactly as the fixed _spawn_locked does.
    env = hermes_subprocess_env(inherit_credentials=True)
    env["HERMES_COMPUTE_HOST_HEARTBEAT_SECS"] = "5"
    env.setdefault("PYTHONPATH", str(tmp_path))

    probe = (
        "import json, os; print(json.dumps({"
        "'relay': 'GATEWAY_RELAY_SECRET' in os.environ, "
        "'session': 'HERMES_DASHBOARD_SESSION_TOKEN' in os.environ, "
        "'heartbeat': os.environ.get('HERMES_COMPUTE_HOST_HEARTBEAT_SECS', ''), "
        "'pythonpath_present': bool(os.environ.get('PYTHONPATH', ''))}))"
    )

    out = _sp.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    result = _json.loads(out.stdout.strip().splitlines()[-1])
    assert result["relay"] is False, "Tier-1 relay secret leaked to compute host"
    assert result["session"] is False, "session token leaked to compute host"
    assert result["heartbeat"] == "5", "heartbeat must survive"
    assert result["pythonpath_present"] is True, "PYTHONPATH must survive"
