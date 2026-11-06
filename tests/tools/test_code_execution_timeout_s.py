"""Per-call ``timeout_s`` ceiling contract for execute_code (Élya3 timeout prototype).

Behaviour contracts, real session kernel, no source-reading:
- an explicit unknown call key is rejected BEFORE execution (the 06/11 audit
  found a silently ignored parameter made a model believe a limit was applied);
- ``timeout_s`` (1-300) actually shortens the kernel timeout: the cell is
  killed, its state is explicitly lost, and the next call starts a fresh kernel;
- absent ``timeout_s``: the configured timeout applies (no regression);
- on timeout the killed cell leaves no descendant process alive.
"""

import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from tests.tools._child_env_fixtures import child_env
from tools.code_execution_tool import _execute_code_handler, build_execute_code_schema

TASK = "timeout-proto-test"


def _parse(raw: str) -> dict:
    return json.loads(raw)


# ---- Schema contract -------------------------------------------------------

def test_schema_declares_bounded_timeout_s():
    schema = build_execute_code_schema()["parameters"]["properties"]
    assert set(schema) == {"code", "reset", "timeout_s"}
    assert schema["timeout_s"]["minimum"] == 1 and schema["timeout_s"]["maximum"] == 300


# ---- Handler contract (real dispatch path, no kernel needed) ---------------

def _handler_error(raw: str) -> str | None:
    payload = _parse(raw)
    assert payload.get("status") != "success", payload
    return payload.get("error")


def test_unknown_key_rejected_before_execution():
    error = _handler_error(_execute_code_handler({"code": "print('x')", "bogus": 1}))
    assert error and "unknown parameter" in error and "bogus" in error


def test_timeout_s_range_rejected_before_execution():
    for bad in (0, 301, "five", 2.5, True):
        error = _handler_error(_execute_code_handler({"code": "print('x')", "timeout_s": bad}))
        assert error and "timeout_s" in error, bad


def test_command_alias_still_redirects():
    error = _handler_error(_execute_code_handler({"command": "ls"}))
    assert error and "code=..." in error


# ---- Real kernel: custom timeout honoured, state loss explicit -------------

def _call(code, timeout_s=None, reset=False, task=TASK):
    with patch("tools.code_execution_tool._load_config", return_value={}):
        from tools.code_execution_tool import execute_code
        raw = execute_code(code=code, task_id=task, enabled_tools=["read_file"],
                           reset=reset, timeout_s=timeout_s)
    return _parse(raw)


def test_custom_timeout_kills_cell_state_lost_next_kernel_fresh(child_env):  # health: allow F811 -- pytest injects the imported child_env fixture by parameter name
    before = _call("STATE = 'alive'; print('set')", reset=True, task=TASK)
    assert before["status"] == "success"
    assert before["kernel"]["reused"] is False  # reset spawned a fresh kernel

    timed = _call("import time\nimport time as t\nt.sleep(60)\nprint('must not print')", timeout_s=3)
    assert timed["status"] == "timeout", timed
    assert 2.0 <= timed["duration_seconds"] < 20, timed["duration_seconds"]
    assert "killed" in timed.get("error", "") and "lost" in timed.get("error", "")

    after = _call("print(STATE)", task=TASK)
    assert after["status"] == "error"  # STATE gone: per-call timeout did NOT preserve the kernel
    assert after["kernel"]["reused"] is False and after["kernel"]["execution_count"] == 1


def test_descendant_processes_killed_on_timeout(child_env):  # health: allow F811 -- pytest injects the imported child_env fixture by parameter name
    code = ("import subprocess, time\n"
            "subprocess.Popen(['sleep', '60'])\n"
            "with open('%s', 'w') as f:\n"
            "    f.write('witness')\n"
            "time.sleep(60)\n") % (child_env / "timeout-witness.txt")
    timed = _call(code, timeout_s=3)
    assert timed["status"] == "timeout"
    probe = subprocess.run(["pgrep", "-fl", "sleep 60"], capture_output=True, text=True, check=False)
    assert probe.returncode != 0 and not probe.stdout, probe.stdout
    _call("print('fresh')", reset=True, task=TASK)  # leave a clean kernel for later files


@pytest.mark.platforms("macos", "linux")
def test_descendants_killed_live_posix(child_env):  # health: allow F811 -- pytest injects the imported child_env fixture by parameter name
    test_descendant_processes_killed_on_timeout(child_env)


# ---- Remote-path propagation (SIMULATED transport, real code path) ---------

def test_remote_path_ceiling_propagates_to_launch_call(tmp_path, monkeypatch):
    """SIMULATED transport (stub env, no remote host): proves timeout_s
    reaches the remote launch command's deadline via the per-call ship."""
    recorded: list[tuple[str, object]] = []

    class _StubEnv:
        def get_temp_dir(self):
            return str(tmp_path)

        def execute(self, command, cwd=None, timeout=None, stdin_data=None):
            recorded.append((command, timeout))
            if "command -v python3" in command:
                return {"returncode": 0, "output": "OK"}
            if "exec python3 script.py" in command:  # the real wrapped launch of the per-call ship
                return {"returncode": 124, "output": ""}  # hit its ceiling wrapper
            return {"returncode": 0, "output": ""}  # chmod / file ship / rm cleanup

    import tools.code_execution_tool as cet
    monkeypatch.setattr(cet, "_get_or_create_env", lambda task_id: (_StubEnv(), "ssh"))
    from tools import code_kernel_remote as ckr
    monkeypatch.setattr(ckr, "execute_in_remote_kernel",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated: no remote kernel")))
    with patch("tools.code_execution_tool._load_config", return_value={"timeout": 300}):
        result = _parse(cet._execute_remote("print('x')", timeout_s=7,
                                            task_id="remote-test", enabled_tools=["read_file"]))
    assert result["status"] == "timeout", result
    assert "timed out after 7s" in result.get("error", ""), result.get("error")
    launch_timeouts = [t for cmd, t in recorded if "exec python3 script.py" in cmd]
    assert launch_timeouts and launch_timeouts[-1] == 7, recorded


# ---- Concurrency: two cells sharing one kernel when one times out ----------

def test_timeout_on_one_cell_of_shared_kernel_does_not_hang_sibling_cell(child_env):  # health: allow F811 -- pytest injects the imported child_env fixture by parameter name
    """OBSERVED-BEHAVIOUR contract: the timeout kill is kernel-wide by design
    (state loss is deliberate). This pins what a sibling cell on the SAME
    kernel actually gets — success (if it settled first) or a kernel-disposed
    error — and that the next call starts a fresh kernel either way."""
    import threading

    barrier = threading.Event()

    def slow():
        barrier.wait()
        return _call("import time; time.sleep(30); print('never')", timeout_s=3, task="share-kernel")

    def fast():
        barrier.wait()
        return _call("print('sibling')", task="share-kernel")

    # threads via wrappers storing into a dict
    out = {}
    def _run(key, fn):
        barrier.wait()
        out[key] = fn()
    t_slow = threading.Thread(target=_run, args=("slow", slow))
    t_fast = threading.Thread(target=_run, args=("fast", fast))
    t_slow.start(); t_fast.start(); barrier.set()
    t_slow.join(timeout=60); t_fast.join(timeout=60)
    assert all(k in out for k in ("slow", "fast")), out  # neither call hung past its budget
    assert out["slow"]["status"] == "timeout", out["slow"]
    assert out["fast"]["status"] in ("success", "error", "interrupted"), out["fast"]
    after = _call("print('fresh-again')", task="share-kernel")
    assert after["kernel"]["reused"] is False  # a kernel-wide kill leaves no stale kernel behind


# ---- No regression: absent timeout_s keeps the configured behaviour --------

def test_absent_timeout_s_keeps_default_and_persists_state(child_env):  # health: allow F811 -- pytest injects the imported child_env fixture by parameter name
    first = _call("PERSIST = 'kept'", reset=True, task="no-ceiling-task")
    assert first["status"] == "success"
    second = _call("print(PERSIST)", task="no-ceiling-task")
    assert second["status"] == "success"
    assert second["kernel"]["reused"] is True
    assert "kept" in second["output"]
