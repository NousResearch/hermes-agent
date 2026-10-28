"""Unit Stop must retain bytes written to the original log after primary exit."""

import ctypes
import json
import os
from pathlib import Path
import shlex
import sys

import psutil
import pytest

from tests.waiting import wait_until


pytestmark = pytest.mark.platforms("linux")


@pytest.fixture
def unit_log_job(tmp_path, monkeypatch, request):
    from tools import process_registry as module

    # Adopt and reap the actual detached child rather than leaving a zombie behind.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    home = tmp_path / "home"
    logdir = home / "logs" / "process-output"
    logdir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(module, "CHECKPOINT_PATH", home / "processes.json")
    monkeypatch.setattr(module, "worker_output_log", lambda sid: logdir / (sid + ".log"))
    registry = module.ProcessRegistry()

    def unit_launch(session, command, suffix, label):
        session.systemd_unit = "hermes-worker-" + suffix + ".scope"
        return [module._find_shell(), "-lic", "set +m; " + command]

    monkeypatch.setattr(registry, "_scope_argv", unit_launch)
    gate, childfile = tmp_path / "release-child", tmp_path / "child-pid"
    split = getattr(request, 'param', None) == 'split-utf8'
    late = " os.write(1, " + repr(b'\x82\xacLATE ORIGINAL OUTPUT\n') + "); os._exit(0)\n" if split else " print('LATE ORIGINAL OUTPUT',flush=True); os._exit(0)\n"
    primary = "os.write(1, " + repr(b'PRIMARY ORIGINAL OUTPUT\n\xe2') + ")\n" if split else "print('PRIMARY ORIGINAL OUTPUT',flush=True)\n"
    payload = (
        "import os,time,threading; from pathlib import Path\n"
        "child=os.fork()\n"
        "if child==0:\n"
        " os.setsid()\n"
        f" gate=Path({str(gate)!r}); deadline=time.monotonic()+15\n"
        " while not gate.exists() and time.monotonic()<deadline: threading.Event().wait(.01)\n"
        + late +
        f"Path({str(childfile)!r}).write_text(str(child))\n"
        + primary +
        "raise SystemExit(7)\n"
    )
    session = registry.spawn_local(shlex.join([sys.executable, "-c", payload]),
                                   cwd=str(tmp_path), persist_on_release=True, task_id='worker',
                                   owner_task_id='worker', session_key='route')
    child_pid = None
    try:
        assert session._completion_event.wait(5)
        assert session.exit_code == 7
        assert session.output_buffer.count("PRIMARY ORIGINAL OUTPUT") == 1
        assert "LATE ORIGINAL OUTPUT" not in session.output_buffer
        child_pid = int(childfile.read_text())
        assert psutil.Process(child_pid).is_running()

        def reap_child():
            gate.touch()
            wait_until(lambda: _child_has_exited(child_pid), timeout=5)
            assert os.waitpid(child_pid, 0)[0] == child_pid
            assert not psutil.pid_exists(child_pid)

        def stop_empty_unit(unit):
            # A mock unit verdict is valid only after the real fixture is empty.
            assert child_pid is not None and not psutil.pid_exists(child_pid)
            assert unit.startswith("hermes-worker-")
            return True

        monkeypatch.setattr(module, "_stop_systemd_unit", stop_empty_unit)
        yield module, registry, session, home, reap_child
    finally:
        gate.touch()
        if child_pid is None and childfile.exists():
            child_pid = int(childfile.read_text())
        if child_pid is not None and psutil.pid_exists(child_pid):
            wait_until(lambda: _child_has_exited(child_pid), timeout=5)
            os.waitpid(child_pid, 0)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def _child_has_exited(pid):
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _retained(home, session_id):
    return json.loads((home / "logs" / "process-results" / (session_id + ".json")).read_text())


def _checkpoint_ids(module):
    return {row["session_id"] for row in json.loads(module._checkpoint_path().read_text())}


def _owner(module, registry, session, reload_owner):
    if not reload_owner:
        return registry, session
    cold = module.ProcessRegistry()
    assert cold.recover_from_checkpoint() == 1
    restored = cold.get(session.id)
    assert restored is not None and restored is not session
    assert restored._completion_event.wait(5)
    return cold, restored


def _assert_final_output(output):
    assert output.count("PRIMARY ORIGINAL OUTPUT") == 1
    assert output.count("LATE ORIGINAL OUTPUT") == 1
    assert output.index("PRIMARY ORIGINAL OUTPUT") < output.index("LATE ORIGINAL OUTPUT")


@pytest.mark.parametrize("reload_owner", [False, True], ids=["live", "cold"])
def test_unit_stop_collects_late_original_tail_and_rejects_stale_primary_writer(unit_log_job, reload_owner):
    from tools.process_registry_results import load_completed_results, save_completed_result

    module, registry, session, home, reap_child = unit_log_job
    stale = load_completed_results(owns_entry=lambda row: row["id"] == session.id, strict=True)[session.id]
    stale_registry = module.ProcessRegistry()
    stale_registry._finished[stale.id] = stale
    assert "PRIMARY ORIGINAL OUTPUT" in stale.output_buffer
    assert "LATE ORIGINAL OUTPUT" not in stale.output_buffer
    reap_child()
    _assert_final_output(Path(session.output_log).read_text())
    registry, current = _owner(module, registry, session, reload_owner)
    stopped = registry.kill_process(session.id)
    assert stopped["status"] == "already_exited", stopped
    _assert_final_output(stopped["output"])
    _assert_final_output(current.output_buffer)
    saved = _retained(home, session.id)
    assert saved["result_complete"] is True and saved["exit_code"] == 7
    _assert_final_output(saved["output"])
    # A primary-era cache may still publish consumption metadata after final Stop.
    stale.completion_consumed = True
    assert save_completed_result(stale)
    saved = _retained(home, session.id)
    assert saved["result_complete"] is True and saved["exit_code"] == 7
    assert saved["completion_consumed"] is True
    _assert_final_output(saved["output"])
    retained = load_completed_results(owns_entry=lambda row: row["id"] == session.id, strict=True)[session.id]
    assert retained.exit_code == 7
    _assert_final_output(retained.output_buffer)
    assert stale_registry._write_checkpoint()
    assert session.id not in _checkpoint_ids(module)


@pytest.mark.parametrize("reload_owner", [False, True], ids=["live", "cold"])
@pytest.mark.parametrize("damage", ["unreadable", "replaced", "truncated", "regrown"])
def test_unit_stop_preserves_unknown_source_until_genuine_restore(unit_log_job, reload_owner, damage):
    module, registry, session, home, reap_child = unit_log_job
    reap_child()
    log = Path(session.output_log)
    original = log.read_bytes()
    _assert_final_output(original.decode())
    identity = log.stat().st_dev, log.stat().st_ino
    backup = log.with_suffix(".original")
    if damage == "unreadable":
        log.chmod(0)
    elif damage == "replaced":
        log.rename(backup)
        log.write_bytes(original)
    elif damage == 'truncated':
        log.write_bytes(b"PRIMARY")
    else:
        log.write_bytes(b'X' * (len(original) + 7))
    try:
        registry, current = _owner(module, registry, session, reload_owner)
        stopped = registry.kill_process(session.id)
        assert stopped["status"] == "error" and stopped["error"] == "process_stop_unknown", stopped
        assert current._scope_stop_pending
        assert session.id in _checkpoint_ids(module)
        assert log.exists()
        if damage == "replaced":
            assert backup.read_bytes() == original
    finally:
        if damage == "unreadable":
            log.chmod(0o600)
        elif damage == "replaced":
            log.unlink()
            backup.rename(log)
        else:
            log.write_bytes(original)
    assert (log.stat().st_dev, log.stat().st_ino) == identity
    assert log.read_bytes() == original
    stopped = registry.kill_process(session.id)
    assert stopped["status"] == "already_exited", stopped
    _assert_final_output(stopped["output"])
    saved = _retained(home, session.id)
    assert saved["result_complete"] is True and saved["exit_code"] == 7
    _assert_final_output(saved["output"])
    assert session.id not in _checkpoint_ids(module)


@pytest.mark.parametrize('unit_log_job', ['split-utf8'], indirect=True)
def test_unit_final_collection_redecodes_split_utf8_without_duplicate_tail(unit_log_job):
    _module, registry, session, home, reap_child = unit_log_job
    assert '\ufffd' in session.output_buffer  # The primary ended inside this codepoint.
    reap_child()
    stopped = registry.kill_process(session.id)
    assert stopped['status'] == 'already_exited'
    _assert_final_output(stopped['output'])
    assert stopped['output'].count('\u20ac') == 1 and '\ufffd' not in stopped['output']
    assert _retained(home, session.id)['output'] == session.output_buffer


def test_finalized_retained_payload_survives_later_source_read_denial(unit_log_job):
    from tools.process_registry_results import load_completed_results
    module, registry, session, _home, reap_child = unit_log_job
    reap_child()
    assert registry.kill_process(session.id)['status'] == 'already_exited'
    log = Path(session.output_log)
    log.chmod(0)
    try:
        cold = module.ProcessRegistry()
        cold._finished.update(load_completed_results(owns_entry=lambda row: row['id'] == session.id, strict=True))
        stopped = cold.kill_process(session.id)
        assert stopped['status'] == 'already_exited'
        _assert_final_output(stopped['output'])
    finally:
        log.chmod(0o600)


def test_legacy_missing_original_fd_witness_does_not_invent_final_payload(unit_log_job):
    module, registry, session, home, reap_child = unit_log_job
    reap_child()
    session.output_log_state = None
    for path in [module._checkpoint_path(), home / 'logs' / 'process-results' / (session.id + '.json')]:
        value = json.loads(path.read_text())
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            row.pop('output_log_state', None)
        path.write_text(json.dumps(value))
    stopped = registry.kill_process(session.id)
    assert stopped['status'] == 'error' and stopped['error'] == 'process_stop_unknown'
    assert session._scope_stop_pending and not session._result_complete
    assert session.id in _checkpoint_ids(module)


def test_unit_quiescence_cannot_invent_missing_original_status(unit_log_job):
    module, registry, session, home, reap_child = unit_log_job
    reap_child()
    exit_path = Path(session.output_log).with_suffix('.exit')
    original_status = exit_path.read_bytes()
    exit_path.unlink()
    (home / 'logs' / 'process-results' / (session.id + '.json')).unlink()
    cold, current = _owner(module, registry, session, True)
    stopped = cold.kill_process(session.id)
    assert stopped['status'] == 'error' and stopped['error'] == 'process_stop_unknown'
    assert current._scope_stop_pending and not current._result_complete
    exit_path.write_bytes(original_status)  # Actual original producer receipt, no guessed status.
    stopped = cold.kill_process(session.id)
    assert stopped['status'] == 'already_exited' and stopped['exit_code'] == 7
    _assert_final_output(stopped['output'])


def test_bound_unit_original_log_does_not_expire_after_primary_exit(unit_log_job, monkeypatch):
    from agent import runtime_session_store
    from tools import process_registry_output_log as logs
    from tools.process_registry_results import RESULT_RETENTION_SECONDS
    import time
    _module, registry, session, _home, reap_child = unit_log_job
    log = Path(session.output_log)
    exit_path = log.with_suffix('.exit')
    old = time.time() - RESULT_RETENTION_SECONDS - 60
    os.utime(exit_path, (old, old))
    monkeypatch.setattr(runtime_session_store, 'is_worker_process', lambda: True)
    logs.worker_output_log('proc_unrelated_new')
    assert log.exists() and exit_path.exists(), 'A retained native unit still needs its original source'
    reap_child()
    stopped = registry.kill_process(session.id)
    assert stopped['status'] == 'already_exited' and stopped['exit_code'] == 7
    _assert_final_output(stopped['output'])
