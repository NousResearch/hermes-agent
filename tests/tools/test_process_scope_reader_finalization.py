"""Native unit Stop preserves the original PTY and log readers until finalization."""

import json
import threading

import pytest


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("transport,primary_exit", [("pty", False), ("log", False), ("log", True)],
                         ids=["pty", "log", "reconciled-log"])
def test_unit_stop_waits_for_original_reader(tmp_path, monkeypatch, transport, primary_exit):
    from tools import process_registry as module
    from tools.process_registry_job import job_quiescent
    from tools.process_registry_results import load_completed_results

    monkeypatch.setattr(module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = module.ProcessRegistry()

    def unit_transport(session, command, suffix, label):
        session.systemd_unit = "hermes-worker-" + suffix + ".scope"
        return [module._find_shell(), "-lic", "set +m; " + command]

    monkeypatch.setattr(registry, "_scope_argv", unit_transport)
    attempts = []
    monkeypatch.setattr(module, "_stop_systemd_unit", lambda unit: attempts.append(unit) or True)
    if transport == "log":
        monkeypatch.setattr(module, "worker_output_log", lambda sid: tmp_path / (sid + ".log"))
    entered, release = threading.Event(), threading.Event()
    ingest = registry._ingest_output

    def paused_ingest(session, text, **kwargs):
        if "RACE-MARKER" in text:
            entered.set()
            assert release.wait(20), "test must release the original reader"
        ingest(session, text, **kwargs)

    monkeypatch.setattr(registry, "_ingest_output", paused_ingest)
    command = "echo RACE-MARKER" if primary_exit else "echo RACE-MARKER; exec sleep 30"
    session = registry.spawn_local(command, cwd=str(tmp_path), use_pty=transport == "pty")
    try:
        assert entered.wait(5)
        original_reader = session._reader_thread
        unit = session.systemd_unit
        assert unit and not session.native_job_scope
        if primary_exit:
            assert session.process.wait(5) == 0
            registry.poll(session.id)  # Real local-exit reconciliation, with the reader still held.
            assert session.exited and session._reader_finish_requested.is_set()
        result = registry.kill_process(session.id)
        assert result["status"] == "error" and result["error"] == "process_stop_unknown"
        assert session._scope_stop_pending
        assert job_quiescent(session) is not primary_exit
        assert not session._completion_event.is_set()
        assert original_reader.is_alive()
        assert "RACE-MARKER" not in session.output_buffer
        checkpoint = next(row for row in json.loads(module._checkpoint_path().read_text())
                          if row["session_id"] == session.id)
        assert checkpoint["_scope_stop_pending"]

        release.set()
        assert session._completion_event.wait(5)
        original_reader.join(5)
        assert not original_reader.is_alive()
        assert session._reader_thread is original_reader
        result = registry.kill_process(session.id)
        assert result["status"] in {"already_exited", "killed"}
        assert not session._scope_stop_pending
        assert attempts == [unit] and job_quiescent(session)
        assert session.output_buffer.count("RACE-MARKER") == 1
        assert result["output"].count("RACE-MARKER") == 1
        retained = load_completed_results(owns_entry=lambda row: row["id"] == session.id, strict=True)[session.id]
        assert retained.output_buffer.count("RACE-MARKER") == 1
        assert retained.exit_code == session.exit_code
        assert retained.exit_code == 0 if primary_exit else retained.exit_code != 0
        assert session.id not in {row["session_id"] for row in json.loads(module._checkpoint_path().read_text())}
    finally:
        release.set()
        session._reader_thread.join(5)
        registry.kill_process(session.id)
