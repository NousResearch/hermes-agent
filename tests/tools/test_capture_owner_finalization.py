"""Capture failure must reach the final owner before publication."""
from unittest.mock import Mock


def test_adoption_finalizes_capture_before_reader_publication(tmp_path, monkeypatch):
    import subprocess
    import sys
    from tools.process_registry import ProcessRegistry
    from tools.environments.base_output import _BoundedOutputCollector
    registry = ProcessRegistry()
    proc = subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.PIPE, text=True)
    proc.wait(timeout=5)
    collector = _BoundedOutputCollector(20, spill_path=tmp_path / "spill")
    collector.append("x" * 100)
    proc._hermes_capture_collector = collector
    observed = []
    read = registry._reader_loop
    def observe(session):
        observed.append(session.capture_error)
        read(session)
    monkeypatch.setattr(registry, "_reader_loop", observe)
    from tools.terminal_tool_background import yield_to_background_handler
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    early_completions = []
    def fail_redaction(*args, **kwargs):
        registry.list_sessions()  # Status-bar reconciliation cannot see a pending transfer.
        early_completions.extend(registry._finished)
        raise OSError("redaction")
    monkeypatch.setattr("tools.spill_safety.write_text_exclusive", fail_redaction)
    handler = yield_to_background_handler(command="job", env_type="local", cwd=str(tmp_path),
        effective_task_id="task", task_id="task", session_key="")
    try:
        result = handler(proc, collector.render())
        session = registry._running.get(result["yielded_session_id"]) or registry._finished[result["yielded_session_id"]]
        session._reader_thread.join(5)
        assert session._completion_event.is_set()
        assert registry.poll(session.id)["output_incomplete"] is True
        assert not early_completions
        assert result["output_incomplete"] is True
        assert observed and observed[0]
        assert session.capture_error == collector.capture_error
    finally:
        proc.stdout.close()


def test_malformed_executed_cell_result_preserves_partial_output(monkeypatch):
    import json
    from types import SimpleNamespace
    from tools import code_kernel_remote as remote, code_execution_tool as tool
    kernel = SimpleNamespace(kernel_dir="/tmp/kernel", sh=Mock(return_value=""), rpc_token="test")
    monkeypatch.setattr(tool, "_rpc_poll_loop", lambda *a: None)
    monkeypatch.setattr(remote, "_run_remote_cell", lambda *a: (
        "ok", {"stdout": "partial", "execution_count": "invalid"}))
    monkeypatch.setattr(remote._REGISTRY, "discard", Mock())
    result = remote._run_attached_cell(kernel, ("key",), "job", env=Mock(), task_env_id="task",
        sandbox_tools=frozenset(), timeout=5, max_tool_calls=1, reused=True,
        state_reset=False, state_lost=False)
    final = json.loads(tool._finish_remote_kernel_result(result, timeout=5, exec_start=0))
    assert final["output_incomplete"] is True
    assert final["output"] == "partial"
    assert "retry" in final["error"]
