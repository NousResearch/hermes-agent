"""Manual /compress rebuilds the system prompt on the RPC thread; it must see the session's workspace.

Otherwise the rebuilt prompt records the backend process cwd, and every later resume rejects it as
stale runtime and rebuilds it again (a second prompt-cache miss right after the compaction).
"""
import threading
from unittest.mock import MagicMock


def test_manual_compress_runs_under_the_session_cwd(tmp_path, monkeypatch):
    from agent.runtime_cwd import resolve_agent_cwd
    from tui_gateway.server import _compress_session_history

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tui_gateway.server._get_usage", lambda _agent: {})
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(6)]
    seen = []
    agent = MagicMock(_cached_system_prompt="", tools=None, session_id="sess-cwd",
                      _pending_context_engine_compression_notification=None,
                      _compression_skipped_due_to_lock=None)

    def _fake_compress(msgs=None, *_args, **_kwargs):
        seen.append(resolve_agent_cwd())
        return list(msgs or history)[-2:], ""

    agent._compress_context.side_effect = _fake_compress
    session = {"agent": agent, "history_lock": threading.Lock(), "history": list(history),
               "history_version": 1, "running": False, "session_key": "sess-cwd", "cwd": str(workspace)}

    _compress_session_history(session)

    assert seen == [workspace]
    assert resolve_agent_cwd() == tmp_path  # the binding does not leak past the call
