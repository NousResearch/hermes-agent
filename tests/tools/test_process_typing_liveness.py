"""Typing liveness reads bounded completion work, never probes the host."""

from tools.process_registry import ProcessRegistry, ProcessSession


def test_completion_work_is_scoped_and_excludes_silent_or_exited_processes():
    registry = ProcessRegistry()
    silent = ProcessSession(id="silent", command="server", session_key="agent:a:discord:100")
    bounded = ProcessSession(id="bounded", command="build", session_key="agent:b:discord:100", notify_on_complete=True)
    sibling = ProcessSession(id="sibling", command="test", session_key=bounded.session_key, notify_on_complete=True)
    registry._running.update({s.id: s for s in (silent, bounded, sibling)})
    assert not registry.has_completion_work_for_session(silent.session_key)
    assert not registry.has_completion_work_for_session("agent:b:discord:200")
    assert registry.has_completion_work_for_session(bounded.session_key)
    bounded.exited = True  # readers may observe exit before moving the record
    assert registry.has_completion_work_for_session(bounded.session_key)
    sibling.exited = True
    assert not registry.has_completion_work_for_session(bounded.session_key)
