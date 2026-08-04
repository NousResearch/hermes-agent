def test_spawn_via_env_forwards_selected_cwd():
    from tools.process_registry import ProcessRegistry

    calls = []

    class FakeEnvironment:
        # Simulate another session changing mutable shared-environment cwd.
        cwd = "/srv/other-session"

        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "", "returncode": 1}

    registry = ProcessRegistry()
    session = registry.spawn_via_env(
        env=FakeEnvironment(),
        command="pwd",
        cwd="/srv/named-target",
        task_id="process-cwd",
        session_key="named-session",
        target="remote",
        backend="ssh",
    )

    assert session.exited is True
    assert session.cwd == "/srv/named-target"
    assert calls[0][1]["cwd"] == session.cwd
    assert calls[0][1]["cwd"] != FakeEnvironment.cwd
    assert calls[0][1]["rewrite_compound_background"] is False
