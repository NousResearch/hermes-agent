"""The registered terminal receipt agrees with its armed watcher and delivery policy."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("promoted", [False, True])
@pytest.mark.parametrize("surface_child", [False, True])
def test_child_receipt_preserves_watcher_intent_through_handoff(tmp_path, monkeypatch, promoted, surface_child):
    import tools.terminal_tool as terminal
    from agent.delegation_context import delegated_child_context
    from tools.process_registry import ProcessSession, process_registry
    from tools.process_registry_notifications import should_surface_notification
    from tools.registry import registry

    monkeypatch.setattr(terminal, "_get_env_config", lambda: {
        "env_type": "local", "cwd": str(tmp_path), "timeout": 30,
        "docker_image": "", "singularity_image": "", "modal_image": "", "daytona_image": "",
    })
    monkeypatch.setattr(terminal, "_acquire_env", lambda *args: SimpleNamespace(env={}))
    monkeypatch.setattr(terminal, "_check_all_guards", lambda *args, **kwargs: {"approved": True})
    monkeypatch.setattr(process_registry, "_surface_child_process_notifications", lambda: surface_child)
    # The shared delivery predicate reads the class policy, just like real drains.
    monkeypatch.setattr(type(process_registry), "_surface_child_process_notifications", staticmethod(lambda: surface_child))
    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: True)
    session = ProcessSession(id="proc_receipt", command="echo receipt", task_id="sa-child", owner_task_id="sa-child")
    monkeypatch.setitem(process_registry._running, session.id, session)
    monkeypatch.setattr(process_registry, "spawn_local", lambda **kwargs: session)
    args = {"command": "echo receipt", "background": not promoted, "notify": True}
    if promoted:
        args.pop("notify")
        args["timeout"] = terminal.FOREGROUND_MAX_TIMEOUT + 1
    with delegated_child_context("child"):
        result = json.loads(registry.dispatch("terminal", args, task_id="sa-child"))
    assert result.get("error") is None, result
    assert result["notify_on_complete"] is session.notify_on_complete is True
    assert result["notification_suppressed"] is (not surface_child)
    assert should_surface_notification({"owner_task_id": session.owner_task_id}) is surface_child
    if promoted and not surface_child:
        assert "suppressed" in result["promoted_from_foreground"]
    moved = process_registry.transfer_ownership(
        session.id, from_owner="sa-child", to_owner="parent", to_task_id="parent", to_session_key="parent")
    assert moved is session and session.notify_on_complete
    assert should_surface_notification({"owner_task_id": session.owner_task_id})

@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("surface_child", [False, True])
def test_foreground_yield_receipt_matches_adopted_process(tmp_path, monkeypatch, supported, surface_child):
    import tools.terminal_tool as terminal
    from agent.delegation_context import delegated_child_context
    from tools.process_registry import ProcessSession, process_registry
    from tools.registry import registry

    monkeypatch.setattr(terminal, "_get_env_config", lambda: {
        "env_type": "local", "cwd": str(tmp_path), "timeout": 30,
        "docker_image": "", "singularity_image": "", "modal_image": "", "daytona_image": "",
    })
    monkeypatch.setattr(terminal, "_check_all_guards", lambda *args, **kwargs: {"approved": True})
    monkeypatch.setattr(type(process_registry), "_surface_child_process_notifications", staticmethod(lambda: surface_child))
    monkeypatch.setattr("gateway.session_context.async_delivery_supported", lambda: supported)
    session = ProcessSession(id="proc_yield", command="echo receipt", task_id="sa-child",
                             owner_task_id="sa-child", notify_on_complete=True)
    monkeypatch.setattr(process_registry, "adopt_local", lambda *args, **kwargs: session)
    def execute(command, **kwargs):
        return kwargs["yield_handler"](object(), "partial output")
    monkeypatch.setattr(terminal, "_acquire_env", lambda *args: SimpleNamespace(env={}, execute=execute))
    with delegated_child_context("child"):
        result = json.loads(registry.dispatch("terminal", {"command": "echo receipt"}, task_id="sa-child"))
    assert result.get("error") is None, result
    assert result["notify_on_complete"] is session.notify_on_complete is supported
    if supported:
        assert result["notification_suppressed"] is (not surface_child)
    else:
        assert "notify_unsupported" in result
    assert "You will be notified" not in result["note"]
