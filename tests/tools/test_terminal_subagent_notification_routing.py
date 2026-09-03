"""Subagent background-process notifications must return to the parent."""

import json
from types import SimpleNamespace

import pytest

import gateway.session_context as session_context
import tools.approval as approval
import tools.delegate_tool as delegate_tool
import tools.process_registry as process_registry_mod
import tools.terminal_tool as terminal_tool


def _terminal_config():
    return {
        "env_type": "local",
        "cwd": "C:/Projekte",
        "timeout": 1,
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }


@pytest.mark.parametrize(
    "current_session_key",
    ["", "agent:main:discord:dm:123"],
)
def test_subagent_notification_uses_parent_key_without_changing_child_cwd(
    monkeypatch, current_session_key
):
    task_id = "sa-0-notifytest"
    parent_session_id = "parent-session-id"
    process_session_key = current_session_key or task_id
    notification_session_key = current_session_key or parent_session_id
    captured_spawn = {}
    resolved_cwd_keys = []

    class Env:
        env = {}

    class Registry:
        pending_watchers = []

        def spawn_local(self, **kwargs):
            captured_spawn.update(kwargs)
            return SimpleNamespace(
                id="proc_notifytest",
                pid=123,
                watcher_platform="",
                session_key=kwargs["session_key"],
                notification_session_key=kwargs.get("notification_session_key", ""),
                parent_session_id=kwargs.get("parent_session_id", ""),
                notify_on_complete=False,
            )

    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: Env()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {task_id: 0})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_get_env_config", _terminal_config)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool, "_resolve_container_task_id", lambda value: value or "default"
    )
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    def resolve_cwd(**kwargs):
        resolved_cwd_keys.append(kwargs["session_key"])
        return "C:/Projekte"

    monkeypatch.setattr(terminal_tool, "_resolve_command_cwd", resolve_cwd)
    monkeypatch.setattr(process_registry_mod, "process_registry", Registry())
    monkeypatch.setattr(
        approval,
        "get_current_session_key",
        lambda default="": current_session_key,
    )
    monkeypatch.setattr(session_context, "async_delivery_supported", lambda: True)
    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda name, default="": (
            "immediate-child-session" if name == "HERMES_SESSION_ID" else default
        ),
    )
    monkeypatch.setattr(
        delegate_tool,
        "get_subagent_attribution",
        lambda value: {
            "subagent_id": value,
            "delegation_id": "deleg_notifytest",
            "goal": "run a bounded background task",
            "owner_agent_session_id": parent_session_id,
        },
    )

    result = json.loads(
        terminal_tool.terminal_tool(
            command='python -c "raise SystemExit(7)"',
            background=True,
            notify_on_complete=True,
            task_id=task_id,
        )
    )

    assert result["notify_on_complete"] is True
    assert captured_spawn["task_id"] == task_id
    assert set(resolved_cwd_keys) == {process_session_key}
    assert captured_spawn["session_key"] == process_session_key
    assert captured_spawn["notification_session_key"] == notification_session_key
    assert captured_spawn["parent_session_id"] == parent_session_id
    assert captured_spawn["notify_on_complete"] is True


def _patch_terminal_child_spawn(monkeypatch, captured_spawn, task_id, current_session_key=""):
    class Env:
        env = {}

    class Registry:
        pending_watchers = []

        def spawn_local(self, **kwargs):
            captured_spawn.update(kwargs)
            return SimpleNamespace(
                id="proc_notifytest",
                pid=123,
                watcher_platform="",
                session_key=kwargs["session_key"],
                notification_session_key=kwargs.get("notification_session_key", ""),
                parent_session_id=kwargs.get("parent_session_id", ""),
                notify_on_complete=kwargs.get("notify_on_complete", False),
            )

    monkeypatch.setattr(terminal_tool, "_active_environments", {task_id: Env()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {task_id: 0})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_get_env_config", _terminal_config)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool, "_resolve_container_task_id", lambda value: value or "default"
    )
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    monkeypatch.setattr(
        terminal_tool,
        "_resolve_command_cwd",
        lambda **kwargs: "C:/Projekte",
    )
    monkeypatch.setattr(process_registry_mod, "process_registry", Registry())
    monkeypatch.setattr(
        approval,
        "get_current_session_key",
        lambda default="": current_session_key,
    )
    monkeypatch.setattr(session_context, "async_delivery_supported", lambda: True)
    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda name, default="": (
            "immediate-child-session" if name == "HERMES_SESSION_ID" else default
        ),
    )


def test_child_notification_identity_survives_container_key_collapse(monkeypatch):
    """Execution cleanup and notification attribution use separate keys."""
    task_id = "sa-0-container-collapse"
    captured_spawn = {}
    _patch_terminal_child_spawn(monkeypatch, captured_spawn, task_id)
    monkeypatch.setattr(
        terminal_tool,
        "_resolve_container_task_id",
        lambda value: "default",
    )
    monkeypatch.setattr(
        delegate_tool,
        "get_subagent_attribution",
        lambda value: {
            "subagent_id": value,
            "delegation_id": "deleg_container_collapse",
            "goal": "verify process identity",
            "owner_agent_session_id": "root-session",
        },
    )

    result = json.loads(
        terminal_tool.terminal_tool(
            command='python -c "raise SystemExit(7)"',
            background=True,
            notify_on_complete=True,
            task_id=task_id,
        )
    )

    assert result["notify_on_complete"] is True
    assert captured_spawn["task_id"] == "default"
    assert captured_spawn["owner_task_id"] == task_id
    assert captured_spawn["session_key"] == task_id
    assert captured_spawn["notification_session_key"] == "root-session"
    assert captured_spawn["parent_session_id"] == "root-session"


def test_host_local_child_keeps_cleanup_and_notification_identities(monkeypatch):
    """Host-local cleanup namespace and delegate attribution must both survive."""
    task_id = "sa-0-host-local"
    captured_spawn = {}
    _patch_terminal_child_spawn(monkeypatch, captured_spawn, task_id)
    monkeypatch.setattr(
        terminal_tool,
        "_resolve_container_task_id",
        lambda value: "shared",
    )
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {"host-local-shared": SimpleNamespace(env={})},
    )
    monkeypatch.setattr(terminal_tool, "_last_activity", {"host-local-shared": 0})
    monkeypatch.setattr(
        delegate_tool,
        "get_subagent_attribution",
        lambda value: {
            "subagent_id": value,
            "delegation_id": "deleg_host_local",
            "goal": "verify host-local identities",
            "owner_agent_session_id": "root-session",
        },
    )

    result = json.loads(
        terminal_tool.terminal_tool(
            command="host-runner",
            background=True,
            notify_on_complete=True,
            task_id=task_id,
            _host_local=True,
        )
    )

    assert result["notify_on_complete"] is True
    assert captured_spawn["task_id"] == "host-local-shared"
    assert captured_spawn["owner_task_id"] == task_id
    assert captured_spawn["notification_session_key"] == "root-session"


def test_child_completion_event_retains_raw_identity_after_container_collapse(
    monkeypatch,
):
    """Terminal routing and the real registry must agree on child identity."""
    task_id = "sa-0-event-identity"
    captured_spawn = {}
    _patch_terminal_child_spawn(monkeypatch, captured_spawn, task_id)
    monkeypatch.setattr(
        terminal_tool,
        "_resolve_container_task_id",
        lambda value: "default",
    )
    monkeypatch.setattr(
        delegate_tool,
        "get_subagent_attribution",
        lambda value: {
            "subagent_id": value,
            "delegation_id": "deleg_event_identity",
            "goal": "verify queued event identity",
            "owner_agent_session_id": "root-session",
        },
    )

    event_registry = process_registry_mod.ProcessRegistry()
    event_registry._write_checkpoint = lambda: None
    while not event_registry.completion_queue.empty():
        event_registry.completion_queue.get_nowait()

    class _EventRegistry:
        pending_watchers = []

        def spawn_local(self, **kwargs):
            session = process_registry_mod.ProcessSession(
                id="proc_event_identity",
                command=kwargs["command"],
                task_id=kwargs["task_id"],
                session_key=kwargs["session_key"],
                notification_session_key=kwargs.get("notification_session_key", ""),
                owner_task_id=kwargs.get("owner_task_id", ""),
                parent_session_id=kwargs.get("parent_session_id", ""),
                notify_on_complete=kwargs.get("notify_on_complete", False),
                started_at=1234.5,
            )
            event_registry._running[session.id] = session
            session.exited = True
            session.exit_code = 7
            session.completion_reason = "exited"
            session.output_buffer = "EVENT_IDENTITY_CANARY\n"
            event_registry._move_to_finished(session)
            return session

    monkeypatch.setattr(process_registry_mod, "process_registry", _EventRegistry())

    result = json.loads(
        terminal_tool.terminal_tool(
            command='python -c "raise SystemExit(7)"',
            background=True,
            notify_on_complete=True,
            task_id=task_id,
        )
    )
    event = event_registry.completion_queue.get_nowait()

    assert result["notify_on_complete"] is True
    assert event["task_id"] == "default"
    assert event["owner_task_id"] == task_id
    assert event["session_key"] == "root-session"
    assert event["parent_session_id"] == "root-session"
    assert event["exit_code"] == 7
    assert process_registry_mod.ProcessRegistry.should_surface_process_notification(event)


@pytest.mark.parametrize("broken_attribution", [None, {}, {"owner_agent_session_id": ""}])
def test_unresolved_child_attribution_fail_closes_notification(
    monkeypatch, broken_attribution
):
    task_id = "sa-0-orphan"
    captured_spawn = {}
    _patch_terminal_child_spawn(monkeypatch, captured_spawn, task_id)
    monkeypatch.setattr(
        delegate_tool, "get_subagent_attribution", lambda value: broken_attribution
    )

    result = json.loads(
        terminal_tool.terminal_tool(
            command='python -c "raise SystemExit(7)"',
            background=True,
            notify_on_complete=True,
            task_id=task_id,
        )
    )

    assert result.get("notify_on_complete") is False
    assert captured_spawn["session_key"] == task_id
    assert captured_spawn.get("parent_session_id") != "immediate-child-session"
    assert captured_spawn.get("notify_on_complete") is not True


def test_failed_attribution_lookup_does_not_stamp_child_session(monkeypatch):
    task_id = "sa-0-orphan-exc"
    captured_spawn = {}
    _patch_terminal_child_spawn(monkeypatch, captured_spawn, task_id)

    def _boom(value):
        raise RuntimeError("attribution unavailable")

    monkeypatch.setattr(delegate_tool, "get_subagent_attribution", _boom)

    result = json.loads(
        terminal_tool.terminal_tool(
            command='python -c "raise SystemExit(7)"',
            background=True,
            notify_on_complete=True,
            task_id=task_id,
        )
    )

    assert result.get("notify_on_complete") is False
    assert captured_spawn.get("parent_session_id") != "immediate-child-session"
    assert captured_spawn.get("notify_on_complete") is not True
