import json
from types import SimpleNamespace

import tools.terminal_tool as terminal_tool_module
from tools import process_registry as process_registry_module


def _base_config(tmp_path):
    return {
        "env_type": "local",
        "docker_image": "",
        "modal_image": "",
        "daytona_image": "",
        "cwd": str(tmp_path),
        "timeout": 30,
    }


def test_command_requires_pipe_stdin_detects_gh_with_token():
    assert terminal_tool_module._command_requires_pipe_stdin(
        "gh auth login --hostname github.com --git-protocol https --with-token"
    ) is True
    assert terminal_tool_module._command_requires_pipe_stdin(
        "gh auth login --web"
    ) is False


def test_terminal_background_keeps_pty_for_regular_interactive_commands(monkeypatch, tmp_path):
    config = _base_config(tmp_path)
    dummy_env = SimpleNamespace(env={})
    captured = {}

    def fake_spawn_local(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id="proc_test", pid=1234, notify_on_complete=False)

    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool_module, "_check_all_guards", lambda *_args, **_kwargs: {"approved": True})
    monkeypatch.setattr(process_registry_module.process_registry, "spawn_local", fake_spawn_local)
    monkeypatch.setitem(terminal_tool_module._active_environments, "default", dummy_env)
    monkeypatch.setitem(terminal_tool_module._last_activity, "default", 0.0)

    try:
        result = json.loads(
            terminal_tool_module.terminal_tool(
                command="python3 -c \"print(input())\"",
                background=True,
                pty=True,
            )
        )
    finally:
        terminal_tool_module._active_environments.pop("default", None)
        terminal_tool_module._last_activity.pop("default", None)

    assert captured["use_pty"] is True
    assert "pty_note" not in result


def test_terminal_handler_infers_background_when_pty_dependency_is_omitted(monkeypatch):
    captured = {}

    def fake_terminal_tool(**kwargs):
        captured.update(kwargs)
        return '{"ok":true}'

    monkeypatch.setattr(terminal_tool_module, "terminal_tool", fake_terminal_tool)

    result = json.loads(
        terminal_tool_module._handle_terminal(
            {
                "command": "hermes auth add openai-codex --type oauth --no-browser",
                "pty": True,
                "timeout": 900,
            },
            task_id="test",
        )
    )

    assert result == {"ok": True}
    assert captured["background"] is True
    assert captured["pty"] is True
    assert captured["timeout"] == 900


def test_terminal_handler_rejects_explicit_foreground_pty(monkeypatch):
    called = False

    def fake_terminal_tool(**_kwargs):
        nonlocal called
        called = True
        return '{"ok":true}'

    monkeypatch.setattr(terminal_tool_module, "terminal_tool", fake_terminal_tool)

    result = json.loads(
        terminal_tool_module._handle_terminal(
            {"command": "python3", "background": False, "pty": True},
            task_id="test",
        )
    )

    assert "background=true" in result["error"]
    assert called is False


def test_terminal_handler_inferred_pty_background_keeps_background_modifiers(monkeypatch):
    captured = {}

    def fake_terminal_tool(**kwargs):
        captured.update(kwargs)
        return '{"ok":true}'

    monkeypatch.setattr(terminal_tool_module, "terminal_tool", fake_terminal_tool)

    result = json.loads(
        terminal_tool_module._handle_terminal(
            {"command": "python3", "pty": True, "notify": True, "heartbeat": 120, "persist_on_release": True},
            task_id="test",
        )
    )

    assert result == {"ok": True}
    assert captured["background"] is True
    assert captured["notify_on_complete"] is True
    assert captured["heartbeat"] == 120
    assert captured["persist_on_release"] is True


def test_prepared_approval_keys_on_the_inferred_background(monkeypatch, tmp_path):
    """Desktop batch preparation must key the guard on the command the handler will check: a pty
    call without ``background`` runs as a background process, so its trailing ``&`` stays."""
    import threading

    from agent import terminal_approval_batch as tab

    config = _base_config(tmp_path)
    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool_module, "_docker_has_host_access", lambda _c: False)
    checked = []
    monkeypatch.setattr(terminal_tool_module, "_check_all_guards",
                        lambda *key: checked.append(key) or {"approved": True})
    batch = SimpleNamespace(cancelled=threading.Event(), task_id="t",
                            agent=SimpleNamespace(_interrupt_requested=False, session_id="s"))

    def prepared_command(args):
        slot = tab._TerminalSlot(batch, parsed=None, index=0)
        slot.release.set()
        slot.prepare(SimpleNamespace(args=args, call_id="c1"))
        return slot.guard_key[0]

    assert prepared_command({"command": "sleep 5 &", "pty": True}) == "sleep 5 &"
    assert prepared_command({"command": "sleep 5 &", "pty": True, "background": True}) == "sleep 5 &"
    assert prepared_command({"command": "sleep 5 &"}) == "sleep 5"
    assert prepared_command({"command": "sleep 5 &", "pty": False}) == "sleep 5"
    assert [key[0] for key in checked] == ["sleep 5 &", "sleep 5 &", "sleep 5", "sleep 5"]
