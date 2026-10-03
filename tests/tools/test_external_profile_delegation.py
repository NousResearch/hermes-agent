"""Contract tests for explicit external-profile delegate_task dispatch."""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "parent-session"
    return parent


def test_external_profile_validation_rejects_path_and_missing_profile(monkeypatch):
    import tools.delegate_tool_external_profile as external

    with pytest.raises(ValueError):
        external.validate_external_profile("../escape")

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _profile: False)
    with pytest.raises(ValueError, match="does not exist"):
        external.validate_external_profile("expertkodare")


def test_external_profile_dispatch_uses_async_ledger_and_origin(monkeypatch, tmp_path):
    import tools.delegate_tool_dispatch as dispatch

    captured = {}
    monkeypatch.setattr(dispatch, "_resolve_async_wake_sid", lambda sid, history: "origin-session")
    monkeypatch.setattr(dispatch, "_resolve_async_session_key", lambda parent, ui: ("origin-key", "origin-ui"))
    monkeypatch.setattr("tools.delegate_tool_external_profile.validate_external_profile",
                        lambda name: ("expertkodare", tmp_path / "profile"))
    monkeypatch.setattr("tools.delegate_tool_external_profile.external_profile_is_authorized", lambda name: True)
    monkeypatch.setattr("tools.delegate_tool_external_profile.make_external_profile_runner",
                        lambda **kw: (lambda: {"status": "completed", "summary": "ok"}, lambda: None))
    monkeypatch.setattr("tools.delegate_tool._get_max_async_children", lambda: 3)

    def fake_dispatch(**kwargs):
        captured.update(kwargs)
        return {"status": "dispatched", "delegation_id": "deleg_profile"}

    monkeypatch.setattr("tools.async_delegation.dispatch_async_delegation", fake_dispatch)
    result = json.loads(dispatch.dispatch_external_profile_task(
        task={"goal": "review", "profile": "expertkodare"}, parent_agent=_parent(), context="ctx",
        role="leaf", origin=("wake", "ui", None, None, True)))

    assert result["delegation_id"] == "deleg_profile"
    assert result["profile"] == "expertkodare"
    assert captured["session_key"] == "origin-key"
    assert captured["origin_session_id"] == "origin-session"
    assert captured["origin_ui_session_id"] == "origin-ui"
    assert captured["parent_session_id"] == "parent-session"


def test_external_profile_dispatch_rejects_unauthorized_profile(monkeypatch, tmp_path):
    import tools.delegate_tool_dispatch as dispatch

    monkeypatch.setattr("tools.delegate_tool_external_profile.validate_external_profile",
                        lambda name: ("expertkodare", tmp_path / "profile"))
    monkeypatch.setattr("tools.delegate_tool_external_profile.external_profile_is_authorized", lambda name: False)

    with pytest.raises(ValueError, match="not authorized"):
        dispatch.dispatch_external_profile_task(
            task={"goal": "review", "profile": "expertkodare"}, parent_agent=_parent(), context="ctx",
            role="leaf", origin=("wake", "ui", None, None, True))


def test_external_runner_pins_profile_cwd_and_redacts_output(monkeypatch, tmp_path):
    import tools.delegate_tool_external_profile as external

    captured = {}

    class Proc:
        pid = 123
        returncode = 0
        def communicate(self):
            return "token=super-secret\nDONE", None
        def poll(self):
            return None

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return Proc()

    stage = tmp_path / "leaf-profile"
    profile_stage = stage / "profiles" / "expertkodare"
    profile_stage.mkdir(parents=True)
    (profile_stage / "auth.json").write_text("{}")
    (profile_stage / ".env").write_text("PROVIDER_KEY=value\n")
    monkeypatch.setattr(external, "stage_profile_state", lambda _profile, _home: stage)
    monkeypatch.setattr(external, "_docker_client_env", lambda: {"PATH": "/usr/bin"})
    monkeypatch.setattr(external.shutil, "rmtree", lambda path, ignore_errors: captured.setdefault("removed", path))
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("agent.redact.redact_terminal_output", lambda value, command, force=False: value.replace("super-secret", "***"))

    runner, _interrupt = external.make_external_profile_runner(
        profile="expertkodare", profile_home=Path("/profiles/expertkodare"), goal="goal", context="ctx", cwd=str(tmp_path))
    result = runner()

    assert captured["cwd"] == str(tmp_path)
    assert captured["argv"][:3] == ["docker", "run", "--rm"]
    assert "--read-only" in captured["argv"]
    assert any(value.endswith("dst=/workspace,readonly") for value in captured["argv"])
    assert "--cap-drop" in captured["argv"] and "ALL" in captured["argv"]
    assert "--security-opt" in captured["argv"] and "no-new-privileges" in captured["argv"]
    assert not any("docker.sock" in value for value in captured["argv"])
    assert "--profile" in captured["argv"] and "expertkodare" in captured["argv"]
    assert "/workspace" in captured["argv"]
    assert "HERMES_DELEGATED_CHILD_CONTEXT=1" in captured["argv"]
    assert any("auth.json,readonly" in value for value in captured["argv"])
    assert any(".env,readonly" in value for value in captured["argv"])
    assert captured["env"] == {"PATH": "/usr/bin"}
    assert captured["removed"] == stage
    assert result["status"] == "completed"
    assert "super-secret" not in result["summary"]


def test_external_runner_interrupted_before_launch_never_starts_process(monkeypatch, tmp_path):
    import tools.delegate_tool_external_profile as external

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: pytest.fail("Popen must not run after interrupt"))

    runner, interrupt = external.make_external_profile_runner(
        profile="expertkodare", profile_home=Path("/profiles/expertkodare"), goal="goal", context=None, cwd=str(tmp_path))
    interrupt()
    assert runner()["status"] == "interrupted"


def test_external_runner_interrupt_after_launch_terminates_process_group(monkeypatch, tmp_path):
    import tools.delegate_tool_external_profile as external

    started = threading.Event()
    release = threading.Event()
    killed = []

    class Proc:
        pid = 123
        returncode = -15
        def poll(self):
            return None
        def communicate(self):
            started.set()
            assert release.wait(timeout=2)
            return "stopped", None

    stage = tmp_path / "leaf-profile"
    stage.mkdir()
    monkeypatch.setattr(external, "stage_profile_state", lambda _profile, _home: stage)
    monkeypatch.setattr(external, "_docker_client_env", lambda: {"PATH": "/usr/bin"})
    monkeypatch.setattr(external.shutil, "rmtree", lambda *args, **kwargs: None)
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: Proc())
    monkeypatch.setattr("os.killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr("agent.redact.redact_terminal_output", lambda value, command, force=False: value)

    runner, interrupt = external.make_external_profile_runner(
        profile="expertkodare", profile_home=Path("/profiles/expertkodare"), goal="goal", context=None, cwd=str(tmp_path))
    result_box = {}
    thread = threading.Thread(target=lambda: result_box.setdefault("result", runner()))
    thread.start()
    assert started.wait(timeout=2)
    interrupt()
    release.set()
    thread.join(timeout=2)

    assert killed and killed[0][0] == 123
    assert result_box["result"]["status"] == "interrupted"


def test_leaf_stage_is_allowlisted_and_disposable(monkeypatch, tmp_path):
    import tools.delegate_tool_external_profile as external

    source = tmp_path / "profile"
    source.mkdir()
    (source / "config.yaml").write_text("agent: {}")
    (source / "auth.json").write_text("{}")
    (source / "state.db").write_text("must-not-copy")
    (source / "skills").mkdir()
    (source / "skills" / "leaf.md").write_text("allowed")
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(external, "_leaf_stage_dir", lambda: stage)

    assert external.stage_profile_state("expertkodare", source) == stage
    target = stage / "profiles" / "expertkodare"
    assert (target / "config.yaml").exists()
    assert (target / "auth.json").exists()
    assert (target / "skills" / "leaf.md").exists()
    assert not (target / "state.db").exists()


def test_external_profile_authorization_fails_closed(monkeypatch):
    import tools.delegate_tool_external_profile as external

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"delegation": {}})
    assert not external.external_profile_is_authorized("expertkodare")
    monkeypatch.setattr("hermes_cli.config.load_config",
                        lambda: {"delegation": {"external_profile_allowlist": ["expertkodare"]}})
    assert external.external_profile_is_authorized("expertkodare")


def test_leaf_stage_is_removed_after_copy_failure(monkeypatch, tmp_path):
    import tools.delegate_tool_external_profile as external

    source = tmp_path / "profile"
    source.mkdir()
    (source / "skills").mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(external, "_leaf_stage_dir", lambda: stage)
    monkeypatch.setattr(external.shutil, "copytree", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("copy failed")))

    with pytest.raises(OSError, match="copy failed"):
        external.stage_profile_state("expertkodare", source)
    assert not stage.exists()


def test_external_profile_child_cannot_spawn_delegate_task(monkeypatch):
    import tools.delegate_tool as delegate

    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    result = delegate.delegate_task(goal="must not spawn", parent_agent=_parent())
    assert "cannot spawn delegate_task children" in result


def test_immutable_docker_leaf_cannot_spawn_even_without_env_marker(monkeypatch):
    import tools.delegate_tool as delegate

    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setattr("agent.delegation_context.is_external_profile_leaf_runtime", lambda: True)
    result = delegate.delegate_task(goal="must not spawn", parent_agent=_parent())
    assert "Docker leaf runtimes cannot spawn" in result
