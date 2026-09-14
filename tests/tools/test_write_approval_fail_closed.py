"""Step 17 regression coverage for fail-closed write approval."""

import builtins
import json
from unittest.mock import patch

import pytest


def _deny_write_approval_import(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tools" and "write_approval" in (fromlist or ()):
            raise ImportError("disposable Step 17 import failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.mark.parametrize(
    "config_value",
    [
        {},
        {"memory": {"write_approval": None}},
        {"memory": {"write_approval": {"invalid": True}}},
    ],
)
def test_missing_or_invalid_config_requires_memory_approval(config_value, monkeypatch):
    from hermes_cli import config
    from tools import write_approval as wa

    monkeypatch.setattr(config, "load_config", lambda: config_value)
    assert wa.write_approval_enabled(wa.MEMORY) is True
    decision = wa.evaluate_gate(wa.MEMORY)
    assert decision.allow is False
    assert decision.stage is True
    assert decision.blocked is False


def test_raising_config_loader_requires_memory_and_skill_approval(monkeypatch):
    from hermes_cli import config
    from tools import write_approval as wa

    def boom():
        raise RuntimeError("disposable Step 17 config failure")

    monkeypatch.setattr(config, "load_config", boom)
    for subsystem in (wa.MEMORY, wa.SKILLS):
        assert wa.write_approval_enabled(subsystem) is True
        decision = wa.evaluate_gate(subsystem)
        assert decision.allow is False
        assert decision.stage is True


def test_memory_single_write_is_refused_when_gate_import_fails(
    isolated_home, monkeypatch
):
    from tools.memory_tool import MemoryStore, memory_tool

    store = MemoryStore()
    store.load_from_disk()
    _deny_write_approval_import(monkeypatch)
    result = json.loads(
        memory_tool("add", "memory", "step17-single-canary", store=store)
    )
    assert result["success"] is False
    assert "approval gate" in result["error"]
    assert store.memory_entries == []
    assert not (isolated_home / "memories" / "MEMORY.md").exists()


def test_memory_batch_write_is_refused_when_gate_import_fails(
    isolated_home, monkeypatch
):
    from tools.memory_tool import MemoryStore, memory_tool

    store = MemoryStore()
    store.load_from_disk()
    _deny_write_approval_import(monkeypatch)
    result = json.loads(
        memory_tool(
            target="memory",
            operations=[{"action": "add", "content": "step17-batch-canary"}],
            store=store,
        )
    )
    assert result["success"] is False
    assert "approval gate" in result["error"]
    assert store.memory_entries == []
    assert not (isolated_home / "memories" / "MEMORY.md").exists()


def test_skill_create_is_refused_when_gate_import_fails(
    isolated_home, tmp_path, monkeypatch
):
    from tools import skill_manager_tool as sm

    skills = tmp_path / "isolated-skills"
    content = (
        "---\n"
        "name: step17-import-fail\n"
        "description: Disposable Step 17 import-failure canary.\n"
        "---\n\n"
        "# Canary\n"
    )
    _deny_write_approval_import(monkeypatch)
    with (
        patch.object(sm, "SKILLS_DIR", skills),
        patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills]),
    ):
        result = json.loads(
            sm.skill_manage(action="create", name="step17-import-fail", content=content)
        )
    assert result["success"] is False
    assert "approval gate" in result["error"]
    assert not (skills / "step17-import-fail" / "SKILL.md").exists()
    assert not (isolated_home / "pending" / "skills").exists()


def test_stage_write_failure_is_reported_and_caller_does_not_claim_staged(
    isolated_home, tmp_path, monkeypatch
):
    from hermes_cli import config
    from tools import write_approval as wa
    from tools.memory_tool import MemoryStore, memory_tool

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"memory": {"write_approval": True}},
    )
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block mkdir", encoding="utf-8")
    impossible = blocker / "pending"
    monkeypatch.setattr(wa, "_pending_dir", lambda subsystem: impossible)

    store = MemoryStore()
    store.load_from_disk()
    result = json.loads(memory_tool("add", "memory", "must-stage", store=store))

    assert result["success"] is False
    assert "could not be persisted" in result["error"]
    assert store.memory_entries == []
    assert not impossible.exists()


@pytest.mark.parametrize("failure_point", ["mkdir", "write", "replace"])
def test_direct_stage_write_failures_raise(tmp_path, monkeypatch, failure_point):
    from tools import write_approval as wa

    pending = tmp_path / "pending"
    if failure_point == "mkdir":
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("block mkdir", encoding="utf-8")
        pending = blocker / "pending"
    elif failure_point == "write":
        # stage_write persists via utils.atomic_json_write -> utils._atomic_write, whose
        # write(f) callback is utils._dump_json; failing that exercises the real temp-file
        # creation and on-failure cleanup path inside _atomic_write, not just a raise at the
        # call site.
        import utils as _utils_mod

        def fail_dump_json(*_args, **_kwargs):
            raise OSError("disposable write failure")

        monkeypatch.setattr(_utils_mod, "_dump_json", fail_dump_json)
    else:
        # The replace step is utils.atomic_replace, called as a bare name inside
        # utils._atomic_write (same module, so patching the module attribute intercepts it).
        import utils as _utils_mod

        def fail_replace(*_args, **_kwargs):
            raise OSError("disposable replace failure")

        monkeypatch.setattr(_utils_mod, "atomic_replace", fail_replace)

    monkeypatch.setattr(wa, "_pending_dir", lambda subsystem: pending)
    with pytest.raises(wa.PendingWriteError):
        wa.stage_write(
            wa.SKILLS,
            {"action": "create", "name": "step17"},
            summary="disposable",
            origin="foreground",
        )
    assert not list(tmp_path.rglob("*.json"))
    assert not list(tmp_path.rglob("*.json.tmp"))


@pytest.mark.parametrize("caller", ["single", "batch", "skill"])
def test_all_callers_refuse_when_pending_persistence_fails(
    caller, isolated_home, tmp_path, monkeypatch
):
    from hermes_cli import config
    from tools import write_approval as wa

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "memory": {"write_approval": True},
            "skills": {"write_approval": True},
        },
    )
    monkeypatch.setattr(
        wa,
        "stage_write",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            wa.PendingWriteError("disposable persistence failure")
        ),
    )

    if caller in {"single", "batch"}:
        from tools.memory_tool import MemoryStore, memory_tool

        store = MemoryStore()
        store.load_from_disk()
        if caller == "single":
            raw = memory_tool("add", "memory", "must-not-land", store=store)
        else:
            raw = memory_tool(
                target="memory",
                operations=[{"action": "add", "content": "must-not-land"}],
                store=store,
            )
        result = json.loads(raw)
        assert result["success"] is False
        assert "could not be persisted" in result["error"]
        assert store.memory_entries == []
        assert not (isolated_home / "memories" / "MEMORY.md").exists()
    else:
        from tools import skill_manager_tool as sm

        skills = tmp_path / "isolated-skills"
        content = (
            "---\nname: step17-stage-fail\n"
            "description: Disposable persistence failure canary.\n---\n\n# Canary\n"
        )
        with (
            patch.object(sm, "SKILLS_DIR", skills),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills]),
        ):
            result = json.loads(
                sm.skill_manage(
                    action="create", name="step17-stage-fail", content=content
                )
            )
        assert result["success"] is False
        assert "could not be persisted" in result["error"]
        assert not (skills / "step17-stage-fail" / "SKILL.md").exists()


def test_successful_skill_stage_and_approved_replay(
    isolated_home, tmp_path, monkeypatch
):
    from hermes_cli import config
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import skill_manager_tool as sm
    from tools import write_approval as wa

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"skills": {"write_approval": True}},
    )
    skills = tmp_path / "isolated-skills"
    content = (
        "---\nname: step17-approved\n"
        "description: Disposable approved replay canary.\n---\n\n# Canary\n"
    )
    with (
        patch.object(sm, "SKILLS_DIR", skills),
        patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills]),
    ):
        staged = json.loads(
            sm.skill_manage(action="create", name="step17-approved", content=content)
        )
        assert staged["success"] is True
        assert staged["staged"] is True
        assert not (skills / "step17-approved" / "SKILL.md").exists()
        pending = wa.get_pending(wa.SKILLS, staged["pending_id"])
        assert pending is not None

        response = handle_pending_subcommand(
            wa.SKILLS, ["approve", staged["pending_id"]]
        )
        assert "Approved 1" in response
        assert (skills / "step17-approved" / "SKILL.md").read_text(
            encoding="utf-8"
        ) == content
        assert wa.get_pending(wa.SKILLS, staged["pending_id"]) is None


def test_explicit_skill_opt_out_still_allows_write(
    isolated_home, tmp_path, monkeypatch
):
    from hermes_cli import config
    from tools import skill_manager_tool as sm

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"skills": {"write_approval": False}},
    )
    skills = tmp_path / "isolated-skills"
    content = (
        "---\nname: step17-explicit-opt-out\n"
        "description: Disposable explicit opt-out canary.\n---\n\n# Canary\n"
    )
    with (
        patch.object(sm, "SKILLS_DIR", skills),
        patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills]),
    ):
        result = json.loads(
            sm.skill_manage(
                action="create", name="step17-explicit-opt-out", content=content
            )
        )
    assert result["success"] is True
    assert result.get("staged") is None
    assert (skills / "step17-explicit-opt-out" / "SKILL.md").exists()
    assert not (isolated_home / "pending" / "skills").exists()
