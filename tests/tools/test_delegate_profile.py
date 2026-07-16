"""Focused coverage for profile-persona delegation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.delegate_tool as delegate_module
from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_child_system_prompt,
    _load_profile_soul,
    delegate_task,
)


@pytest.fixture()
def profile_home(tmp_path, monkeypatch):
    """Keep named-profile discovery and HERMES_HOME inside the test temp dir."""
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    def create(name: str, soul: str | None = None) -> Path:
        profile_dir = default_home / "profiles" / name
        profile_dir.mkdir(parents=True)
        if soul is not None:
            (profile_dir / "SOUL.md").write_text(soul, encoding="utf-8")
        return profile_dir

    return create


def _parent() -> SimpleNamespace:
    return SimpleNamespace(
        _delegate_depth=0,
        _interrupt_requested=False,
        _memory_manager=None,
        _active_children=[],
        _active_children_lock=None,
        _current_turn_id="",
        _print_fn=None,
        session_id="parent-session",
        session_estimated_cost_usd=0.0,
        session_cost_source="none",
        session_cost_status="unknown",
        tool_progress_callback=None,
        thinking_callback=None,
    )


@pytest.fixture()
def isolated_delegate(monkeypatch):
    """Replace child execution while preserving delegate_task's routing logic."""
    built: list[dict] = []

    def fake_build_child_agent(**kwargs):
        built.append(kwargs)
        return SimpleNamespace(
            session_id=f"child-{kwargs['task_index']}",
            _delegate_role=kwargs["role"],
            _delegate_saved_tool_names=[],
        )

    def fake_run_single_child(task_index, goal, child, parent_agent):
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": goal,
            "api_calls": 0,
            "duration_seconds": 0.0,
        }

    monkeypatch.setattr(delegate_module, "_build_child_agent", fake_build_child_agent)
    monkeypatch.setattr(delegate_module, "_run_single_child", fake_run_single_child)
    monkeypatch.setattr(
        delegate_module,
        "_resolve_delegation_credentials",
        lambda _cfg, _parent: {
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "model": None,
        },
    )
    monkeypatch.setattr(
        delegate_module,
        "_load_config",
        lambda: {"max_iterations": 50, "max_concurrent_children": 3},
    )
    return built


def test_load_profile_soul_uses_named_profile_identity(profile_home):
    profile_home("reviewer", "\nYou are the independent reviewer.\n")

    assert _load_profile_soul(" Reviewer ") == "You are the independent reviewer."


def test_load_profile_soul_does_not_switch_or_parse_profile_runtime(profile_home):
    profile_dir = profile_home("reviewer", "You are the independent reviewer.")
    (profile_dir / "config.yaml").write_text("not: [valid", encoding="utf-8")
    (profile_dir / ".env").write_text("PROFILE_ONLY_TOKEN=unused", encoding="utf-8")
    parent_home = os.environ["HERMES_HOME"]

    assert _load_profile_soul("reviewer") == "You are the independent reviewer."
    assert os.environ["HERMES_HOME"] == parent_home


@pytest.mark.parametrize(
    ("name", "soul", "message"),
    [
        ("missing", None, "Unknown profile"),
        ("no-soul", None, "has no SOUL.md"),
        ("empty", "  \n", "empty SOUL.md"),
    ],
)
def test_load_profile_soul_rejects_unusable_profile(profile_home, name, soul, message):
    if name != "missing":
        profile_home(name, soul)

    with pytest.raises(ValueError, match=message):
        _load_profile_soul(name)


def test_load_profile_soul_rejects_default_profile(profile_home):
    with pytest.raises(ValueError, match="non-default"):
        _load_profile_soul("default")


def test_profile_soul_is_first_child_prompt_segment():
    prompt = _build_child_system_prompt(
        "Review the patch",
        profile_soul="You are the independent reviewer.",
    )

    assert prompt.startswith("You are the independent reviewer.\n")
    assert "You are a focused subagent" in prompt
    assert "YOUR TASK:\nReview the patch" in prompt


def test_schema_exposes_identity_only_profile_top_level_and_per_task():
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]

    assert "profile" in props
    assert "SOUL.md identity prompt" in props["profile"]["description"]
    task_props = props["tasks"]["items"]["properties"]
    assert "profile" in task_props
    assert "SOUL.md identity prompt" in task_props["profile"]["description"]


def test_no_profile_keeps_delegate_identity_unset(isolated_delegate, monkeypatch):
    load_soul = pytest.fail
    monkeypatch.setattr(delegate_module, "_load_profile_soul", load_soul)

    result = json.loads(delegate_task(goal="plain task", parent_agent=_parent()))

    assert result["results"][0]["status"] == "completed"
    assert isolated_delegate[0]["profile_soul"] is None


def test_top_level_profile_loads_soul_for_child(isolated_delegate, monkeypatch):
    monkeypatch.setattr(
        delegate_module,
        "_load_profile_soul",
        lambda name: f"identity:{name}",
    )

    result = json.loads(
        delegate_task(
            goal="implement",
            profile="implementer",
            parent_agent=_parent(),
        )
    )

    assert result["results"][0]["status"] == "completed"
    assert isolated_delegate[0]["profile_soul"] == "identity:implementer"


def test_per_task_profile_overrides_top_level(isolated_delegate, monkeypatch):
    loaded: list[str] = []

    def fake_load(name):
        loaded.append(name)
        return f"identity:{name}"

    monkeypatch.setattr(delegate_module, "_load_profile_soul", fake_load)

    result = json.loads(
        delegate_task(
            tasks=[
                {"goal": "implement"},
                {"goal": "review", "profile": "reviewer"},
            ],
            profile="implementer",
            parent_agent=_parent(),
        )
    )

    assert [entry["status"] for entry in result["results"]] == [
        "completed",
        "completed",
    ]
    assert loaded == ["implementer", "reviewer"]
    assert [call["profile_soul"] for call in isolated_delegate] == [
        "identity:implementer",
        "identity:reviewer",
    ]


def test_invalid_batch_profile_builds_no_children(isolated_delegate, monkeypatch):
    def fake_load(name):
        if name == "missing":
            raise ValueError("Unknown profile 'missing'")
        return f"identity:{name}"

    monkeypatch.setattr(delegate_module, "_load_profile_soul", fake_load)

    result = json.loads(
        delegate_task(
            tasks=[
                {"goal": "implement", "profile": "implementer"},
                {"goal": "review", "profile": "missing"},
            ],
            parent_agent=_parent(),
        )
    )

    assert "Task 1 profile error" in result["error"]
    assert isolated_delegate == []


def test_live_dispatch_forwards_profile(monkeypatch):
    import run_agent

    captured = {}

    def fake_delegate_task(**kwargs):
        captured.update(kwargs)
        return "{}"

    monkeypatch.setattr(delegate_module, "delegate_task", fake_delegate_task)

    run_agent.AIAgent._dispatch_delegate_task(
        _parent(),
        {"goal": "review", "profile": "reviewer"},
    )

    assert captured["profile"] == "reviewer"


def test_registry_fallback_forwards_profile(monkeypatch):
    from tools.registry import registry

    captured = {}

    def fake_delegate_task(**kwargs):
        captured.update(kwargs)
        return "{}"

    monkeypatch.setattr(delegate_module, "delegate_task", fake_delegate_task)

    entry = registry.get_entry("delegate_task")
    assert entry is not None
    entry.handler(
        {"goal": "review", "profile": "reviewer"},
        parent_agent=_parent(),
    )

    assert captured["profile"] == "reviewer"
