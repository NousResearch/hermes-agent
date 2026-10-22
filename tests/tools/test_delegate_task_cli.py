"""A task-level ``cli`` runs that child on a different ACP agent CLI.

``delegate_task`` children normally inherit the parent's transport, so a Hermes parent's children
all run on Hermes. ``cli`` lets one task pin a different ACP agent (``codex acp``, ``hermes acp``,
…) without changing the rest of the batch, and an unknown command must fail the spawn loudly
instead of silently falling back to the parent's transport.
"""
import types

import pytest

import tools.delegate_tool as dt


@pytest.fixture
def built(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    calls = []
    monkeypatch.setattr(
        dt, "_run_single_child",
        lambda task_index, goal, child=None, parent_agent=None, **kw: {
            "task_index": task_index, "status": "completed", "summary": "ok",
            "error": None, "api_calls": 1, "duration_seconds": 1,
        },
    )

    def fake_build(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(tool_progress_callback=None)

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", fake_build)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: {
        "model": "m", "provider": "openrouter", "base_url": "https://x/v1",
        "api_key": "k", "api_mode": "chat_completions"})
    return calls


def _parent():
    return types.SimpleNamespace(
        session_id="root", model="m", tool_progress_callback=None, _delegate_spinner=None,
        _safe_print=lambda line: None,
    )


def test_cli_pins_that_child_transport_only(built):
    out = dt.delegate_task(
        tasks=[
            {"goal": "first task runs on the parent transport"},
            {"goal": "second task runs on codex", "cli": "codex acp"},
        ],
        parent_agent=_parent(),
    )
    assert "error" not in str(out)[:20], out
    assert built[0]["override_acp_command"] is None
    assert built[1]["override_acp_command"] == "codex"
    assert built[1]["override_acp_args"] == ["acp"]


def test_unknown_cli_fails_the_spawn(built, monkeypatch):
    def refuse(**kwargs):
        if kwargs.get("override_acp_command"):
            raise ValueError("Pinned delegation command 'no-such-acp' was not found on PATH.")
        return types.SimpleNamespace(tool_progress_callback=None)

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", refuse)
    out = dt.delegate_task(
        tasks=[{"goal": "task on a missing cli", "cli": "no-such-acp --acp"}],
        parent_agent=_parent(),
    )
    assert "not found on PATH" in out
    assert "completed" not in out


@pytest.mark.parametrize("bad", ["", "   "])
def test_blank_cli_is_rejected(built, bad):
    out = dt.delegate_task(
        tasks=[{"goal": "task with a blank cli", "cli": bad}],
        parent_agent=_parent(),
    )
    assert "must be a non-empty command string" in out
    assert built == []
