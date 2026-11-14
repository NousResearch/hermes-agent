"""A child composed without project context must say so.

The defect these cover: ``_build_child_system_prompt`` appends the repo's context files when
discovery finds them and stays **silent** when it does not -- both when no workspace resolved and
when the workspace carries no context file. A child then works blind to ``AGENTS.md`` and the
transcript records nothing, so "the child followed the conventions" and "the child never saw them"
are indistinguishable afterwards.
"""
from __future__ import annotations

import logging

import pytest

import tools.delegate_tool_progress as mod


@pytest.fixture(autouse=True)
def _reset_warning_flag():
    mod._WARNED_MISSING_PROJECT_CONTEXT = False
    yield
    mod._WARNED_MISSING_PROJECT_CONTEXT = False


def _build(**kwargs) -> str:
    return mod._build_child_system_prompt("do the thing", **kwargs)


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_warns_when_no_workspace_resolved(caplog):
    caplog.set_level(logging.WARNING, logger="tools.delegate_tool")
    prompt = _build()
    msgs = _warnings(caplog)
    assert len(msgs) == 1, msgs
    assert "no workspace resolved" in msgs[0]
    assert "Project Context" not in prompt


def test_warns_when_workspace_has_no_context_files(caplog, tmp_path):
    caplog.set_level(logging.WARNING, logger="tools.delegate_tool")
    prompt = _build(workspace_path=str(tmp_path))
    msgs = _warnings(caplog)
    assert len(msgs) == 1, msgs
    assert str(tmp_path) in msgs[0]
    assert "Project Context" not in prompt


def test_quiet_when_context_files_load(caplog, tmp_path):
    caplog.set_level(logging.WARNING, logger="tools.delegate_tool")
    (tmp_path / "AGENTS.md").write_text("CONVENTION-SENTINEL-12345\n", encoding="utf-8")
    prompt = _build(workspace_path=str(tmp_path))
    assert _warnings(caplog) == []
    assert "CONVENTION-SENTINEL-12345" in prompt


def test_warns_once_per_process(caplog):
    caplog.set_level(logging.WARNING, logger="tools.delegate_tool")
    _build()
    _build()
    assert len(_warnings(caplog)) == 1


def test_warning_never_leaks_into_the_prompt(caplog):
    caplog.set_level(logging.WARNING, logger="tools.delegate_tool")
    prompt = _build()
    assert "delegated child has no project context" not in prompt
