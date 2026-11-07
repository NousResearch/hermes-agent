"""Inline invocation content participates in skill_view's task-scoped dedup."""

import json
from types import SimpleNamespace

import pytest

from agent.skill_commands import (
    build_skill_invocation_message,
    build_stacked_skill_invocation_message,
)
from agent.turn_context import _stage_turn_user_message
from cron.scheduler_prompt import _build_job_prompt
from tools.registry import registry
from tools.skills_tool_dedup import reset_skill_view_dedup


NAME = "inline-dedup-demo"
BODY = "Follow the inline demo procedure completely."
TASK = "cron:demo:execution-one"


@pytest.fixture
def skill(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    directory = home / "skills" / NAME
    directory.mkdir(parents=True)
    md = directory / "SKILL.md"
    md.write_text(f"---\nname: {NAME}\ndescription: Inline dedup demo.\n---\n\n{BODY}\n", encoding="utf-8")
    (directory / "references").mkdir()
    (directory / "references" / "guide.md").write_text("Detailed demo reference.", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_skill_view_dedup()
    yield md
    reset_skill_view_dedup()


def _inline(kind):
    # Builders run before the final tool task ID exists. In messaging, this is
    # a route key; cron's builder similarly only knows the job ID on the base.
    if kind == "cron":
        return _build_job_prompt({"id": "demo", "skills": [NAME], "prompt": "Run the demo."})
    if kind == "stacked":
        return build_stacked_skill_invocation_message([f"/{NAME}"], task_id="route-key")[0]
    return build_skill_invocation_message(f"/{NAME}", task_id="route-key")


def _stage(message, task=TASK):
    agent = SimpleNamespace(_current_task_id=task)
    row, _ = _stage_turn_user_message(agent, message, None, None, None, None, None)
    assert row["content"] == message


def _view(task=TASK, file_path=None):
    args = {"name": NAME}
    if file_path:
        args["file_path"] = file_path
    return json.loads(registry.dispatch("skill_view", args, task_id=task))


@pytest.mark.parametrize("kind", ["cron", "slash", "stacked"])
def test_inline_then_view_returns_truthful_stub(skill, kind):
    message = _inline(kind)
    assert BODY in message
    # Gateway context may surround the invocation without changing its bytes.
    _stage("Current turn:\n" + message)
    for _ in range(2):
        result = _view()
        assert result.get("dedup") is True
        assert result["content_returned"] is False
        assert "content" not in result
        assert "invocation message" in result["message"]
        assert "earlier skill_view result" not in result["message"]


@pytest.mark.parametrize("kind", ["cron", "slash"])
@pytest.mark.parametrize("change", ["rewrite", "other_task", "reset", "reference"])
def test_inline_exceptions_return_full_content(skill, kind, change):
    _stage(_inline(kind))
    task, file_path, expected = TASK, None, BODY
    if change == "rewrite":
        expected = "Follow the NEWEST demo procedure completely."
        skill.write_text(skill.read_text(encoding="utf-8").replace(BODY, expected), encoding="utf-8")
    elif change == "other_task":
        task = "cron:demo:execution-two"
    elif change == "reset":
        reset_skill_view_dedup(TASK)
    else:
        file_path, expected = "references/guide.md", "Detailed demo reference."
    result = _view(task, file_path)
    assert expected in result.get("content", "")
    assert not result.get("dedup")


def test_unsubmitted_inline_does_not_mark_an_unrelated_turn(skill):
    assert BODY in _inline("slash")
    _stage("An unrelated request.")
    assert BODY in _view().get("content", "")


def test_background_review_returns_full_after_inline(skill):
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin

    _stage(_inline("slash"))
    token = set_current_write_origin("background_review")
    try:
        assert BODY in _view().get("content", "")
    finally:
        reset_current_write_origin(token)
    assert _view().get("dedup") is True


def test_setup_needed_returns_full_after_inline(skill, monkeypatch):
    import tools.skills_tool as st

    monkeypatch.setattr(st, "_skill_readiness", lambda *_: ({"setup_needed": True}, {}))
    _stage(_inline("slash"))
    for _ in range(2):
        result = _view()
        assert result["setup_needed"] is True
        assert BODY in result["content"]


def test_preprocessed_body_must_match_what_was_inlined(skill):
    skill.write_text(skill.read_text(encoding="utf-8") + "\n${HERMES_SESSION_ID}\n", encoding="utf-8")
    message = _inline("slash")
    assert "route-key" in message
    _stage(message)
    result = _view()
    assert TASK in result.get("content", "")
    assert not result.get("dedup")
