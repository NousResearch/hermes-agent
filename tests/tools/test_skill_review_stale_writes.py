"""Concurrent reviewers must preserve edits made after the content they actually read.

Real skill reads and mutations run in separate review contexts against temporary
profile files. Existing write-lock tests cover torn writes; these protect the
read-to-model-to-write gap and permit sequential patches from one review.
"""

import contextvars
import json
import threading

import pytest

from tools import skill_manager_tool, skills_tool
from tools.skill_manager_guards import _reset_background_review_read_marks
from tools.skill_provenance import set_current_write_origin


@pytest.fixture
def skill(tmp_path, monkeypatch):
    home = tmp_path / "home"
    root = home / "skills"
    folder = root / "workflow"
    folder.mkdir(parents=True)
    target = folder / "SKILL.md"
    target.write_text("---\nname: workflow\ndescription: Use for deployment.\n---\n\nFirst step.\nSecond step.\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", root)
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", root)
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [root])
    return target


def _review_context():
    context = contextvars.copy_context()
    context.run(set_current_write_origin, "background_review")
    context.run(_reset_background_review_read_marks)
    return context


def _write(context, **kwargs):
    return json.loads(context.run(skill_manager_tool.skill_manage, name="workflow", **kwargs))


@pytest.mark.parametrize("reader", ["skill_view", "read_file"])
def test_stale_review_rewrite_preserves_intervening_edit_and_can_retry(skill, reader):
    first, second = _review_context(), _review_context()
    if reader == "skill_view":
        read = lambda: skills_tool.skill_view("workflow")
    else:
        from tools.file_tools import read_file_tool
        read = lambda: read_file_tool(str(skill), task_id="stale-review")
    original = skill.read_text()
    first.run(read)
    second.run(skills_tool.skill_view, "workflow")
    changed = _write(second, action="patch", old_string="First step.", new_string="Validated first step.")
    assert changed["success"], changed
    refused = _write(first, action="edit", content=original.replace("Second step.", "Final step."))
    assert refused.get("_stale_review_read"), refused
    assert "Validated first step." in skill.read_text()
    first.run(read)
    retried = _write(first, action="patch", old_string="Second step.", new_string="Final step.")
    assert retried["success"], retried
    assert "Validated first step." in skill.read_text()
    assert "Final step." in skill.read_text()


def test_review_batch_accepts_its_own_updated_baseline(skill):
    context = _review_context()
    context.run(skills_tool.skill_view, "workflow")
    result = _write(context, action="patch", operations=[
        {"action": "patch", "old_string": "First step.", "new_string": "Checked first step."},
        {"action": "patch", "old_string": "Second step.", "new_string": "Checked second step."},
    ])
    assert result["success"], result
    assert "Checked first step." in skill.read_text()
    assert "Checked second step." in skill.read_text()


def test_skill_view_marks_bytes_returned_not_a_later_re_read(skill, monkeypatch):
    context = _review_context()
    original_reader = skills_tool._read_skill_text

    def raced_read(path):
        content = original_reader(path)
        path.write_text(content.replace("First step.", "Someone else's change."))
        return content

    monkeypatch.setattr(skills_tool, "_read_skill_text", raced_read)
    context.run(skills_tool.skill_view, "workflow")
    result = _write(context, action="patch", old_string="Second step.", new_string="Final step.")
    assert result.get("_stale_review_read"), result
    assert "Someone else's change." in skill.read_text()


def test_stale_supporting_file_replacement_is_refused(skill):
    target = skill.parent / "references" / "deploy.md"
    target.parent.mkdir()
    target.write_text("Original deployment notes.")
    context = _review_context()
    context.run(skills_tool.skill_view, "workflow", file_path="references/deploy.md")
    target.write_text("Updated deployment notes from another session.")
    result = _write(context, action="write_file", file_path="references/deploy.md", file_content="Stale rewrite.")
    assert result.get("_stale_review_read"), result
    assert target.read_text() == "Updated deployment notes from another session."


def test_cancelled_review_cannot_write_after_slow_provider_returns(skill):
    from tools.skill_manager_guards import background_review_write_scope
    context = _review_context()
    context.run(skills_tool.skill_view, "workflow")
    before = skill.read_text()
    cancelled = threading.Event()
    cancelled.set()

    def late_write():
        with background_review_write_scope(cancelled):
            return json.loads(skill_manager_tool.skill_manage(
                "patch", "workflow", old_string="First step.", new_string="Obsolete change."))

    result = context.run(late_write)
    assert result.get("_review_cancelled"), result
    assert skill.read_text() == before
