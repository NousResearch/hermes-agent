"""Tests for skill_view repeat-view dedup (unchanged-skill stub)."""

import json
import time

import pytest

from tools.skills_tool import (
    _skill_view_with_bump,
    reset_skill_view_dedup,
)

@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    d = skills / "demo-dedup-skill"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: demo-dedup-skill\ndescription: Demo skill for dedup tests.\n---\n"
        "# Demo\n\nStep one: run the demo procedure fully.\n",
        encoding="utf-8",
    )
    refs = d / "references"
    refs.mkdir()
    (refs / "guide.md").write_text("# Guide\n\nDetailed reference content here.\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_skill_view_dedup()
    from tools.skill_manager_guards import _reset_background_review_read_marks
    _reset_background_review_read_marks()
    return home

def _view(name, file_path=None, task="t-svd"):
    args = {"name": name}
    if file_path:
        args["file_path"] = file_path
    return json.loads(_skill_view_with_bump(args, task_id=task))

class TestSkillViewDedup:
    def test_first_view_returns_full_content(self, skills_home):
        r = _view("demo-dedup-skill")
        assert r["success"] is True
        assert "Step one" in r.get("content", "")

    def test_repeat_view_returns_stub(self, skills_home):
        _view("demo-dedup-skill")
        r2 = _view("demo-dedup-skill")
        assert r2["success"] is True
        assert r2.get("dedup") is True
        assert r2.get("content_returned") is False
        assert "content" not in r2

    def test_modified_skill_returns_full_content(self, skills_home):
        _view("demo-dedup-skill")
        md = skills_home / "skills" / "demo-dedup-skill" / "SKILL.md"
        time.sleep(0.01)
        md.write_text(md.read_text(encoding="utf-8") + "\nStep two: new instruction.\n", encoding="utf-8")
        r2 = _view("demo-dedup-skill")
        assert "Step two" in r2.get("content", "")
        assert r2.get("dedup") is None

    def test_linked_file_dedup_is_independent(self, skills_home):
        _view("demo-dedup-skill")
        # First view of a DIFFERENT file within the skill: full content.
        r = _view("demo-dedup-skill", file_path="references/guide.md")
        assert "Detailed reference" in r.get("content", "")
        # Repeat of that file: stub.
        r2 = _view("demo-dedup-skill", file_path="references/guide.md")
        assert r2.get("dedup") is True

    def test_different_tasks_do_not_share_cache(self, skills_home):
        _view("demo-dedup-skill", task="task-A")
        r = _view("demo-dedup-skill", task="task-B")
        assert "Step one" in r.get("content", "")

    def test_reset_returns_full_content(self, skills_home):
        _view("demo-dedup-skill")
        reset_skill_view_dedup("t-svd")
        r2 = _view("demo-dedup-skill")
        assert "Step one" in r2.get("content", "")

    def test_no_task_id_never_dedups(self, skills_home):
        args = {"name": "demo-dedup-skill"}
        json.loads(_skill_view_with_bump(args, task_id=None))
        r2 = json.loads(_skill_view_with_bump(args, task_id=None))
        assert "Step one" in r2.get("content", "")

    def test_background_review_skips_dedup_and_marks_read(self, skills_home):
        from tools.skill_provenance import (
            reset_current_write_origin,
            set_current_write_origin,
        )

        _view("demo-dedup-skill")

        token = set_current_write_origin("background_review")
        try:
            review = _view("demo-dedup-skill")
        finally:
            reset_current_write_origin(token)

        assert review["success"] is True
        assert "Step one" in review.get("content", "")
        assert review.get("dedup") is None
        assert review.get("content_returned") is None
        # The real read marks the file, so the fork's read-before-write guard now admits the patch.
        from tools.skill_manager_guards import _background_review_has_read
        assert _background_review_has_read(skills_home / "skills" / "demo-dedup-skill" / "SKILL.md")

    def test_background_review_does_not_pollute_foreground_cache(self, skills_home):
        from tools.skill_provenance import (
            reset_current_write_origin,
            set_current_write_origin,
        )

        token = set_current_write_origin("background_review")
        try:
            _view("demo-dedup-skill")
        finally:
            reset_current_write_origin(token)

        foreground = _view("demo-dedup-skill")
        assert "Step one" in foreground.get("content", "")

        repeat = _view("demo-dedup-skill")
        assert repeat.get("dedup") is True
        assert repeat.get("content_returned") is False


class TestResetKeepsSurvivingResults:
    """A transcript rewrite releases only the stubs whose full result it removed (#132177)."""

    def _transcript(self, *bodies):
        return [{"role": "tool", "tool_call_id": f"c{i}", "content": b} for i, b in enumerate(bodies)]

    def test_surviving_full_result_keeps_stub(self, skills_home):
        body = _skill_view_with_bump({"name": "demo-dedup-skill"}, task_id="t-svd")
        reset_skill_view_dedup("t-svd", surviving_messages=self._transcript(body))
        assert _view("demo-dedup-skill").get("dedup") is True

    def test_demoted_result_serves_content_again(self, skills_home):
        _view("demo-dedup-skill")
        marker = "[skill_view] name=demo-dedup-skill (2,000 chars)"
        reset_skill_view_dedup("t-svd", surviving_messages=self._transcript(marker))
        assert "Step one" in _view("demo-dedup-skill").get("content", "")

    def test_only_the_removed_file_is_released(self, skills_home):
        main = _skill_view_with_bump({"name": "demo-dedup-skill"}, task_id="t-svd")
        _view("demo-dedup-skill", file_path="references/guide.md")
        reset_skill_view_dedup("t-svd", surviving_messages=self._transcript(main))
        assert _view("demo-dedup-skill").get("dedup") is True
        assert "Detailed reference" in _view("demo-dedup-skill", file_path="references/guide.md").get("content", "")

    def test_stub_in_transcript_does_not_count_as_surviving(self, skills_home):
        _view("demo-dedup-skill")
        stub = _skill_view_with_bump({"name": "demo-dedup-skill"}, task_id="t-svd")
        reset_skill_view_dedup("t-svd", surviving_messages=self._transcript(stub))
        assert "Step one" in _view("demo-dedup-skill").get("content", "")
