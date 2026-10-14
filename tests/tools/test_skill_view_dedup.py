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


class TestSkillViewDedupSessionScope:
    """The dedup cache is process-global, so it must be scoped to the active
    session: a new session_id drops the cache, session_id=None is a no-op,
    and the fork path (task_id=None) stays out of the cache."""

    @pytest.fixture(autouse=True)
    def _clean_session_state(self):
        from tools import skills_tool_dedup
        skills_tool_dedup._active_session_id = None
        skills_tool_dedup._skill_view_tracker.clear()
        yield
        skills_tool_dedup._active_session_id = None
        skills_tool_dedup._skill_view_tracker.clear()

    def _view(self, name, file_path=None, task="t-svd", session_id=None):
        args = {"name": name}
        if file_path:
            args["file_path"] = file_path
        return json.loads(_skill_view_with_bump(args, task_id=task, session_id=session_id))

    def test_session_change_drops_cache(self, skills_home):
        # Session A views the skill; a repeat within A dedups.
        self._view("demo-dedup-skill", session_id="sess-A")
        r = self._view("demo-dedup-skill", session_id="sess-A")
        assert r.get("dedup") is True
        # A fresh session reusing the SAME task_id must NOT inherit A's entry:
        # it never loaded the skill into its own context, so it gets full content.
        r2 = self._view("demo-dedup-skill", session_id="sess-B")
        assert "Step one" in r2.get("content", "")
        assert r2.get("dedup") is None

    def test_session_id_none_is_noop(self, skills_home):
        # Views without a session_id never drop the cache and still dedup
        # (the fork / pre-session path keeps working).
        self._view("demo-dedup-skill", session_id=None)
        r = self._view("demo-dedup-skill", session_id=None)
        assert r.get("dedup") is True

    def test_none_session_does_not_drop_populated_cache(self, skills_home):
        # A None session_id must not clobber a cache populated by a real session.
        self._view("demo-dedup-skill", session_id="sess-A")
        r = self._view("demo-dedup-skill", session_id=None)
        assert r.get("dedup") is True

    def test_fork_path_stays_out_of_cache(self, skills_home):
        # task_id=None (background-review fork) is never recorded and never dedups,
        # regardless of session_id.
        r1 = self._view("demo-dedup-skill", task=None, session_id="sess-A")
        assert "Step one" in r1.get("content", "")
        r2 = self._view("demo-dedup-skill", task=None, session_id="sess-A")
        assert "Step one" in r2.get("content", "")
        assert r2.get("dedup") is None
