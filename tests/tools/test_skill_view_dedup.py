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
        r1 = json.loads(_skill_view_with_bump(args, task_id=None))
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
    """The dedup cache is process-global, so its entries must be attributed to the
    session that served them: a different session_id drops only the foreign entries
    of the task's bucket (other sessions' buckets are left alone), session_id of
    None/"" is a no-op, and the fork path (task_id=None) stays out of the cache."""

    @pytest.fixture(autouse=True)
    def _clean_session_state(self):
        from tools import skills_tool_dedup
        skills_tool_dedup._skill_view_tracker.clear()
        yield
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

    def test_interleaved_A_B(self, skills_home):
        # The review's scenario 1:1 — two live sessions in one process (gateway:
        # task_id == session_id, so each session has its own bucket): A runs 4 turns,
        # B interleaved, one view per turn, deterministic (no threads). B's activity
        # must NOT wipe A's bucket — A's cross-turn dedup survives. The old
        # process-global clear-on-session-change wiped the whole table whenever the
        # other session was live (0 deduped, 8 full); this matches the review's
        # "alternative" row: 6 deduped, 2 full.
        a_results, b_results = [], []
        for turn in range(4):
            a = self._view("demo-dedup-skill", task="task-A", session_id="sess-A")
            a_results.append(a.get("dedup") is True)
            b = self._view("demo-dedup-skill", task="task-B", session_id="sess-B")
            b_results.append(b.get("dedup") is True)
        # A: full, dedup, dedup, dedup; B: full, dedup, dedup, dedup. 6 deduped, 2 full.
        assert a_results == [False, True, True, True]
        assert b_results == [False, True, True, True]

    def test_shared_task_id_sessions_alternate(self, skills_home):
        # A fresh session reusing the SAME task_id drops only the foreign entries of
        # that bucket: A's repeat dedups; B's first view is full (A's entry is
        # dropped); A's next view is full (B's entry is dropped); A's repeat dedups.
        self._view("demo-dedup-skill", session_id="sess-A")
        r = self._view("demo-dedup-skill", session_id="sess-A")
        assert r.get("dedup") is True
        r2 = self._view("demo-dedup-skill", session_id="sess-B")
        assert "Step one" in r2.get("content", "")
        assert r2.get("dedup") is None
        r3 = self._view("demo-dedup-skill", session_id="sess-A")
        assert "Step one" in r3.get("content", "")
        assert r3.get("dedup") is None
        r4 = self._view("demo-dedup-skill", session_id="sess-A")
        assert r4.get("dedup") is True

    def test_session_id_flip_flop(self, skills_home):
        # "" is not a session id — the tool layer passes ``agent.session_id or ""``.
        # "" -> "real-id" -> "" must not clear the cache on the "" flip.
        r1 = self._view("demo-dedup-skill", session_id="")
        assert "Step one" in r1.get("content", "")  # first view: full
        r2 = self._view("demo-dedup-skill", session_id="real-id")
        # A real session taking over the bucket: the ""-written entry is foreign to
        # it (a fresh session must not get a stale stub) -> full content.
        assert "Step one" in r2.get("content", "")
        assert r2.get("dedup") is None
        r3 = self._view("demo-dedup-skill", session_id="")
        # The "" flip back must NOT clear the cache: the real session's entry is
        # still there, so the unattributed view dedups against it.
        assert r3.get("dedup") is True
        assert r3.get("content_returned") is False
        from tools import skills_tool_dedup
        # The cache held state through the flip: the entry (written by "real-id")
        # survived the "" flip.
        (entry,) = skills_tool_dedup._skill_view_tracker["t-svd"].values()
        assert entry[3] == "real-id"
