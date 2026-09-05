"""Regression tests for #32114 / P-0057: a skill_view reload must never return an
"unchanged" dedup stub (or a SKILL_PRUNED placeholder) once compression has demoted
the skill body out of the transcript. While the on-disk source still exists, reload
must fall back to a full re-read; a genuinely unchanged-and-still-present body
keeps the dedup stub.
"""

import json

import pytest

from tools.skills_tool import _skill_view_with_bump
from tools.skills_tool_dedup import (
    _is_ghosted_skill_view,
    _mark_ghosted_skill_views,
    reset_skill_view_dedup,
)


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    d = home / "skills" / "demo-dedup-skill"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: demo-dedup-skill\ndescription: Demo skill for ghost tests.\n---\n"
        "# Demo\n\nStep one: run the demo procedure fully.\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_skill_view_dedup()
    return home


def _view(name, task="t-ghost"):
    return json.loads(_skill_view_with_bump({"name": name}, task_id=task))


class TestGhostReload:
    def test_reload_after_ghost_returns_full_content(self, skills_home):
        assert "Step one" in _view("demo-dedup-skill")["content"]
        # Compression demoted the body out of the transcript (no boundary reset ran):
        _mark_ghosted_skill_views(["demo-dedup-skill"])
        r2 = _view("demo-dedup-skill")
        assert r2["success"] is True
        assert "Step one" in r2.get("content", "")
        assert r2.get("dedup") is None
        assert r2.get("content_returned") is not False
        assert "SKILL_PRUNED" not in json.dumps(r2)
        assert "unchanged" not in r2.get("message", "")

    def test_ghost_heals_across_tasks(self, skills_home):
        _view("demo-dedup-skill", task="task-A")
        _mark_ghosted_skill_views(["demo-dedup-skill"])
        r = _view("demo-dedup-skill", task="task-B")
        assert "Step one" in r.get("content", "")

    def test_fresh_full_view_re_arms_dedup(self, skills_home):
        assert "Step one" in _view("demo-dedup-skill")["content"]
        _mark_ghosted_skill_views(["demo-dedup-skill"])
        # Self-heal: full content again, and the fresh view re-records the cache.
        assert "Step one" in _view("demo-dedup-skill")["content"]
        # Body is present again and NOT ghosted: the ordinary stub returns.
        r = _view("demo-dedup-skill")
        assert r.get("dedup") is True

    def test_reset_clears_ghosts(self, skills_home):
        _view("demo-dedup-skill")
        _mark_ghosted_skill_views(["demo-dedup-skill"])
        reset_skill_view_dedup()
        assert _is_ghosted_skill_view("demo-dedup-skill") is False
        assert "Step one" in _view("demo-dedup-skill")["content"]

    def test_task_scoped_reset_keeps_ghosts(self, skills_home):
        # A sibling in-process task crossing a compaction boundary resets only its own
        # cache; ghost flags must survive (they force a harmless full re-read, while
        # dropping them would resurrect the "unchanged"-stub deadlock for tasks whose
        # bodies were already demoted out of their transcripts).
        _view("demo-dedup-skill")
        _mark_ghosted_skill_views(["demo-dedup-skill"])
        reset_skill_view_dedup(task_id="t-ghost")
        assert _is_ghosted_skill_view("demo-dedup-skill") is True
        # Self-heal still works after the task-scoped reset.
        assert "Step one" in _view("demo-dedup-skill")["content"]
        reset_skill_view_dedup()

    def test_ghost_name_variants_match(self):
        _mark_ghosted_skill_views(["demo-skill"])
        assert _is_ghosted_skill_view("demo-skill") is True
        assert _is_ghosted_skill_view("PLUGIN:demo-skill") is True
        assert _is_ghosted_skill_view("category/demo-skill") is True
        assert _is_ghosted_skill_view("other-skill") is False
        reset_skill_view_dedup()
