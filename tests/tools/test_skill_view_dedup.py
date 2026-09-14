"""Tests for skill_view repeat-view dedup (unchanged-skill stub)."""

import json
import os
import time
from pathlib import Path

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
        "# Demo\n\nStep one: run the demo procedure fully.\n"
    )
    refs = d / "references"
    refs.mkdir()
    (refs / "guide.md").write_text("# Guide\n\nDetailed reference content here.\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_skill_view_dedup()
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
        assert "unchanged" in r2["message"]
        assert "content" not in r2

    def test_modified_skill_returns_full_content(self, skills_home):
        _view("demo-dedup-skill")
        md = skills_home / "skills" / "demo-dedup-skill" / "SKILL.md"
        time.sleep(0.01)
        md.write_text(md.read_text() + "\nStep two: new instruction.\n")
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

    def test_compression_hook_importable(self):
        # conversation_compression imports this lazily; keep the seam stable.
        from tools.skills_tool import reset_skill_view_dedup as f
        f(None)


class TestCompactionResetsDedup:
    """Both compaction paths must re-arm skill_view, not just file reads.

    A skill re-viewed after compaction has to return full content: the earlier
    result the stub points at was summarised away. The Codex app-server path
    opted out of the skill half (`skills=False`), so a parent that reloaded a
    contract mid-run got a stub naming content no longer in its context.
    """

    def _calls(self, monkeypatch):
        import agent.conversation_compression as cc
        import tools.file_tools_read_tracking as frt
        import tools.skills_tool as st

        seen = []
        monkeypatch.setattr(frt, "reset_file_dedup", lambda t: seen.append(("file", t)))
        monkeypatch.setattr(st, "reset_skill_view_dedup", lambda t: seen.append(("skill", t)))
        return cc, seen

    def test_hermes_path_helper_resets_both(self, monkeypatch):
        cc, seen = self._calls(monkeypatch)
        cc._reset_read_dedup_caches("t-1")
        assert seen == [("file", "t-1"), ("skill", "t-1")]

    def test_codex_app_server_path_resets_skill_view(self, monkeypatch):
        cc, seen = self._calls(monkeypatch)

        class _Result:
            interrupted = False
            error = None
            should_retire = False
            thread_id = "th-1"
            turn_id = "tu-1"

        class _Session:
            def compact_thread(self):
                return _Result()

        class _Agent:
            session_id = "s-1"
            codex_app_server_auto_compaction = "hermes"
            _codex_session = _Session()
            context_compressor = object()
            _cached_system_prompt = "SYS"
            status_callback = None

            def _emit_status(self, *a, **k):
                pass

        messages = [{"role": "user", "content": "hi"}]
        out, prompt = cc._compress_context_via_codex_app_server(
            _Agent(), messages, "SYS", task_id="t-2", force=True
        )
        assert out is messages and prompt == "SYS"
        assert ("skill", "t-2") in seen, "codex compaction left skill_view dedup armed"
