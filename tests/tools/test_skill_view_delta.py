"""skill_view changed-file deltas: served only against a baseline visible in the transcript."""

import json

import pytest

from agent.context_compressor import _skill_pruned_marker
from agent.tool_dispatch_helpers import make_tool_result_message
from tools.skill_manager_tool import skill_manage
from tools.skills_tool import _skill_view_with_bump, reset_skill_view_dedup
from tools.skills_tool_delta import (
    apply_skill_view_delta,
    skill_view_result_for_transcript as _skill_view_transcript_delta,
    transcript_skill_view_state,
)

SKILL = "demo-delta-skill"
_BODY = "".join(f"## Section {i}\n\n- Rule {i}: key on ids, poll operations, honour Retry-After.\n\n"
                for i in range(1, 200))


@pytest.fixture
def skill_dir(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    d = home / "skills" / SKILL
    (d / "references").mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {SKILL}\ndescription: Demo skill for delta tests.\n---\n"
                                "# Demo\n\n## When to Use\nTests.\n\n" + _BODY, encoding="utf-8")
    (d / "references" / "guide.md").write_text("# Guide\n\n" + _BODY, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_skill_view_dedup()
    from tools.skill_manager_guards import _reset_background_review_read_marks
    _reset_background_review_read_marks()
    return d


class Session:
    """A transcript plus the executor path: handler -> transcript delta -> tool message."""

    def __init__(self, task="t-delta"):
        self.messages: list = []
        self.task = task
        self.n = 0

    def view(self, file_path=None, **extra):
        self.n += 1
        call_id = f"call_{self.n}"
        args = {"name": SKILL, **({"file_path": file_path} if file_path else {}), **extra}
        self.messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "skill_view", "arguments": json.dumps(args)}}]})
        result = _skill_view_with_bump(args, task_id=self.task)
        result = _skill_view_transcript_delta(result, self.messages, args)
        self.messages.append(make_tool_result_message("skill_view", result, call_id))
        return json.loads(result)

    def visible(self, file_path=None):
        state = transcript_skill_view_state(self.messages, (SKILL, file_path or ""))
        return state and state[0]


def _patch(old, new, file_path=None):
    r = json.loads(skill_manage(action="patch", name=SKILL, file_path=file_path, old_string=old,
                                new_string=new, task_id="t-delta"))
    assert r["success"], r


def _disk(skill_dir, rel="SKILL.md"):
    return (skill_dir / rel).read_text(encoding="utf-8")


def test_readback_after_patch_is_a_small_lossless_delta(skill_dir):
    s = Session()
    first = s.view()
    _patch("- Rule 42: key on ids", "- Rule 42: key on ids (never display names)")
    r = s.view()
    assert r.get("delta") is True and "content" not in r
    assert "+- Rule 42: key on ids (never display names)" in r["diff"]
    assert r["changed_sections"] == ["## Section 42"]
    assert apply_skill_view_delta(first["content"], r["diff"]) == _disk(skill_dir)
    assert len(json.dumps(r)) < 0.1 * len(json.dumps(first))


def test_chained_deltas_rebuild_the_current_file(skill_dir):
    s = Session()
    s.view()
    for n in (10, 20, 30):
        _patch(f"- Rule {n}: key", f"- Rule {n}: KEY")
        assert s.view().get("delta") is True
    assert s.visible() == _disk(skill_dir)


def test_rewind_that_drops_the_baseline_gets_full_content(skill_dir):
    s = Session()
    s.view()
    del s.messages[:]  # edit-and-resend / rewind truncation removed the earlier result
    _patch("- Rule 5: key", "- Rule 5: KEY")
    reset_skill_view_dedup()  # isolate the transcript check from the dedup stub (#139793)
    r = s.view()
    assert r.get("delta") is None and "Rule 5: KEY" in r["content"]


def test_pruned_baseline_gets_full_content(skill_dir):
    s = Session()
    s.view()
    s.messages[-1]["content"] = _skill_pruned_marker(SKILL)  # compaction/prune demoted the body
    _patch("- Rule 6: key", "- Rule 6: KEY")
    r = s.view()
    assert r.get("delta") is None and "Rule 6: KEY" in r["content"]


def test_pruned_earlier_delta_breaks_the_chain(skill_dir):
    s = Session()
    s.view()
    _patch("- Rule 7: key", "- Rule 7: KEY")
    assert s.view().get("delta") is True
    s.messages[-1]["content"] = "[skill_view] demoted"  # the delta row itself was demoted
    _patch("- Rule 8: key", "- Rule 8: KEY")
    r = s.view()
    assert r.get("delta") is None and "Rule 8: KEY" in r["content"]


def test_annotated_or_persisted_baseline_gets_full_content(skill_dir):
    s = Session()
    s.view()
    s.messages[-1]["content"] += "\n[guardrail note appended]"
    _patch("- Rule 9: key", "- Rule 9: KEY")
    assert s.view().get("delta") is None


def test_fork_or_subagent_with_its_own_transcript_gets_full_content(skill_dir):
    parent = Session(task="shared-task")
    parent.view()
    _patch("- Rule 11: key", "- Rule 11: KEY")
    child = Session(task="child-task")
    r = child.view()
    assert r.get("delta") is None and "Rule 11: KEY" in r["content"]


def test_concurrent_writers_diff_against_what_the_transcript_holds(skill_dir):
    s = Session()
    s.view()
    other = Session(task="other-task")
    _patch("- Rule 12: key", "- Rule 12: KEY")
    other.view()  # another task observes the intermediate version; ours never did
    _patch("- Rule 13: key", "- Rule 13: KEY")
    r = s.view()
    assert r.get("delta") is True
    assert s.visible() == _disk(skill_dir)


def test_large_rewrite_falls_back_to_full(skill_dir):
    s = Session()
    s.view()
    md = skill_dir / "SKILL.md"
    md.write_text(md.read_text(encoding="utf-8").replace("key on ids", "KEY ON IDS"), encoding="utf-8")
    r = s.view()
    assert r.get("delta") is None and "KEY ON IDS" in r["content"]


def test_full_true_always_returns_the_whole_file(skill_dir):
    s = Session()
    s.view()
    assert s.view().get("dedup") is True  # existing stub behaviour unchanged
    r = s.view(full=True)
    assert r.get("dedup") is None and r["content"] == _disk(skill_dir)
    _patch("- Rule 14: key", "- Rule 14: KEY")
    r = s.view(full=True)
    assert r.get("delta") is None and "Rule 14: KEY" in r["content"]


def test_reference_files_are_tracked_independently(skill_dir):
    s = Session()
    s.view()
    first = s.view("references/guide.md")
    _patch("- Rule 15: key", "- Rule 15: KEY", file_path="references/guide.md")
    r = s.view("references/guide.md")
    assert r.get("delta") is True
    assert apply_skill_view_delta(first["content"], r["diff"]) == _disk(skill_dir, "references/guide.md")
    assert s.visible() == _disk(skill_dir)  # SKILL.md state untouched by the reference delta


def test_new_linked_file_is_reported_in_changed_fields(skill_dir):
    s = Session()
    s.view()
    (skill_dir / "references" / "new.md").write_text("# New\n", encoding="utf-8")
    _patch("- Rule 16: key", "- Rule 16: KEY")
    r = s.view()
    assert r.get("delta") is True and "linked_files" in r["changed_fields"]


def test_background_review_fork_never_gets_a_delta(skill_dir):
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin
    s = Session()
    s.view()
    _patch("- Rule 17: key", "- Rule 17: KEY")
    token = set_current_write_origin("background_review")
    try:
        r = s.view()
    finally:
        reset_current_write_origin(token)
    assert r.get("delta") is None and "Rule 17: KEY" in r["content"]


def test_without_a_transcript_the_result_is_untouched(skill_dir):
    result = _skill_view_with_bump({"name": SKILL}, task_id=None)
    assert _skill_view_transcript_delta(result, None, {"name": SKILL}) == result
    assert _skill_view_transcript_delta(result, [], {"name": SKILL}) == result


def test_strict_apply_rejects_a_wrong_base():
    from tools.skills_tool_delta import skill_view_delta
    diff = skill_view_delta("a\nb\nc\n", "a\nB\nc\n")
    assert apply_skill_view_delta("a\nb\nc\n", diff) == "a\nB\nc\n"
    with pytest.raises(ValueError):
        apply_skill_view_delta("a\nx\nc\n", diff)
    tail = skill_view_delta("a\nb", "a\nb\nc")  # no trailing newline round-trips
    assert apply_skill_view_delta("a\nb", tail) == "a\nb\nc"
