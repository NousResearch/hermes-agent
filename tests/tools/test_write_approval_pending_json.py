"""Regression tests for the pending review surfaces being lossy (#143181).

`/memory pending` (and `/skills pending`) show summaries capped at staging time, and a
background consolidation's cap used to slice prefix+detail together, so the cut always
landed in the new text. Two fixes: the staged summary now caps only the detail (prefix
survives intact), and `pending --json` returns every record's full payload so review
panes get the uncut content without re-reading pending/<id>.json.
"""

import json
import os
import shutil
import tempfile

import pytest


@pytest.fixture
def hermes_home(monkeypatch):
    d = tempfile.mkdtemp(prefix="hermes_wa_json_")
    home = os.path.join(d, ".hermes")
    os.makedirs(home)
    monkeypatch.setenv("HERMES_HOME", home)
    yield home
    shutil.rmtree(d, ignore_errors=True)


def test_pending_json_returns_full_uncapped_payload(hermes_home):
    """--json carries the whole staged payload, not the staging-time-capped summary."""
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa

    new_text = "自洽≠验证通过，须主动找反例交叉验证。（别用当下的观点当结论。）" * 4  # >200 chars
    old_text = "旧观点：自洽即验证通过"
    entry = f"{old_text}，其余保留部分。"
    wa.stage_write("memory",
                   {"action": "patch", "target": "memory", "old_text": old_text,
                    "content": new_text, "matched_entry": entry},
                   summary="background review consolidation (patch on memory): patch span '…",  # capped
                   origin="background_review")

    out = json.loads(handle_pending_subcommand(wa.MEMORY, ["pending", "--json"]))

    assert out["count"] == 1
    rec = out["records"][0]
    assert rec["action"] == "patch" and rec["origin"] == "background_review"
    assert rec["content"] == new_text  # full — never the capped summary
    assert rec["old_text"] == old_text
    assert rec["matched_entries"] == [f"patchs entry: {entry}"]
    # The human list stays short; --json must expose strictly more than the summary.
    assert len(rec["content"]) > len(rec["summary"])


def test_pending_json_covers_skills_records(hermes_home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa

    skill_md = "# Skill\n" + "step. " * 80
    wa.stage_write("skills", {"action": "write_file", "name": "triage", "file": "SKILL.md",
                              "content": skill_md},
                   summary="write SKILL.md in triage (…", origin="foreground")

    out = json.loads(handle_pending_subcommand(wa.SKILLS, ["pending", "--json"]))

    assert out["count"] == 1
    rec = out["records"][0]
    assert rec["content"] == skill_md
    assert rec["summary"].startswith("write SKILL.md")
    # matched_entries is the memory pinning feature; skills records don't carry it.
    assert "matched_entries" not in rec


def test_background_consolidation_summary_caps_detail_not_prefix(hermes_home):
    """The old `(...)[:200]` capped prefix+detail together, so a long consolidation lost its
    new text; now only the detail is capped and the fixed prefix survives intact."""
    from tools.memory_tool import MemoryStore, memory_tool
    from tools.skill_provenance import (reset_current_write_origin, reset_review_attended,
                                        set_current_write_origin, set_review_attended)
    from tools import write_approval as wa

    store = MemoryStore(); store.load_from_disk()
    old_text = "自洽即验证通过"
    assert store.add("memory", f"{old_text}，尾巴保留。")["success"]
    new_text = "自洽≠验证通过，须主动找反例交叉验证。" * 8  # pushes detail well past 200 chars
    origin, attended = set_current_write_origin("background_review"), set_review_attended(False)
    try:
        staged = json.loads(memory_tool(action="patch", old_text=old_text, content=new_text, store=store))
    finally:
        reset_review_attended(attended)
        reset_current_write_origin(origin)
    assert staged["staged"] is True, staged

    record = wa.get_pending(wa.MEMORY, staged["pending_id"])
    prefix = "background review consolidation (patch on memory): "
    detail = f"- patch span '{old_text}' -> '{new_text}' (rest of the entry is kept)"

    assert record["summary"] == f"{prefix}{detail[:200]}".rstrip()  # stage_write strips
    assert len(record["summary"]) >= len(prefix) + 199  # old code returned exactly 200
