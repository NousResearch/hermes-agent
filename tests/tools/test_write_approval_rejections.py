"""Disapproved memory proposals stay suppressed until changed or explicitly reset."""

import json

import pytest

from hermes_cli.write_approval_commands import handle_pending_subcommand
from tools import write_approval as wa


def test_rejection_suppression_changed_content_force_and_reset(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sent = []
    monkeypatch.setattr(wa, "_prompt_for_record", lambda sub, rec: sent.append(rec["id"]))
    payload = {"action": "replace", "target": "user", "matched_entry": "full old entry",
               "old_text": "old", "content": "proposal"}
    first = wa.stage_write("memory", payload, summary="first", origin="pytest")
    assert "Rejected" in handle_pending_subcommand("memory", ["reject", first["id"]])
    skipped = wa.stage_write("memory", payload, summary="same", origin="pytest")
    assert skipped["skipped"] is True
    assert wa.list_pending("memory") == [] and len(sent) == 1
    changed = wa.stage_write("memory", {**payload, "content": "changed proposal"}, summary="changed", origin="pytest")
    forced = wa.stage_write("memory", payload, summary="forced", origin="pytest", force=True)
    assert wa.get_pending("memory", changed["id"]) and wa.get_pending("memory", forced["id"])
    assert len(sent) == 3
    assert "Forgot 1" in handle_pending_subcommand("memory", ["forget-rejections"])
    assert not wa.stage_write("memory", payload, summary="reset", origin="pytest").get("skipped")
    # Skills have exactly their prior reject-and-stage behavior.
    skill = wa.stage_write("skills", payload, summary="skill", origin="pytest")
    handle_pending_subcommand("skills", ["reject", skill["id"]])
    assert not wa.stage_write("skills", payload, summary="skill", origin="pytest").get("skipped")
    assert handle_pending_subcommand("skills", ["forget-rejections"]) is None


@pytest.mark.parametrize("unattended", [False, True])
def test_memory_tool_does_not_report_a_phantom_pending_id(tmp_path, monkeypatch, unattended):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("memory:\n  write_approval: true\n")
    monkeypatch.setattr(wa, "_prompt_for_record", lambda *args: None)
    from tools.memory_tool import load_on_disk_store, memory_tool
    from tools.skill_provenance import set_current_write_origin, reset_current_write_origin, set_review_attended, reset_review_attended

    store = load_on_disk_store()
    store.add("user", "old entry")
    origin = set_current_write_origin("background_review" if unattended else "foreground")
    attended = set_review_attended(False)
    try:
        arguments = dict(action="patch", target="user", old_text="old entry", content="replacement", store=store)
        first = json.loads(memory_tool(**arguments))
        handle_pending_subcommand("memory", ["reject", first["pending_id"]])
        suppressed = json.loads(memory_tool(**arguments))
        assert suppressed["suppressed"] and not suppressed.get("staged")
        assert "pending_id" not in suppressed
        assert store.user_entries == ["old entry"]
        assert wa.list_pending("memory") == []
    finally:
        reset_review_attended(attended)
        reset_current_write_origin(origin)


def test_fingerprints_are_full_content_specific_and_home_scoped(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tools.memory_rejections import proposal_fingerprint

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(wa, "_prompt_for_record", lambda *args: None)
    payload = {"action": "replace", "target": "user", "matched_entry": "old" * 100, "content": "new" * 100}
    first = wa.stage_write("memory", payload, summary="a", origin="pytest")
    handle_pending_subcommand("memory", ["reject", first["id"]])
    assert proposal_fingerprint(payload) != proposal_fingerprint({**payload, "matched_entry": payload["matched_entry"] + "tail"})
    assert proposal_fingerprint(payload) != proposal_fingerprint({**payload, "content": payload["content"] + "tail"})
    token = set_hermes_home_override(tmp_path / "other")
    try:
        assert not wa.stage_write("memory", payload, summary="b", origin="pytest").get("skipped")
    finally:
        reset_hermes_home_override(token)
    assert wa.stage_write("memory", payload, summary="a again", origin="pytest")["skipped"]
