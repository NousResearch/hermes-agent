"""Real sync -> agent mutation -> pending/approval -> sync, with isolated homes."""
import json

import pytest

from tools import skill_manager_tool as sm, skills_sync as sync, write_approval as wa
from tools.registry import registry
from tools.skill_provenance import BACKGROUND_REVIEW, reset_current_write_origin, set_current_write_origin


CONTENT = "---\nname: example\ndescription: Use when testing skill ownership.\n---\n\nOriginal instructions.\n"


def package(path):
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(CONTENT, encoding="utf-8")
    (path / "references").mkdir()
    (path / "references/upstream.md").write_text("Original reference.", encoding="utf-8")
    return path


@pytest.fixture
def install(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("skills:\n  write_approval: false\n", encoding="utf-8")
    source = tmp_path / "bundled"
    upstream = package(source / "category/example")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(source))
    assert "example" in sync.sync_skills(quiet=True)["copied"]
    return home, upstream, home / "skills/category/example"


def invoke(operations):
    # Exercise the advertised registry tool, not a bare private mutation helper.
    entry = registry.get_entry("skill_manage")
    assert entry is not None
    return json.loads(entry.handler({"operations": operations}))


def patch_op(name="example"):
    return {"action": "patch", "name": name, "old_string": "Original instructions.",
            "new_string": "Local lesson."}


@pytest.mark.parametrize("kind", ["bundled", "hub"])
@pytest.mark.parametrize("shape", ["flat", "batch"])
@pytest.mark.parametrize("op", [
    patch_op(),
    {"action": "patch", "name": "example", "content": CONTENT.replace("Original", "Local")},
    {"action": "write_file", "name": "example", "file_path": "references/local.md", "file_content": "Local addition."},
    {"action": "patch", "name": "example", "file_path": "references/upstream.md", "old_string": "Original", "new_string": "Local"},
    {"action": "remove_file", "name": "example", "file_path": "references/upstream.md"},
    {"action": "delete", "name": "example"},
])
def test_managed_mutations_wait_without_changing_package(install, kind, shape, op):
    home, _, local = install
    if kind == "hub":
        (home / "skills/.bundled_manifest").unlink()
        hub = home / "skills/.hub"
        hub.mkdir()
        (hub / "lock.json").write_text(json.dumps({"installed": {
            "vendor-id": {"install_path": "category/example"}}}), encoding="utf-8")
    before = sync._dir_hash(local)
    result = invoke([op]) if shape == "batch" else json.loads(sm.skill_manage(**op))
    assert result.get("staged") is True, result
    assert sync._dir_hash(local) == before
    pending = wa.get_pending(wa.SKILLS, result["pending_id"])
    assert pending is not None
    warning = "freeze future package updates" if kind == "bundled" else "overwritten"
    assert warning in result["message"]
    assert warning in pending["summary"]  # The human must also see the risk, not only the model.
    preview = wa.skill_pending_diff(pending)
    if op["action"] in {"patch", "write_file"}:
        assert "Local" in preview, preview
    if "operations" in pending["payload"]:
        assert "[0]" in preview and "example" in preview, preview


def test_profile_ownership_and_explicit_approval_preserve_update_lifecycle(install, tmp_path, monkeypatch):
    home, upstream, local = install
    result = invoke([
        {"action": "create", "name": "personal", "content": CONTENT.replace("name: example", "name: personal")},
        {"action": "write_file", "name": "personal", "file_path": "references/local.md", "file_content": "Lesson."},
    ])
    assert result["success"] and not result.get("staged"), result
    personal = home / "skills/personal"
    assert (personal / "references/local.md").read_text() == "Lesson."

    # A -> B -> A in the same interpreter, including a name-only manifest collision.
    other = tmp_path / "other"
    other_local = package(other / "skills/custom/example")
    (other / "skills/.bundled_manifest").write_text("example:old-hash\n", encoding="utf-8")
    for target, staged in [(home, True), (other, False), (home, True)]:
        monkeypatch.setenv("HERMES_HOME", str(target))
        result = invoke([patch_op()])
        assert result["success"], result
        assert bool(result.get("staged")) is staged, result
    assert "Local lesson." in (other_local / "SKILL.md").read_text()
    assert not (other / "pending").exists()

    # A mixed call must wait as a unit, not commit its local op before asking.
    operations = [patch_op("personal"), patch_op("category/example"),
                  {"action": "write_file", "name": "example", "file_path": "references/local.md",
                   "file_content": "Explicit customization."}]
    operations.append({**patch_op(), "old_string": "Local lesson.", "new_string": "Reviewed lesson."})
    operations[1].pop("name")  # legacy top-level default must survive staged replay
    result = json.loads(sm.skill_manage(action="", name="category/example", operations=operations))
    assert result.get("staged") is True, result
    assert "Original instructions." in (personal / "SKILL.md").read_text()
    assert (local / "SKILL.md").read_text() == CONTENT
    pending = wa.get_pending(wa.SKILLS, result["pending_id"])
    assert pending is not None
    preview = wa.skill_pending_diff(pending)
    assert "[0]" in preview and "personal" in preview
    assert "[1]" in preview and "category/example" in preview
    assert "[2]" in preview and "Explicit customization." in preview
    assert "[3]" in preview and "-Local lesson." in preview and "+Reviewed lesson." in preview
    queued = len(wa.list_pending(wa.SKILLS))

    # Background ownership refusal still wins over the foreground approval path.
    token = set_current_write_origin(BACKGROUND_REVIEW)
    try:
        refused = invoke([patch_op()])
    finally:
        reset_current_write_origin(token)
    assert refused["success"] is False, refused
    assert "bundled" in refused["error"] or "built-in" in refused["error"]
    assert len(wa.list_pending(wa.SKILLS)) == queued

    # Until human approval, both upstream instructions and new references arrive.
    (upstream / "SKILL.md").write_text(CONTENT + "\nUpstream v2.\n", encoding="utf-8")
    (upstream / "references/new.md").write_text("New upstream reference.", encoding="utf-8")
    assert "example" in sync.sync_skills(quiet=True)["updated"]
    assert (local / "references/new.md").read_text() == "New upstream reference."
    assert "Upstream v2." in (local / "SKILL.md").read_text()

    # Explicit approval remains possible and does not re-stage the atomic batch.
    approved = json.loads(sm.apply_skill_pending(pending["payload"]))
    assert approved["success"] and not approved.get("staged"), approved
    assert "Reviewed lesson." in (local / "SKILL.md").read_text()
    assert "Upstream v2." in (local / "SKILL.md").read_text()
    assert (local / "references/local.md").read_text() == "Explicit customization."
    assert len(wa.list_pending(wa.SKILLS)) == queued
    (upstream / "SKILL.md").write_text(CONTENT + "\nUpstream v3.\n", encoding="utf-8")
    assert "example" in sync.sync_skills(quiet=True)["user_modified"]
    assert "Reviewed lesson." in (local / "SKILL.md").read_text()
