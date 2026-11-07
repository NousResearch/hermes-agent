"""/skills diff must preview a full-rewrite patch (``patch`` + ``content``) as the rewrite
approval applies, not as an empty-anchor find-and-replace that "would fail"."""
import json

import pytest

OLD = "---\nname: probe\ndescription: Probe skill.\n---\n# Probe\nold body\n"
NEW = OLD.replace("old body", "NEW body")


@pytest.fixture
def gated_home(tmp_path, monkeypatch):
    (tmp_path / "skills" / "probe").mkdir(parents=True)
    (tmp_path / "skills" / "probe" / "SKILL.md").write_text(OLD, encoding="utf-8")
    (tmp_path / "config.yaml").write_text("skills:\n  write_approval: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("batched", [False, True])
def test_full_rewrite_preview_matches_what_approval_writes(gated_home, batched):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    from tools.skill_manager_tool import skill_manage

    op = {"action": "patch", "name": "probe", "content": NEW}
    staged = json.loads(skill_manage(action="", name="", operations=[op]) if batched
                        else skill_manage(action="patch", name="probe", content=NEW))
    assert staged.get("staged"), staged

    preview = handle_pending_subcommand(wa.SKILLS, ["diff", staged["pending_id"]]) or ""
    assert "would fail" not in preview
    assert "-old body" in preview and "+NEW body" in preview

    handle_pending_subcommand(wa.SKILLS, ["approve", staged["pending_id"]])
    assert (gated_home / "skills" / "probe" / "SKILL.md").read_text(encoding="utf-8") == NEW
