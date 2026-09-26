"""Support-file drift must not collapse distinct skill packages (#119959)."""

import json
from pathlib import Path

import pytest

from tools.skills_tool import skill_view


@pytest.mark.parametrize("support", [
    "scripts/run.py", "references/guide.md", "templates/item.txt", "assets/data.bin",
    "assets/.config.json", "assets/.settings/data.json", "scripts/shipped.pyc",
])
def test_skill_view_refuses_support_drift_but_preserves_explicit_selection(tmp_path, monkeypatch, support):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    roots = [tmp_path / "home/skills/demo", tmp_path / "home/skills/category/demo"]
    for root in roots:
        root.mkdir(parents=True)
        (root / "SKILL.md").write_text("---\nname: demo\ndescription: Demo.\n---\nUse the support files.\n", encoding="utf-8")
        path = root / support
        path.parent.mkdir(parents=True)
        path.write_text("same", encoding="utf-8")

    def no_whole_support_read(path):
        raise AssertionError("copy identity must stream files, not read_bytes")

    monkeypatch.setattr(Path, "read_bytes", no_whole_support_read)

    def view(name="demo"):
        return json.loads(skill_view(name, preprocess=False))

    assert view()["success"] is True
    # Incidental generated files must not bring back duplicate-copy init failures.
    for rel in ["scripts/__pycache__/run.cpython-311.pyc", ".DS_Store"]:
        path = roots[0] / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated")
    assert view()["success"] is True

    (roots[0] / support).write_text("stale", encoding="utf-8")
    result = view()
    assert result["success"] is False, result
    assert "Ambiguous" in result["error"]
    assert len(result["matches"]) == 2
    chosen = view("category/demo")
    assert chosen["success"] is True
    assert Path(chosen["skill_dir"]) == roots[1]
    (roots[0] / support).unlink()
    assert view()["success"] is False  # missing files are drift too
    (roots[0] / support).write_text("same", encoding="utf-8")
    assert view()["success"] is True


def test_legacy_copy_and_unreadable_support_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "home/skills"
    package = root / "demo"
    package.mkdir(parents=True)
    body = "---\nname: demo\ndescription: Demo.\n---\nInstructions.\n"
    (root / "demo.md").write_text(body, encoding="utf-8")
    (package / "SKILL.md").write_text(body, encoding="utf-8")
    (root / "unrelated.txt").write_text("not part of the flat skill", encoding="utf-8")
    initial = json.loads(skill_view("demo", preprocess=False))
    assert initial["success"] is True, initial
    support = package / "helper.txt"
    support.write_text("additional content", encoding="utf-8")
    assert json.loads(skill_view("demo", preprocess=False))["success"] is False
    real_read = Path.open

    def unreadable(path, *args, **kwargs):
        if path == support:
            raise PermissionError("fixture denies this support file")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unreadable)
    result = json.loads(skill_view("demo", preprocess=False))
    assert result["success"] is False
    assert "Ambiguous" in result["error"]
