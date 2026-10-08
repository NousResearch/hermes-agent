"""Reference discoverability stays advisory across real skill writes (#132893)."""

import json
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.skill_linter import WARNING, lint_skill
from tools.registry import registry
import tools.skill_manager_tool  # registers the production handler


CONTENT = """---
name: reference-guide
description: Read a guide for maintaining project references.
version: 1.0.0
author: Test Author
license: MIT
metadata:
  hermes:
    tags: [references]
    related_skills: []
---
# Reference Guide
## When to Use
When maintaining project references.
"""


def _long(prefix="", count=101):
    lines = prefix.splitlines()
    return "\n".join(lines + ["Details."] * (count - len(lines))) + "\n"


@pytest.mark.parametrize("files,links,expected", [
    ({"references/a.md": _long(count=100)}, "references/a.md", set()),
    ({"references/a.md": _long()}, "references/a.md", {("reference-toc", "references/a.md")}),
    ({"references/a.md": _long("## Table of Contents")}, "references/a.md", set()),
    ({"references/a.md": _long("### contents")}, "references/a.md", set()),
    ({"references/a.md": _long("\n" * 39 + "# Contents")}, "references/a.md", set()),
    ({"references/a.md": _long("\n" * 40 + "# Contents")}, "references/a.md", {("reference-toc", "references/a.md")}),
    ({"templates/a.md": _long("- [Setup](#setup)\n- [Run](#run)")}, "[Guide](templates/a.md)", set()),
    ({"assets/a.md": _long("1. [Setup](#setup)\n2. [Run](#run)")}, "`assets/a.md`", set()),
    ({"references/a.md": _long("```md\n## Contents\n```")}, "references/a.md", {("reference-toc", "references/a.md")}),
    ({"references/a.md": _long("~~~md\n## Contents\n~~~")}, "references/a.md", {("reference-toc", "references/a.md")}),
    ({"references/a.md": "[More](b.md)", "references/b.md": "Details."}, "[Start](references/a.md)", {("reference-depth", "references/b.md")}),
    ({"references/a.md": "[More](b.md)", "references/b.md": "Details."}, "references/a.md\nreferences/b.md", set()),
    ({"references/a.md": "[More](../templates/b.md#run)", "templates/b.md": "[More](../assets/c.md)", "assets/c.md": "[Back](../references/a.md)"}, "references/a.md", {("reference-depth", "templates/b.md"), ("reference-depth", "assets/c.md")}),
    ({"references/a.md": "[More](<file name.md>)", "references/file name.md": "Details."}, "references/a.md", {("reference-depth", "references/file name.md")}),
    ({"references/a.md": "[More](file%20name.md)", "references/file name.md": "Details."}, "references/a.md", {("reference-depth", "references/file name.md")}),
    ({"references/a.md": "Details.", "references/orphan.md": "Unused."}, "references/a.md", set()),
    ({"references/_private/a.md": _long(), "_generated.md": _long()}, "", set()),
    ({"examples/a.md": _long()}, "[Guide](examples/a.md)", {("reference-toc", "examples/a.md")}),
])
def test_reference_warnings_describe_partial_read_risks(tmp_path, files, links, expected):
    # An underscore in the owning profile path must not hide public support files.
    skill_dir = tmp_path / "_profile" / "reference-guide"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(CONTENT + "\n" + links, encoding="utf-8")
    for rel, content in files.items():
        path = skill_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    findings = [f for f in lint_skill(skill_md) if f.rule in {"reference-toc", "reference-depth"}]
    actual = {(f.rule, rel) for f in findings for rel in files if "'" + rel + "'" in f.message}
    assert actual == expected
    assert len(findings) == len(expected)
    assert all(f.severity == WARNING for f in findings)


def _dispatch(**args):
    return json.loads(registry.dispatch("skill_manage", args))


@pytest.mark.parametrize("launch_name", ["default", "launch"])
def test_support_writes_surface_advisories_only_for_the_owning_profile(tmp_path, monkeypatch, launch_name):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default = tmp_path / ".hermes"
    homes = {"default": default, "launch": default / "profiles" / "launch",
             "a": default / "profiles" / "a", "b": default / "profiles" / "b"}
    for home in homes.values():
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("skills:\n  approval: false\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(homes[launch_name]))
    for name in ("a", "b"):
        token = set_hermes_home_override(homes[name])
        try:
            result = _dispatch(action="create", name="reference-guide", content=CONTENT + "\n[Start](templates/a.md)\n")
            assert result["success"] and not result.get("staged")
            result = _dispatch(action="write_file", name="reference-guide", file_path="templates/a.md", file_content="[More](b.md)")
            assert result["success"]
        finally:
            reset_hermes_home_override(token)
    for name, count in (("a", 101), ("b", 100), ("a", 101)):
        token = set_hermes_home_override(homes[name])
        try:
            result = _dispatch(action="write_file", name="reference-guide", file_path="templates/b.md", file_content=_long(count=count))
            assert result["success"] and not result.get("staged")
            rules = {f["rule"] for f in result.get("lint_warnings", [])}
            assert "reference-depth" in rules
            assert ("reference-toc" in rules) == (count > 100)
            path = homes[name] / "skills" / "reference-guide" / "templates" / "b.md"
            assert path.read_text(encoding="utf-8") == _long(count=count)
            patched = _dispatch(action="patch", name="reference-guide", file_path="templates/b.md",
                                old_string=_long(count=count), new_string=_long("## Contents", count=count))
            assert patched["success"]
            patch_rules = {f["rule"] for f in patched.get("lint_warnings", [])}
            assert "reference-depth" in patch_rules and "reference-toc" not in patch_rules
        finally:
            reset_hermes_home_override(token)
    assert not (homes[launch_name] / "skills" / "reference-guide").exists()
