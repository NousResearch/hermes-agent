import json
import shutil
import sys
from pathlib import Path

import pytest

HERMES_ROOT = Path(r'C:\\Users\\ZpLp\\AppData\\Local\\hermes\\hermes-agent')
if str(HERMES_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_ROOT))

from tools.skill_validation import validate_skill
from tools.skill_manager_tool import _create_skill
from hermes_constants import get_hermes_home


def _skill_home():
    return Path(get_hermes_home()) / "skills"


def _make_skill_dir(name, frontmatter_valid=True, script_body="print('ok')\n"):
    SKILL_HOME = _skill_home()
    d = SKILL_HOME / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    if frontmatter_valid:
        fm = f"name: {name}\ndescription: test\n"
    else:
        fm = "bad: x\n"
    md = f"---\n{fm}---\n\n# {name}\n"
    (d / "SKILL.md").write_text(md, encoding="utf-8")
    scripts = d / "scripts"
    scripts.mkdir()
    (scripts / "test.py").write_text(script_body, encoding="utf-8")
    return d


def _cleanup(name):
    d = _skill_home() / name
    if d.exists():
        shutil.rmtree(d)


def test_valid_skill_passes():
    _cleanup("svl4_valid")
    _make_skill_dir("svl4_valid")
    report = json.loads(validate_skill("svl4_valid", "", run_scripts=False))
    assert report["valid"] is True
    _cleanup("svl4_valid")


def test_runtime_script_execution_captures_stdout():
    _cleanup("svl4_exec")
    _make_skill_dir("svl4_exec", script_body="print('hello from script')\n")
    report = json.loads(validate_skill("svl4_exec", "", run_scripts=True))
    assert report["valid"] is True
    script = report["scripts"][0]
    assert script["status"] == "exec_ok"
    assert "hello from script" in script["stdout"]
    _cleanup("svl4_exec")


def test_runtime_script_failure_detected():
    _cleanup("svl4_exec_fail")
    _make_skill_dir("svl4_exec_fail", script_body="raise RuntimeError('bad')\n")
    report = json.loads(validate_skill("svl4_exec_fail", "", run_scripts=True))
    assert report["valid"] is False
    script = report["scripts"][0]
    assert script["status"] == "exec_failed"
    assert "bad" in script["stderr"]
    _cleanup("svl4_exec_fail")


def test_bad_frontmatter_detected():
    _cleanup("svl4_fm")
    _make_skill_dir("svl4_fm", frontmatter_valid=False)
    report = json.loads(validate_skill("svl4_fm", "", run_scripts=False))
    assert report["valid"] is False
    assert any("description" in e for e in report["errors"])
    _cleanup("svl4_fm")


def test_creation_blocks_invalid_skill(monkeypatch):
    import tools.skill_manager_tool as smt
    monkeypatch.setattr(smt, "SKILLS_DIR", _skill_home())
    name = "svl4_create_invalid"
    _cleanup(name)
    content = "---\nname: svl4_create_invalid\n---\n\n# Bad\n"
    result = _create_skill(name, content)
    assert result["success"] is False
    assert (_skill_home() / name).exists() is False


def test_creation_allows_valid_skill(monkeypatch):
    import tools.skill_manager_tool as smt
    monkeypatch.setattr(smt, "SKILLS_DIR", _skill_home())
    name = "svl4_create_valid"
    _cleanup(name)
    content = f"---\nname: {name}\ndescription: ok\n---\n\n# OK\n"
    result = _create_skill(name, content)
    assert result["success"] is True
    assert (_skill_home() / name).exists() is True
    _cleanup(name)