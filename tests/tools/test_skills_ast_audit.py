"""Tests for tools.skills_ast_audit — opt-in AST diagnostic scanner."""

import sys

from tools.skills_ast_audit import ast_scan_path, format_ast_report


def _pids(findings):
    return [pid for (_f, _l, pid, _d) in findings]


def test_bypass_payload_detected(tmp_path):
    """The exact bypass shape from #7072 is caught."""
    f = tmp_path / "exfil.py"
    f.write_text(
        "import importlib\n"
        "parts = ['o', 's']\n"
        "m = importlib.import_module(''.join(parts))\n"
        "e = m.__dict__[''.join(['e','n','v'])]\n"
    )
    pids = _pids(ast_scan_path(f))
    assert "dynamic_import" in pids
    assert "importlib_import" in pids
    assert "dict_access" in pids


def test_syntax_error_does_not_crash(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("def broken(\n")
    assert ast_scan_path(f) == []


def test_recursion_error_does_not_crash(tmp_path):
    f = tmp_path / "deep.py"
    f.write_text("a" + ".x" * 5000 + "\n")
    orig = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        result = ast_scan_path(f)
    finally:
        sys.setrecursionlimit(orig)
    assert isinstance(result, list)


def test_importer_lookalike_not_flagged(tmp_path):
    """`import importer` must NOT match — dot-bounded prefix."""
    f = tmp_path / "ok.py"
    f.write_text("import importer\nfrom importer import x\n")
    assert _pids(ast_scan_path(f)) == []


def test_literal_dunder_import_not_flagged(tmp_path):
    """__import__('os') with a literal is not flagged (regex catches those)."""
    f = tmp_path / "ok.py"
    f.write_text("m = __import__('os')\n")
    assert "dynamic_import_computed" not in _pids(ast_scan_path(f))


def test_non_python_file_returns_empty(tmp_path):
    f = tmp_path / "script.sh"
    f.write_text("import importlib\n")
    assert ast_scan_path(f) == []


def test_directory_scans_recursively_and_skips_cache_dirs(tmp_path):
    skill = tmp_path / "s"
    skill.mkdir()
    (skill / "main.py").write_text("import importlib\n")
    (skill / "sub").mkdir()
    (skill / "sub" / "u.py").write_text("from importlib.util import find_spec\n")
    for d in ("__pycache__", ".venv", "venv", "node_modules"):
        ignored = skill / d
        ignored.mkdir()
        (ignored / "junk.py").write_text("import importlib\n")
    pids = _pids(ast_scan_path(skill))
    assert pids.count("importlib_import") == 2


def test_missing_path_returns_empty(tmp_path):
    assert ast_scan_path(tmp_path / "does_not_exist") == []


def test_dynamic_getattr_and_dict_access_detected(tmp_path):
    f = tmp_path / "g.py"
    f.write_text("name = 'x'\nv = getattr(o, name)\nv = o.__dict__[name]\n")
    pids = _pids(ast_scan_path(f))
    assert "dynamic_getattr" in pids
    assert "dict_access" in pids


def test_format_report_empty():
    assert "No dynamic" in format_ast_report([])


def test_format_report_with_findings():
    findings = [
        ("a.py", 1, "importlib_import", "import importlib — ..."),
        ("a.py", 3, "dynamic_import", "importlib.import_module() — ..."),
    ]
    out = format_ast_report(findings, skill_name="test")
    assert "test" in out and "a.py" in out and "L1" in out and "L3" in out
    assert "diagnostic hints" in out
