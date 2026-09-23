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


def test_format_report_with_findings():
    findings = [
        ("a.py", 1, "importlib_import", "import importlib — ..."),
        ("a.py", 3, "dynamic_import", "importlib.import_module() — ..."),
    ]
    out = format_ast_report(findings, skill_name="test")
    assert "test" in out and "a.py" in out and "L1" in out and "L3" in out
    assert "diagnostic hints" in out
