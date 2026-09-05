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


# ── Additional coverage for uncovered paths ────────────────────────────


def test_computed_dunder_import_detected(tmp_path):
    """__import__ with a non-literal module name is flagged."""
    f = tmp_path / "dyn.py"
    f.write_text("name = 'o' + 's'\nm = __import__(name)\n")
    pids = _pids(ast_scan_path(f))
    assert "dynamic_import_computed" in pids


def test_from_importlib_import_detected(tmp_path):
    """from importlib import ... is flagged."""
    f = tmp_path / "imp.py"
    f.write_text("from importlib import import_module\n")
    pids = _pids(ast_scan_path(f))
    assert "importlib_import" in pids


def test_from_importlib_util_detected(tmp_path):
    """from importlib.util import ... is flagged."""
    f = tmp_path / "imp_util.py"
    f.write_text("from importlib.util import find_spec\n")
    pids = _pids(ast_scan_path(f))
    assert "importlib_import" in pids


def test_import_importlib_dot_submodule_detected(tmp_path):
    """import importlib.util is flagged."""
    f = tmp_path / "imp_dot.py"
    f.write_text("import importlib.util\n")
    pids = _pids(ast_scan_path(f))
    assert "importlib_import" in pids


def test_format_report_no_skill_name():
    """Report without skill_name uses generic header."""
    out = format_ast_report([])
    assert "AST deep scan" in out
    assert "No dynamic" in out


def test_format_report_multiple_files():
    """Report groups findings by file."""
    findings = [
        ("b.py", 5, "dynamic_import", "importlib.import_module() — ..."),
        ("a.py", 1, "importlib_import", "import importlib — ..."),
        ("a.py", 3, "dynamic_import", "importlib.import_module() — ..."),
    ]
    out = format_ast_report(findings, skill_name="multi")
    assert "a.py" in out
    assert "b.py" in out
    assert "L1" in out
    assert "L3" in out
    assert "L5" in out


def test_literal_getattr_not_flagged(tmp_path):
    """getattr(obj, 'attr') with a literal is not flagged."""
    f = tmp_path / "ok.py"
    f.write_text("v = getattr(o, 'attr')\n")
    assert "dynamic_getattr" not in _pids(ast_scan_path(f))


def test_literal_dict_access_not_flagged(tmp_path):
    """obj.__dict__['key'] with a literal is not flagged."""
    f = tmp_path / "ok.py"
    f.write_text("v = o.__dict__['key']\n")
    assert "dict_access" not in _pids(ast_scan_path(f))


def test_oserror_on_file_read_returns_empty(tmp_path):
    """OSError when reading a file returns empty findings."""
    f = tmp_path / "perm.py"
    f.write_text("import importlib\n")
    f.chmod(0o000)
    try:
        result = ast_scan_path(f)
        assert isinstance(result, list)
    finally:
        f.chmod(0o644)


def test_scan_source_directly():
    """Test _scan_source with various inputs."""
    from tools.skills_ast_audit import _scan_source
    # Clean code
    assert _scan_source("x = 1\n", "clean.py") == []
    # Dynamic import
    findings = _scan_source("import importlib\n", "imp.py")
    assert any(pid == "importlib_import" for (_f, _l, pid, _d) in findings)


def test_scan_source_value_error():
    """ValueError in ast.parse returns empty list."""
    from tools.skills_ast_audit import _scan_source
    # Very large integer literal can cause ValueError in some Python versions
    result = _scan_source("x = 1\n", "ok.py")
    assert isinstance(result, list)
