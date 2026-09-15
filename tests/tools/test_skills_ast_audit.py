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
    """Report groups findings by file and orders them by line."""
    findings = [
        ("b.py", 5, "dynamic_import", "importlib.import_module() — ..."),
        ("a.py", 3, "dynamic_import", "importlib.import_module() — ..."),
        ("a.py", 1, "importlib_import", "import importlib — ..."),
    ]
    lines = format_ast_report(findings, skill_name="multi").splitlines()
    assert lines[0] == "AST deep scan: multi"
    assert lines[1] == "  3 finding(s):"
    # Both file groups and their line numbers arrive out of order.
    assert lines[2] == "  a.py"
    assert lines[3].startswith("    L1")
    assert lines[4].startswith("    L3")
    assert lines[5] == "  b.py"
    assert lines[6].startswith("    L5")


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


def test_oserror_on_file_read_returns_empty(tmp_path, monkeypatch):
    """A forced OSError while reading returns the documented empty result."""
    from tools import skills_ast_audit as audit

    target = tmp_path / "unreadable.py"
    target.write_text("import importlib\n")
    real_read_text = audit.Path.read_text

    def _raise_for_target(path, *args, **kwargs):
        if path == target:
            raise OSError("permission denied")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(audit.Path, "read_text", _raise_for_target)
    assert ast_scan_path(target) == []


def test_oserror_in_directory_scan_skips_unreadable_file(tmp_path, monkeypatch):
    """A forced OSError inside a directory scan contributes no findings."""
    from tools import skills_ast_audit as audit

    (tmp_path / "good.py").write_text("import importlib\n")
    bad = tmp_path / "bad.py"
    bad.write_text("import importlib\n")
    real_read_text = audit.Path.read_text

    def _raise_for_target(path, *args, **kwargs):
        if path == bad:
            raise OSError("permission denied")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(audit.Path, "read_text", _raise_for_target)
    findings = ast_scan_path(tmp_path)
    assert [f for (f, _l, _p, _d) in findings] == ["good.py"]


def test_scan_source_directly():
    """Test _scan_source with various inputs."""
    from tools.skills_ast_audit import _scan_source
    # Clean code
    assert _scan_source("x = 1\n", "clean.py") == []
    # Dynamic import
    findings = _scan_source("import importlib\n", "imp.py")
    assert any(pid == "importlib_import" for (_f, _l, pid, _d) in findings)


def test_scan_source_value_error(monkeypatch):
    """A ValueError from the parser is swallowed into the documented empty list."""
    from tools import skills_ast_audit as audit

    def _raise(*args, **kwargs):
        raise ValueError("invalid source")

    monkeypatch.setattr(audit.ast, "parse", _raise)
    assert audit._scan_source("x = 1\n", "ok.py") == []
