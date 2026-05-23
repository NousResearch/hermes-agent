"""
Tests for tools.skills_ast_audit — the opt-in AST diagnostic scanner.

These tests verify detection of dynamic import/access patterns that can
bypass line-by-line regex scanning, without crashing on hostile or
pathological input.
"""

import sys
from pathlib import Path

from tools.skills_ast_audit import (
    AstFinding,
    ast_scan_bundle_files,
    ast_scan_file,
    ast_scan_skill,
    format_ast_report,
)


# ---------------------------------------------------------------------------
# Core detection tests
# ---------------------------------------------------------------------------


class TestAstScanPython:
    """AST scanner detects dynamic import and access patterns."""

    def test_importlib_import_module_detected(self, tmp_path):
        """importlib.import_module() calls are flagged."""
        f = tmp_path / "evil.py"
        f.write_text("import importlib\nm = importlib.import_module('os')\n")
        findings = ast_scan_file(f)
        pids = [f.pattern_id for f in findings]
        assert "ast_dynamic_import" in pids
        assert "ast_importlib_import" in pids

    def test_importlib_submodule_import_detected(self, tmp_path):
        """`import importlib.util` and similar submodules are flagged."""
        f = tmp_path / "evil.py"
        f.write_text("import importlib.util\n")
        findings = ast_scan_file(f)
        pids = [f.pattern_id for f in findings]
        assert "ast_importlib_import" in pids

    def test_importlib_submodule_aliased_import_detected(self, tmp_path):
        """`import importlib.machinery as m` (aliased submodule) is flagged."""
        f = tmp_path / "evil.py"
        f.write_text("import importlib.machinery as m\n")
        findings = ast_scan_file(f)
        pids = [f.pattern_id for f in findings]
        assert "ast_importlib_import" in pids

    def test_from_importlib_import_detected(self, tmp_path):
        """`from importlib import import_module` is flagged."""
        f = tmp_path / "evil.py"
        f.write_text("from importlib import import_module\n")
        findings = ast_scan_file(f)
        pids = [f.pattern_id for f in findings]
        assert "ast_importlib_import" in pids

    def test_from_importlib_submodule_import_detected(self, tmp_path):
        """`from importlib.util import find_spec` is flagged."""
        f = tmp_path / "evil.py"
        f.write_text("from importlib.util import find_spec\n")
        findings = ast_scan_file(f)
        pids = [f.pattern_id for f in findings]
        assert "ast_importlib_import" in pids

    def test_importer_lookalike_not_flagged(self, tmp_path):
        """`import importer` must NOT match — prefix check is dot-bounded."""
        f = tmp_path / "ok.py"
        f.write_text("import importer\n")
        findings = ast_scan_file(f)
        pids = [f.pattern_id for f in findings]
        assert "ast_importlib_import" not in pids

    def test_from_importer_lookalike_not_flagged(self, tmp_path):
        """`from importer import something` must NOT match the importlib check."""
        f = tmp_path / "ok.py"
        f.write_text("from importer import something\n")
        findings = ast_scan_file(f)
        pids = [f.pattern_id for f in findings]
        assert "ast_importlib_import" not in pids

    def test_dunder_import_with_computed_arg_detected(self, tmp_path):
        """__import__ with non-literal argument is flagged."""
        f = tmp_path / "evil.py"
        f.write_text("name = 'os'\nm = __import__(name)\n")
        findings = ast_scan_file(f)
        pids = [f.pattern_id for f in findings]
        assert "ast_dynamic_import_computed" in pids

    def test_dunder_dict_computed_key_detected(self, tmp_path):
        """__dict__[<computed>] access is flagged."""
        f = tmp_path / "evil.py"
        f.write_text("key = 'environ'\nval = obj.__dict__[key]\n")
        findings = ast_scan_file(f)
        pids = [f.pattern_id for f in findings]
        assert "ast_dict_access" in pids

    def test_getattr_with_computed_name_detected(self, tmp_path):
        """getattr(obj, computed_name) is flagged."""
        f = tmp_path / "evil.py"
        f.write_text("name = 'system'\nfn = getattr(os, name)\n")
        findings = ast_scan_file(f)
        pids = [f.pattern_id for f in findings]
        assert "ast_dynamic_getattr" in pids

    def test_syntax_error_handled_gracefully(self, tmp_path):
        """Files with syntax errors should not crash the scanner."""
        f = tmp_path / "bad.py"
        f.write_text("def broken(\n")
        findings = ast_scan_file(f)
        assert isinstance(findings, list)

    def test_literal_dunder_import_not_flagged_by_ast(self, tmp_path):
        """__import__('os') with literal string is NOT flagged by AST."""
        f = tmp_path / "ok.py"
        f.write_text("m = __import__('os')\n")
        findings = ast_scan_file(f)
        pids = [f.pattern_id for f in findings]
        assert "ast_dynamic_import_computed" not in pids

    def test_full_bypass_payload_now_detected(self, tmp_path):
        """The exact bypass payload from #7072 should now be caught."""
        payload = """
import importlib
parts = ['o', 's']
m = importlib.import_module(''.join(parts))
e = m.__dict__[''.join(['e','n','v','i','r','o','n'])]
"""
        f = tmp_path / "exfil.py"
        f.write_text(payload)
        findings = ast_scan_file(f)
        pids = [f.pattern_id for f in findings]
        assert "ast_dynamic_import" in pids
        assert "ast_dict_access" in pids
        assert "ast_importlib_import" in pids

    def test_non_python_files_return_empty(self, tmp_path):
        """AST scan returns empty list for non-.py files."""
        f = tmp_path / "script.sh"
        f.write_text("import importlib\nimportlib.import_module('os')\n")
        findings = ast_scan_file(f)
        assert findings == []

    def test_scan_handles_recursion_error_gracefully(self, tmp_path):
        """Deeply-nested expressions that blow the visitor recursion limit
        must not crash the scan — return whatever findings were collected so far."""
        src = "a" + ".x" * 5000 + "\n"
        f = tmp_path / "deep.py"
        f.write_text(src)

        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(200)
        try:
            findings = ast_scan_file(f)
        finally:
            sys.setrecursionlimit(original_limit)

        assert isinstance(findings, list)


# ---------------------------------------------------------------------------
# Directory scanner tests
# ---------------------------------------------------------------------------


class TestAstScanSkill:
    """Directory-level scanning via ast_scan_skill()."""

    def test_scans_all_py_files_in_tree(self, tmp_path):
        """All .py files in a skill directory are scanned recursively."""
        skill = tmp_path / "my-skill"
        skill.mkdir()
        sub = skill / "subpkg"
        sub.mkdir()

        (skill / "main.py").write_text("import importlib\n")
        (sub / "utils.py").write_text("import importlib.util\n")

        findings = ast_scan_skill(skill)
        pids = [f.pattern_id for f in findings]
        # Both files should have importlib findings
        assert pids.count("ast_importlib_import") == 2

    def test_skips_ignored_dirs(self, tmp_path):
        """__pycache__, venv, .venv, and node_modules directories are skipped."""
        skill = tmp_path / "my-skill"
        skill.mkdir()
        for dirname in ("__pycache__", "venv", ".venv", "node_modules"):
            ignored = skill / dirname
            ignored.mkdir()
            (ignored / "cached.py").write_text("import importlib\n")

        findings = ast_scan_skill(skill)
        assert findings == []

    def test_skips_non_existent_dir(self, tmp_path):
        """Non-existent directory returns empty list."""
        findings = ast_scan_skill(Path("/nonexistent/skill/path"))
        assert findings == []

    def test_non_dir_path(self, tmp_path):
        """A file path (not a directory) returns empty list."""
        f = tmp_path / "not_a_dir.py"
        f.write_text("import importlib\n")
        findings = ast_scan_skill(f)
        assert findings == []


class TestAstScanBundleFiles:
    """In-memory bundle scanning for pre-install inspect diagnostics."""

    def test_scans_python_files_from_bundle(self):
        """Python files in source adapter bundle mappings are scanned."""
        findings = ast_scan_bundle_files({
            "SKILL.md": "---\nname: test\n---\n",
            "scripts/run.py": "import importlib\n",
            "references/readme.md": "import importlib\n",
        })
        assert [f.pattern_id for f in findings] == ["ast_importlib_import"]
        assert findings[0].file == "scripts/run.py"

    def test_decodes_bytes_bundle_content(self):
        """Bundle file content may be bytes; decode with replacement."""
        findings = ast_scan_bundle_files({
            "scripts/run.py": b"from importlib.util import find_spec\n",
        })
        assert [f.pattern_id for f in findings] == ["ast_importlib_import"]

    def test_skips_bundle_cache_dirs(self):
        """Virtualenv/cache paths in a bundle are ignored."""
        findings = ast_scan_bundle_files({
            "venv/lib/run.py": "import importlib\n",
            "__pycache__/cached.py": "import importlib\n",
        })
        assert findings == []


# ---------------------------------------------------------------------------
# Report formatting tests
# ---------------------------------------------------------------------------


class TestFormatAstReport:
    """Rich report formatting."""

    def test_empty_findings(self):
        """Empty findings list produces a clean 'nothing found' message."""
        report = format_ast_report([])
        assert "No AST-level patterns detected" in report

    def test_empty_with_skill_name(self):
        """Report with skill name but no findings."""
        report = format_ast_report([], skill_name="test-skill")
        assert "test-skill" in report
        assert "No AST-level patterns detected" in report

    def test_findings_grouped_by_file(self):
        """Findings from the same file appear together."""
        findings = [
            AstFinding(
                pattern_id="ast_importlib_import",
                severity="medium",
                category="obfuscation",
                file="main.py",
                line=1,
                match="import importlib",
                description="importlib imported",
            ),
            AstFinding(
                pattern_id="ast_dynamic_import",
                severity="high",
                category="obfuscation",
                file="main.py",
                line=3,
                match="importlib.import_module()",
                description="dynamic import via importlib",
            ),
        ]
        report = format_ast_report(findings)
        assert "main.py" in report
        assert "importlib imported" in report
        assert "dynamic import via importlib" in report
        assert "2 finding" in report  # summary line
        assert "Note: AST findings are diagnostic hints" in report

    def test_severity_summary(self):
        """Report header includes severity counts."""
        findings = [
            AstFinding("id1", "high", "x", "f.py", 1, "m", "desc"),
            AstFinding("id2", "high", "x", "f.py", 2, "m", "desc"),
            AstFinding("id3", "medium", "x", "f.py", 3, "m", "desc"),
        ]
        report = format_ast_report(findings)
        assert "2 high" in report
        assert "1 medium" in report
