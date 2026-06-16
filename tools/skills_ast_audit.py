"""
AST-level deep audit for skill Python files — opt-in diagnostic, not a security gate.

Per SECURITY.md §2.4, Skills Guard is in-process heuristics ("useful — not
boundaries"). This module is a separate opt-in diagnostic that flags dynamic
import / dynamic attribute access patterns operators may want to eyeball when
reviewing third-party skill code. Every pattern flagged here has legitimate
uses; findings are hints for human review, not verdicts.

CLI: ``hermes skills audit --deep``
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

# (file, line, pattern_id, description)
Finding = Tuple[str, int, str, str]

_IGNORED_DIRS = {"__pycache__", ".venv", "venv", "node_modules"}


def _scan_source(content: str, rel_path: str) -> List[Finding]:
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError, RecursionError):
        return []

    findings: List[Finding] = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node):
            f = node.func
            # importlib.import_module(...)
            if isinstance(f, ast.Attribute) and f.attr == "import_module":
                findings.append((rel_path, node.lineno, "dynamic_import",
                                 "importlib.import_module() — loads arbitrary modules at runtime"))
            # __import__(<computed>)
            elif isinstance(f, ast.Name) and f.id == "__import__":
                if node.args and not isinstance(node.args[0], ast.Constant):
                    findings.append((rel_path, node.lineno, "dynamic_import_computed",
                                     "__import__ with non-literal module name"))
            # getattr(obj, <computed>)
            elif isinstance(f, ast.Name) and f.id == "getattr":
                if len(node.args) >= 2 and not isinstance(node.args[1], ast.Constant):
                    findings.append((rel_path, node.lineno, "dynamic_getattr",
                                     "getattr with non-literal attribute name"))
            self.generic_visit(node)

        def visit_Subscript(self, node):
            # obj.__dict__[<computed>]
            if (isinstance(node.value, ast.Attribute)
                    and node.value.attr == "__dict__"
                    and not isinstance(node.slice, ast.Constant)):
                findings.append((rel_path, node.lineno, "dict_access",
                                 "__dict__[<computed>] — dynamic attribute access"))
            self.generic_visit(node)

        def visit_Import(self, node):
            for a in node.names:
                if a.name == "importlib" or a.name.startswith("importlib."):
                    findings.append((rel_path, node.lineno, "importlib_import",
                                     f"import {a.name} — enables dynamic module loading"))
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            m = node.module or ""
            if m == "importlib" or m.startswith("importlib."):
                findings.append((rel_path, node.lineno, "importlib_import",
                                 f"from {m} import ... — enables dynamic module loading"))
            self.generic_visit(node)

    try:
        V().visit(tree)
    except (RecursionError, ValueError, RuntimeError):
        # Hostile/pathological input: return what we collected so far.
        pass

    return findings


def ast_scan_path(path: Path) -> List[Finding]:
    """Scan a single .py file or recursively scan all .py under a directory.

    Returns a list of (file, line, pattern_id, description) tuples. Empty for
    non-Python paths, missing paths, or paths with no matching patterns.
    """
    if path.is_file():
        if path.suffix.lower() != ".py":
            return []
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return _scan_source(content, path.name)

    if not path.is_dir():
        return []

    out: List[Finding] = []
    for py in sorted(path.rglob("*.py")):
        if set(py.parent.parts) & _IGNORED_DIRS:
            continue
        try:
            content = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            rel = py.relative_to(path).as_posix()
        except ValueError:
            rel = py.name
        out.extend(_scan_source(content, rel))
    return out


def format_ast_report(findings: List[Finding], skill_name: str = "") -> str:
    """Plain-text report (Rich-markup-free) grouped by file."""
    header = f"AST deep scan: {skill_name}" if skill_name else "AST deep scan"
    if not findings:
        return f"{header}\n  No dynamic import/access patterns detected."

    lines = [header, f"  {len(findings)} finding(s):"]
    current = None
    for f, line, pid, desc in sorted(findings):
        if f != current:
            current = f
            lines.append(f"  {f}")
        lines.append(f"    L{line}  {pid}  — {desc}")
    lines.append("")
    lines.append("  Note: diagnostic hints for human review, not security verdicts.")
    return "\n".join(lines)
