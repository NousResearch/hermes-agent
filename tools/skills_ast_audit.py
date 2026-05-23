"""
AST-level deep audit for skill Python files — opt-in diagnostic, not a security gate.

This is a standalone diagnostic tool per SECURITY.md spirit: it helps operators
inspect skill code for patterns that *could* enable dynamic import/access
obfuscation, but it is NOT a security boundary. Every pattern flagged here has
legitimate uses. Use your judgment.

Usage::

    from tools.skills_ast_audit import ast_scan_skill, format_ast_report

    findings = ast_scan_skill(Path("~/.hermes/skills/some-skill"))
    if findings:
        print(format_ast_report(findings))

CLI integration: ``hermes skills audit --deep``
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, List, Optional, Union


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AstFinding:
    """A single finding from AST-level analysis."""

    pattern_id: str
    """Short identifier for deduplication and grouping (e.g. 'ast_importlib_import')."""

    severity: str
    """One of 'high', 'medium', 'low' — for display only, not a security claim."""

    category: str
    """Grouping label — currently always 'obfuscation'."""

    file: str
    """Relative path to the file containing the finding."""

    line: int
    """1-based line number."""

    match: str
    """The matched source construct (human-readable snippet)."""

    description: str
    """Why this pattern is worth reviewing."""


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _ast_scan_python(content: str, rel_path: str) -> List[AstFinding]:
    """Detect obfuscation via dynamic imports, attribute access, and string construction.

    Hostile or pathological input (deeply-nested expressions, malformed source)
    must not crash the scan. Both ``ast.parse`` and the visitor traversal are
    guarded so parse/visit failures degrade gracefully to "no AST findings"
    rather than raising.
    """
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError, RecursionError):
        return []

    findings: List[AstFinding] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node):
            # Detect importlib.import_module(...)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            ):
                findings.append(
                    AstFinding(
                        pattern_id="ast_dynamic_import",
                        severity="high",
                        category="obfuscation",
                        file=rel_path,
                        line=node.lineno,
                        match="importlib.import_module()",
                        description="dynamic import via importlib — can load arbitrary modules at runtime",
                    )
                )
            # Detect __import__ with non-literal argument
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                if node.args and not isinstance(node.args[0], ast.Constant):
                    findings.append(
                        AstFinding(
                            pattern_id="ast_dynamic_import_computed",
                            severity="high",
                            category="obfuscation",
                            file=rel_path,
                            line=node.lineno,
                            match="__import__(<computed>)",
                            description="__import__ with dynamically constructed module name",
                        )
                    )
            # Detect getattr with computed attribute name
            if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                if len(node.args) >= 2 and not isinstance(
                    node.args[1], ast.Constant
                ):
                    findings.append(
                        AstFinding(
                            pattern_id="ast_dynamic_getattr",
                            severity="medium",
                            category="obfuscation",
                            file=rel_path,
                            line=node.lineno,
                            match="getattr(<obj>, <computed>)",
                            description="getattr with dynamically constructed attribute name",
                        )
                    )
            self.generic_visit(node)

        def visit_Subscript(self, node):
            # Detect obj.__dict__[<computed>]
            if (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "__dict__"
            ):
                if not isinstance(node.slice, ast.Constant):
                    findings.append(
                        AstFinding(
                            pattern_id="ast_dict_access",
                            severity="high",
                            category="obfuscation",
                            file=rel_path,
                            line=node.lineno,
                            match="__dict__[<computed>]",
                            description="dynamic attribute access via __dict__ with computed key",
                        )
                    )
            self.generic_visit(node)

        def visit_Import(self, node):
            # Flag importlib and any importlib.* submodule.
            for alias in node.names:
                if alias.name == "importlib" or alias.name.startswith(
                    "importlib."
                ):
                    findings.append(
                        AstFinding(
                            pattern_id="ast_importlib_import",
                            severity="medium",
                            category="obfuscation",
                            file=rel_path,
                            line=node.lineno,
                            match=f"import {alias.name}",
                            description="importlib imported — enables dynamic module loading",
                        )
                    )
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            module = node.module or ""
            if module == "importlib" or module.startswith("importlib."):
                findings.append(
                    AstFinding(
                        pattern_id="ast_importlib_import",
                        severity="medium",
                        category="obfuscation",
                        file=rel_path,
                        line=node.lineno,
                        match=f"from {module} import ...",
                        description="importlib imported — enables dynamic module loading",
                    )
                )
            self.generic_visit(node)

    try:
        _Visitor().visit(tree)
    except (RecursionError, ValueError, RuntimeError):
        # Visitor traversal can fail on hostile input even when ast.parse
        # succeeded (e.g. deeply-nested call/attribute chains). Return
        # whatever findings we collected before the failure.
        return findings

    return findings


def ast_scan_file(file_path: Path, rel_path: Optional[str] = None) -> List[AstFinding]:
    """Scan a single Python file and return AST-level findings.

    Args:
        file_path: Absolute path to the .py file.
        rel_path: Relative path for display (defaults to file_path.name).

    Returns:
        List of :class:`AstFinding` — empty if the file isn't Python or scan yields nothing.
    """
    if file_path.suffix.lower() != ".py":
        return []

    if rel_path is None:
        rel_path = file_path.name

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return []

    return _ast_scan_python(content, rel_path)


def ast_scan_skill(skill_path: Path) -> List[AstFinding]:
    """Recursively scan all Python files in a skill directory.

    Args:
        skill_path: Path to the installed skill directory.

    Returns:
        Combined list of :class:`AstFinding` across all .py files.
    """
    if not skill_path.is_dir():
        return []

    all_findings: List[AstFinding] = []

    for py_file in sorted(skill_path.rglob("*.py")):
        # Skip __pycache__ and .venv/venv directories
        parts = set(py_file.parent.parts)
        if parts & {"__pycache__", ".venv", "venv", "node_modules"}:
            continue
        try:
            rel = py_file.relative_to(skill_path).as_posix()
        except ValueError:
            rel = py_file.name
        all_findings.extend(ast_scan_file(py_file, rel))

    return all_findings


def ast_scan_bundle_files(
    files: Mapping[str, Union[str, bytes]],
) -> List[AstFinding]:
    """Scan Python files from an in-memory skill bundle.

    This powers ``hermes skills inspect --ast-deep`` so operators can review
    a skill before installing it. The input is the bundle's filename -> content
    mapping, as returned by the skills hub source adapters.
    """
    all_findings: List[AstFinding] = []

    for rel_path, content in sorted(files.items()):
        path = Path(rel_path)
        if path.suffix.lower() != ".py":
            continue
        if set(path.parts) & {"__pycache__", ".venv", "venv", "node_modules"}:
            continue
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content)
        all_findings.extend(_ast_scan_python(text, rel_path))

    return all_findings


# ---------------------------------------------------------------------------
# Rich formatting
# ---------------------------------------------------------------------------


def format_ast_report(
    findings: List[AstFinding],
    skill_name: str = "",
) -> str:
    """Format AST findings as a Rich-markup string.

    Args:
        findings: List of findings from :func:`ast_scan_skill`.
        skill_name: Optional skill name for the report header.

    Returns:
        Rich-markup string suitable for ``console.print()``.
    """
    if not findings:
        header = (
            f"[bold]AST Deep Scan: {skill_name}[/]"
            if skill_name
            else "[bold]AST Deep Scan[/]"
        )
        return f"{header}\n[dim green]No AST-level patterns detected.[/]"

    lines: List[str] = []
    severity_order = {"high": 0, "medium": 1, "low": 2}
    findings_sorted = sorted(
        findings,
        key=lambda f: (
            severity_order.get(f.severity, 99),
            f.file,
            f.line,
        ),
    )

    if skill_name:
        lines.append(f"[bold]AST Deep Scan: {skill_name}[/]")
    else:
        lines.append("[bold]AST Deep Scan[/]")

    total = len(findings_sorted)
    high_count = sum(1 for f in findings_sorted if f.severity == "high")
    med_count = sum(1 for f in findings_sorted if f.severity == "medium")
    low_count = sum(1 for f in findings_sorted if f.severity == "low")

    summary_parts = []
    if high_count:
        summary_parts.append(f"[bold red]{high_count} high[/]")
    if med_count:
        summary_parts.append(f"[yellow]{med_count} medium[/]")
    if low_count:
        summary_parts.append(f"[dim]{low_count} low[/]")
    lines.append(
        f"[dim]{total} finding(s)[/] — "
        + ", ".join(summary_parts)
        if summary_parts
        else f"[dim]{total} finding(s)[/]"
    )
    lines.append("")

    current_file = None
    for f in findings_sorted:
        if f.file != current_file:
            current_file = f.file
            lines.append(f"  [bold cyan]{f.file}[/]")
        sev_color = {"high": "bold red", "medium": "yellow", "low": "dim"}.get(
            f.severity, "dim"
        )
        lines.append(
            f"    L{f.line:>4} [{sev_color}]{f.severity:6}[/] {f.description}"
        )
        lines.append(f"          [dim]{f.match}[/]")

    lines.append("")
    lines.append(
        "[dim]Note: AST findings are diagnostic hints, not security verdicts. "
        "Review each pattern in context.[/]"
    )

    return "\n".join(lines)
