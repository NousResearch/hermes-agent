#!/usr/bin/env python3
"""
Code Structure Reading Tool
============================

Single ``read_code_structure`` tool that parses source code files and returns
a structured summary of function and class definitions — including their
names, start line numbers, and docstrings — without reading the full file
contents.

Supported languages:
- Python (.py) — full AST parsing
- JavaScript / TypeScript (.js, .ts, .jsx, .tsx) — regex-based extraction
- Go (.go) — regex-based extraction
- Java (.java) — regex-based extraction
- C / C++ (.c, .cpp, .h, .hpp) — regex-based extraction
- Ruby (.rb) — regex-based extraction
- Rust (.rs) — regex-based extraction
- PHP (.php) — regex-based extraction
- R (.r, .R) — regex-based extraction

For Python, the tool uses the ``ast`` module to reliably extract:
- Functions (including async) with their decorators
- Classes with their decorators and base classes
- Nested functions/classes within classes (methods)

For other languages, a robust regex-based extraction is used that captures:
- Function / method / procedure declarations
- Class / struct / interface / trait declarations

Output format
-------------
Each entry in the result is a dict with:

- ``name``: the function/method/class name
- ``type``: ``"function"`` or ``"class"`` (or ``"method"`` for Python class members)
- ``line``: the 1-indexed start line number
- ``end_line``: the 1-indexed end line number (Python only, ``null`` for regex)
- ``docstring``: the docstring text, or ``null`` if absent
- ``decorators``: list of decorator names (Python only)
- ``parent``: the parent class name for methods (Python only)

The tool belongs to the ``file`` toolset alongside ``read_file``,
``write_file``, ``patch``, and ``search_files``.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported extensions and their parsing strategies
# ---------------------------------------------------------------------------

_PYTHON_EXTENSIONS = {".py"}
_REGEX_EXTENSIONS = {
    ".js", ".jsx", ".ts", ".tsx",  # JS/TS family
    ".go",                          # Go
    ".java",                        # Java
    ".c", ".cpp", ".h", ".hpp",    # C/C++
    ".rb",                          # Ruby
    ".rs",                          # Rust
    ".php",                         # PHP
    ".r", ".R",                     # R
}
ALL_CODE_EXTENSIONS = _PYTHON_EXTENSIONS | _REGEX_EXTENSIONS


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

READ_CODE_STRUCTURE_SCHEMA: Dict[str, Any] = {
    "name": "read_code_structure",
    "description": (
        "Parse a source code file and return its function/class structure "
        "(names, line numbers, docstrings) without reading the full file contents. "
        "Supports Python (full AST), JavaScript/TypeScript, Go, Java, C/C++, "
        "Ruby, Rust, PHP, and R (regex-based). "
        "Use this when you need an overview of a code file's structure — "
        "function names, class names, their line numbers, and docstrings — "
        "rather than the full source. Faster and more token-efficient than "
        "read_file for large files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path to the source code file (absolute, relative, or "
                    "~/path). Must be a code file with a supported extension "
                    f"({', '.join(sorted(ALL_CODE_EXTENSIONS))})."
                ),
            },
        },
        "required": ["path"],
    },
}


# ---------------------------------------------------------------------------
# Python AST extraction
# ---------------------------------------------------------------------------

def _extract_python_structure(source: str, filename: str = "<module>") -> List[Dict[str, Any]]:
    """Use ``ast.parse`` to extract functions, classes, and methods."""

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        logger.warning("Python AST parse error in %s: %s", filename, exc)
        return []

    results: List[Dict[str, Any]] = []

    def _docstring(node: ast.AST) -> Optional[str]:
        ds = ast.get_docstring(node)
        return ds.strip() if ds else None

    def _decorators(node: ast.AST) -> List[str]:
        if not hasattr(node, "decorator_list"):
            return []
        decs: List[str] = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decs.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decs.append(ast.dump(dec))
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decs.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decs.append(ast.dump(dec.func))
            else:
                decs.append(ast.dump(dec))
        return decs

    def _visit_class(class_node: ast.ClassDef, parent: Optional[str] = None) -> None:
        entry = {
            "name": class_node.name,
            "type": "class",
            "line": class_node.lineno,
            "end_line": class_node.end_lineno,
            "docstring": _docstring(class_node),
            "decorators": _decorators(class_node),
            "parent": parent,
        }
        results.append(entry)
        # Visit nested classes and methods
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                _visit_function(node, parent=class_node.name, parent_is_class=True)
            elif isinstance(node, ast.ClassDef):
                _visit_class(node, parent=class_node.name)

    def _visit_function(func_node, parent: Optional[str] = None, parent_is_class: bool = False) -> None:
        is_method = parent_is_class
        entry = {
            "name": func_node.name,
            "type": "method" if is_method else "function",
            "line": func_node.lineno,
            "end_line": func_node.end_lineno,
            "docstring": _docstring(func_node),
            "decorators": _decorators(func_node),
            "parent": parent,
        }
        results.append(entry)
        # Visit nested classes and functions within function body
        for node in func_node.body:
            if isinstance(node, ast.ClassDef):
                _visit_class(node, parent=f"{parent}.{func_node.name}" if parent else func_node.name)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                _visit_function(node, parent=f"{parent}.{func_node.name}" if parent else func_node.name, parent_is_class=False)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            _visit_class(node)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            _visit_function(node)

    return results


# ---------------------------------------------------------------------------
# Regex-based extraction for non-Python languages
# ---------------------------------------------------------------------------

# Language-specific regex patterns for function declarations
_FUNCTION_PATTERNS = {
    # JavaScript / TypeScript — function declarations and arrow functions
    ".js": [
        re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>|\([^)]*\)\s*:\s*\w+\s*=>)",
            re.MULTILINE,
        ),
    ],
    ".jsx": [  # same as .js
        re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>|\([^)]*\)\s*:\s*\w+\s*=>)",
            re.MULTILINE,
        ),
    ],
    ".ts": [
        re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[<(]",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>|\([^)]*\)\s*:\s*\w+\s*=>)",
            re.MULTILINE,
        ),
    ],
    ".tsx": [  # same as .ts
        re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[<(]",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>|\([^)]*\)\s*:\s*\w+\s*=>)",
            re.MULTILINE,
        ),
    ],
    # Go — func declarations
    ".go": [
        re.compile(
            r"^\s*func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(",
            re.MULTILINE,
        ),
    ],
    # Java — method and class declarations
    ".java": [
        re.compile(
            r"^\s*(?:public|private|protected|static|\s)*\s+(?:class|interface|enum)\s+(\w+)",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:public|private|protected|static|synchronized|final|abstract|\s)+\s+(\w+)\s*\(",
            re.MULTILINE,
        ),
    ],
    # C / C++ — function declarations
    ".c": [
        re.compile(
            r"^\s*(?:[\w:*<>\s]+?)\s+(\w+)\s*\([^)]*\)\s*(?:\{|;)",
            re.MULTILINE,
        ),
    ],
    ".cpp": [
        re.compile(
            r"^\s*(?:[\w:*<>\s]+?)\s+(\w+)\s*\([^)]*\)\s*(?:\{|;|const)",
            re.MULTILINE,
        ),
    ],
    ".h": [  # same as .c
        re.compile(
            r"^\s*(?:[\w:*<>\s]+?)\s+(\w+)\s*\([^)]*\)\s*(?:\{|;)",
            re.MULTILINE,
        ),
    ],
    ".hpp": [  # same as .cpp
        re.compile(
            r"^\s*(?:[\w:*<>\s]+?)\s+(\w+)\s*\([^)]*\)\s*(?:\{|;|const)",
            re.MULTILINE,
        ),
    ],
    # Ruby — method definitions
    ".rb": [
        re.compile(
            r"^\s*def\s+(?:self\.)?(\w+[?!]?)",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*class\s+(\w+)",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*module\s+(\w+)",
            re.MULTILINE,
        ),
    ],
    # Rust — fn declarations
    ".rs": [
        re.compile(
            r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:pub\s+)?struct\s+(\w+)",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:pub\s+)?(?:trait|enum|impl)\s+(\w+)",
            re.MULTILINE,
        ),
    ],
    # PHP — function and class declarations
    ".php": [
        re.compile(
            r"^\s*function\s+(\w+)\s*\(",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:abstract\s+|final\s+)?class\s+(\w+)",
            re.MULTILINE,
        ),
    ],
    # R — function definitions
    ".r": [
        re.compile(
            r"^\s*(\w+)\s*<-\s*function\s*\(",
            re.MULTILINE,
        ),
    ],
    ".R": [  # same as .r
        re.compile(
            r"^\s*(\w+)\s*<-\s*function\s*\(",
            re.MULTILINE,
        ),
    ],
}

# Class/struct patterns for non-Python languages
_CLASS_PATTERNS = {
    ".js": [
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+(\w+)", re.MULTILINE),
    ],
    ".jsx": [
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+(\w+)", re.MULTILINE),
    ],
    ".ts": [
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)", re.MULTILINE),
    ],
    ".tsx": [
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)", re.MULTILINE),
    ],
    ".go": [
        re.compile(r"^\s*type\s+(\w+)\s+struct\s*\{", re.MULTILINE),
    ],
    ".java": [
        re.compile(r"^\s*(?:public|private|protected|static|\s)*\s+(?:class|interface|enum)\s+(\w+)", re.MULTILINE),
    ],
    ".c": [
        re.compile(r"^\s*(?:typedef\s+)?struct\s+(\w+)", re.MULTILINE),
    ],
    ".cpp": [
        re.compile(r"^\s*(?:typedef\s+)?struct\s+(\w+)", re.MULTILINE),
        re.compile(r"^\s*class\s+(\w+)", re.MULTILINE),
    ],
    ".h": [
        re.compile(r"^\s*(?:typedef\s+)?struct\s+(\w+)", re.MULTILINE),
        re.compile(r"^\s*class\s+(\w+)", re.MULTILINE),
    ],
    ".hpp": [
        re.compile(r"^\s*(?:typedef\s+)?struct\s+(\w+)", re.MULTILINE),
        re.compile(r"^\s*class\s+(\w+)", re.MULTILINE),
    ],
    ".rb": [
        re.compile(r"^\s*class\s+(\w+)", re.MULTILINE),
        re.compile(r"^\s*module\s+(\w+)", re.MULTILINE),
    ],
    ".rs": [
        re.compile(r"^\s*(?:pub\s+)?struct\s+(\w+)", re.MULTILINE),
        re.compile(r"^\s*(?:pub\s+)?trait\s+(\w+)", re.MULTILINE),
        re.compile(r"^\s*(?:pub\s+)?enum\s+(\w+)", re.MULTILINE),
    ],
    ".php": [
        re.compile(r"^\s*(?:abstract\s+|final\s+)?class\s+(\w+)", re.MULTILINE),
        re.compile(r"^\s*interface\s+(\w+)", re.MULTILINE),
        re.compile(r"^\s*trait\s+(\w+)", re.MULTILINE),
    ],
    ".r": [],
    ".R": [],
}


def _find_docstring_above_line(source: str, line_number: int, ext: str) -> Optional[str]:
    """Attempt to find a docstring or comment block immediately above a line.

    For multi-line block comments (e.g. JSDoc ``/** ... */``), the function
    scans backwards from the line just above ``line_number``, detects the
    ``*/`` closer, then continues scanning back to the ``/**`` opener, and
    returns the whole block as a single docstring.

    For single-line comment styles (``//``, ``#``, ``///``), contiguous
    comment lines above the target are collected.
    """
    lines = source.splitlines()
    if line_number < 2 or len(lines) < line_number - 1:
        return None

    # Check if the line just above the target ends a block comment (*/)
    line_above = lines[line_number - 2].rstrip()

    # --- Multi-line block comment (JSDoc / JavaDoc / Doxygen) ---
    if line_above.endswith("*/"):
        # Collect lines from */ back to /**
        block_lines: List[str] = []
        i = line_number - 2  # 0-indexed, the */ line
        found_opener = False
        while i >= 0:
            l = lines[i]
            block_lines.insert(0, l)
            if "/**" in l or "/*" in l:
                found_opener = True
                break
            i -= 1
        if found_opener and block_lines:
            # Strip comment markers and join
            cleaned = []
            for bl in block_lines:
                bl_clean = re.sub(
                    r"^\s*(?:/\*\*?|\*\s?|\*/)\s*", "", bl
                ).strip()
                # Also handle standalone * lines inside the block
                bl_clean = re.sub(r"^\*\s?", "", bl_clean).strip()
                if bl_clean:
                    cleaned.append(bl_clean)
            return " ".join(cleaned) if cleaned else None

    # --- Single-line comment style ---
    # Determine the single-line comment prefix pattern for this extension
    _SINGLE_LINE_PREFIX: Dict[str, re.Pattern] = {
        ".js": re.compile(r"^\s*//"),
        ".jsx": re.compile(r"^\s*//"),
        ".ts": re.compile(r"^\s*//"),
        ".tsx": re.compile(r"^\s*//"),
        ".go": re.compile(r"^\s*//"),
        ".java": re.compile(r"^\s*//"),
        ".c": re.compile(r"^\s*//"),
        ".cpp": re.compile(r"^\s*//"),
        ".h": re.compile(r"^\s*//"),
        ".hpp": re.compile(r"^\s*//"),
        ".rb": re.compile(r"^\s*#"),
        ".rs": re.compile(r"^\s*///|^\s*//!"),
        ".php": re.compile(r"^\s*//|^\s*#"),
        ".r": re.compile(r"^\s*#"),
        ".R": re.compile(r"^\s*#"),
    }
    single_pat = _SINGLE_LINE_PREFIX.get(ext)
    if single_pat is None:
        return None

    doc_lines: List[str] = []
    i = line_number - 2  # 0-indexed, one line above the target
    while i >= 0:
        line = lines[i] if i < len(lines) else ""
        if single_pat.match(line):
            doc_lines.insert(0, line)
            i -= 1
        else:
            break
    if not doc_lines:
        return None
    # Strip comment markers and join
    cleaned = []
    for dl in doc_lines:
        dl_clean = re.sub(r"^\s*(?://|///|//!|#!|#)\s*", "", dl).strip()
        if dl_clean:
            cleaned.append(dl_clean)
    return " ".join(cleaned) if cleaned else None


def _extract_regex_structure(source: str, ext: str) -> List[Dict[str, Any]]:
    """Regex-based extraction for non-Python languages."""
    results: List[Dict[str, Any]] = []

    # Extract functions
    func_patterns = _FUNCTION_PATTERNS.get(ext, [])
    for pat in func_patterns:
        for match in pat.finditer(source):
            name = match.group(1)
            line = source[:match.start()].count("\n") + 1
            docstring = _find_docstring_above_line(source, line, ext)
            results.append({
                "name": name,
                "type": "function",
                "line": line,
                "end_line": None,
                "docstring": docstring,
                "decorators": [],
                "parent": None,
            })

    # Extract classes / structs / interfaces / traits
    class_patterns = _CLASS_PATTERNS.get(ext, [])
    for pat in class_patterns:
        for match in pat.finditer(source):
            name = match.group(1)
            line = source[:match.start()].count("\n") + 1
            docstring = _find_docstring_above_line(source, line, ext)
            results.append({
                "name": name,
                "type": "class",
                "line": line,
                "end_line": None,
                "docstring": docstring,
                "decorators": [],
                "parent": None,
            })

    # Sort by line number
    results.sort(key=lambda r: r["line"])
    return results


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_code_structure_requirements() -> bool:
    """Always available — no external dependencies needed."""
    return True


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def _handle_read_code_structure(args: Dict[str, Any], **_kw: Any) -> str:
    path = (args.get("path") or "").strip()
    if not path:
        return tool_error("path is required for read_code_structure")

    resolved = Path(path).expanduser()
    if not resolved.exists():
        # Suggest similar filenames using fuzzy matching
        try:
            from difflib import get_close_matches
            parent = resolved.parent
            if parent.exists():
                sibling_names = [p.name for p in parent.iterdir() if p.is_file()]
                matches = get_close_matches(resolved.name, sibling_names, n=5, cutoff=0.4)
                if matches:
                    suggestion = f" Did you mean: {', '.join(matches)}?"
                else:
                    suggestion = ""
            else:
                suggestion = ""
        except Exception:
            suggestion = ""
        return tool_error(f"File not found: {path}.{suggestion}")

    ext = resolved.suffix.lower()
    if ext not in ALL_CODE_EXTENSIONS:
        return tool_error(
            f"Unsupported file type: '{ext}'. Supported extensions: "
            f"{', '.join(sorted(ALL_CODE_EXTENSIONS))}"
        )

    try:
        source = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return tool_error(f"Cannot read file: {exc}")

    if ext in _PYTHON_EXTENSIONS:
        entries = _extract_python_structure(source, filename=str(resolved))
    else:
        entries = _extract_regex_structure(source, ext)

    if not entries:
        return tool_result({"path": str(resolved), "entries": [], "language": ext.lstrip(".")})

    return tool_result({
        "path": str(resolved),
        "language": ext.lstrip("."),
        "entries": entries,
    })


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

registry.register(
    name="read_code_structure",
    toolset="file",
    schema=READ_CODE_STRUCTURE_SCHEMA,
    handler=_handle_read_code_structure,
    check_fn=check_code_structure_requirements,
    requires_env=[],
    is_async=False,
    emoji="🗂️",
)
