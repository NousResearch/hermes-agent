#!/usr/bin/env python3
"""Code-intel capability abstraction and read-only Python MVP helpers."""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path
from typing import Any

from tools.registry import registry, tool_result

SUPPORTED_OPERATIONS = ("rename", "references", "definition", "symbols", "diagnostics")
SUPPORTED_LANGUAGES = ("python",)
PYTHON_BACKEND = "python_ast"

_PYTHON_CAPABILITIES = {
    "rename": True,
    "references": True,
    "definition": True,
    "symbols": True,
    "diagnostics": True,
}

_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_DEFINITION_NODES = (ast.arg, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.alias)


def _normalize_language(path: str | Path | None = None, language: str | None = None) -> str | None:
    if language:
        lowered = language.lower()
        aliases = {
            "py": "python",
            "python": "python",
            "js": "javascript",
            "javascript": "javascript",
            "ts": "typescript",
            "typescript": "typescript",
        }
        return aliases.get(lowered, lowered)
    if path is None:
        return None
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
    }.get(suffix)


def _unsupported_language_result(language: str | None, *, operation: str | None = None) -> dict[str, Any]:
    result = {
        "ok": False,
        "backend": None,
        "language": language,
        "code": "unsupported_language",
        "error": "Unsupported language for code_intel.",
        "supported_languages": list(SUPPORTED_LANGUAGES),
    }
    if operation:
        result["operation"] = operation
    return result


def _path_not_found_result(path: str, *, operation: str | None = None) -> dict[str, Any]:
    result = {
        "ok": False,
        "backend": None,
        "code": "path_not_found",
        "error": "Path does not exist or is not a file.",
        "path": str(path),
    }
    if operation:
        result["operation"] = operation
    return result


def _parse_python(path: str | Path) -> tuple[str, ast.AST | None, dict[str, Any] | None]:
    file_path = Path(path)
    if not file_path.is_file():
        return "", None, _path_not_found_result(str(file_path))
    source = file_path.read_text(encoding="utf-8")
    try:
        return source, ast.parse(source, filename=str(file_path)), None
    except SyntaxError as exc:
        return source, None, {
            "ok": False,
            "backend": PYTHON_BACKEND,
            "language": "python",
            "code": "parse_error",
            "error": "Python source could not be parsed.",
            "details": {
                "line": exc.lineno,
                "column": exc.offset - 1 if exc.offset else None,
                "message": exc.msg,
            },
        }


def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _node_contains(node: ast.AST, line: int, column: int) -> bool:
    end_lineno = getattr(node, "end_lineno", None)
    end_col = getattr(node, "end_col_offset", None)
    if end_lineno is None or end_col is None:
        return False
    if line < node.lineno or line > end_lineno:
        return False
    if line == node.lineno and column < node.col_offset:
        return False
    if line == end_lineno and column >= end_col:
        return False
    return True


def _cursor_symbol_name(source: str, line: int, column: int) -> str | None:
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.NAME or token.start[0] != line:
            continue
        for candidate_column in dict.fromkeys([column, column - 1]):
            if candidate_column < 0:
                continue
            if token.start[1] <= candidate_column < token.end[1]:
                return token.string
    return None


def _nearest_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST | None:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, _SCOPE_NODES):
            return current
    return None


def _scope_key(node: ast.AST | None, parents: dict[ast.AST, ast.AST]) -> tuple[str, int, int] | None:
    scope = _nearest_scope(node, parents) if node is not None else None
    if scope is None:
        return None
    return (scope.__class__.__name__, getattr(scope, "lineno", -1), getattr(scope, "col_offset", -1))


def _candidate_symbol_nodes(tree: ast.AST) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.arg, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.alias)):
            nodes.append(node)
    return sorted(nodes, key=lambda item: (getattr(item, "lineno", 0), getattr(item, "col_offset", 0)))


def _symbol_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.alias):
        return node.asname or node.name.split(".")[0]
    return None


def _find_symbol_node(tree: ast.AST, source: str, line: int, column: int) -> tuple[ast.AST | None, str | None]:
    token_name = _cursor_symbol_name(source, line, column)
    best: ast.AST | None = None
    for node in _candidate_symbol_nodes(tree):
        if token_name is not None and _symbol_name(node) != token_name:
            continue
        if _node_contains(node, line, column) or _node_contains(node, line, column - 1):
            best = node
    return best, _symbol_name(best) if best is not None else token_name


def _line_snippet(source: str, line: int) -> str:
    lines = source.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1]
    return ""


def _reference_item(path: Path, source: str, symbol_name: str, line: int, column: int) -> dict[str, Any]:
    return {
        "file": str(path),
        "line": line,
        "column": column,
        "snippet": _line_snippet(source, line),
        "symbol_name": symbol_name,
        "backend": PYTHON_BACKEND,
    }


def _is_definition_node(node: ast.AST) -> bool:
    if isinstance(node, _DEFINITION_NODES):
        return True
    return isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))


def _scope_binds_name(scope: ast.AST | None, symbol_name: str) -> bool:
    if scope is None:
        return False
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = scope.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            if arg.arg == symbol_name:
                return True
        if args.vararg and args.vararg.arg == symbol_name:
            return True
        if args.kwarg and args.kwarg.arg == symbol_name:
            return True
    for child in ast.walk(scope):
        if child is not scope and isinstance(child, _SCOPE_NODES):
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)) and child.id == symbol_name:
            return True
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == symbol_name:
            return True
        if isinstance(child, ast.alias) and _symbol_name(child) == symbol_name:
            return True
    return False


def _definition_candidates(tree: ast.AST, symbol_name: str, target_scope: tuple[str, int, int] | None, parents: dict[ast.AST, ast.AST]) -> list[ast.AST]:
    local: list[ast.AST] = []
    module: list[ast.AST] = []
    for node in _candidate_symbol_nodes(tree):
        if _symbol_name(node) != symbol_name or not _is_definition_node(node):
            continue
        if _scope_key(node, parents) == target_scope:
            local.append(node)
        if _scope_key(node, parents) is None:
            module.append(node)
    return local if local else module


def _reference_matches_target_scope(
    node: ast.AST,
    *,
    symbol_name: str,
    target_scope: tuple[str, int, int] | None,
    parents: dict[ast.AST, ast.AST],
) -> bool:
    node_scope = _scope_key(node, parents)
    if target_scope is not None:
        return node_scope == target_scope
    if node_scope is None:
        return True
    scope = _nearest_scope(node, parents)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return not _scope_binds_name(scope, symbol_name)
    return False


def code_intel_capabilities(
    *, path: str | None = None, language: str | None = None, backend: str | None = None
) -> dict[str, Any]:
    normalized_language = _normalize_language(path=path, language=language)
    if normalized_language != "python":
        return _unsupported_language_result(normalized_language)
    return {
        "ok": True,
        "language": "python",
        "backend": backend or PYTHON_BACKEND,
        "external_server_required": False,
        "operations": dict(_PYTHON_CAPABILITIES),
        "supported_operations": list(SUPPORTED_OPERATIONS),
    }


def code_references(
    path: str,
    line: int,
    column: int,
    project_root: str | None = None,
    max_files: int = 50,
) -> dict[str, Any]:
    del project_root, max_files
    normalized_language = _normalize_language(path=path)
    if normalized_language != "python":
        return _unsupported_language_result(normalized_language, operation="references")
    source, tree, error = _parse_python(path)
    if error:
        error.setdefault("operation", "references")
        return error

    assert tree is not None
    parents = _build_parent_map(tree)
    target_node, symbol_name = _find_symbol_node(tree, source, line, column)
    if target_node is None or not symbol_name:
        return {
            "ok": False,
            "backend": PYTHON_BACKEND,
            "language": "python",
            "operation": "references",
            "code": "symbol_not_found",
            "error": "No symbol found at the requested position.",
        }

    target_scope = _scope_key(target_node, parents)
    candidates = _definition_candidates(tree, symbol_name, target_scope, parents)
    if candidates:
        target_scope = _scope_key(candidates[0], parents)

    references = []
    for node in _candidate_symbol_nodes(tree):
        if _symbol_name(node) != symbol_name:
            continue
        if not _reference_matches_target_scope(node, symbol_name=symbol_name, target_scope=target_scope, parents=parents):
            continue
        references.append(
            _reference_item(Path(path), source, symbol_name, getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
        )

    return {
        "ok": True,
        "backend": PYTHON_BACKEND,
        "language": "python",
        "operation": "references",
        "symbol_name": symbol_name,
        "references": references,
    }


def code_definition(path: str, line: int, column: int, project_root: str | None = None) -> dict[str, Any]:
    del project_root
    normalized_language = _normalize_language(path=path)
    if normalized_language != "python":
        return _unsupported_language_result(normalized_language, operation="definition")
    source, tree, error = _parse_python(path)
    if error:
        error.setdefault("operation", "definition")
        return error

    assert tree is not None
    parents = _build_parent_map(tree)
    target_node, symbol_name = _find_symbol_node(tree, source, line, column)
    if target_node is None or not symbol_name:
        return {
            "ok": False,
            "backend": PYTHON_BACKEND,
            "language": "python",
            "operation": "definition",
            "code": "symbol_not_found",
            "error": "No symbol found at the requested position.",
        }

    target_scope = _scope_key(target_node, parents)
    candidates = _definition_candidates(tree, symbol_name, target_scope, parents)
    if not candidates:
        return {
            "ok": False,
            "backend": PYTHON_BACKEND,
            "language": "python",
            "operation": "definition",
            "code": "definition_not_found",
            "error": "No definition found for symbol.",
            "symbol_name": symbol_name,
        }

    definition = min(candidates, key=lambda item: (getattr(item, "lineno", 0), getattr(item, "col_offset", 0)))
    item = _reference_item(
        Path(path),
        source,
        symbol_name,
        getattr(definition, "lineno", 0),
        getattr(definition, "col_offset", 0),
    )
    return {
        "ok": True,
        "backend": PYTHON_BACKEND,
        "language": "python",
        "operation": "definition",
        "symbol_name": symbol_name,
        "definition": item,
    }


def code_symbols(path: str, project_root: str | None = None) -> dict[str, Any]:
    del project_root
    normalized_language = _normalize_language(path=path)
    if normalized_language != "python":
        return _unsupported_language_result(normalized_language, operation="symbols")
    source, tree, error = _parse_python(path)
    if error:
        error.setdefault("operation", "symbols")
        return error

    assert tree is not None
    symbols = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(_reference_item(Path(path), source, node.name, node.lineno, node.col_offset))
    return {
        "ok": True,
        "backend": PYTHON_BACKEND,
        "language": "python",
        "operation": "symbols",
        "symbols": symbols,
    }


def code_diagnostics(path: str, project_root: str | None = None) -> dict[str, Any]:
    del project_root
    normalized_language = _normalize_language(path=path)
    if normalized_language != "python":
        return _unsupported_language_result(normalized_language, operation="diagnostics")
    source, tree, error = _parse_python(path)
    if error:
        return {
            "ok": True,
            "backend": PYTHON_BACKEND,
            "language": "python",
            "operation": "diagnostics",
            "diagnostics": [error["details"]],
        }
    assert tree is not None
    return {
        "ok": True,
        "backend": PYTHON_BACKEND,
        "language": "python",
        "operation": "diagnostics",
        "diagnostics": [],
    }


_CODE_REFERENCES_SCHEMA = {
    "name": "code_references",
    "description": "Find read-only Python symbol references using the bounded code-intel backend.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to a Python source file."},
            "line": {"type": "integer", "description": "1-based line number of the symbol occurrence."},
            "column": {"type": "integer", "description": "0-based column of the symbol occurrence."},
            "project_root": {"type": "string", "description": "Optional project root reserved for future backend breadth."},
            "max_files": {"type": "integer", "default": 50, "description": "Maximum files to inspect when a backend supports project references."},
        },
        "required": ["path", "line", "column"],
    },
}

_CODE_DEFINITION_SCHEMA = {
    "name": "code_definition",
    "description": "Go to a Python symbol definition using the bounded code-intel backend.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to a Python source file."},
            "line": {"type": "integer", "description": "1-based line number of the symbol occurrence."},
            "column": {"type": "integer", "description": "0-based column of the symbol occurrence."},
            "project_root": {"type": "string", "description": "Optional project root reserved for future backend breadth."},
        },
        "required": ["path", "line", "column"],
    },
}


def check_code_intel_requirements() -> bool:
    return True


registry.register(
    name="code_references",
    toolset="code_intel",
    schema=_CODE_REFERENCES_SCHEMA,
    handler=lambda args, **kwargs: tool_result(
        code_references(
            path=args.get("path", ""),
            line=args.get("line"),
            column=args.get("column"),
            project_root=args.get("project_root"),
            max_files=args.get("max_files", 50),
        )
    ),
    check_fn=check_code_intel_requirements,
    description=_CODE_REFERENCES_SCHEMA["description"],
    emoji="🔎",
)

registry.register(
    name="code_definition",
    toolset="code_intel",
    schema=_CODE_DEFINITION_SCHEMA,
    handler=lambda args, **kwargs: tool_result(
        code_definition(
            path=args.get("path", ""),
            line=args.get("line"),
            column=args.get("column"),
            project_root=args.get("project_root"),
        )
    ),
    check_fn=check_code_intel_requirements,
    description=_CODE_DEFINITION_SCHEMA["description"],
    emoji="🎯",
)
