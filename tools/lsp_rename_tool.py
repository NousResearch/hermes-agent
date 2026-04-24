#!/usr/bin/env python3
"""Bounded first-class code-intel rename tool."""

from __future__ import annotations

import ast
import difflib
import io
import keyword
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tools import file_state
from tools.file_tools import (
    _check_file_staleness,
    _get_stale_edit_mode,
    _resolve_path_for_task,
    _strict_stale_error,
    _update_read_timestamp,
)
from tools.registry import registry, tool_error, tool_result

SUPPORTED_LANGUAGES = ["python"]
DEFAULT_MAX_FILES = 1
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_BLOCKED_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_COMPREHENSION_NODES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


LSP_RENAME_SCHEMA = {
    "name": "lsp_rename",
    "description": (
        "Safely rename a Python local identifier using bounded single-file code intelligence. "
        "Supports preview mode via apply=false and refuses unsafe targets like attributes, "
        "globals, and unsupported languages."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the source file containing the symbol to rename.",
            },
            "line": {
                "type": "integer",
                "description": "1-based line number of the identifier occurrence.",
            },
            "column": {
                "type": "integer",
                "description": "0-based column of the identifier occurrence. A 1-based fallback is also tolerated.",
            },
            "new_name": {
                "type": "string",
                "description": "New identifier name.",
            },
            "apply": {
                "type": "boolean",
                "default": True,
                "description": "When false, return a diff preview and do not modify files.",
            },
            "project_root": {
                "type": "string",
                "description": "Optional project root. Current MVP stays bounded to a single Python file.",
            },
            "max_files": {
                "type": "integer",
                "default": DEFAULT_MAX_FILES,
                "minimum": 1,
                "description": "Maximum number of files the rename may touch. Current AST fallback is bounded to 1.",
            },
            "language": {
                "type": "string",
                "description": "Optional language override. Defaults to auto-detection from the file path.",
            },
        },
        "required": ["path", "line", "column", "new_name"],
    },
}


@dataclass(frozen=True)
class Replacement:
    start: tuple[int, int]
    end: tuple[int, int]
    new_text: str


def _detect_language(path: Path, language: str | None) -> str | None:
    if language:
        return language.lower()
    if path.suffix.lower() == ".py":
        return "python"
    return None


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


def _collect_name_like_nodes(tree: ast.AST) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.arg, ast.Attribute)):
            nodes.append(node)
    return sorted(
        nodes,
        key=lambda item: (
            getattr(item, "lineno", 0),
            getattr(item, "col_offset", 0),
            -(getattr(item, "end_lineno", 0) or 0),
            -(getattr(item, "end_col_offset", 0) or 0),
        ),
    )


def _find_target_node(tree: ast.AST, line: int, column: int) -> ast.AST | None:
    for candidate_column in dict.fromkeys([column, column - 1]):
        if candidate_column < 0:
            continue
        best: ast.AST | None = None
        for node in _collect_name_like_nodes(tree):
            if _node_contains(node, line, candidate_column):
                best = node
        if best is not None:
            return best
    return None


def _nearest_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST | None:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, _SCOPE_NODES):
            return current
    return None


def _find_target_identifier_at_cursor(source: str, line: int, column: int) -> tuple[str | None, str | None]:
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.NAME:
            continue
        if token.start[0] != line:
            continue
        for candidate_column in dict.fromkeys([column, column - 1]):
            if candidate_column < 0:
                continue
            if token.start[1] <= candidate_column < token.end[1]:
                line_text = source.splitlines()[line - 1]
                before = line_text[: token.start[1]].rstrip()
                if before.endswith("."):
                    return token.string, "attribute_access"
                return token.string, None
    return None, None


def _get_scope_globals_and_nonlocals(scope: ast.AST) -> tuple[set[str], set[str]]:
    global_names: set[str] = set()
    nonlocal_names: set[str] = set()
    for node in getattr(scope, "body", []):
        if isinstance(node, ast.Global):
            global_names.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            nonlocal_names.update(node.names)
    return global_names, nonlocal_names


def _scope_local_bindings(scope: ast.AST) -> set[str]:
    bindings: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = scope.args
        for arg in [
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
        ]:
            bindings.add(arg.arg)
        if args.vararg:
            bindings.add(args.vararg.arg)
        if args.kwarg:
            bindings.add(args.kwarg.arg)
    for node in ast.walk(scope):
        if node is not scope and isinstance(node, _BLOCKED_SCOPE_NODES):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bindings.add(node.id)
    return bindings


def _scope_has_comprehension(scope: ast.AST) -> bool:
    for node in ast.walk(scope):
        if isinstance(node, _COMPREHENSION_NODES):
            return True
    return False


def _nested_scope_references_name(scope: ast.AST, target_name: str) -> bool:
    for node in ast.iter_child_nodes(scope):
        if isinstance(node, _BLOCKED_SCOPE_NODES):
            for nested in ast.walk(node):
                if isinstance(nested, ast.Name) and nested.id == target_name:
                    return True
                if isinstance(nested, ast.arg) and nested.arg == target_name:
                    return True
            continue
        if _nested_scope_references_name(node, target_name):
            return True
    return False


def _signature_expression_references_name(scope: ast.AST, target_name: str) -> bool:
    """Return True when signature-time expressions mention target_name.

    Python evaluates annotations/defaults outside normal local variable use
    semantics. Rewriting those names during a local rename can silently alter
    global/enclosing references, so the bounded fallback refuses these cases
    rather than pretending to be a full language server.
    """
    expression_nodes: list[ast.AST | None] = []
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = scope.args
        expression_nodes.extend(args.defaults)
        expression_nodes.extend(item for item in args.kw_defaults if item is not None)
        expression_nodes.append(args.vararg.annotation if args.vararg else None)
        expression_nodes.append(args.kwarg.annotation if args.kwarg else None)
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            expression_nodes.append(arg.annotation)
        expression_nodes.append(scope.returns)
    elif isinstance(scope, ast.Lambda):
        args = scope.args
        expression_nodes.extend(args.defaults)
        expression_nodes.extend(item for item in args.kw_defaults if item is not None)

    for expression in expression_nodes:
        if expression is None:
            continue
        for node in ast.walk(expression):
            if isinstance(node, ast.Name) and node.id == target_name:
                return True
    return False


def _iter_scope_name_occurrences(scope: ast.AST, target_name: str) -> Iterable[ast.AST]:
    def visit(node: ast.AST, *, is_root: bool = False):
        if not is_root and isinstance(node, _BLOCKED_SCOPE_NODES):
            return
        if isinstance(node, ast.Name) and node.id == target_name:
            yield node
        elif isinstance(node, ast.arg) and node.arg == target_name:
            yield node
        for child in ast.iter_child_nodes(node):
            yield from visit(child)

    yield from visit(scope, is_root=True)


def _replacement_for_node(node: ast.AST, new_name: str) -> Replacement:
    if isinstance(node, ast.Name):
        return Replacement((node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset), new_name)
    if isinstance(node, ast.arg):
        start_col = node.col_offset
        end_col = start_col + len(node.arg)
        return Replacement((node.lineno, start_col), (node.lineno, end_col), new_name)
    raise TypeError(f"Unsupported replacement node: {type(node)!r}")


def _offsets_for_lines(source: str) -> list[int]:
    offsets = [0]
    running = 0
    for line in source.splitlines(keepends=True):
        running += len(line)
        offsets.append(running)
    return offsets


def _to_index(offsets: list[int], position: tuple[int, int]) -> int:
    line, column = position
    return offsets[line - 1] + column


def _apply_replacements(source: str, replacements: list[Replacement]) -> str:
    offsets = _offsets_for_lines(source)
    ordered = sorted(
        replacements,
        key=lambda item: (_to_index(offsets, item.start), _to_index(offsets, item.end)),
        reverse=True,
    )
    updated = source
    for replacement in ordered:
        start = _to_index(offsets, replacement.start)
        end = _to_index(offsets, replacement.end)
        updated = updated[:start] + replacement.new_text + updated[end:]
    return updated


def _make_diff(path: Path, old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )


def _python_single_file_rename(
    path: Path,
    source: str,
    *,
    line: int,
    column: int,
    new_name: str,
    apply: bool,
    max_files: int,
    project_root: str | None,
    task_id: str,
) -> str:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return tool_error(
            "Failed to parse Python source for rename.",
            code="parse_error",
            details=str(exc),
        )

    parents = _build_parent_map(tree)
    target_node = _find_target_node(tree, line, column)
    token_name, token_reason = _find_target_identifier_at_cursor(source, line, column)
    if token_reason == "attribute_access":
        return tool_error(
            "Refusing unsafe rename target.",
            code="unsafe_target",
            reason="attribute_access",
            target_name=token_name,
        )
    if isinstance(target_node, ast.Attribute):
        return tool_error(
            "Refusing unsafe rename target.",
            code="unsafe_target",
            reason="attribute_access",
        )
    if not isinstance(target_node, (ast.Name, ast.arg)):
        return tool_error(
            "Could not resolve a supported identifier at the requested position.",
            code="target_not_found",
        )

    target_name = target_node.id if isinstance(target_node, ast.Name) else target_node.arg
    scope = _nearest_scope(target_node, parents)
    if scope is None:
        return tool_error(
            "Refusing unsafe rename target.",
            code="unsafe_target",
            reason="module_scope_not_supported",
            target_name=target_name,
        )

    global_names, nonlocal_names = _get_scope_globals_and_nonlocals(scope)
    if target_name in global_names:
        return tool_error(
            "Refusing unsafe rename target.",
            code="unsafe_target",
            reason="global_binding",
            target_name=target_name,
        )
    if target_name in nonlocal_names:
        return tool_error(
            "Refusing unsafe rename target.",
            code="unsafe_target",
            reason="nonlocal_binding",
            target_name=target_name,
        )

    bindings = _scope_local_bindings(scope)
    if target_name not in bindings:
        return tool_error(
            "Refusing unsafe rename target.",
            code="unsafe_target",
            reason="not_local_identifier",
            target_name=target_name,
        )
    if new_name != target_name and new_name in bindings:
        return tool_error(
            "Refusing unsafe rename target.",
            code="unsafe_target",
            reason="binding_collision",
            target_name=target_name,
            new_name=new_name,
        )
    if _scope_has_comprehension(scope):
        return tool_error(
            "Refusing unsafe rename target.",
            code="unsafe_target",
            reason="comprehension_scope_not_supported",
            target_name=target_name,
        )
    if _nested_scope_references_name(scope, target_name):
        return tool_error(
            "Refusing unsafe rename target.",
            code="unsafe_target",
            reason="nested_scope_reference_not_supported",
            target_name=target_name,
        )
    if _signature_expression_references_name(scope, target_name):
        return tool_error(
            "Refusing unsafe rename target.",
            code="unsafe_target",
            reason="signature_expression_reference_not_supported",
            target_name=target_name,
        )

    replacements = [_replacement_for_node(node, new_name) for node in _iter_scope_name_occurrences(scope, target_name)]
    if not replacements:
        return tool_error(
            "No safe rename edits were produced.",
            code="no_edits",
            target_name=target_name,
        )

    updated_source = _apply_replacements(source, replacements)
    diff = _make_diff(path, source, updated_source)
    if updated_source == source:
        return tool_result(
            success=True,
            applied=False,
            engine="python_ast_single_file",
            path=str(path),
            target_name=target_name,
            new_name=new_name,
            changed_files=0,
            max_files=max_files,
            project_root=project_root,
            diff="",
        )

    if apply:
        resolved_path = str(path)
        with file_state.lock_path(resolved_path):
            cross_warning = file_state.check_stale(task_id, resolved_path)
            stale_warning = cross_warning or _check_file_staleness(str(path), task_id)
            strict_error = None
            if _get_stale_edit_mode() == "strict":
                strict_error = _strict_stale_error(str(path), task_id, cross_warning=cross_warning)
            if strict_error:
                return tool_error(strict_error, code="stale_edit_blocked", path=str(path))
            path.write_text(updated_source, encoding="utf-8")
            _update_read_timestamp(str(path), task_id)
            file_state.note_write(task_id, resolved_path)
    else:
        stale_warning = None

    result = tool_result(
        success=True,
        applied=apply,
        engine="python_ast_single_file",
        path=str(path),
        target_name=target_name,
        new_name=new_name,
        changed_files=1,
        max_files=max_files,
        project_root=project_root,
        diff=diff,
        files=[str(path)],
    )
    if stale_warning:
        data = tool_result(
            success=True,
            applied=apply,
            engine="python_ast_single_file",
            path=str(path),
            target_name=target_name,
            new_name=new_name,
            changed_files=1,
            max_files=max_files,
            project_root=project_root,
            diff=diff,
            files=[str(path)],
            _warning=stale_warning,
        )
        return data
    return result


def lsp_rename_tool(
    *,
    path: str,
    line: int,
    column: int,
    new_name: str,
    apply: bool = True,
    project_root: str | None = None,
    max_files: int = DEFAULT_MAX_FILES,
    language: str | None = None,
    task_id: str = "default",
) -> str:
    file_path = _resolve_path_for_task(path, task_id)
    if project_root:
        root_path = _resolve_path_for_task(project_root, task_id)
        try:
            file_path.relative_to(root_path)
        except ValueError:
            return tool_error(
                "Path is outside project_root.",
                code="path_outside_project_root",
                path=str(file_path),
                project_root=str(root_path),
            )
    if not file_path.exists() or not file_path.is_file():
        return tool_error("Path does not exist or is not a file.", code="path_not_found", path=str(file_path))
    if not isinstance(line, int) or line < 1:
        return tool_error("line must be a positive integer.", code="invalid_line")
    if not isinstance(column, int) or column < 0:
        return tool_error("column must be a non-negative integer.", code="invalid_column")
    if not isinstance(max_files, int) or max_files < 1:
        return tool_error("max_files must be >= 1.", code="invalid_max_files")
    if max_files > DEFAULT_MAX_FILES:
        return tool_error(
            "Workspace rename is not supported by bounded lsp_rename.",
            code="workspace_rename_not_supported",
            max_files=max_files,
            supported_max_files=DEFAULT_MAX_FILES,
        )
    if not isinstance(new_name, str) or not new_name.isidentifier() or keyword.iskeyword(new_name):
        return tool_error("Invalid Python identifier for rename target.", code="invalid_new_name")

    detected_language = _detect_language(file_path, language)
    if detected_language not in SUPPORTED_LANGUAGES:
        return tool_error(
            "Unsupported language for lsp_rename.",
            code="unsupported_language",
            language=detected_language or language or "unknown",
            supported_languages=SUPPORTED_LANGUAGES,
        )

    source = file_path.read_text(encoding="utf-8")
    return _python_single_file_rename(
        file_path,
        source,
        line=line,
        column=column,
        new_name=new_name,
        apply=apply,
        max_files=max_files,
        project_root=str(root_path) if project_root else None,
        task_id=task_id,
    )


def check_lsp_rename_requirements() -> bool:
    return True


registry.register(
    name="lsp_rename",
    toolset="code_intel",
    schema=LSP_RENAME_SCHEMA,
    handler=lambda args, **kwargs: lsp_rename_tool(
        path=args.get("path", ""),
        line=args.get("line"),
        column=args.get("column"),
        new_name=args.get("new_name", ""),
        apply=args.get("apply", True),
        project_root=args.get("project_root"),
        max_files=args.get("max_files", DEFAULT_MAX_FILES),
        language=args.get("language"),
        task_id=kwargs.get("task_id", "default"),
    ),
    check_fn=check_lsp_rename_requirements,
    description=LSP_RENAME_SCHEMA["description"],
    emoji="✏️",
)
