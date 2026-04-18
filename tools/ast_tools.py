#!/usr/bin/env python3
"""Hermes-native structural code inspection for Python, JSON, JavaScript, and TypeScript."""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ("python", "javascript", "typescript", "json")
LANGUAGE_ALIASES = {
    "py": "python",
    "python": "python",
    "js": "javascript",
    "jsx": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "typescript": "typescript",
    "json": "json",
}
LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "json",
}

AST_LIST_DEFS_SCHEMA = {
    "name": "ast_list_defs",
    "description": "List structural definitions from local Python, JSON, JavaScript, or TypeScript files using Hermes-native parsers.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to a Python, JSON, JavaScript, or TypeScript source file"},
            "language": {
                "type": "string",
                "description": "Optional language override. Supported: python, javascript, typescript, json",
            },
            "include_nested": {"type": "boolean", "description": "Include nested class/function/method or nested structural definitions", "default": False},
        },
        "required": ["path"],
    },
}

AST_FIND_NODES_SCHEMA = {
    "name": "ast_find_nodes",
    "description": "Find structural nodes from local Python, JSON, JavaScript, or TypeScript files by node type and/or display name.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to a Python, JSON, JavaScript, or TypeScript source file"},
            "language": {
                "type": "string",
                "description": "Optional language override. Supported: python, javascript, typescript, json",
            },
            "node_type": {"type": "string", "description": "Optional structural node type, for example FunctionDef, ClassDeclaration, MethodDefinition, ObjectProperty"},
            "name": {"type": "string", "description": "Optional display-name filter (case-insensitive substring match)"},
            "include_source": {"type": "boolean", "description": "Include a source snippet or structural value preview for each match", "default": False},
            "max_results": {"type": "integer", "description": "Maximum number of matches to return", "default": 100},
        },
        "required": ["path"],
    },
}


_CLASS_RE = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)\b")
_FUNCTION_RE = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(")
_VARIABLE_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|[A-Za-z_$][\w$]*\s*=>)"
)
_INTERFACE_RE = re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)\b")
_TYPE_ALIAS_RE = re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\b")
_METHOD_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|static|readonly|abstract|override)\s+)*(?:async\s+)?([A-Za-z_$][\w$]*)\s*\("
)
_METHOD_EXCLUDED = {"if", "for", "while", "switch", "catch", "function", "class", "return"}


PYTHON_LIMITATIONS = []
JSON_LIMITATIONS = [
    "JSON support is structural and does not provide exact source spans.",
    "JSON nodes are limited to object properties and array items rather than executable AST constructs.",
]
JS_TS_LIMITATIONS = [
    "JavaScript/TypeScript support is heuristic and focuses on declarations, methods, and assignment-based function definitions.",
    "Heuristic parsing does not attempt full ECMAScript syntax coverage or call-expression traversal.",
]


def _check_ast_requirements() -> bool:
    return True


def _supported_languages() -> list[str]:
    return list(SUPPORTED_LANGUAGES)


def _normalize_language(language: str | None) -> str | None:
    if not language:
        return None
    return LANGUAGE_ALIASES.get(language.strip().lower())


def _detect_language(source_path: Path, language: str | None) -> str | None:
    normalized = _normalize_language(language)
    if language and not normalized:
        return None
    if normalized:
        return normalized
    return LANGUAGE_BY_SUFFIX.get(source_path.suffix.lower())


def _resolve_source_path(path: str) -> Path | None:
    source_path = Path(path).expanduser()
    if not source_path.exists():
        return None
    return source_path.resolve()


def _unsupported_language_error(path: Path, language: str | None = None) -> str:
    return tool_error(
        "Unsupported language for Hermes AST tooling v1",
        error_type="UnsupportedLanguage",
        path=str(path),
        detected_language=language or "unknown",
        supported_languages=_supported_languages(),
    )


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _attach_parents(node: ast.AST) -> None:
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "_hermes_parent", parent)


def _find_parent_name(node: ast.AST) -> str | None:
    current = getattr(node, "_hermes_parent", None)
    while current is not None:
        name = getattr(current, "name", None)
        if name:
            return name
        current = getattr(current, "_hermes_parent", None)
    return None


def _node_range(node: ast.AST) -> dict[str, Any]:
    return {
        "lineno": getattr(node, "lineno", None),
        "end_lineno": getattr(node, "end_lineno", None),
        "col_offset": getattr(node, "col_offset", None),
        "end_col_offset": getattr(node, "end_col_offset", None),
    }


def _expr_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expr_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _expr_name(node.func)
    if isinstance(node, ast.alias):
        return node.asname or node.name
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return None


def _decorators(node: ast.AST) -> list[str]:
    values = []
    for decorator in getattr(node, "decorator_list", []):
        values.append(_expr_name(decorator) or ast.dump(decorator, include_attributes=False))
    return values


def _serialize_named_node(node: ast.AST, parent: str | None) -> dict[str, Any]:
    payload = {
        "name": getattr(node, "name", None),
        "display_name": getattr(node, "name", None),
        "node_type": type(node).__name__,
        "parent": parent,
        "docstring": ast.get_docstring(node, clean=False) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else None,
        "decorators": _decorators(node),
        "is_async": isinstance(node, ast.AsyncFunctionDef),
    }
    payload.update(_node_range(node))
    return payload


def _is_named_definition(node: ast.AST) -> bool:
    return isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))


def _list_nested_defs(node: ast.AST, items: list[dict[str, Any]]) -> None:
    parent_name = getattr(node, "name", None)
    body = getattr(node, "body", [])
    for child in body:
        if _is_named_definition(child):
            items.append(_serialize_named_node(child, parent_name))
            _list_nested_defs(child, items)


def _display_name(node: ast.AST) -> str | None:
    name = getattr(node, "name", None)
    if isinstance(name, str):
        return name
    return _expr_name(node)


def _serialize_match(node: ast.AST, source: str, include_source: bool) -> dict[str, Any]:
    payload = {
        "node_type": type(node).__name__,
        "name": getattr(node, "name", None),
        "display_name": _display_name(node),
        "parent": _find_parent_name(node),
    }
    payload.update(_node_range(node))
    if include_source:
        payload["source"] = ast.get_source_segment(source, node)
    return payload


def _parse_python_module(path: Path) -> dict[str, Any] | str:
    try:
        source = _read_source(path)
        module = ast.parse(source, filename=str(path), type_comments=True)
        _attach_parents(module)
        return {
            "path": path,
            "source": source,
            "module": module,
            "language": "python",
            "parser": "python_ast",
            "confidence": "high",
            "limitations": list(PYTHON_LIMITATIONS),
        }
    except SyntaxError as exc:
        return tool_error(
            f"SyntaxError: {exc.msg}",
            error_type="SyntaxError",
            path=str(path),
            lineno=exc.lineno,
            offset=exc.offset,
            text=exc.text,
        )
    except Exception as exc:
        logger.debug("AST parse failed", exc_info=True)
        return tool_error(str(exc), error_type=type(exc).__name__, path=str(path))


def _json_value_type(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def _json_preview(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_node(
    *,
    name: str,
    node_type: str,
    parent: str | None,
    json_path: str,
    value: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "display_name": name,
        "node_type": node_type,
        "parent": parent,
        "json_path": json_path,
        "value_type": _json_value_type(value),
        "source": _json_preview(value),
        "lineno": None,
        "end_lineno": None,
        "col_offset": None,
        "end_col_offset": None,
    }


def _json_child_path(base_path: str, child: str) -> str:
    if base_path == "$":
        return f"$.{child}"
    return f"{base_path}.{child}"


def _collect_json_nodes(value: Any, *, parent: str | None, json_path: str, items: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = _json_child_path(json_path, key)
            items.append(
                _json_node(
                    name=key,
                    node_type="ObjectProperty",
                    parent=parent,
                    json_path=child_path,
                    value=child,
                )
            )
            if isinstance(child, (dict, list)):
                _collect_json_nodes(child, parent=key, json_path=child_path, items=items)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            item_name = f"[{index}]"
            child_path = f"{json_path}{item_name}"
            items.append(
                _json_node(
                    name=item_name,
                    node_type="ArrayItem",
                    parent=parent,
                    json_path=child_path,
                    value=child,
                )
            )
            if isinstance(child, (dict, list)):
                _collect_json_nodes(child, parent=item_name, json_path=child_path, items=items)


def _parse_json_document(path: Path) -> dict[str, Any] | str:
    try:
        source = _read_source(path)
        document = json.loads(source)
    except json.JSONDecodeError as exc:
        return tool_error(
            f"JSONDecodeError: {exc.msg}",
            error_type="JSONDecodeError",
            path=str(path),
            lineno=exc.lineno,
            colno=exc.colno,
            pos=exc.pos,
        )
    except Exception as exc:
        logger.debug("JSON parse failed", exc_info=True)
        return tool_error(str(exc), error_type=type(exc).__name__, path=str(path))

    definitions: list[dict[str, Any]] = []
    _collect_json_nodes(document, parent=None, json_path="$", items=definitions)
    return {
        "path": path,
        "source": source,
        "document": document,
        "definitions": definitions,
        "language": "json",
        "parser": "json_structure_v1",
        "confidence": "high",
        "limitations": list(JSON_LIMITATIONS),
    }


def _strip_js_comments(source: str) -> list[str]:
    cleaned_lines = []
    in_block_comment = False
    for raw_line in source.splitlines():
        line = raw_line
        if in_block_comment:
            if "*/" in line:
                line = line.split("*/", 1)[1]
                in_block_comment = False
            else:
                cleaned_lines.append("")
                continue
        while "/*" in line:
            before, after = line.split("/*", 1)
            if "*/" in after:
                after = after.split("*/", 1)[1]
                line = before + " " + after
            else:
                line = before
                in_block_comment = True
                break
        if "//" in line:
            line = line.split("//", 1)[0]
        cleaned_lines.append(line)
    return cleaned_lines


def _count_braces(line: str) -> tuple[int, int]:
    return line.count("{"), line.count("}")


def _line_has_block_start(line: str) -> bool:
    return bool(re.search(r"\{\s*$", line))


def _heuristic_node(
    *,
    name: str,
    node_type: str,
    parent: str | None,
    line_number: int,
    raw_line: str,
    is_async: bool = False,
) -> dict[str, Any]:
    col_offset = raw_line.find(name)
    return {
        "name": name,
        "display_name": name,
        "node_type": node_type,
        "parent": parent,
        "docstring": None,
        "decorators": [],
        "is_async": is_async,
        "lineno": line_number,
        "end_lineno": line_number,
        "col_offset": col_offset if col_offset >= 0 else None,
        "end_col_offset": (col_offset + len(name)) if col_offset >= 0 else None,
        "source": raw_line.strip() or None,
    }


def _match_js_ts_declaration(line: str, *, inside_class: bool) -> tuple[str, str, bool, bool] | None:
    match = _CLASS_RE.match(line)
    if match:
        return match.group(1), "ClassDeclaration", _line_has_block_start(line), False

    match = _FUNCTION_RE.match(line)
    if match:
        return match.group(1), "FunctionDeclaration", _line_has_block_start(line), "async function" in line

    match = _VARIABLE_RE.match(line)
    if match:
        return match.group(1), "VariableDeclarator", _line_has_block_start(line), bool(re.search(r"=\s*async\b", line))

    match = _INTERFACE_RE.match(line)
    if match:
        return match.group(1), "InterfaceDeclaration", _line_has_block_start(line), False

    match = _TYPE_ALIAS_RE.match(line)
    if match:
        return match.group(1), "TypeAliasDeclaration", False, False

    if inside_class:
        match = _METHOD_RE.match(line)
        if match and match.group(1) not in _METHOD_EXCLUDED:
            return match.group(1), "MethodDefinition", _line_has_block_start(line), line.lstrip().startswith("async ") or " async " in line

    return None


def _parse_js_ts_structure(path: Path, language: str) -> dict[str, Any] | str:
    try:
        source = _read_source(path)
    except Exception as exc:
        logger.debug("JS/TS read failed", exc_info=True)
        return tool_error(str(exc), error_type=type(exc).__name__, path=str(path))

    cleaned_lines = _strip_js_comments(source)
    raw_lines = source.splitlines()
    definitions: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    depth = 0

    for line_number, (raw_line, line) in enumerate(zip(raw_lines, cleaned_lines), start=1):
        while stack and depth <= stack[-1]["depth"]:
            stack.pop()

        parent = stack[-1]["name"] if stack else None
        inside_class = bool(stack) and stack[-1]["node_type"] == "ClassDeclaration"
        matched = _match_js_ts_declaration(line, inside_class=inside_class)
        if matched:
            name, node_type, opens_block, is_async = matched
            item = _heuristic_node(
                name=name,
                node_type=node_type,
                parent=parent,
                line_number=line_number,
                raw_line=raw_line,
                is_async=is_async,
            )
            definitions.append(item)
            if opens_block:
                stack.append({"name": name, "node_type": node_type, "depth": depth})

        open_braces, close_braces = _count_braces(line)
        depth += open_braces - close_braces
        if depth < 0:
            depth = 0

    return {
        "path": path,
        "source": source,
        "definitions": definitions,
        "language": language,
        "parser": "hermes_js_structure_v1",
        "confidence": "medium",
        "limitations": list(JS_TS_LIMITATIONS),
    }


def _parse_source(path: str, language: str | None) -> dict[str, Any] | str:
    source_path = _resolve_source_path(path)
    if source_path is None:
        return tool_error(f"File not found: {path}", path=str(Path(path).expanduser()))

    detected_language = _detect_language(source_path, language)
    if detected_language == "python":
        return _parse_python_module(source_path)
    if detected_language == "json":
        return _parse_json_document(source_path)
    if detected_language in {"javascript", "typescript"}:
        return _parse_js_ts_structure(source_path, detected_language)
    return _unsupported_language_error(source_path, language=detected_language)


def _result_metadata(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(parsed["path"]),
        "language": parsed["language"],
        "parser": parsed["parser"],
        "confidence": parsed["confidence"],
        "limitations": parsed["limitations"],
    }


def _structural_matches(
    definitions: list[dict[str, Any]],
    *,
    node_type: str | None,
    name: str | None,
    include_source: bool,
    max_results: int,
) -> list[dict[str, Any]]:
    target_name = name.lower() if name else None
    matches: list[dict[str, Any]] = []
    for item in definitions:
        if node_type and item.get("node_type") != node_type:
            continue
        display_name = item.get("display_name") or item.get("name")
        if target_name and not display_name:
            continue
        if target_name and target_name not in display_name.lower():
            continue
        match = {
            key: value
            for key, value in item.items()
            if include_source or key != "source"
        }
        matches.append(match)
        if len(matches) >= max_results:
            break
    return matches


def ast_list_defs_tool(*, path: str, include_nested: bool = False, language: str | None = None) -> str:
    parsed = _parse_source(path, language)
    if isinstance(parsed, str):
        return parsed

    if parsed["language"] == "python":
        module = parsed["module"]
        definitions = []
        for node in module.body:
            if _is_named_definition(node):
                definitions.append(_serialize_named_node(node, None))
                if include_nested:
                    _list_nested_defs(node, definitions)
    else:
        definitions = parsed["definitions"] if include_nested else [
            item for item in parsed["definitions"] if item.get("parent") is None
        ]

    payload = {
        "ok": True,
        **_result_metadata(parsed),
        "definitions": definitions,
    }
    return tool_result(payload)


def ast_find_nodes_tool(
    *,
    path: str,
    node_type: str | None = None,
    name: str | None = None,
    include_source: bool = False,
    max_results: int = 100,
    language: str | None = None,
) -> str:
    parsed = _parse_source(path, language)
    if isinstance(parsed, str):
        return parsed

    if not node_type and not name:
        return tool_error("Provide at least one filter: node_type or name", path=str(parsed["path"]))

    if parsed["language"] == "python":
        source = parsed["source"]
        module = parsed["module"]
        target_name = name.lower() if name else None
        matches = []
        for node in ast.walk(module):
            if node_type and type(node).__name__ != node_type:
                continue
            display_name = _display_name(node)
            if target_name and not display_name:
                continue
            if target_name and target_name not in display_name.lower():
                continue
            matches.append(_serialize_match(node, source, include_source))
            if len(matches) >= max_results:
                break
    else:
        matches = _structural_matches(
            parsed["definitions"],
            node_type=node_type,
            name=name,
            include_source=include_source,
            max_results=max_results,
        )

    payload = {
        "ok": True,
        **_result_metadata(parsed),
        "count": len(matches),
        "matches": matches,
    }
    return tool_result(payload)


AST_TOOLS = [
    {"name": "ast_list_defs", "function": ast_list_defs_tool},
    {"name": "ast_find_nodes", "function": ast_find_nodes_tool},
]


def _handle_ast_list_defs(args, **_kwargs):
    return ast_list_defs_tool(
        path=args.get("path", ""),
        include_nested=args.get("include_nested", False),
        language=args.get("language"),
    )


def _handle_ast_find_nodes(args, **_kwargs):
    return ast_find_nodes_tool(
        path=args.get("path", ""),
        language=args.get("language"),
        node_type=args.get("node_type"),
        name=args.get("name"),
        include_source=args.get("include_source", False),
        max_results=args.get("max_results", 100),
    )


registry.register(
    name="ast_list_defs",
    toolset="code_intel",
    schema=AST_LIST_DEFS_SCHEMA,
    handler=_handle_ast_list_defs,
    check_fn=_check_ast_requirements,
    emoji="🌳",
)
registry.register(
    name="ast_find_nodes",
    toolset="code_intel",
    schema=AST_FIND_NODES_SCHEMA,
    handler=_handle_ast_find_nodes,
    check_fn=_check_ast_requirements,
    emoji="🔍",
)
