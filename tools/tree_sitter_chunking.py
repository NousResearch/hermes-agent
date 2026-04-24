from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TREE_SITTER_EXPANSION_MAX_CHARS = 32 * 1024
TREE_SITTER_EXPANSION_MAX_LINES = 300

_LANGUAGE_BY_EXTENSION = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".md": "markdown",
    ".markdown": "markdown",
}

_BLOCK_NODE_TYPES = {
    "javascript": {
        "function_declaration",
        "generator_function_declaration",
        "method_definition",
        "class_declaration",
        "class",
        "arrow_function",
        "function_expression",
    },
    "typescript": {
        "function_declaration",
        "generator_function_declaration",
        "method_definition",
        "class_declaration",
        "class",
        "arrow_function",
        "function_expression",
        "abstract_class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "module",
        "internal_module",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
    },
    "markdown": {
        "section",
        "fenced_code_block",
    },
}


@dataclass(frozen=True)
class SyntaxChunkResult:
    start_line: int
    end_line: int
    language: str | None
    strategy: str
    fallback_reason: str | None = None


def supported_language_for_path(path: str) -> str | None:
    return _LANGUAGE_BY_EXTENSION.get(Path(path).suffix.lower())


def maybe_expand_syntax_read_window(
    *,
    path: str,
    source: str,
    requested_start_line: int,
    requested_end_line: int,
    limit: int,
) -> SyntaxChunkResult:
    language = supported_language_for_path(path)
    if language is None:
        return SyntaxChunkResult(
            start_line=requested_start_line,
            end_line=requested_end_line,
            language=None,
            strategy="line",
            fallback_reason="unsupported_language",
        )

    parser = _get_parser_for_language(language)
    if parser is None:
        return SyntaxChunkResult(
            start_line=requested_start_line,
            end_line=requested_end_line,
            language=language,
            strategy="line",
            fallback_reason="tree_sitter_unavailable",
        )

    try:
        tree = parser.parse(source.encode("utf-8"))
    except Exception:
        return SyntaxChunkResult(
            start_line=requested_start_line,
            end_line=requested_end_line,
            language=language,
            strategy="line",
            fallback_reason="tree_sitter_parse_failed",
        )

    root_node = getattr(tree, "root_node", None)
    if root_node is None:
        return SyntaxChunkResult(
            start_line=requested_start_line,
            end_line=requested_end_line,
            language=language,
            strategy="line",
            fallback_reason="tree_missing_root",
        )

    ranges = _collect_candidate_ranges(root_node, language)
    if not ranges:
        return SyntaxChunkResult(
            start_line=requested_start_line,
            end_line=requested_end_line,
            language=language,
            strategy="line",
            fallback_reason="no_syntax_blocks_found",
        )

    expanded_start = requested_start_line
    expanded_end = requested_end_line
    for boundary_line in (requested_start_line, requested_end_line):
        container = _smallest_containing_range(ranges, boundary_line)
        if container is None:
            continue
        expanded_start = min(expanded_start, container[0])
        expanded_end = max(expanded_end, container[1])

    if expanded_start == requested_start_line and expanded_end == requested_end_line:
        return SyntaxChunkResult(
            start_line=requested_start_line,
            end_line=requested_end_line,
            language=language,
            strategy="line",
            fallback_reason="no_boundary_expansion_needed",
        )

    expanded_line_count = expanded_end - expanded_start + 1
    if expanded_line_count > max(limit * 2, TREE_SITTER_EXPANSION_MAX_LINES):
        return SyntaxChunkResult(
            start_line=requested_start_line,
            end_line=requested_end_line,
            language=language,
            strategy="line",
            fallback_reason="expanded_window_too_large",
        )

    raw_lines = source.splitlines()[expanded_start - 1 : expanded_end]
    if len("\n".join(raw_lines)) > TREE_SITTER_EXPANSION_MAX_CHARS:
        return SyntaxChunkResult(
            start_line=requested_start_line,
            end_line=requested_end_line,
            language=language,
            strategy="line",
            fallback_reason="expanded_window_too_large",
        )

    return SyntaxChunkResult(
        start_line=expanded_start,
        end_line=expanded_end,
        language=language,
        strategy="tree_sitter",
        fallback_reason=None,
    )


def _collect_candidate_ranges(root_node, language: str) -> list[tuple[int, int]]:
    block_types = _BLOCK_NODE_TYPES.get(language, set())
    ranges: list[tuple[int, int]] = []
    for node in _walk_nodes(root_node):
        node_type = getattr(node, "type", None)
        if node_type not in block_types:
            continue
        start_line = _point_to_line(getattr(node, "start_point", None))
        end_line = _point_to_line(getattr(node, "end_point", None))
        if start_line is None or end_line is None or end_line < start_line:
            continue
        ranges.append((start_line, end_line))
    return ranges


def _walk_nodes(node) -> Iterable:
    yield node
    for child in getattr(node, "children", []) or []:
        yield from _walk_nodes(child)


def _point_to_line(point) -> int | None:
    if not isinstance(point, tuple) or not point:
        return None
    try:
        return int(point[0]) + 1
    except (TypeError, ValueError):
        return None


def _smallest_containing_range(ranges: list[tuple[int, int]], line_number: int) -> tuple[int, int] | None:
    containing = [span for span in ranges if span[0] <= line_number <= span[1]]
    if not containing:
        return None
    return min(containing, key=lambda span: (span[1] - span[0], -span[0]))


def _get_parser_for_language(language: str):
    if language == "javascript":
        return _build_parser(("tree_sitter_javascript", ("language", "LANGUAGE")))
    if language == "typescript":
        return _build_parser(
            (
                "tree_sitter_typescript",
                (
                    "language_typescript",
                    "typescript_language",
                    "language",
                    "LANGUAGE_TYPESCRIPT",
                    "LANGUAGE",
                ),
            )
        )
    if language == "go":
        return _build_parser(("tree_sitter_go", ("language", "LANGUAGE")))
    if language == "rust":
        return _build_parser(("tree_sitter_rust", ("language", "LANGUAGE")))
    if language == "markdown":
        return _build_parser(
            ("tree_sitter_markdown", ("language", "LANGUAGE")),
            ("tree_sitter_md", ("language", "LANGUAGE")),
        )
    return None


def _build_parser(*grammar_candidates):
    try:
        from tree_sitter import Parser
    except Exception:
        return None

    language_obj = None
    for module_name, attr_names in grammar_candidates:
        try:
            module = __import__(module_name, fromlist=["__name__"])
        except Exception:
            continue
        for attr_name in attr_names:
            if not hasattr(module, attr_name):
                continue
            candidate = getattr(module, attr_name)
            try:
                language_obj = candidate() if callable(candidate) else candidate
            except Exception:
                continue
            if language_obj is not None:
                break
        if language_obj is not None:
            break

    if language_obj is None:
        return None

    try:
        parser = Parser()
    except Exception:
        return None

    try:
        if hasattr(parser, "set_language"):
            parser.set_language(language_obj)
        else:
            parser.language = language_obj
    except Exception:
        return None
    return parser
