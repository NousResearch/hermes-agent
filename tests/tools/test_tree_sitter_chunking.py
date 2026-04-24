import textwrap

from tools.tree_sitter_chunking import SyntaxChunkResult, maybe_expand_syntax_read_window


class FakeNode:
    def __init__(self, node_type, start_line, end_line, children=None):
        self.type = node_type
        self.start_point = (start_line - 1, 0)
        self.end_point = (end_line - 1, 0)
        self.children = children or []


class FakeTree:
    def __init__(self, root_node):
        self.root_node = root_node


class FakeParser:
    def __init__(self, root_node):
        self._root_node = root_node

    def parse(self, _source_bytes):
        return FakeTree(self._root_node)


def test_missing_parser_returns_requested_window_with_fallback_metadata(monkeypatch):
    source = textwrap.dedent(
        """
        const before = 1;
        function demo() {
          return before + 1;
        }
        const after = 2;
        """
    ).lstrip()

    monkeypatch.setattr("tools.tree_sitter_chunking._get_parser_for_language", lambda _language: None)

    result = maybe_expand_syntax_read_window(
        path="demo.js",
        source=source,
        requested_start_line=3,
        requested_end_line=3,
        limit=1,
    )

    assert result == SyntaxChunkResult(
        start_line=3,
        end_line=3,
        language="javascript",
        strategy="line",
        fallback_reason="tree_sitter_unavailable",
    )


def test_javascript_function_boundary_expands_when_parser_available(monkeypatch):
    source = textwrap.dedent(
        """
        function demo() {
          const value = 1;
          return value;
        }
        const after = 2;
        """
    ).lstrip()
    root = FakeNode(
        "program",
        1,
        5,
        children=[
            FakeNode("function_declaration", 1, 4),
            FakeNode("lexical_declaration", 5, 5),
        ],
    )

    monkeypatch.setattr(
        "tools.tree_sitter_chunking._get_parser_for_language",
        lambda _language: FakeParser(root),
    )

    result = maybe_expand_syntax_read_window(
        path="demo.js",
        source=source,
        requested_start_line=2,
        requested_end_line=2,
        limit=1,
    )

    assert result == SyntaxChunkResult(
        start_line=1,
        end_line=4,
        language="javascript",
        strategy="tree_sitter",
        fallback_reason=None,
    )


def test_real_environment_gracefully_falls_back_when_tree_sitter_is_unavailable():
    source = "function demo() {\n  return 1;\n}\n"

    result = maybe_expand_syntax_read_window(
        path="demo.js",
        source=source,
        requested_start_line=2,
        requested_end_line=2,
        limit=1,
    )

    assert result.start_line == 2
    assert result.end_line == 2
    assert result.language == "javascript"
    assert result.strategy in {"line", "tree_sitter"}
    if result.strategy == "line":
        assert result.fallback_reason == "tree_sitter_unavailable"
