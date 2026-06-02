from tools.file_operations import _parse_search_context_line


def test_context_line_parser_uses_known_path_before_dash_number_content():
    line = "src/file-12-name.py-7-before mentions ticket -42- here"

    assert _parse_search_context_line(line, {"src/file-12-name.py"}) == (
        "src/file-12-name.py",
        7,
        "before mentions ticket -42- here",
    )


def test_context_line_parser_falls_back_to_rightmost_separator_without_known_path():
    line = "src/file-12-name.py-7-before"

    assert _parse_search_context_line(line) == (
        "src/file-12-name.py",
        7,
        "before",
    )
