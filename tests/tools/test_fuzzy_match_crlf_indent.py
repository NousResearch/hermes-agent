"""Tests for fuzzy_find_and_replace CRLF/indentation edge cases."""

from tools.fuzzy_match import fuzzy_find_and_replace


def test_crlf_column0_first_line_preserves_indent():
    """When old_string's first line is at column 0, the reindent logic
    should NOT double-indent the new_string's indented body lines."""
    content = (
        "def outer():\r\n"
        "    x = 1\r\n"
        "    if x:\r\n"
        "        y = 2\r\n"
        "        print(y)\r\n"
        "    return x\r\n"
    )
    old_string = "if x:\n        y = 2\n        print(y)"
    new_string = "if x:\n        y = 999\n        print(y * 2)"

    new_content, count, strategy, err = fuzzy_find_and_replace(
        content, old_string, new_string, False
    )

    assert err is None, f"unexpected error: {err}"
    assert count == 1
    indents = [len(l) - len(l.lstrip(" ")) for l in new_content.splitlines()]
    # Expected: [0, 4, 4, 8, 8, 4] — body lines stay at 8 spaces
    assert indents == [0, 4, 4, 8, 8, 4], f"indentation corrupted: {indents}"


def test_lf_column0_first_line_preserves_indent():
    """Same bug on LF files — column-0 anchor should not double-indent."""
    content = (
        "def outer():\n"
        "    x = 1\n"
        "    if x:\n"
        "        y = 2\n"
        "        print(y)\n"
        "    return x\n"
    )
    old_string = "if x:\n        y = 2\n        print(y)"
    new_string = "if x:\n        y = 999\n        print(y * 2)"

    new_content, count, strategy, err = fuzzy_find_and_replace(
        content, old_string, new_string, False
    )

    assert err is None
    assert count == 1
    indents = [len(l) - len(l.lstrip(" ")) for l in new_content.splitlines()]
    assert indents == [0, 4, 4, 8, 8, 4], f"indentation corrupted: {indents}"


def test_non_empty_old_indent_still_works():
    """When old_string has a non-empty base indent, the existing swap
    behavior should still work (regression check)."""
    content = (
        "def outer():\n"
        "    x = 1\n"
        "    if x:\n"
        "        y = 2\n"
        "        print(y)\n"
        "    return x\n"
    )
    # old_string has 4-space base indent
    old_string = "    if x:\n        y = 2\n        print(y)"
    new_string = "    if x:\n        y = 999\n        print(y * 2)"

    new_content, count, strategy, err = fuzzy_find_and_replace(
        content, old_string, new_string, False
    )

    assert err is None
    assert count == 1
    indents = [len(l) - len(l.lstrip(" ")) for l in new_content.splitlines()]
    assert indents == [0, 4, 4, 8, 8, 4], f"indentation corrupted: {indents}"
