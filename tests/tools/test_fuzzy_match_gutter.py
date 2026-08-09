"""old_string pasted verbatim from read_file's ``N|`` gutter is recovered, never misapplied."""

import pytest

from tools.fuzzy_match import GUTTER_STRIPPED_NOTE, fuzzy_find_and_replace

FILE = "def foo():\n    x = 1\n\n    return x\n"


@pytest.mark.parametrize("new", [
    "def foo():\n    x = 2\n\n    return x",          # new_string without the gutter
    "41|def foo():\n42|    x = 2\n43|\n44|    return x",  # both strings pasted with it
    "41|def foo():\n42|    x = 2\n\n44|    return x",     # blank line lost its prefix
])
def test_gutter_pasted_old_string_is_recovered(new):
    notes: list[str] = []
    old = "41|def foo():\n42|    x = 1\n43|\n44|    return x"
    result, count, _strategy, err = fuzzy_find_and_replace(FILE, old, new, correction_notes=notes)
    assert (err, count) == (None, 1)
    assert result == "def foo():\n    x = 2\n\n    return x\n"
    assert notes == [GUTTER_STRIPPED_NOTE]


@pytest.mark.parametrize("content, old, new", [
    # The file holds gutter-shaped data: the pipes may be content, never strip.
    ("ids:\n7|x\nnames:\nalice\nbob\n", "1|alice\n2|bob", "X"),
    # Non-consecutive numbers are not a read_file paste (pipelines, tables).
    ("a\nb\n", "3|a\n9|b", "X"),
    # A bare line among numbered ones: genuine content, not a gutter.
    ("a\nb\n", "1|a\nb", "X"),
    # A stripped old_string reaching only a similarity match would land on a guess.
    ("def g():\n    return 1\n\ndef foo():\n    x = 1\n", "1|def foo():\n2|    x = 9", "1|def foo():\n2|    x = 2"),
    # new_string partly prefixed: writing it would put `N|` into the file.
    (FILE, "1|def foo():\n2|    x = 1", "1|def foo():\n    y = 0\n2|    x = 1"),
])
def test_gutter_recovery_never_rewrites_a_wrong_or_literal_region(content, old, new):
    result, count, _strategy, err = fuzzy_find_and_replace(content, old, new)
    assert count == 0 and err
    assert result == content
