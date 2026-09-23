"""Tests for ``read_file`` Python outline mode (``outline=True``).

Covers the stdlib ``ast`` scanner (``tools/python_outline.py``) and the
``read_file_tool`` wiring: opt-in outline output, full-text default
preserved, graceful fallback for non-Python and unparseable sources.
"""

import json

import pytest

from tools.python_outline import PYTHON_OUTLINE_MAX_ENTRIES, python_outline
from tools.file_tools import READ_FILE_SCHEMA, read_file_tool

SAMPLE = '''"""Module docstring first line.
Second line ignored.
"""

import os


class Store:
    """Store things."""

    def save(self, key, value=None, *tags, mode="w", **opts):
        """Save an item."""
        return True

    async def load(self, key):
        return None


def helper(a, b=2, /, *args, extra, **kw):
    pass


def _private():
    pass
'''


# =========================================================================
# Scanner unit tests
# =========================================================================


class TestPythonOutlineScanner:
    def test_classes_functions_and_methods_found(self):
        entries = python_outline(SAMPLE)
        names = [(e["kind"], e["name"]) for e in entries]
        assert ("class", "Store") in names
        assert ("function", "helper") in names
        assert ("function", "save") in names
        assert ("async function", "load") in names

    def test_source_order_and_line_numbers(self):
        entries = python_outline(SAMPLE)
        lines = [e["line"] for e in entries]
        assert lines == sorted(lines)
        by_name = {e["name"]: e for e in entries}
        assert by_name["Store"]["line"] == 8
        assert by_name["helper"]["line"] == 19

    def test_signatures(self):
        by_name = {e["name"]: e for e in python_outline(SAMPLE)}
        assert by_name["save"]["signature"] == "(self, key, value, *tags, mode, **opts)"
        assert by_name["helper"]["signature"] == "(a, b, /, *args, extra, **kw)"
        assert by_name["Store"]["signature"] == "()"

    def test_first_docstring_line_only(self):
        by_name = {e["name"]: e for e in python_outline(SAMPLE)}
        assert by_name["save"]["doc"] == "Save an item."
        assert by_name["_private"]["doc"] == ""

    def test_nested_one_level_in(self):
        src = "def outer():\n    def inner():\n        pass\n"
        entries = python_outline(src)
        assert [(e["name"], e["depth"]) for e in entries] == [
            ("outer", 0),
            ("inner", 1),
        ]

    def test_body_lines_never_included(self):
        entries = python_outline(SAMPLE)
        assert all(
            "return True" not in (e["name"] + e["signature"] + e["doc"])
            for e in entries
        )

    def test_syntax_error_raises(self):
        with pytest.raises(SyntaxError):
            python_outline("def broken(:\n")

    def test_empty_source(self):
        assert python_outline("") == []

    def test_deterministic(self):
        assert python_outline(SAMPLE) == python_outline(SAMPLE)


# =========================================================================
# read_file_tool wiring
# =========================================================================


class TestReadFileOutlineWiring:
    def test_outline_python_file(self, tmp_path):
        target = tmp_path / "mod.py"
        target.write_text(SAMPLE)
        result = json.loads(read_file_tool(str(target), outline=True))
        assert result.get("outline") is True
        assert result.get("entries", 0) >= 4
        assert "class Store()" in result["content"]
        assert "return True" not in result["content"]

    def test_default_read_unchanged(self, tmp_path):
        target = tmp_path / "mod.py"
        target.write_text(SAMPLE)
        result = json.loads(read_file_tool(str(target)))
        assert "outline" not in result
        assert "return True" in result["content"]

    def test_non_python_falls_back_to_full_read(self, tmp_path):
        target = tmp_path / "notes.txt"
        target.write_text("just some text\nsecond line\n")
        result = json.loads(read_file_tool(str(target), outline=True))
        assert "outline" not in result
        assert "just some text" in result["content"]

    def test_unparseable_python_falls_back_to_full_read(self, tmp_path):
        target = tmp_path / "broken.py"
        target.write_text("def broken(:\n")
        result = json.loads(read_file_tool(str(target), outline=True))
        assert "outline" not in result
        assert "def broken(:" in result["content"]

    def test_schema_advertises_opt_in_outline(self):
        outline = READ_FILE_SCHEMA["parameters"]["properties"]["outline"]
        assert outline["type"] == "boolean"
        assert outline["default"] is False
        assert "path" in READ_FILE_SCHEMA["parameters"]["required"]

    def test_outline_byte_bar(self, tmp_path):
        """Outline output must be a fraction of source bytes (<= 50%).

        Uses a body-heavy source (the realistic case: long function
        bodies under short signatures), mirroring the byte bar in the
        architecture brief.
        """
        import tools.python_outline as scanner_mod

        src = tmp_path / "big.py"
        chunk = (
            "def worker_{i}(job, retries=3):\n"
            '    """Process one job."""\n'
            + "".join(
                f"    step_{j} = job.transform({j})  # body line {j}\n"
                for j in range(30)
            )
            + "    return step_0\n\n\n"
        )
        body = "".join(chunk.format(i=i) for i in range(20))
        src.write_text(body)
        result = json.loads(read_file_tool(str(src), outline=True))
        assert result.get("outline") is True
        assert result.get("entries") == 20
        outline_bytes = len(result["content"].encode("utf-8"))
        source_bytes = len(body.encode("utf-8"))
        assert outline_bytes <= 0.5 * source_bytes
        assert scanner_mod.__file__ is not None  # module resolves
