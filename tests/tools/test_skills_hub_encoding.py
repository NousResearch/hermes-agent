"""Encoding-safety guard for the Skills Hub file I/O.

The hub state files (lock.json, taps.json, the skill-index caches) are written
with ``json.dumps(..., ensure_ascii=False)`` and carry skill metadata that can
contain emoji / CJK / accented text. ``Path.write_text()`` / ``Path.read_text()``
without an explicit ``encoding=`` use the platform locale codec — ``cp1252`` /
``GBK`` on Windows — which raises ``UnicodeEncodeError`` (or silently mojibakes)
on that content, corrupting installs.

``scripts/check-windows-footguns.py`` only flags the builtin ``open()`` and
deliberately skips ``.write_text()`` / ``.read_text()`` method calls, so this
guard covers that gap for the two skills-hub I/O modules. Mirrors the static
AST checks in ``test_windows_compat.py``.
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Modules that own the skills-hub state files and must round-trip non-ASCII.
GUARDED_FILES = [
    "tools/skills_hub.py",
    "tools/skills_sync.py",
]

_TEXT_IO_METHODS = {"write_text", "read_text"}


def _text_io_without_encoding(relpath: str) -> list[int]:
    """Line numbers of Path.write_text/read_text calls missing encoding=."""
    source = (PROJECT_ROOT / relpath).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relpath)
    missing = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _TEXT_IO_METHODS
            and not any(kw.arg == "encoding" for kw in node.keywords)
        ):
            missing.append(node.lineno)
    return missing


@pytest.mark.parametrize("relpath", GUARDED_FILES)
def test_text_io_specifies_encoding(relpath):
    missing = _text_io_without_encoding(relpath)
    assert not missing, (
        f"{relpath}: Path.write_text/read_text without encoding= at line(s) "
        f"{missing}. Pass encoding='utf-8' — cp1252/GBK locales corrupt the "
        f"non-ASCII skill metadata these hub files carry."
    )
