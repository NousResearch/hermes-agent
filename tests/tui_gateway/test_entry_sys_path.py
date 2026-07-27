"""Tests for tui_gateway/entry.py sys.path hardening (issues #15989, #51286).

When the TUI backend is spawned by Node.js, the launch directory may shadow
Hermes's own top-level modules (``utils``, ``proxy``, ``ui``).  entry.py must
neutralize this before any non-stdlib import is resolved, by delegating to the
shared ``hermes_bootstrap.harden_import_path`` guard.

These tests assert the entry point wires up the real guard (rather than
re-implementing it inline) and that the guard's behavior covers both the
relative-cwd form and the absolute-cwd-path form that was the actual #51286
failure.
"""

import ast
import pathlib

import hermes_bootstrap


def _entry_source() -> str:
    here = pathlib.Path(__file__).resolve()
    repo_root = here.parent.parent.parent  # tests/tui_gateway/ -> repo root
    return (repo_root / "tui_gateway" / "entry.py").read_text(encoding="utf-8")


def test_entry_calls_shared_harden_guard_before_heavy_imports():
    """entry.py must call hermes_bootstrap.harden_import_path() before it
    imports tui_gateway.server (which pulls ``from utils import ...``)."""
    source = _entry_source()
    tree = ast.parse(source)

    harden_call_line = None
    server_import_line = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "harden_import_path"
        ):
            harden_call_line = node.lineno
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "tui_gateway"
        ):
            if server_import_line is None:
                server_import_line = node.lineno

    assert harden_call_line is not None, (
        "entry.py must call hermes_bootstrap.harden_import_path()"
    )
    assert server_import_line is not None, "entry.py must import from tui_gateway"
    assert harden_call_line < server_import_line, (
        "harden_import_path() must run before tui_gateway.server is imported"
    )


def test_entry_does_not_reimplement_guard_inline():
    """The old inline ``{'', '.'}`` strip lived in entry.py; the dedicated
    helper now owns it.  Guard against the inline logic creeping back."""
    source = _entry_source()
    assert '{"", "."}' not in source and "{'', '.'}" not in source, (
        "entry.py should delegate to hermes_bootstrap.harden_import_path, "
        "not re-implement the sys.path strip inline"
    )


def test_guard_handles_absolute_cwd_path():
    """The #51286 case: the launch dir is on sys.path as its own absolute
    path, ahead of the Hermes root.  harden_import_path must relocate the
    Hermes root to the front so ``from utils import ...`` resolves to Hermes."""
    import sys

    original = sys.path[:]
    try:
        sys.path[:] = ["/home/user/tg-ws-proxy", "/opt/hermes", "/usr/lib"]
        hermes_bootstrap.harden_import_path(src_root="/opt/hermes")
        assert sys.path[0] == "/opt/hermes"
        assert sys.path.index("/opt/hermes") < sys.path.index(
            "/home/user/tg-ws-proxy"
        )
    finally:
        sys.path[:] = original
