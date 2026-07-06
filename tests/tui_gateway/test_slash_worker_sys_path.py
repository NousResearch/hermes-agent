"""Regression tests for tui_gateway/slash_worker.py sys.path hardening (issue #51286).

The slash-command worker is spawned as ``-m tui_gateway.slash_worker`` and
inherits the user's CWD. A local package (e.g. ``utils/``) in that CWD shadows
the installed hermes ``utils`` module and crashes the worker on ``import cli``
(``ImportError: cannot import name 'atomic_replace' from 'utils'``).

#51693 added this guard to the sibling entrypoints ``tui_gateway/entry.py`` and
``acp_adapter/entry.py`` (via the shared ``hermes_bootstrap.harden_import_path``
helper) but missed this child, so the crash still reproduced. slash_worker.py
must run the guard before its first non-stdlib import.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_slash_worker_imports_from_cwd_with_colliding_utils(tmp_path):
    """Importing the worker from a CWD that ships its own ``utils/`` package
    must succeed — the guard strips CWD so the installed module wins."""
    # Mimic the user's project (tg-ws-proxy ships utils/, proxy/, ui/).
    for pkg in ("utils", "proxy", "ui"):
        (tmp_path / pkg).mkdir()
        (tmp_path / pkg / "__init__.py").write_text("")  # no atomic_replace, etc.

    env = {k: v for k, v in os.environ.items() if k != "HERMES_PYTHON_SRC_ROOT"}
    # Keep the source importable via PYTHONPATH; CWD ('') still precedes it on
    # sys.path for ``-c``, so the shadow (and thus the guard) is still exercised.
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    result = subprocess.run(
        [sys.executable, "-c", "import tui_gateway.slash_worker"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        "slash_worker failed to import from a CWD containing a colliding "
        "utils/ package — sys.path guard regressed (issue #51286).\n"
        f"stderr:\n{result.stderr}"
    )


def test_sys_path_guard_runs_before_cli_import():
    """The guard must execute before ``import cli`` — reordering it below the
    import would re-introduce the shadowing crash. Assert via AST that the
    ``hermes_bootstrap.harden_import_path()`` call precedes ``import cli``."""
    src = (PROJECT_ROOT / "tui_gateway" / "slash_worker.py").read_text()
    tree = ast.parse(src)

    harden_call_line = None
    cli_import_line = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "harden_import_path"
        ):
            if harden_call_line is None:
                harden_call_line = node.lineno
        if isinstance(node, ast.Import) and any(a.name == "cli" for a in node.names):
            if cli_import_line is None:
                cli_import_line = node.lineno

    assert harden_call_line is not None, (
        "slash_worker.py must call hermes_bootstrap.harden_import_path()"
    )
    assert cli_import_line is not None, "slash_worker.py must 'import cli'"
    assert harden_call_line < cli_import_line, (
        "harden_import_path() must run before 'import cli' (issue #51286)"
    )


def test_guard_delegates_to_shared_helper_not_inline():
    """slash_worker should delegate to the shared guard, not re-implement the
    old inline ``{"", "."}`` sys.path filter that #51693 replaced."""
    src = (PROJECT_ROOT / "tui_gateway" / "slash_worker.py").read_text()
    assert '{"", "."}' not in src and "{'', '.'}" not in src, (
        "slash_worker.py should delegate to hermes_bootstrap.harden_import_path, "
        "not re-implement the guard inline"
    )
    assert "hermes_bootstrap.harden_import_path()" in src, (
        "slash_worker.py must call the shared hermes_bootstrap.harden_import_path guard"
    )
