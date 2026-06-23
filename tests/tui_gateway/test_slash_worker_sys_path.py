"""Regression tests for tui_gateway/slash_worker.py sys.path hardening (issue #51286).

The slash-command worker is spawned as ``-m tui_gateway.slash_worker`` and
inherits the user's CWD. A local package (e.g. ``utils/``) in that CWD shadows
the installed hermes ``utils`` module and crashes the worker on ``import cli``
(``ImportError: cannot import name 'atomic_replace' from 'utils'``).

#15989 added this guard to the sibling entrypoint ``tui_gateway/entry.py`` but
missed this child, so the crash still reproduced. slash_worker.py must sanitize
sys.path before its first non-stdlib import.
"""

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
    import would re-introduce the shadowing crash."""
    src = (PROJECT_ROOT / "tui_gateway" / "slash_worker.py").read_text()
    guard = 'sys.path = [p for p in sys.path if p not in {"", "."}]'
    cli_import = "import cli as cli_mod"
    assert guard in src, "sys.path shadowing guard missing from slash_worker.py"
    assert cli_import in src, "expected 'import cli as cli_mod' in slash_worker.py"
    assert src.index(guard) < src.index(cli_import), (
        "sys.path guard must run before 'import cli' (issue #51286)"
    )
