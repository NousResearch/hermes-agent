"""Regression test for tui_gateway/entry.py sys.path hardening (#15989, #51286).

The TUI backend is spawned by Node with the user's launch directory as CWD. A
local package there (e.g. ``utils/``, ``proxy/``, ``ui/`` in tg-ws-proxy) shadowed
Hermes's own top-level modules and crashed the backend on import
(``ImportError: cannot import name ... from 'utils'``). entry.py must run
``hermes_bootstrap.harden_import_path()`` before its first non-stdlib import.
Sibling guard for the slash worker: test_slash_worker_sys_path.py.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_entry_imports_from_cwd_with_colliding_packages(tmp_path):
    """Importing the TUI entry point from a CWD that ships its own ``utils/``
    (and friends) must succeed — the guard strips CWD so Hermes's modules win."""
    for pkg in ("utils", "proxy", "ui"):
        (tmp_path / pkg).mkdir()
        (tmp_path / pkg / "__init__.py").write_text("", encoding="utf-8")

    env = {k: v for k, v in os.environ.items() if k != "HERMES_PYTHON_SRC_ROOT"}
    # Source importable via PYTHONPATH; CWD ('') still precedes it on sys.path
    # for ``-c``, so the shadow (and thus the guard) is exercised.
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["HERMES_HOME"] = str(tmp_path / "hermes_home")

    result = subprocess.run(
        [sys.executable, "-c", "import tui_gateway.entry"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, (
        "tui_gateway.entry failed to import from a CWD with a colliding utils/ "
        f"package (#51286):\n{result.stderr[-2000:]}"
    )
