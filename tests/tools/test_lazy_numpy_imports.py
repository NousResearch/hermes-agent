"""Smoke tests: tools.neutts_synth and tools.voice_mode must not import numpy
at module-load time (lazy-import discipline).

Each test spawns a fresh subprocess so that numpy is guaranteed absent from
sys.modules when the target module is imported.  This is more reliable than
manipulating sys.modules inside the test process, which can be contaminated
by other tests that already loaded numpy.
"""

from __future__ import annotations

import subprocess
import sys


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_CHECK_SCRIPT = """\
import sys
import importlib

# Ensure numpy is not pre-loaded
sys.modules.pop("numpy", None)

# Import the target module
importlib.import_module({module!r})

# If numpy crept in, fail loudly
if "numpy" in sys.modules:
    print("FAIL: numpy was imported as a side-effect of loading {module}", file=sys.stderr)
    sys.exit(1)

print("PASS")
"""


def _assert_no_numpy_on_import(module: str) -> None:
    """Run a subprocess that imports *module* and asserts numpy stays out."""
    script = _CHECK_SCRIPT.format(module=module)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"numpy was imported at module load time for {module!r}.\n"
        f"stderr: {result.stderr.strip()}\n"
        f"stdout: {result.stdout.strip()}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_neutts_synth_imports_without_numpy():
    """Loading tools.neutts_synth must NOT trigger a numpy import."""
    _assert_no_numpy_on_import("tools.neutts_synth")


def test_voice_mode_imports_without_numpy():
    """Loading tools.voice_mode must NOT trigger a numpy import."""
    _assert_no_numpy_on_import("tools.voice_mode")
