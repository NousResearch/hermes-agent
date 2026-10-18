"""Regression tests for the ``tools.connectors`` import surface.

The package is imported concurrently with its own submodules: the tool registry's discovery
scan imports ``tools.connectors.tool`` while the gateway's Group Chat worker imports the
package (``tui_gateway.methods_connectors`` -> ``tools.connectors``). When ``__init__``
imported its submodules eagerly, one thread held the package's module lock while waiting for
a submodule another thread already held, and Python 3.14 aborted the loser with
``_frozen_importlib._DeadlockError: deadlock detected by _ModuleLock('tools.connectors.tool')``
— the Group Chat worker then failed to start ("mutating Group Chat commands will fail closed").

Both checks run in a subprocess so the assertions see a pristine ``sys.modules`` and cannot
perturb the test session's own imports.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_NO_EAGER_SUBMODULE = """
import sys
import tools.connectors  # noqa: F401
loaded = sorted(k for k in sys.modules if k.startswith("tools.connectors."))
print("\\n".join(loaded))
"""

_RACE = """
import importlib, sys, threading
errors = []
ITER = 25

def purge():
    for name in [k for k in list(sys.modules) if k.startswith("tools.connectors")]:
        sys.modules.pop(name, None)

def run(target):
    for _ in range(ITER):
        try:
            importlib.import_module(target)
        except Exception as exc:  # noqa: BLE001
            errors.append((target, type(exc).__name__))
        purge()

threads = [threading.Thread(target=run, args=("tools.connectors",)) for _ in range(3)]
threads += [threading.Thread(target=run, args=("tools.connectors.tool",)) for _ in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=120)
print("alive", sum(1 for t in threads if t.is_alive()))
print("errors", [e for e in errors if e[1] == "_DeadlockError"])
"""


def _run_snippet(snippet: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        cwd=str(REPO_ROOT),
        check=False,
    )


def test_package_import_does_not_load_submodules():
    """Importing the package must not import any submodule: that eager edge is the cycle."""
    proc = _run_snippet(_NO_EAGER_SUBMODULE)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"eagerly imported: {proc.stdout.strip()}"


def test_concurrent_package_and_submodule_import_never_deadlocks():
    """The two real-world import paths racing must not raise _DeadlockError."""
    proc = _run_snippet(_RACE)
    assert proc.returncode == 0, proc.stderr
    assert "alive 0" in proc.stdout, proc.stdout
    assert "errors []" in proc.stdout, proc.stdout
