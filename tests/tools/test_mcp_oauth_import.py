"""tools.mcp_oauth must import on every supported Python (requires-python >=3.11).

Before 3.13 ``threading.Lock`` is a factory function, not a class, so an annotation such as
``threading.Lock | None`` raised TypeError when its ``def`` ran, at import, and every
gateway-managed worker exited during startup (#160188). The child process gives
``threading.Lock`` that older shape on any interpreter and evaluates the module's annotations,
which 3.14 otherwise defers (PEP 649) and 3.11/3.12 evaluate at ``def`` time. A child process
keeps the fresh import out of this test session.
"""

import subprocess
import sys
from pathlib import Path

_CHILD = """
import _thread
import inspect
import threading

threading.Lock = lambda: _thread.allocate_lock()  # the pre-3.13 shape: a function, not a class

import tools.mcp_oauth as module

functions = [
    obj for obj in vars(module).values()
    if inspect.isfunction(obj) and obj.__module__ == module.__name__
]
assert functions
for function in functions:
    inspect.get_annotations(function)
"""


def test_mcp_oauth_annotations_evaluate_where_threading_lock_is_a_function():
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", _CHILD], cwd=repo_root, capture_output=True, text=True, timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
