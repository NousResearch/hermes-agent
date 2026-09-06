"""Local POSIX content/file search runs rg as a direct argv subprocess.

The shell path pays two bash spawns per search (``test -e`` probe + ``set -o
pipefail; rg ... | head``). On a ``LocalEnvironment`` the same rg argv can run
natively with a bounded stdout read; the parser and every argument builder are
shared, so the two transports must agree on results.
"""

import json
import sys

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="native rg lane is POSIX-only")


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "a.py").write_text("needle one\nplain\nneedle two\n")
    (tmp_path / "b.txt").write_text("needle three\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("needle four\n")
    return tmp_path


def _ops(tree, spy):
    env = LocalEnvironment(cwd=str(tree))
    real = type(env).execute.__get__(env, type(env))

    def recording(command, *a, **kw):
        spy.append(command)
        return real(command, *a, **kw)

    env.execute = recording
    return ShellFileOperations(env, cwd=str(tree))


def _normalized(result):
    d = result.to_dict()
    for key in ("matches", "files"):
        if key in d:
            d[key] = sorted(json.dumps(item, sort_keys=True) for item in d[key])
    return d


def test_native_search_never_touches_the_shell_and_matches_shell_results(tree, monkeypatch):
    cases = [
        dict(pattern="needle", path=str(tree)),
        # (no offset/limit slicing here: rg's parallel walk orders files
        # nondeterministically, so a page differs run-to-run on either transport)
        dict(pattern="needle", path=str(tree), output_mode="count"),
        dict(pattern="needle", path=str(tree), output_mode="files_only", file_glob="*.py"),
        dict(pattern="NEEDLE_NOPE", path=str(tree)),  # zero-match probes
        dict(pattern="*.py", path=str(tree), target="files"),
        dict(pattern="needle", path=str(tree / "missing")),
    ]
    for case in cases:
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
        shell = _normalized(_ops(tree, []).search(**case))
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "1")
        calls = []
        native = _normalized(_ops(tree, calls).search(**case))
        assert native == shell, case
        # rg resolution (``command -v rg``) still goes through the shell once; the
        # existence probe and the rg pipeline itself must not.
        assert not [c for c in calls if "pipefail" in c or c.startswith("test -e")], case


def test_kill_switch_routes_search_back_to_the_shell(tree, monkeypatch):
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
    calls = []
    result = _ops(tree, calls).search(pattern="needle", path=str(tree))
    assert result.total_count == 4
    assert any(c.startswith("test -e") for c in calls)
    assert any("pipefail" in c and "rg" in c for c in calls)
