"""The runtime venv's site-packages must not ride on PYTHONPATH into a cell's subprocesses.

A cell's own subprocesses inherit the session kernel's environment, and pip treats a
site-packages directory on PYTHONPATH as an environment it may uninstall from. One
``other-env/bin/pip install "pydantic<2.10"`` inside a cell could therefore remove the
gateway's pinned pydantic from the Hermes runtime venv and crash-loop the gateway on the
next restart (#155952).

The kernel still needs that directory on ``sys.path`` — a bare bundled Python keeps its
dependencies there — so the fix keeps it on the startup path and names it in the child env
for the kernel to drop from ``os.environ`` before the first cell runs.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.tools._child_env_fixtures import child_env, run_code  # noqa: F401  (fixtures)
from tools import code_execution_env as ce


def _fake_runtime_site(tmp_path: Path) -> Path:
    """A site-packages directory that looks like the runtime venv's, one importable module."""
    site = tmp_path / "runtime-venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    site.mkdir(parents=True)
    (site / "runtime_only_probe.py").write_text("VALUE = 'runtime-only'\n", encoding="utf-8")
    return site


def _with_runtime_site(site: Path, venv: Path):
    """Patch the two lookups `_build_child_env` uses to decide the runtime path."""
    return (
        patch("tools.environments.local_pythonpath._validated_runtime_venv", return_value=venv),
        patch("pm.environments.site_packages", return_value=site),
    )


@pytest.mark.platforms("linux", "macos", "windows")
def test_build_child_env_names_the_runtime_site_path(child_env, tmp_path, monkeypatch):  # noqa: F811
    """The parent keeps the directory on the startup path and names it for the kernel to strip."""
    site = _fake_runtime_site(tmp_path)
    venv = site.parents[2]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(site), str(tmp_path / "user-lib")]))
    validated, resolver = _with_runtime_site(site, venv)
    with validated, resolver:
        env = ce._build_child_env(
            rpc_endpoint="unix:/tmp/x.sock", rpc_token="t", tmpdir=str(tmp_path / "staging"),
            child_python=sys.executable,
        )
    assert str(site) in env["PYTHONPATH"].split(os.pathsep), (
        "the kernel's own imports need this entry on the startup path"
    )
    assert env.get("HERMES_CHILD_STRIP_PYTHONPATH") == str(site)


@pytest.mark.platforms("linux", "macos", "windows")
def test_no_marker_without_a_runtime_site(child_env, monkeypatch):  # noqa: F811
    """An external interpreter has no Hermes runtime tree to strip — the marker stays absent."""
    monkeypatch.setenv("PYTHONPATH", str(Path(sys.executable).parent))
    env = ce._build_child_env(
        rpc_endpoint="unix:/tmp/x.sock", rpc_token="t", tmpdir=str(child_env / "staging"),
        child_python="/nonexistent/project/python",
    )
    assert "HERMES_CHILD_STRIP_PYTHONPATH" not in env


@pytest.mark.platforms("linux", "macos", "windows")
def test_cell_and_its_children_do_not_inherit_the_runtime_site(child_env, tmp_path, monkeypatch):  # noqa: F811
    """End to end through a real session kernel: sys.path keeps the tree, os.environ loses it."""
    site = _fake_runtime_site(tmp_path)
    venv = site.parents[2]
    monkeypatch.setenv("PYTHONPATH", str(site))
    validated, resolver = _with_runtime_site(site, venv)
    with validated, resolver:
        result = run_code(
            """
import json, os, subprocess, sys
import runtime_only_probe
child = subprocess.run(
    [sys.executable, "-c", "import os; print(os.environ.get('PYTHONPATH', ''))"],
    capture_output=True, text=True, check=True,
).stdout.strip()
print(json.dumps({
    "imported": runtime_only_probe.VALUE,
    "cell_pythonpath": os.environ.get("PYTHONPATH", ""),
    "child_pythonpath": child,
    "strip_marker": os.environ.get("HERMES_CHILD_STRIP_PYTHONPATH"),
}))
""",
            mode="strict",
        )
    # The kernel can still import it: this is the bundled-Python case the entry exists for.
    assert result["imported"] == "runtime-only"
    # Nothing a cell spawns can see the tree, so pip has no install location to uninstall from.
    assert str(site) not in result["cell_pythonpath"].split(os.pathsep)
    assert str(site) not in result["child_pythonpath"].split(os.pathsep)
    # The marker is consumed by the kernel and never handed onward.
    assert result["strip_marker"] is None


@pytest.mark.platforms("linux", "macos", "windows")
def test_blank_pythonpath_components_survive_the_scrub(child_env, tmp_path, monkeypatch):  # noqa: F811
    """A blank component means "the current directory" to CPython — the scrub must not eat it."""
    site = _fake_runtime_site(tmp_path)
    venv = site.parents[2]
    user_lib = tmp_path / "user-lib"
    user_lib.mkdir()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(site), "", str(user_lib)]))
    validated, resolver = _with_runtime_site(site, venv)
    with validated, resolver:
        result = run_code(
            """
import json, os, sys
print(json.dumps({"cell_pythonpath": os.environ.get("PYTHONPATH", "")}))
""",
            mode="strict",
        )
    entries = result["cell_pythonpath"].split(os.pathsep)
    assert str(site) not in entries, "the runtime tree is still the thing being removed"
    assert "" in entries, "a blank entry is cwd for a child import; dropping it changes behaviour"
    assert str(user_lib) in entries, "an operator-supplied entry is left alone"

