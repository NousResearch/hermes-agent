"""Live regression: a multi-root search keeps every root's partial-result warning.

``path="a,b"`` searches each existing root and merges the results. The merge used
to drop each root's skipped-unreadable-path warning (content mode) or overwrite the
traversal's warning with its own "searched N that exist" note (filename mode), so a
search that silently skipped a mode-000 directory came back looking complete.
"""

import json
import os
import shutil

import pytest

from tools.file_operations import ShellFileOperations
import tools.file_tools  # registers search_files
from tools.registry import registry


@pytest.fixture
def two_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    roots, locked = [], []
    for name in ("alpha", "beta"):
        root = tmp_path / name
        (root / "locked").mkdir(parents=True)
        (root / f"{name}.py").write_text("needle\n")
        (root / "locked" / "secret.py").write_text("needle\n")
        os.chmod(root / "locked", 0o000)
        roots.append(root)
        locked.append(root / "locked")
    yield roots, locked
    for path in locked:
        os.chmod(path, 0o755)


@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads chmod 000 dirs")
@pytest.mark.parametrize("engine", ["rg", "grep"])
@pytest.mark.parametrize("target", ["content", "files"])
def test_multi_root_search_reports_every_skipped_path(two_roots, monkeypatch, target, engine):
    if engine == "rg" and shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")
    if engine == "grep":
        monkeypatch.setattr(ShellFileOperations, "_has_command", lambda self, command: command != "rg")
    roots, locked = two_roots
    pattern = "needle" if target == "content" else "*.py"
    handler = registry.get_entry("search_files").handler
    out = json.loads(handler({"pattern": pattern, "target": target, "path": ",".join(map(str, roots))},
                             task_id=f"multi-root-{target}-{engine}"))

    assert "error" not in out, out
    found = out.get("files") or [m["path"] for m in out.get("matches", [])]
    assert sorted(found) == sorted(str(root / f"{root.name}.py") for root in roots)
    warning = out.get("warning", "")
    assert "Skipped 2 unreadable path(s)" in warning, warning
    assert all(str(path) in warning for path in locked), warning
    assert "Results may be incomplete" in warning
    assert warning.rstrip().endswith("searched 2 that exist"), warning


@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads chmod 000 dirs")
@pytest.mark.parametrize("engine", ["rg", "grep"])
def test_multi_root_skipped_dir_name_with_comma_survives_merge(tmp_path, monkeypatch, engine):
    """A skipped directory named ``a, b`` is one path in the merged count and text."""
    if engine == "rg" and shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")
    if engine == "grep":
        monkeypatch.setattr(ShellFileOperations, "_has_command", lambda self, command: command != "rg")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    roots, locked = [tmp_path / "alpha", tmp_path / "beta"], []
    for root, dirname in zip(roots, ("plain", "one, two")):
        (root / dirname).mkdir(parents=True)
        (root / f"{root.name}.py").write_text("needle\n")
        os.chmod(root / dirname, 0o000)
        locked.append(root / dirname)
    try:
        handler = registry.get_entry("search_files").handler
        out = json.loads(handler({"pattern": "needle", "target": "content", "path": ",".join(map(str, roots))},
                                 task_id=f"multi-root-comma-{engine}"))
    finally:
        for path in locked:
            os.chmod(path, 0o755)

    warning = out.get("warning", "")
    assert "Skipped 2 unreadable path(s)" in warning, warning
    assert f"permissions: {locked[0]}, {locked[1]}. Results may be incomplete" in warning, warning
