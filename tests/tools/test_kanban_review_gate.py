"""Pre-review build gate: a worktree card must build + focused tests green
before ``kanban_request_review`` is accepted.

The gate is the zero-token subprocess check that runs in front of the
``running -> review`` transition.  These tests pin the contract:

* green path: a clean worktree whose focused tests pass proceeds to review.
* failing-build bounce: a worktree with a changed python file that fails to
  build returns the gate output and the card stays in its current lane — no
  ``review`` transition, and no failure is counted against the card.
* comment content: the auto-comment carries the last ~30 lines of gate
  output.
* non-worktree cards skip the gate entirely (no refusal ever fires).

We exercise the tool-facing handler (``_handle_request_review``) so the full
worker -> tool -> DB path is covered, not just the pure helpers.
"""
from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real git repo (the 'primary repo') with a ``venv/bin/python`` shim.

    ``venv/bin/python`` is a shell shim that execs the interpreter running the
    tests, so ``pytest`` resolves for the gate subprocess. The fixture
    returns the primary repo root; each test adds a linked worktree under
    ``<repo>/.worktrees/<name>`` to exercise the project-linked layout.
    """
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
    (root / "base.txt").write_text("x\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "base"], check=True)
    # venv shim so _project_python resolves a working interpreter + pytest.
    venv_bin = root / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    py_shim = venv_bin / "python"
    py_shim.write_text("#!/bin/sh\nexec " + sys.executable + " \"$@\"\n")
    py_shim.chmod(py_shim.stat().st_mode | stat.S_IEXEC)
    return root


def _add_worktree(repo: Path, name: str) -> Path:
    """Create a linked worktree under ``<repo>/.worktrees/<name>``."""
    branch = f"wt/{name}"
    ws = repo / ".worktrees" / name
    ws.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-b", branch, str(ws)],
        check=True,
        capture_output=True,
    )
    return ws


def _make_task(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    ws: Path,
    *,
    kind: str = "worktree",
) -> str:
    """Create + claim a card rooted at ``ws`` and set the worker env to it."""
    monkeypatch.setenv("HERMES_HOME", str(kanban_home))
    monkeypatch.setattr(Path, "home", lambda: kanban_home)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="worktree build gate",
            assignee="builder",
            workspace_kind=kind,
            workspace_path=str(ws),
        )
        claimed = kb.claim_task(conn, tid, claimer="builder:1")
        assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
    return tid


def _change_python_file(w: Path, rel: str, content: str) -> None:
    """Write an uncommitted python change into the worktree."""
    target = w / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def test_gate_green_path_proceeds_to_review(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _add_worktree(repo, "green")
    # A changed module plus a passing focused test.
    _change_python_file(ws, "mymod.py", "GOOD = 1\n")
    _change_python_file(ws, "tests/test_mymod.py", "def test_good():\n    assert 1 == 1\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    resp = json.loads(tools._handle_request_review({"summary": "works"}))
    assert resp.get("ok") is True, resp
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"


def test_gate_failing_build_bounces_card(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _add_worktree(repo, "bad")
    # Syntax error so py_compile (the import/build sanity check) fails.
    _change_python_file(ws, "broken.py", "def x(:\n    pass\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    resp = json.loads(tools._handle_request_review({"summary": "broken"}))
    assert "error" in resp
    assert "Pre-review build gate failed" in resp["error"]

    with kb.connect() as conn:
        # Card never left the builder lane.
        assert kb.get_task(conn, tid).status == "running"
        # No failure counted against the card.
        assert kb.get_task(conn, tid).consecutive_failures == 0
        comments = kb.list_comments(conn, tid)
        assert len(comments) == 1
        assert "Pre-review build gate FAILED" in comments[0].body


def test_gate_comment_carries_output_tail(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _add_worktree(repo, "tail")
    # A test that fails → focused pytest gate returns output tail.
    _change_python_file(ws, "mymod.py", "OK = 1\n")
    _change_python_file(ws, "tests/test_mymod.py", "def test_bad():\n    assert 1 == 2\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    resp = json.loads(tools._handle_request_review({"summary": "flaky"}))
    assert "error" in resp
    with kb.connect() as conn:
        comments = kb.list_comments(conn, tid)
        assert len(comments) == 1
        body = comments[0].body
        assert "```" in body
        assert "import/build sanity" in body or "focused tests" in body


def test_gate_skipped_for_non_worktree_card(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _add_worktree(repo, "dir-card")
    _change_python_file(ws, "mymod.py", "def f():\n    pass\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws, kind="dir")
    from tools import kanban_tools as tools

    resp = json.loads(tools._handle_request_review({"summary": "dir card"}))
    assert resp.get("ok") is True, resp
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"


def test_gate_command_override_is_per_project(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config override replaces the default pytest gate command."""
    import hermes_cli.config as hcfg
    from tools import kanban_tools as tools

    cfg = {"kanban": {"review_gate": {"enabled": True, "command": ["{python}", "myrunner", "{tests}"]}}}
    monkeypatch.setattr(tools, "load_config", lambda: cfg)
    monkeypatch.setattr(hcfg, "load_config", lambda: cfg)

    cmd = tools._gate_command("/venv/bin/python", ["a.py", "b.py"])
    assert cmd == ["/venv/bin/python", "myrunner", "a.py b.py"], cmd


def test_gate_command_defaults_to_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no config override, the gate runs pytest on the focused tests."""
    import hermes_cli.config as hcfg
    from tools import kanban_tools as tools

    cfg = {"kanban": {"review_gate": {"enabled": True, "command": None}}}
    monkeypatch.setattr(tools, "load_config", lambda: cfg)
    monkeypatch.setattr(hcfg, "load_config", lambda: cfg)

    cmd = tools._gate_command("/venv/bin/python", ["a.py", "b.py"])
    assert cmd == ["/venv/bin/python", "-m", "pytest", "a.py", "b.py", "-q"], cmd


def test_gate_obeys_enabled_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """review_gate.enabled=false lets a worktree card straight through."""
    import hermes_cli.config as hcfg
    from tools import kanban_tools as tools

    cfg = {"kanban": {"review_gate": {"enabled": False, "command": None}}}
    monkeypatch.setattr(tools, "load_config", lambda: cfg)
    monkeypatch.setattr(hcfg, "load_config", lambda: cfg)

    import types
    task = types.SimpleNamespace(workspace_kind="worktree", workspace_path="/nonexistent")
    assert tools._run_pre_review_gate(task) is None


def test_base_ref_guard_ignores_diverged_remote(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote-tracking ref with no merge-base must not be used as base.

    Regression: if origin/main points at fully-diverged history (no shared
    ancestor with HEAD), using it as the diff base reports every file ever
    created as 'changed', which would fail an otherwise-green gate.
    """
    ws = _add_worktree(repo, "diverged")
    _change_python_file(ws, "mymod.py", "GOOD = 1\n")
    _change_python_file(ws, "tests/test_mymod.py", "def test_good():\n    assert 1 == 1\n")

    # Simulate a stale remote: create a true orphan commit (no ancestor shared
    # with HEAD) in the same object store and point origin/main at it.
    tree_sha = subprocess.run(
        ["git", "-C", str(repo), "mktree"],
        input="", capture_output=True, text=True, check=True,
    ).stdout.strip()
    orphan_env = {
        "GIT_AUTHOR_NAME": "o", "GIT_AUTHOR_EMAIL": "o@o",
        "GIT_COMMITTER_NAME": "o", "GIT_COMMITTER_EMAIL": "o@o",
    }
    orphan_sha = subprocess.run(
        ["git", "-C", str(repo), "commit-tree", tree_sha, "-m", "orphan"],
        env={**__import__("os").environ, **orphan_env},
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repo), "update-ref", "refs/remotes/origin/main", orphan_sha],
        check=True,
    )

    from tools import kanban_tools as tools

    # merge-base origin/main HEAD should now be empty (diverged), so the base
    # guard must fall through to local main and detect only the real changes.
    changed = tools._changed_python_files(str(ws))
    assert "mymod.py" in changed, changed
    assert "tests/test_mymod.py" in changed, changed
    # And not every file in the repo.
    assert "base.txt" not in changed

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    resp = json.loads(tools._handle_request_review({"summary": "diff vs unfair base"}))
    assert resp.get("ok") is True, resp
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"