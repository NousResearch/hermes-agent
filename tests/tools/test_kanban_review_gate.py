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
from hermes_cli import kanban_db_connect as kbc


def _gate_comments(conn, tid):
    """The card's comments minus the charter §4 auto-points placeholder.

    Every minted card carries that placeholder (est-mintpath-20260911), so
    these tests count gate comments rather than all comments.
    """
    return [c for c in kb.list_comments(conn, tid) if c.author != "(system)"]


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
    body: str | None = None,
) -> str:
    """Create + claim a card rooted at ``ws`` and set the worker env to it.

    2026-09-06 (G1): the focused rung runs ONLY the test command the card body
    names; pass ``body`` when a test needs the rung to run something.
    """
    monkeypatch.setenv("HERMES_HOME", str(kanban_home))
    monkeypatch.setattr(Path, "home", lambda: kanban_home)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="worktree build gate",
            body=body,
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
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"


def test_gate_failing_build_bounces_card(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _add_worktree(repo, "bad")
    # Syntax error so the import/build sanity check fails.
    _change_python_file(ws, "broken.py", "def x(:\n    pass\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    resp = json.loads(tools._handle_request_review({"summary": "broken"}))
    assert "error" in resp
    assert "Pre-review gate failed" in resp["error"]

    with kbc.connect() as conn:
        # Card never left the builder lane.
        assert kb.get_task(conn, tid).status == "running"
        # No failure counted against the card.
        assert kb.get_task(conn, tid).consecutive_failures == 0
        comments = _gate_comments(conn, tid)
        assert len(comments) == 1
        assert "Pre-review gate FAILED" in comments[0].body
        # The bounce names WHICH rung failed.
        assert "import/build sanity" in comments[0].body


def test_gate_skips_browser_ui_test_when_dist_absent(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A focused UI/browser test is skipped (logged) when frontend/dist is
    absent in the worktree, so an env gap does not false-bounce the card.

    A fresh worktree ships gitignored node_modules/dist absent; a browser/UI
    test in the focused set would ERROR (cannot build the served bundle it
    drives) rather than exercise anything.  The gate skips that rung on
    build-less worktree-only runs; the card's own static tests still gate.
    The worktree has an index build skip (no dist) and the changed file maps
    to a browser/UI test — the gate must proceed to review, not bounce.
    """
    ws = _add_worktree(repo, "uiship")
    # A UI-test module that would need a built bundle; dist is absent (the
    # fresh worktree ships none).
    _change_python_file(ws, "app/mod.py", "X = 1\n")
    _change_python_file(ws, "tests/test_ui_v2.py",
                        "def test_ui_pass():\n    assert 1 == 1\n")

    from tools import kanban_tools as tools

    # The helper itself classifies it as a browser/UI test.
    assert tools._is_browser_ui_test("tests/test_ui_v2.py")
    assert not tools._is_browser_ui_test("tests/test_mymod.py")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    resp = json.loads(tools._handle_request_review({"summary": "ui ship"}))
    assert resp.get("ok") is True, resp
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"


def test_gate_runs_non_ui_test_even_without_dist(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-UI focused test still runs (and gates) even without frontend/dist
    — the skip only drops the browser/UI rung, never masks a real unit test."""
    ws = _add_worktree(repo, "plain")
    _change_python_file(ws, "mymod.py", "GOOD = 1\n")
    _change_python_file(ws, "tests/test_mymod.py", "def test_bad():\n    assert 1 == 2\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws,
                     body="AC: `pytest tests/test_mymod.py -q` exits 0")
    from tools import kanban_tools as tools

    resp = json.loads(tools._handle_request_review({"summary": "plain fail"}))
    # The non-UI focused test failing must still bounce the card.
    assert "error" in resp
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"


def test_gate_comment_carries_output_tail(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _add_worktree(repo, "tail")
    # A test that fails → focused pytest gate returns output tail.
    _change_python_file(ws, "mymod.py", "OK = 1\n")
    _change_python_file(ws, "tests/test_mymod.py", "def test_bad():\n    assert 1 == 2\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws,
                     body="AC: `pytest tests/test_mymod.py -q` exits 0")
    from tools import kanban_tools as tools

    resp = json.loads(tools._handle_request_review({"summary": "flaky"}))
    assert "error" in resp
    with kbc.connect() as conn:
        comments = _gate_comments(conn, tid)
        assert len(comments) == 1
        body = comments[0].body
        assert "```" in body
        assert "import/build sanity" in body or "focused tests" in body


def test_gate_import_ok_for_relative_import_module(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed module using a relative import passes the sanity check.

    Regression: loading a changed module under a synthetic flat name with no
    package context raised ``ImportError: attempted relative import with no
    known parent package`` and falsely bounced any module that does ``from
    .sibling import ...`` (the majority of ``gateway/``, ``agent/``, and
    ``hermes_cli/observability/``). Importing by dotted name via the real
    import machinery preserves the package context, so relative imports
    resolve.
    """
    ws = _add_worktree(repo, "relimport")
    # A real package: pkg/__init__.py so dotted ``pkg.mod`` is importable.
    _change_python_file(ws, "pkg/__init__.py", "")
    _change_python_file(ws, "pkg/sibling.py", "VAL = 42\n")
    _change_python_file(ws, "pkg/mod.py", "from .sibling import VAL\nresult = VAL + 1\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    # The pure helper reports green on the relative-import module.
    changed = tools._changed_python_files(str(ws))
    assert "pkg/mod.py" in changed, changed
    proc = subprocess.run(
        tools._build_sanity_command("python", str(ws), changed),
        capture_output=True, text=True, cwd=str(ws),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    resp = json.loads(tools._handle_request_review({"summary": "rel import ok"}))
    assert resp.get("ok") is True, resp
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"


def test_gate_missing_import_bounces_card(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed module importing a missing module bounces the card.

    The card exists because of ce14358 (a module importing from untracked
    files reached review).  ``py_compile`` cannot see a missing import — this
    proves the real import-resolution check catches it.
    """
    ws = _add_worktree(repo, "badimport")
    _change_python_file(ws, "broken.py", "import totally_missing_module_xyz\nX = 1\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    resp = json.loads(tools._handle_request_review({"summary": "bad import"}))
    assert "error" in resp
    assert "Pre-review gate failed" in resp["error"]

    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"
        assert kb.get_task(conn, tid).consecutive_failures == 0
        comments = _gate_comments(conn, tid)
        assert len(comments) == 1
        assert "import sanity FAIL broken.py" in comments[0].body


def test_gate_import_ok_when_sibling_module_resolves(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed module importing a present sibling passes the sanity check."""
    ws = _add_worktree(repo, "goodimport")
    _change_python_file(ws, "sibling.py", "VAL = 42\n")
    _change_python_file(ws, "uses_sibling.py", "from sibling import VAL\nresult = VAL + 1\n")
    _change_python_file(ws, "tests/test_uses_sibling.py", "def test_ok():\n    assert True\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    resp = json.loads(tools._handle_request_review({"summary": "ok import"}))
    assert resp.get("ok") is True, resp
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"


def test_changed_python_files_handles_porcelain_rename(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A porcelain rename entry feeds only the destination path.

    "R  old.py -> new.py" must not pass the pseudo-path "old.py -> new.py"
    to the gate (it would be a false FileNotFound bounce).
    """
    ws = _add_worktree(repo, "rename")
    old = ws / "renamed.py"
    old.write_text("OLD = 1\n")
    # Stage + `git mv` inside the linked worktree so porcelain emits a rename
    # entry ("R  renamed.py -> renamed_new.py"), not an add/delete pair.
    subprocess.run(["git", "-C", str(ws), "add", "renamed.py"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(ws), "mv", "renamed.py", "renamed_new.py"],
        check=True, capture_output=True,
    )

    from tools import kanban_tools as tools
    changed = tools._changed_python_files(str(ws))
    assert "renamed_new.py" in changed, changed
    assert "renamed.py -> renamed_new.py" not in changed
    assert "->" not in " ".join(changed)


def test_resolve_base_ref_prefers_deepest_ancestor(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale remote-tracking ref must not win the base-ref selection.

    When ``origin/main`` lags behind local ``main``, ``_resolve_base_ref``
    must pick the DEEPEST common ancestor (closest to HEAD) so the gate diff
    scopes to the card's own changes.  Picking stale ``origin/main`` sweeps
    in unrelated backlog commits and maps them to red baseline tests that
    are not the card's responsibility (t_13af5268: frontend-only card
    bounced on backend search/ws AC15 failures).
    """
    from tools import kanban_tools as tools

    # Move local main ahead of origin/main by 3 commits of backend work.
    def commit(path: str, content: str, msg: str) -> None:
        f = repo / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
        subprocess.run(["git", "-C", str(repo), "add", path], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-q", "-m", msg],
            check=True, capture_output=True,
        )

    # origin/main must exist and LAG behind local main.  Point it at the
    # original base commit, then advance local main past it.
    base_sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    # Create the stale remote-tracking ref (no remote configured, so just
    # write the ref directly — exactly what git fetch leaves behind).
    subprocess.run(
        ["git", "-C", str(repo), "update-ref", "refs/remotes/origin/main", base_sha],
        check=True, capture_output=True,
    )
    stale_origin = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "origin/main"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert stale_origin == base_sha

    commit("backend/app/search.py", "def search():\n    pass\n", "backend search")
    commit("backend/app/ws.py", "def ws():\n    pass\n", "backend ws")
    commit("backend/app/db.py", "def db():\n    pass\n", "backend db")
    new_main = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "main"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    # Sanity: local main is now ahead of the stale origin/main.
    assert new_main != base_sha

    # Now a worktree cut off the NEW main — its true base is main/HEAD~1,
    # which is deeper (closer to HEAD) than the stale origin/main.
    ws = _add_worktree(repo, "deepbase")
    _change_python_file(ws, "frontend/Widget.jsx", "export const x = 1\n")

    base = tools._resolve_base_ref(str(ws))
    # Must NOT be the stale origin/main.
    assert base != "origin/main", base

    changed = tools._changed_python_files(str(ws))
    # Only the frontend file must be seen as changed — NOT the backend
    # backlog that a stale-origin diff would sweep in.
    assert changed == [], changed
    # And a frontend-only change must not map to any focused test, so the
    # focused-tests rung is skipped and the gate passes.
    tests = tools._focused_test_paths(str(ws), changed)
    assert tests == [], tests


def test_gate_skipped_for_non_worktree_card(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _add_worktree(repo, "dir-card")
    _change_python_file(ws, "mymod.py", "def f():\n    pass\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws, kind="dir")
    from tools import kanban_tools as tools

    resp = json.loads(tools._handle_request_review({"summary": "dir card"}))
    assert resp.get("ok") is True, resp
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"


def test_gate_command_override_splats_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config override replaces the default pytest gate command.

    ``{tests}`` must expand to one argv element per focused test path (not a
    single space-joined element — pytest would read one bogus path).
    """
    import hermes_cli.config as hcfg
    from tools import kanban_tools as tools

    cfg = {"kanban": {"review_gate": {"enabled": True, "command": ["{python}", "myrunner", "{tests}"]}}}
    monkeypatch.setattr(tools, "load_config", lambda: cfg)
    monkeypatch.setattr(hcfg, "load_config", lambda: cfg)

    cmd = tools._gate_command("/venv/bin/python", ["a.py", "b.py"])
    assert cmd == ["/venv/bin/python", "myrunner", "a.py", "b.py"], cmd


def test_gate_command_override_shared_arg_keeps_own_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A placeholder embedded between fixed text stays one element per test."""
    import hermes_cli.config as hcfg
    from tools import kanban_tools as tools

    cfg = {"kanban": {"review_gate": {"enabled": True, "command": ["{python}", "--tb={tests}", "-q"]}}}
    monkeypatch.setattr(tools, "load_config", lambda: cfg)
    monkeypatch.setattr(hcfg, "load_config", lambda: cfg)

    cmd = tools._gate_command("/venv/bin/python", ["a.py", "b.py"])
    assert cmd == ["/venv/bin/python", "--tb=a.py", "--tb=b.py", "-q"], cmd


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
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"


def _write_tool_script(tmp_path: Path, name: str, body: str) -> str:
    """Write an executable script the gate can run as a lint/typecheck tool."""
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _set_rung_overrides(
    monkeypatch: pytest.MonkeyPatch,
    *,
    lint: list[str] | None = None,
    typecheck: list[str] | None = None,
) -> None:
    """Pin lint_command / typecheck_command in the worker config."""
    import hermes_cli.config as hcfg
    from tools import kanban_tools as tools

    cfg = {
        "kanban": {
            "review_gate": {
                "enabled": True,
                "command": None,
                "lint_command": lint,
                "typecheck_command": typecheck,
            }
        }
    }
    monkeypatch.setattr(tools, "load_config", lambda: cfg)
    monkeypatch.setattr(hcfg, "load_config", lambda: cfg)
    # Reset the per-project tool cache so a previous test's resolution can't
    # bleed in (cache key is (python, key); fixture python differs per test,
    # but be explicit).
    tools._TOOL_CACHE.clear()
    tools._PYTEST_CACHE.clear()


def test_ladder_lint_fail_bounces_before_typecheck(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Short-circuit: a lint failure bounces before typecheck ever runs.

    Both lint and typecheck commands would fail here; because lint runs first
    and short-circuits, the bounce must name lint, carry lint's output only,
    and never execute typecheck.
    """
    ws = _add_worktree(repo, "ladder-lint-first")
    _change_python_file(ws, "mymod.py", "OK = 1\n")
    _change_python_file(ws, "tests/test_mymod.py", "def test_good():\n    assert 1 == 1\n")  # would pass

    lint_script = _write_tool_script(tmp_path, "bad_lint.sh", "echo LINT PROBLEM\nexit 1")
    tc_script = _write_tool_script(tmp_path, "bad_tc.sh", "echo TYPECHECK PROBLEM\nexit 1")
    _set_rung_overrides(
        monkeypatch,
        lint=[lint_script, "{files}"],
        typecheck=[tc_script, "{files}"],
    )

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    resp = json.loads(tools._handle_request_review({"summary": "lint fails"}))
    assert "error" in resp
    assert "'lint'" in resp["error"]  # names the rung
    assert "LINT PROBLEM" in resp["error"]  # carries lint's tail
    assert "TYPECHECK PROBLEM" not in resp["error"]  # typecheck never ran

    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"
        assert kb.get_task(conn, tid).consecutive_failures == 0
        comments = _gate_comments(conn, tid)
        assert len(comments) == 1
        body = comments[0].body
        assert "FAILED on the 'lint' rung" in body
        assert "LINT PROBLEM" in body
        assert "TYPECHECK PROBLEM" not in body
        # The ladder's own value is measurable: which rung bounced is recorded.
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert "gate_bounced" in kinds
        bounced = [e for e in kb.list_events(conn, tid) if e.kind == "gate_bounced"]
        assert bounced[0].payload == {"rung": "lint"}


def test_ladder_missing_linter_skipped_not_failed(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rung whose tool is absent is SKIPPED (not a failure).

    Here no linter is available (default resolution → nothing in the venv),
    but a typecheck override is present and fails. The card must bounce on
    typecheck — the skipped lint rung is not counted as a failure and the
    bounce must name typecheck, not lint.
    """
    ws = _add_worktree(repo, "ladder-skip-lint")
    _change_python_file(ws, "mymod.py", "OK = 1\n")
    _change_python_file(ws, "tests/test_mymod.py", "def test_good():\n    assert 1 == 1\n")

    tc_script = _write_tool_script(tmp_path, "bad_tc.sh", "echo TYPECHECK PROBLEM\nexit 1")
    # lint_command left None → no linter in venv default chain → skipped.
    _set_rung_overrides(monkeypatch, typecheck=[tc_script, "{files}"])

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    resp = json.loads(tools._handle_request_review({"summary": "typecheck fails"}))
    assert "error" in resp
    assert "'typecheck'" in resp["error"]
    assert "TYPECHECK PROBLEM" in resp["error"]

    with kbc.connect() as conn:
        comments = _gate_comments(conn, tid)
        assert len(comments) == 1
        assert "FAILED on the 'typecheck' rung" in comments[0].body
        bounced = [e for e in kb.list_events(conn, tid) if e.kind == "gate_bounced"]
        assert bounced[0].payload == {"rung": "typecheck"}


def test_ladder_all_green_proceeds_to_review(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rung green (lint, typecheck, import/build, focused tests) → review,
    exactly as before the ladder existed."""
    ws = _add_worktree(repo, "ladder-all-green")
    _change_python_file(ws, "mymod.py", "GOOD = 1\n")
    _change_python_file(ws, "tests/test_mymod.py", "def test_good():\n    assert 1 == 1\n")

    lint_ok = _write_tool_script(tmp_path, "ok_lint.sh", "exit 0")
    tc_ok = _write_tool_script(tmp_path, "ok_tc.sh", "exit 0")
    _set_rung_overrides(
        monkeypatch,
        lint=[lint_ok, "{files}"],
        typecheck=[tc_ok, "{files}"],
    )

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    resp = json.loads(tools._handle_request_review({"summary": "all green"}))
    assert resp.get("ok") is True, resp
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"


def test_ladder_tool_detection_is_venv_scoped(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool availability is scoped to the project venv, not the ambient PATH.

    `ruff` is on many hosts' PATH but a project that does nothing special
    should still skip the lint rung (default resolution). With an override the
    rung runs regardless — proving the override is the project-scoped escape
    hatch and the default probes the venv only.
    """
    ws = _add_worktree(repo, "ladder-venv-scope")
    _change_python_file(ws, "mymod.py", "OK = 1\n")
    _change_python_file(ws, "tests/test_mymod.py", "def test_good():\n    assert 1 == 1\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    # No override: venv has no ruff/flake8/pylint/mypy → rung skipped, not failed.
    assert tools._resolve_tool(
        str(repo / "venv" / "bin" / "python"), "lint", tools._LINT_TOOLS
    ) is None
    # The full gate still proceeds to review (skipped rungs are not failures).
    resp = json.loads(tools._handle_request_review({"summary": "no tools, no override"}))
    assert resp.get("ok") is True, resp
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"


# ---------------------------------------------------------------------------
# 09-01 self-test defects: (1) invoke the tool the way it was detected, not
# ``python -m``; (2) a worktree venv without pytest never false-bounces.
# ---------------------------------------------------------------------------


def _write_venv_bin_tool(repo: Path, name: str, body: str) -> Path:
    """Write an executable script into the repo venv ``bin`` (the detection
    surface ``_resolve_tool`` probes) usable as a standalone tool binary."""
    path = repo / "venv" / "bin" / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_rung_command_invokes_detected_binary_directly(repo: Path) -> None:
    """Regression: a standalone default tool is invoked as its own binary,
    never ``python -m <tool>`` (which dies with 'No module named X' and turns
    the missing-tool -> skip contract into a false bounce)."""
    script = _write_venv_bin_tool(repo, "ruff", "echo RAN")
    from tools import kanban_tools as tools

    tools._TOOL_CACHE.clear()
    tools._PYTEST_CACHE.clear()
    pypath = str(repo / "venv" / "bin" / "python")
    cmd = tools._rung_command(pypath, "lint", tools._LINT_TOOLS, ["a.py"])
    # The exact detected binary + ruff's `check` subcommand — no `-m`.
    assert cmd == [str(script), "check", "a.py"], cmd
    # The binary is the one _resolve_tool saw in the same venv bin.
    assert cmd[0] == str(repo / "venv" / "bin" / "ruff")


def test_ladder_standalone_binary_gates(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a standalone ruff binary present in the venv is executed and
    a failing lint bounces the card, carrying the binary's own output (proving
    the binary itself ran, not a failing `python -m ruff`)."""
    ws = _add_worktree(repo, "standalone-ruff")
    _change_python_file(ws, "mymod.py", "OK = 1\n")
    _change_python_file(ws, "tests/test_mymod.py", "def test_good():\n    assert 1 == 1\n")
    _write_venv_bin_tool(repo, "ruff", "echo STANDALONE-RUFF-RAN\nexit 1")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    tools._TOOL_CACHE.clear()
    tools._PYTEST_CACHE.clear()
    resp = json.loads(tools._handle_request_review({"summary": "ruff fails"}))
    assert "error" in resp
    assert "on the 'lint' rung" in resp["error"]
    assert "STANDALONE-RUFF-RAN" in resp["error"]  # the binary itself ran
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"
        assert kb.get_task(conn, tid).consecutive_failures == 0


def test_gate_preexisting_failure_passes_baseline_aware(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-existing focused failure (present on merge-base main) must NOT
    block review — the card is not responsible for it."""
    # Commit a failing test on main.  A worktree change to the corresponding
    # module pulls that (still-failing) test into the focused selection; the
    # merge-base archive run reproduces the same failure, so the card passes.
    subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True, capture_output=True,
    )
    (repo / "prefail.py").write_text("X = 1\n")
    repo / "tests"
    tdir = repo / "tests"
    tdir.mkdir(exist_ok=True)
    (tdir / "test_prefail.py").write_text("def test_prefail():\n    assert False\n")
    subprocess.run(["git", "-C", str(repo), "add", "prefail.py", "tests/test_prefail.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "prefail"], check=True, capture_output=True)

    ws = _add_worktree(repo, "preexisting")
    # A card-legit change to the module that routes to the failing test.
    _change_python_file(ws, "prefail.py", "X = 2\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws,
                     body="AC: `pytest tests/test_prefail.py -q` exits 0")
    from tools import kanban_tools as tools

    tools._BASE_ARCHIVE_CACHE.clear()
    resp = json.loads(tools._handle_request_review({"summary": "only pre-existing failure"}))
    assert resp.get("ok") is True, resp
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"


def test_gate_new_failure_still_bounces_baseline_aware(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure the worktree INTRODUCES (absent from merge-base main) still
    blocks review even with baseline awareness."""
    # main has a passing test.
    (repo / "tests").mkdir(exist_ok=True)
    (repo / "tests" / "test_newfail.py").write_text("def test_old():\n    assert True\n")
    subprocess.run(["git", "-C", str(repo), "add", "tests/test_newfail.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "passing"], check=True, capture_output=True)

    ws = _add_worktree(repo, "newfail")
    # The worktree adds a NEW failing test to the changed test file.
    _change_python_file(
        ws, "tests/test_newfail.py",
        "def test_old():\n    assert True\n\ndef test_new():\n    assert False\n",
    )

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws,
                     body="AC: `python -m pytest tests/test_newfail.py -v` exits 0")
    from tools import kanban_tools as tools

    tools._BASE_ARCHIVE_CACHE.clear()
    resp = json.loads(tools._handle_request_review({"summary": "introduced failure"}))
    assert "error" in resp
    assert "Pre-review gate failed" in resp["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"
        comments = _gate_comments(conn, tid)
        assert len(comments) == 1
        assert "focused tests" in comments[0].body


def _write_pytest_missing_python(repo: Path) -> None:
    """Replace the repo venv's python with one where ``import pytest`` fails.

    The gate probes pytest via ``python -c 'import pytest'``; a real interpreter
    without pytest returns nonzero.  This shim fails exactly that probe while
    passing everything else through to the real interpreter, so the import rung
    still validates non-test modules under a genuinely pytest-less venv.
    """
    real = sys.executable
    pypath = repo / "venv" / "bin" / "python"
    pypath.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-c" ] && [ "$2" = "import pytest" ]; then\n'
        "  exit 1\n"
        "fi\n"
        f'exec {real} "$@"\n'
    )
    pypath.chmod(pypath.stat().st_mode | stat.S_IEXEC)


def test_worktree_venv_without_pytest_passes_import_rung(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh worktree venv lacking pytest never false-bounces a card whose
    changed test file does ``import pytest``.

    The import rung skips (logged) the pytest-dependent test file while still
    validating the changed non-test module, and the focused-tests rung is
    skipped — so a good card proceeds to review instead of dying on the gate's
    missing toolchain."""
    ws = _add_worktree(repo, "no-pytest")
    _write_pytest_missing_python(repo)
    _change_python_file(ws, "mymod.py", "OK = 1\n")
    # The changed test file imports pytest at module load — under the pytest-
    # less venv a naive import rung would false-bounce on ImportError.
    _change_python_file(ws, "tests/test_mymod.py", "import pytest\ndef test_good():\n    assert 1 == 1\n")

    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    tools._TOOL_CACHE.clear()
    tools._PYTEST_CACHE.clear()
    assert tools._pytest_importable(str(repo / "venv" / "bin" / "python"), str(ws)) is False
    resp = json.loads(tools._handle_request_review({"summary": "venv has no pytest"}))
    assert resp.get("ok") is True, resp
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"




# ---------------------------------------------------------------------------
# 09-06 per-card scoped-test selection: the focused-tests rung must honor the
# card's own scope rather than derive the run set from the worktree diff
# (sibling commits onto shared main pollute the diff with out-of-scope red
# baseline tests — t_4eea8efe / t_058f9fd0 defect).
# ---------------------------------------------------------------------------


def test_scoped_paths_from_body_pytest_command(tmp_path: Path) -> None:
    """A ``pytest tests/test_ws.py -q`` AC line selects exactly that file."""
    from tools import kanban_tools as ktools

    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    (ws / "tests" / "test_ws.py").write_text("")
    (ws / "tests" / "test_search_backend.py").write_text("")
    body = (
        "## Scope\nMake `tests/test_ws.py` pass.\n\n"
        "### Acceptance criteria\n"
        "- `./.venv/bin/python -m pytest tests/test_ws.py -q` exits 0\n"
        "- Do NOT modify `backend/app/search.py` or its scoped tests.\n"
    )
    parsed = ktools._scoped_test_paths_from_body(body, ws)
    assert parsed == ["tests/test_ws.py"], parsed



def test_scoped_paths_no_command_falls_back_empty(tmp_path: Path) -> None:
    """A body with no scoped command yields [] -> the rung runs NOTHING (G1)."""
    from tools import kanban_tools as ktools

    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    (ws / "tests" / "test_other.py").write_text("")
    assert ktools._scoped_test_paths_from_body("Just build it.", ws) == []


def test_scoped_paths_missing_file_degrades_empty(tmp_path: Path) -> None:
    """A stale pointer (file absent) degrades to [] -> nothing runs (G1)."""
    from tools import kanban_tools as ktools

    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    body = "`pytest tests/test_ghost.py -q` exits 0"
    assert ktools._scoped_test_paths_from_body(body, ws) == []


def test_scoped_paths_inline_code_trailing_backtick(tmp_path: Path) -> None:
    """An inline ``...`` span never yields a bogus trailing-backtick path."""
    from tools import kanban_tools as ktools

    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    (ws / "tests" / "test_ws.py").write_text("")
    body = "AC: `pytest tests/test_ws.py -q` exits 0"
    parsed = ktools._scoped_test_paths_from_body(body, ws)
    assert parsed == ["tests/test_ws.py"], parsed
    assert all(not p.endswith("`") for p in parsed)


def test_scoped_paths_multiple_paths_venv_prefix(tmp_path: Path) -> None:
    """A venv-prefixed AC naming MULTIPLE test paths selects exactly those.

    Regression (t_6fa75c84 / t_eafe2bd3): the extractor regex only matched a
    SINGLE path, so a card whose AC runs
    ``.venv/bin/python -m pytest tests/test_worker_pool.py
    tests/test_worker_pool_regressions.py -q`` was NOT matched and the gate
    silently degraded to the diff-derived fallback, sweeping in out-of-scope
    red tests (test_search_backend.py) and bouncing a card whose scoped AC was
    green.  The multi-path invocation must yield both paths, in order, and the
    rung must run exactly those.
    """
    from tools import kanban_tools as ktools

    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    (ws / "tests" / "test_worker_pool.py").write_text("")
    (ws / "tests" / "test_worker_pool_regressions.py").write_text("")
    (ws / "tests" / "test_search_backend.py").write_text("")
    body = (
        "### Acceptance criteria\n"
        "`.venv/bin/python -m pytest tests/test_worker_pool.py "
        "tests/test_worker_pool_regressions.py -q` exits 0\n"
        "- Do NOT modify `backend/app/search.py` or its scoped tests.\n"
    )
    parsed = ktools._scoped_test_paths_from_body(body, ws)
    assert parsed == [
        "tests/test_worker_pool.py",
        "tests/test_worker_pool_regressions.py",
    ], parsed
    assert "tests/test_search_backend.py" not in parsed


def test_scoped_paths_multiple_paths_one_absent_keeps_resolvable(tmp_path: Path) -> None:
    """A multi-file AC with one stale pointer keeps the resolvable path.

    The resolve-under-worktree guard drops only the missing path — it must NOT
    drop the whole invocation (which would degrade to the diff fallback and
    lose the card's real scope).  The guard is per-path, not all-or-nothing.
    """
    from tools import kanban_tools as ktools

    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    (ws / "tests" / "test_present.py").write_text("")
    body = (
        "`python -m pytest tests/test_present.py tests/test_ghost.py -q` "
        "exits 0"
    )
    parsed = ktools._scoped_test_paths_from_body(body, ws)
    assert parsed == ["tests/test_present.py"], parsed


def test_scoped_paths_multiple_paths_all_absent_degrades_empty(tmp_path: Path) -> None:
    """A multi-file AC where ALL paths are absent degrades to [] -> diff fallback.

    A fully-stale multi-path command must degrade cleanly (not crash).
    """
    from tools import kanban_tools as ktools

    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    body = (
        "`pytest tests/test_ghost.py tests/test_spectre.py -q` exits 0"
    )
    parsed = ktools._scoped_test_paths_from_body(body, ws)
    assert parsed == [], parsed


# ---------------------------------------------------------------------------
# G1 (2026-09-06): every real command shape, and no diff-derived fallback
# ---------------------------------------------------------------------------


def test_scoped_paths_node_id_and_verbose_flag(tmp_path: Path) -> None:
    """The shape every 6 Sep card used: venv prefix, ``::node``, ``-v``."""
    from tools import kanban_tools as ktools

    ws = tmp_path / "ws"
    (ws / "tests").mkdir(parents=True)
    (ws / "tests" / "test_ui_v2.py").write_text("")
    body = ("1. `cd <worktree> && .venv/bin/python -m pytest "
            "tests/test_ui_v2.py::test_drag_reorders -v` exits 0.")
    assert ktools._scoped_test_paths_from_body(body, ws) == ["tests/test_ui_v2.py::test_drag_reorders"]


def test_scoped_paths_flag_before_path_and_directory(tmp_path: Path) -> None:
    from tools import kanban_tools as ktools

    ws = tmp_path / "ws"
    (ws / "tests" / "regression").mkdir(parents=True)
    assert ktools._scoped_test_paths_from_body("`pytest -q tests/regression`", ws) == ["tests/regression"]


def test_gate_runs_nothing_when_body_names_no_command(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control for the diff fallback: a worktree that changes a module
    whose mirror test FAILS enters review when the body names no command —
    proof the diff-derived selection is gone. (E1 makes the command mandatory
    at mint, so this is the safe direction: silence, not a false bounce.)"""
    (repo / "tests").mkdir(exist_ok=True)
    (repo / "tests" / "test_mod.py").write_text("def test_x():\n    assert False\n")
    (repo / "mod.py").write_text("X = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "mod.py", "tests/test_mod.py"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "mod"], check=True, capture_output=True)
    ws = _add_worktree(repo, "nocmd")
    _change_python_file(ws, "mod.py", "X = 2\n")
    tid = _make_task(tmp_path / ".hermes", monkeypatch, ws)
    from tools import kanban_tools as tools

    tools._BASE_ARCHIVE_CACHE.clear()
    resp = json.loads(tools._handle_request_review({"summary": "no command in body"}))
    assert resp.get("ok") is True, resp
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "review"
