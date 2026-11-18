"""search_files says when a content search is the whole answer, and never says it of a partial one.

Live routing eval: after a correct search_files result Opus re-ran the same search with grep
"to double-check" on every count / case-insensitive / rename task. A result that states it is
complete removes that reason; a result that claims completeness while rows were dropped would
teach the model to stop trusting the tool, so truncation detection is pinned on both engines.
"""

import json
import os

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.file_operations_common import ExecuteResult
from tools.file_tools import _get_file_ops, search_tool


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    d = tmp_path / "proj"
    d.mkdir()
    (d / "one.py").write_text("hit\n")
    (d / "two.py").write_text("hit\nhit\nhit\n")
    (d / "three.py").write_text("hit\nhit\n")
    return d


def _search(proj, task, **kw):
    return json.loads(search_tool(path=str(proj), task_id=task, **kw))


def _with_home(real_exec, home):
    """The backend's ``echo $HOME`` (how ``~`` expands) answers ``home``; every other command runs."""
    def run(command, *a, **kw):
        return ExecuteResult(stdout=home) if command == "echo $HOME" else real_exec(command, *a, **kw)
    return run


def _force_engine(monkeypatch, task, engine):
    ops = _get_file_ops(task_id=task)
    if not ops._has_command(engine):
        pytest.skip(f"{engine} not installed")
    real = ops._has_command
    monkeypatch.setattr(ops, "_has_command",
                        lambda cmd: real(cmd) if cmd not in ("rg", "grep") or cmd == engine else False)


@pytest.mark.parametrize("engine", ["rg", "grep"])
@pytest.mark.parametrize("mode", ["content", "files_only", "count"])
def test_a_page_that_holds_every_row_is_complete_and_one_row_short_is_truncated(proj, monkeypatch, engine, mode):
    rows = {"content": 6, "files_only": 3, "count": 3}[mode]
    task = f"t-complete-{engine}-{mode}"
    _force_engine(monkeypatch, task, engine)

    full = _search(proj, task, pattern="hit", output_mode=mode, limit=rows)
    assert "truncated" not in full
    assert full["complete"].startswith("Complete: ")

    short = _search(proj, task, pattern="hit", output_mode=mode, limit=rows - 1)
    assert short["truncated"] is True
    assert "complete" not in short
    assert f"offset={rows - 1}" in short["_hint"]


def test_count_mode_ranks_files_highest_first_and_states_the_total(proj):
    r = _search(proj, "t-complete-rank", pattern="hit", output_mode="count")
    assert [n for n in r["counts"].values()] == [3, 2, 1]
    assert r["total_count"] == 6
    assert "6 matching lines in 3 files" in r["complete"]


def test_complete_says_whether_hidden_or_gitignored_files_also_match(proj):
    if not _get_file_ops(task_id="t-complete-hidden")._has_command("rg"):
        pytest.skip("rg not installed")
    clean = _search(proj, "t-complete-hidden", pattern="hit")
    assert "none match" in clean["complete"]
    assert "node_modules" in clean["complete"]  # the claim names the trees the check prunes

    (proj / ".hidden.py").write_text("hit\n")
    (proj / ".gitignore").write_text("build.log\n")
    (proj / "build.log").write_text("hit\n")
    (proj / ".git").mkdir()  # makes .gitignore apply; .git itself stays pruned
    r = _search(proj, "t-complete-hidden", pattern="hit")
    assert r["total_count"] == 6
    assert "2 more file(s) match there" in r["complete"]
    assert ".hidden.py" in r["complete"] and "build.log" in r["complete"]


def test_hidden_check_uses_the_backend_root_and_never_costs_the_primary_result(proj, monkeypatch):
    task = "t-complete-tilde"
    ops = _get_file_ops(task_id=task)
    if not ops._has_command("rg"):
        pytest.skip("rg not installed")
    (proj / ".hidden.py").write_text("hit\n")
    monkeypatch.setattr(ops, "_exec", _with_home(ops._exec, str(proj.parent)))
    tilde = json.loads(search_tool(pattern="hit", path="~/proj", task_id=task))
    assert ".hidden.py" in tilde["complete"], tilde

    def broken(*_a, **_k):
        raise OSError("probe exploded")

    monkeypatch.setattr(ops, "hidden_match_counts", broken)
    r = _search(proj, task, pattern="hit")
    assert r["total_count"] == 6 and "were not verified" in r["complete"]


def test_later_pages_and_context_rows_never_feed_the_claim(proj):
    task = "t-complete-pages"
    second = _search(proj, task, pattern="hit", output_mode="files_only", limit=2, offset=2)
    assert len(second["files"]) == 1 and "complete" not in second  # a final page is not the whole answer
    pages = [_search(proj, task, pattern="hit", output_mode="count", limit=1, offset=o).get("counts", {})
             for o in (0, 1, 2, 3)]
    assert [len(p) for p in pages] == [1, 1, 1, 0]  # one row per page, not the whole prefix

    ctx_dir = proj.parent / "ctx"
    ctx_dir.mkdir()
    (ctx_dir / "a.py").write_text("before\nhit\nafter\n")
    ctx = _search(ctx_dir, task, pattern="hit", context=1)
    assert ctx["complete"].startswith("Complete: 3 result rows in 1 files")  # 1 hit + 2 context rows


def test_a_visible_filename_with_a_space_is_listed_not_called_hidden(proj):
    (proj / "visible file.txt").write_text("hit\n")
    r = _search(proj, "t-complete-space", pattern="hit", output_mode="files_only")
    assert r["total_count"] == 4 and any(f.endswith("visible file.txt") for f in r["files"])
    assert "none match" in r["complete"]


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("lane", ["rg-native", "rg-shell", "grep"])
def test_a_partial_search_still_lists_a_readable_filename_with_a_space(proj, monkeypatch, lane):
    """rg exits 2 beside an unreadable dir; the readable spaced name must survive that partial scan.
    Rooted at "." so the rows are "./visible file.txt", with no "-<digit>" in a tmp path to rescue them."""
    if os.geteuid() == 0:
        pytest.skip("root reads mode-000 directories")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(proj)), cwd=str(proj))
    if not ops._has_command("rg" if lane.startswith("rg") else "grep"):
        pytest.skip("search engine not installed")
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "1" if lane == "rg-native" else "0")
    if lane == "grep":
        real = ops._has_command
        monkeypatch.setattr(ops, "_has_command", lambda cmd: False if cmd == "rg" else real(cmd))
    (proj / "visible file.txt").write_text("hit\n")
    locked = proj / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        r = ops.search("hit", path=".", output_mode="files_only")
    finally:
        locked.chmod(0o755)
    assert r.error is None and "unreadable" in (r.warning or "")
    assert sorted(os.path.basename(f) for f in r.files) == ["one.py", "three.py", "two.py", "visible file.txt"]


@pytest.mark.platforms("linux", "macos")
def test_a_search_that_skipped_unreadable_paths_claims_nothing(proj):
    if os.geteuid() == 0:
        pytest.skip("root reads mode-000 directories")
    locked = proj / "locked"
    locked.mkdir()
    (locked / "x.py").write_text("hit\n")
    locked.chmod(0)
    try:
        r = _search(proj, "t-complete-locked", pattern="hit")
    finally:
        locked.chmod(0o755)
    assert "unreadable" in r["warning"]
    assert "complete" not in r


@pytest.mark.platforms("linux", "macos")
def test_an_unreadable_hidden_dir_leaves_hidden_files_unverified(proj, monkeypatch):
    if os.geteuid() == 0:
        pytest.skip("root reads mode-000 directories")
    task = "t-complete-hidden-locked"
    ops = _get_file_ops(task_id=task)
    if not ops._has_command("rg"):
        pytest.skip("rg not installed")
    monkeypatch.setattr(ops, "_native_read_enabled", lambda: False)  # the shell `| head` lane
    locked = proj / ".private"
    locked.mkdir()
    locked.chmod(0)
    try:
        r = _search(proj, task, pattern="hit")
    finally:
        locked.chmod(0o755)
    assert "were not verified" in r["complete"]
