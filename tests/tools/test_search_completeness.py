"""search_files says when a content search is the whole answer, and never says it of a partial one.

Live routing eval: after a correct search_files result Opus re-ran the same search with grep
"to double-check" on every count / case-insensitive / rename task. A result that states it is
complete removes that reason; a result that claims completeness while rows were dropped would
teach the model to stop trusting the tool, so truncation detection is pinned on both engines.
"""

import json
import os

import pytest

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

    (proj / ".hidden.py").write_text("hit\n")
    (proj / ".gitignore").write_text("build.log\n")
    (proj / "build.log").write_text("hit\n")
    (proj / ".git").mkdir()  # makes .gitignore apply; .git itself stays pruned
    r = _search(proj, "t-complete-hidden", pattern="hit")
    assert r["total_count"] == 6
    assert "2 hidden or gitignored file(s) also match" in r["complete"]
    assert ".hidden.py" in r["complete"] and "build.log" in r["complete"]


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
