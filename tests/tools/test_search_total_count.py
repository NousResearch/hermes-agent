"""Regression tests for search_files total_count under-reporting (#79530).

The rg search pipeline reads at most ``fetch_limit`` rows — ``| head`` on the
shell lane, an early kill on the native lane — but ``total_count`` was derived
from the fetched rows only. Any search with more matches than the fetch ceiling
therefore reported its own page size as the total (5 of 1300 real matches) and
never set ``truncated``. The fix marks the page truncated when the fetch filled
its window and replaces the floor with an untruncated rg count (``-c`` summed
for content/count modes, ``-l | wc -l`` for files_only).
"""

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ExecuteResult, ShellFileOperations


MATCHING_FILES = 1300
FETCH_CEILING = 5


@pytest.fixture(scope="module")
def big_dir(tmp_path_factory):
    """A directory with more matching files than the fetch ceiling."""
    path = tmp_path_factory.mktemp("big79530")
    for i in range(MATCHING_FILES):
        (path / f"file_{i}.md").write_text(f"email: pessoa{i}@teste.com\n", encoding="utf-8")
    return path


def _ops(path):
    return ShellFileOperations(LocalEnvironment(str(path)))


def test_content_mode_total_count_is_true_total(big_dir):
    """Default content mode: all 1300 matches counted, page stays at 5 rows."""
    result = _ops(big_dir)._search_content(
        "email", str(big_dir), None, FETCH_CEILING, 0, "content", 0)

    assert len(result.matches) == FETCH_CEILING
    assert result.total_count == MATCHING_FILES
    assert result.truncated is True
    # The re-count is exact: a cut page is not a lower bound here.
    assert "total_count_is_lower_bound" not in result.to_dict()


def test_shell_lane_total_count_is_true_total(big_dir, monkeypatch):
    """The correction is transport-independent: the ``| head`` lane reports it too."""
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
    result = _ops(big_dir)._search_content(
        "email", str(big_dir), None, FETCH_CEILING, 0, "content", 0)

    assert len(result.matches) == FETCH_CEILING
    assert result.total_count == MATCHING_FILES
    assert result.truncated is True


@pytest.mark.parametrize("output_mode", ["files_only", "count"])
def test_shell_lane_all_output_modes_report_true_total(big_dir, monkeypatch, output_mode):
    """``-l | wc -l`` and ``-c | awk`` also run on the shell lane."""
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
    result = _ops(big_dir)._search_content(
        "email", str(big_dir), None, FETCH_CEILING, 0, output_mode, 0)

    assert result.total_count == MATCHING_FILES
    assert result.truncated is True


def test_files_only_mode_total_count_is_true_total(big_dir):
    result = _ops(big_dir)._search_content(
        "email", str(big_dir), None, FETCH_CEILING, 0, "files_only", 0)

    assert len(result.files) == FETCH_CEILING
    assert result.total_count == MATCHING_FILES
    assert result.truncated is True


def test_count_mode_total_count_is_true_total_with_bounded_per_file_table(big_dir):
    result = _ops(big_dir)._search_content(
        "email", str(big_dir), None, FETCH_CEILING, 0, "count", 0)

    assert result.total_count == MATCHING_FILES
    assert result.truncated is True
    # Only the total is exact: the per-file table stays page-bounded.
    assert len(result.counts) == FETCH_CEILING


def test_offset_page_reports_the_full_total(big_dir):
    """An offset page keeps the full total and stays flagged truncated."""
    result = _ops(big_dir)._search_content(
        "email", str(big_dir), None, FETCH_CEILING, 10, "content", 0)

    assert len(result.matches) == FETCH_CEILING
    assert result.total_count == MATCHING_FILES
    assert result.truncated is True


def test_count_mode_sums_lines_not_individual_hits(tmp_path):
    """``-c`` keeps the row unit: two hits on one line are one content row."""
    for i in range(6):
        (tmp_path / f"f{i}.txt").write_text("foo foo\n", encoding="utf-8")

    result = _ops(tmp_path)._search_content("foo", str(tmp_path), None, 2, 0, "content", 0)

    assert len(result.matches) == 2
    assert result.total_count == 6
    assert result.truncated is True


def test_count_with_colons_in_file_names(tmp_path):
    """The per-file table splits on the LAST colon, so a colon in a name is safe."""
    for i in range(6):
        (tmp_path / f"f{i}:x.txt").write_text("needle\n", encoding="utf-8")

    result = _ops(tmp_path)._search_content("needle", str(tmp_path), None, 2, 0, "count", 0)

    assert result.total_count == 6
    assert result.truncated is True


def test_result_exactly_at_the_ceiling_is_complete_and_not_truncated(tmp_path):
    """A complete result that happens to fill the window stays truncated=False."""
    for i in range(FETCH_CEILING):
        (tmp_path / f"f{i}.txt").write_text(f"needle {i}\n", encoding="utf-8")

    result = _ops(tmp_path)._search_content("needle", str(tmp_path), None, FETCH_CEILING, 0, "content", 0)

    assert result.total_count == FETCH_CEILING
    assert result.truncated is False
    assert "total_count_is_lower_bound" not in result.to_dict()


def test_small_dir_total_count_unchanged_and_no_count_query(tmp_path, monkeypatch):
    """Below the ceiling the fetched rows are the total — no extra rg run."""
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text(f"needle {i}\n", encoding="utf-8")
    ops = _ops(tmp_path)
    calls = []
    monkeypatch.setattr(
        ops, "_count_rg_exact_total",
        lambda *args, **kwargs: calls.append(args), raising=False)

    result = ops._search_content("needle", str(tmp_path), None, 5, 0, "content", 0)

    assert result.total_count == 3
    assert result.truncated is False
    assert calls == []


def test_context_search_keeps_the_bounded_estimate(big_dir, monkeypatch):
    """Context rows are counted as matches, so a line count is a different unit."""
    ops = _ops(big_dir)
    monkeypatch.setattr(
        ops, "_count_rg_exact_total",
        lambda *args, **kwargs: pytest.fail("context searches must not re-count"),
        raising=False)

    result = ops._search_content("email", str(big_dir), None, FETCH_CEILING, 0, "content", 1)

    assert result.total_count < MATCHING_FILES
    assert result.to_dict()["total_count_is_lower_bound"] is True


def test_failed_count_query_keeps_the_fetched_estimate_but_flags_truncation(big_dir, monkeypatch):
    """An untrustworthy count never becomes a total; the cut page is still flagged."""
    ops = _ops(big_dir)
    monkeypatch.setattr(ops, "_count_rg_exact_total", lambda *args, **kwargs: None, raising=False)

    result = ops._search_content("email", str(big_dir), None, FETCH_CEILING, 0, "content", 0)

    assert result.total_count == FETCH_CEILING
    assert result.truncated is True
    assert "total_count_is_lower_bound" in result.to_dict()


def test_grep_fallback_marks_truncation_without_an_exact_count(big_dir, monkeypatch):
    """rg-less hosts get the honest flag even though no re-count runs."""
    ops = _ops(big_dir)
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "grep")

    result = ops._search_content("email", str(big_dir), None, FETCH_CEILING, 0, "content", 0)

    assert result.total_count == FETCH_CEILING
    assert result.truncated is True
    assert "total_count_is_lower_bound" in result.to_dict()


def test_multiline_search_skips_the_recount(tmp_path, monkeypatch):
    """A multiline match spans rows, so a line count is not the content unit."""
    for i in range(6):
        (tmp_path / f"f{i}.txt").write_text("foo\nbar\n", encoding="utf-8")
    ops = _ops(tmp_path)
    monkeypatch.setattr(
        ops, "_count_rg_exact_total",
        lambda *args, **kwargs: pytest.fail("multiline searches must not re-count"),
        raising=False)

    result = ops._search_content("foo\nbar", str(tmp_path), None, 2, 0, "content", 0)

    assert result.total_count < 6
    assert result.truncated is True


def test_timeout_never_launches_a_recount(big_dir, monkeypatch):
    """A timed-out fetch is partial output, not a ceiling hit to count past."""
    ops = _ops(big_dir)
    calls = []
    monkeypatch.setattr(
        ops, "_count_rg_exact_total",
        lambda *args, **kwargs: calls.append(args), raising=False)
    monkeypatch.setattr(
        ops, "_run_rg_bounded",
        lambda words, fetch_limit, timeout, **kwargs: ExecuteResult(
            stdout="\n".join(f"/tmp/f{i}.txt:1:email" for i in range(fetch_limit))
            + "\n[Command timed out after 60s]",
            exit_code=124))

    result = ops._search_content("email", str(big_dir), None, 3, 0, "content", 0)

    assert result.limit_reason == "search_timeout"
    assert result.truncated is True
    assert calls == []


def test_recount_below_the_page_keeps_the_page_total(big_dir, monkeypatch):
    """A smaller count (tree changed under us) never undercuts the returned page."""
    ops = _ops(big_dir)
    monkeypatch.setattr(ops, "_count_rg_exact_total", lambda *args, **kwargs: 3, raising=False)

    result = ops._search_content("email", str(big_dir), None, FETCH_CEILING, 0, "content", 0)

    assert result.total_count == FETCH_CEILING
    assert result.truncated is True
    assert "total_count_is_lower_bound" in result.to_dict()
