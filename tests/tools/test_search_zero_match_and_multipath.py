"""Tests for search_files zero-match probes and multi-path recovery."""

import json
import os

import pytest

from tools.file_tools import search_tool


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    d = tmp_path / "proj"
    d.mkdir()
    (d / "a.py").write_text("TOKEN_ALPHA = 'find_me_value'\nother = 1\n")
    (d / "b.py").write_text("x = compute(TOKEN_ALPHA)\n")
    e = tmp_path / "extra"
    e.mkdir()
    (e / "c.txt").write_text("TOKEN_ALPHA appears here too\n")
    return tmp_path


class TestZeroMatchProbe:
    def test_case_mismatch_gets_hint(self, proj):
        r = json.loads(search_tool("token_alpha", path=str(proj / "proj"), task_id="t-zm"))
        assert r["total_count"] == 0
        assert "case-insensitive" in r.get("warning", "")

    def test_regex_metachar_literal_hint(self, proj):
        d = proj / "proj"
        (d / "meta.py").write_text("result = lookup[key+1]\n")
        r = json.loads(search_tool("lookup[key+1]", path=str(d), task_id="t-zm"))
        assert r["total_count"] == 0
        assert "literal match" in r.get("warning", "")

    def test_true_zero_match_no_hint(self, proj):
        r = json.loads(search_tool("zzz_totally_absent_zzz", path=str(proj / "proj"), task_id="t-zm"))
        assert r["total_count"] == 0
        assert "warning" not in r

    def test_hidden_only_match_gets_hint(self, proj):
        d = proj / "proj"
        (d / ".secretdir").mkdir()
        (d / ".secretdir" / "conf.cfg").write_text("HIDDEN_ONLY_TOKEN = true\n")
        r = json.loads(search_tool("HIDDEN_ONLY_TOKEN", path=str(d), task_id="t-zm"))
        assert r["total_count"] == 0
        # "hidden" (not the full rg-only "hidden or gitignored" wording) so
        # the assertion holds on both engines: rg names gitignored files,
        # the grep fallback has no .gitignore awareness and only names hidden.
        assert "hidden" in r.get("warning", "")

    def test_matching_search_unaffected(self, proj):
        r = json.loads(search_tool("TOKEN_ALPHA", path=str(proj / "proj"), task_id="t-zm"))
        assert r["total_count"] >= 2
        assert "warning" not in r


class TestMultiPathRecovery:
    def test_two_existing_paths_merged(self, proj):
        p = f"{proj / 'proj'} {proj / 'extra'}"
        r = json.loads(search_tool("TOKEN_ALPHA", path=p, task_id="t-mp"))
        assert "error" not in r
        assert r["total_count"] >= 3
        blob = json.dumps(r)
        assert "a.py" in blob and "c.txt" in blob
        assert "2 entries" in r.get("warning", "") or "searched 2" in r.get("warning", "")

    def test_missing_path_skipped_with_note(self, proj):
        p = f"{proj / 'proj'} {proj / 'nonexistent_dir'}"
        r = json.loads(search_tool("TOKEN_ALPHA", path=p, task_id="t-mp"))
        assert "error" not in r
        assert r["total_count"] >= 2
        assert "skipped missing" in r.get("warning", "")

    def test_comma_separated_paths(self, proj):
        p = f"{proj / 'proj'},{proj / 'extra'}"
        r = json.loads(search_tool("TOKEN_ALPHA", path=p, task_id="t-mp"))
        assert "error" not in r
        assert r["total_count"] >= 3

    def test_all_missing_still_errors(self, proj):
        p = f"{proj / 'gone1'} {proj / 'gone2'}"
        r = json.loads(search_tool("TOKEN_ALPHA", path=p, task_id="t-mp"))
        assert "error" in r

    def test_single_missing_path_keeps_similar_hint(self, proj):
        # single-path miss must keep the existing "Similar paths" behavior
        r = json.loads(search_tool("TOKEN_ALPHA", path=str(proj / "pro"), task_id="t-mp"))
        assert "error" in r
        assert "Path not found" in r["error"]

    def test_files_target_multi_path(self, proj):
        p = f"{proj / 'proj'} {proj / 'extra'}"
        r = json.loads(search_tool("*.py", path=p, target="files", task_id="t-mp"))
        assert "error" not in r
        blob = json.dumps(r)
        assert "a.py" in blob


class TestZeroMatchProbeEngineParity:
    """The hint must be attached on BOTH search engines' code paths.

    The probe block originally lived inline after the grep call. Three minutes
    later a separate commit (auto-multiline) added an early ``return result`` to
    the ripgrep branch, which orphaned the block: the entire zero-match
    steering tier was unreachable for every user with rg installed, while the
    feature's own tests reported the absence as a plain assertion failure.
    Fixed in 794d6c434e; these tests pin the wiring per engine so a future
    early return can't silently orphan it again.

    The probe shells out to the host's search engine (rg when installed,
    grep otherwise), so the hint text differs per engine. The probe is
    therefore stubbed to a sentinel here — this isolates the *wiring* rather
    than the probe's own engine dependency, and a naive parity test asserting
    real hint text would fail on the grep leg for an unrelated reason.
    """

    @pytest.mark.parametrize("engine", ["rg", "grep"])
    def test_hint_is_attached_on_each_search_engine(self, proj, monkeypatch, engine):
        from tools.file_tools import _get_file_ops

        ops = _get_file_ops(task_id=f"t-parity-{engine}")
        if not ops._has_command(engine):
            pytest.skip(f"{engine} not installed")
        real = ops._has_command

        def only(cmd, _real=real, _keep=engine):
            # Forces which engine `search` picks; other lookups pass through.
            if cmd in ("rg", "grep"):
                return _real(cmd) if cmd == _keep else False
            return _real(cmd)

        monkeypatch.setattr(ops, "_has_command", only)
        monkeypatch.setattr(ops, "_zero_match_probe", lambda *a, **k: "SENTINEL_HINT")
        r = ops.search("token_alpha", path=str(proj / "proj"), target="content")
        assert r.total_count == 0
        assert "SENTINEL_HINT" in (r.warning or ""), (
            f"zero-match hint not wired on the {engine} path: warning={r.warning!r}"
        )

    def test_hint_not_attached_when_matches_exist(self, proj, monkeypatch):
        from tools.file_tools import _get_file_ops

        ops = _get_file_ops(task_id="t-parity-hit")
        monkeypatch.setattr(ops, "_zero_match_probe", lambda *a, **k: "SENTINEL_HINT")
        r = ops.search("TOKEN_ALPHA", path=str(proj / "proj"), target="content")
        assert r.total_count > 0
        assert "SENTINEL_HINT" not in (r.warning or "")

    def test_rg_path_still_skips_line_oriented_newline_warning(self, proj):
        """The early return existed to skip a grep-only warning — keep that.

        rg auto-enables --multiline for ``\\n`` patterns, so the line-oriented
        explanation must not be attached on the rg path. A fix that merely
        deleted the early return would regress this.
        """
        from tools.file_tools import _get_file_ops

        ops = _get_file_ops(task_id="t-parity-nl")
        if not ops._has_command("rg"):
            pytest.skip("rg not installed")
        r = ops.search("TOKEN_ALPHA\\nother", path=str(proj / "proj"), target="content")
        assert "line-oriented" not in (r.warning or "")


class TestZeroMatchProbeGrepFallback:
    """The real zero-match probe must fire on the grep fallback too.

    Issue #79512: _zero_match_probe returned None immediately when rg was
    missing, so grep-only hosts silently lost the case-insensitive / literal /
    hidden-only hints. These tests pin the real probe (not a stub) against
    the grep engine by faking rg's absence via _has_command.
    """

    @pytest.fixture
    def grep_ops(self, proj, monkeypatch):
        from tools.file_tools import _get_file_ops

        ops = _get_file_ops(task_id="t-grep-probe")
        real = ops._has_command
        if not real("grep"):
            pytest.skip("grep not installed")

        def no_rg(cmd, _real=real):
            # Forces the grep engine for content search AND the probe;
            # other capability lookups pass through.
            if cmd == "rg":
                return False
            return _real(cmd)

        monkeypatch.setattr(ops, "_has_command", no_rg)
        return ops

    def test_case_mismatch_hint_without_rg(self, proj, grep_ops):
        r = grep_ops.search("token_alpha", path=str(proj / "proj"), target="content")
        assert r.total_count == 0
        assert "case-insensitive" in (r.warning or "")

    def test_literal_hint_without_rg(self, proj, grep_ops):
        d = proj / "proj"
        (d / "meta.py").write_text("result = lookup[key+1]\n")
        r = grep_ops.search("lookup[key+1]", path=str(d), target="content")
        assert r.total_count == 0
        assert "literal match" in (r.warning or "")

    def test_hidden_only_hint_without_rg(self, proj, grep_ops):
        d = proj / "proj"
        (d / ".secretdir").mkdir()
        (d / ".secretdir" / "conf.cfg").write_text("HIDDEN_ONLY_TOKEN = true\n")
        r = grep_ops.search("HIDDEN_ONLY_TOKEN", path=str(d), target="content")
        assert r.total_count == 0
        assert "hidden" in (r.warning or "")

    def test_true_zero_match_no_hint_without_rg(self, proj, grep_ops):
        r = grep_ops.search(
            "zzz_totally_absent_zzz", path=str(proj / "proj"), target="content"
        )
        assert r.total_count == 0
        assert not r.warning

    def test_matching_search_unaffected_without_rg(self, proj, grep_ops):
        r = grep_ops.search("TOKEN_ALPHA", path=str(proj / "proj"), target="content")
        assert r.total_count >= 2
        assert not r.warning
