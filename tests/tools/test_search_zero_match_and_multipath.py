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
        assert "hidden or gitignored" in r.get("warning", "")

    def test_matching_search_unaffected(self, proj):
        r = json.loads(search_tool("TOKEN_ALPHA", path=str(proj / "proj"), task_id="t-zm"))
        assert r["total_count"] >= 2
        assert "warning" not in r


class TestZeroMatchProbeGrepFallback:
    """Zero-match probes must still attach hints when rg is unavailable.

    The main content search falls back to grep when ``rg`` is not on the
    executed environment's PATH. The probe used to hard-require rg and
    silently return no hint in that case — search worked (via grep) but
    the zero-match steering was dead. These tests force the grep engine
    and assert the same hints appear. Regression for the CI failure on
    ``tests/tools/test_search_zero_match_and_multipath.py``.
    """

    @staticmethod
    def _grep_ops(tmp_path):
        from tools.environments.local import LocalEnvironment
        from tools.file_operations import ShellFileOperations

        d = tmp_path / "proj"
        d.mkdir()
        env = LocalEnvironment(cwd=str(d.parent))
        ops = ShellFileOperations(env, cwd=str(d.parent))
        # Simulate an environment with grep but no rg.
        ops._has_command = lambda cmd: cmd == "grep"
        return ops, d

    def test_case_mismatch_hint_via_grep(self, tmp_path):
        ops, d = self._grep_ops(tmp_path)
        (d / "a.py").write_text("TOKEN_ALPHA = 'x'\n")
        hint = ops._zero_match_probe("token_alpha", str(d), None)
        assert hint and "case-insensitive" in hint

    def test_literal_hint_via_grep(self, tmp_path):
        ops, d = self._grep_ops(tmp_path)
        (d / "meta.py").write_text("result = lookup[key+1]\n")
        hint = ops._zero_match_probe("lookup[key+1]", str(d), None)
        assert hint and "literal match" in hint

    def test_hidden_only_hint_via_grep(self, tmp_path):
        ops, d = self._grep_ops(tmp_path)
        (d / ".secretdir").mkdir()
        (d / ".secretdir" / "conf.cfg").write_text("HIDDEN_ONLY_TOKEN = true\n")
        hint = ops._zero_match_probe("HIDDEN_ONLY_TOKEN", str(d), None)
        assert hint and "hidden or gitignored" in hint

    def test_true_zero_no_hint_via_grep(self, tmp_path):
        ops, d = self._grep_ops(tmp_path)
        (d / "a.py").write_text("x = 1\n")
        assert ops._zero_match_probe("zzz_absent_zzz", str(d), None) is None

    def test_probe_engine_prefers_rg_when_available(self, tmp_path):
        from tools.environments.local import LocalEnvironment
        from tools.file_operations import ShellFileOperations

        d = tmp_path / "proj"
        d.mkdir()
        ops = ShellFileOperations(LocalEnvironment(cwd=str(d.parent)), cwd=str(d.parent))
        ops._has_command = lambda cmd: cmd in ("rg", "grep")
        engine, flags = ops._probe_engine()
        assert engine == "rg"
        assert "count-matches" in flags

    def test_end_to_end_search_real_hint_via_forced_grep(self, tmp_path):
        """End-to-end ``search()`` with grep forced attaches a real probe hint.

        The whole pipeline — engine pick, ``_search_with_grep``, zero-count
        detection, and the probe itself — runs for real (no sentinel, no
        probe stub). Regression for the probe silently dying when rg is
        absent while the main search itself falls back to grep.
        """
        ops, d = self._grep_ops(tmp_path)
        (d / "a.py").write_text("TOKEN_ALPHA = 'x'\n")
        r = ops.search("token_alpha", path=str(d), target="content")
        assert r.total_count == 0
        assert r.warning and "case-insensitive" in r.warning


class TestNativePathBackendGating:
    """rg native-path conversion is limited to the local Windows backend.

    Commands run through ``self.env.execute``, so the executed backend —
    not the host OS — decides whether the MSYS→native path rewrite
    applies. A Windows host driving a remote backend (SSH, WSL, Docker,
    ...) must never rewrite valid remote paths like ``/mnt/d/...`` into
    ``D:\\...`` (issue #67914). These tests pin the split: conversion on
    the local backend only, pass-through on remote backend paths.
    """

    @staticmethod
    def _local_ops(tmp_path):
        from tools.environments.local import LocalEnvironment
        from tools.file_operations import ShellFileOperations

        d = tmp_path / "proj"
        d.mkdir()
        return ShellFileOperations(LocalEnvironment(cwd=str(d.parent)), cwd=str(d.parent))

    @staticmethod
    def _remote_ops(commands=None):
        from tools.file_operations import ShellFileOperations

        class RemoteEnv:
            """Minimal POSIX backend (SSH/WSL-like); no MSYS path rewriting."""

            cwd = "/home/me"

            def execute(self, command, cwd=None, **kwargs):
                if commands is not None:
                    commands.append(command)
                return {"output": "", "returncode": 0}

        return ShellFileOperations(RemoteEnv())

    def test_local_windows_backend_converts_msys_path(self, tmp_path, monkeypatch):
        import tools.environments.local as local_mod

        ops = self._local_ops(tmp_path)
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        out = ops._escape_native_arg("/c/Users/alice/notes.txt")
        assert "C:" in out  # native drive form for the native rg.exe
        assert "/c/Users" not in out

    def test_remote_backend_keeps_remote_path_unchanged(self, monkeypatch):
        import tools.environments.local as local_mod

        ops = self._remote_ops()
        # Simulate a Windows host driving a remote backend: conversion
        # must still be skipped — the remote POSIX side owns these paths.
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
        assert ops._escape_native_arg("/mnt/d/data") == "'/mnt/d/data'"
        assert ops._escape_native_arg("/mnt/c/Users/alice") == "'/mnt/c/Users/alice'"
        assert ops._escape_native_arg("/home/user/proj") == "'/home/user/proj'"

    def test_remote_backend_probe_command_keeps_remote_path(self, monkeypatch):
        """The zero-match probe emits the raw remote path in its command."""
        import tools.environments.local as local_mod

        commands = []
        ops = self._remote_ops(commands)
        ops._has_command = lambda cmd: cmd == "rg"
        monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)  # Windows host, remote backend
        assert ops._zero_match_probe("needle", "/mnt/d/data", None) is None
        assert commands, "probe should have executed shell commands"
        for cmd in commands:
            assert "/mnt/d/data" in cmd
            assert "D:" not in cmd


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

    The probe itself shells out to rg (by design: bounded, count-only), so a
    grep-only host gets no hints even with correct wiring. The probe is
    therefore stubbed to a sentinel here — this isolates the *wiring* rather
    than the probe's own dependency, and a naive parity test asserting real
    hint text would fail on the grep leg for an unrelated reason.
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
