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

    def test_case_mismatch_hint_names_the_files(self, proj):
        # The probe already ran the -i search; it must hand over the paths,
        # not just a count (issue #80522: hint-only sent weak models into
        # 5-search casing-variant spirals — +6 turns measured on the A/B eval).
        r = json.loads(search_tool("token_alpha", path=str(proj / "proj"), task_id="t-zm"))
        w = r.get("warning", "")
        assert "a.py" in w and "b.py" in w

    def test_regex_metachar_literal_hint(self, proj):
        d = proj / "proj"
        (d / "meta.py").write_text("result = lookup[key+1]\n")
        r = json.loads(search_tool("lookup[key+1]", path=str(d), task_id="t-zm"))
        assert r["total_count"] == 0
        assert "literal match" in r.get("warning", "")
        assert "meta.py" in r.get("warning", "")

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
        # Same class as the casing probe: the path must be in the hint.
        assert "conf.cfg" in r.get("warning", "")

    def test_hidden_probe_prunes_dependency_trees_and_keeps_local_ignored(self, proj, monkeypatch):
        d = proj / "proj"
        dependency = d / "node_modules" / "package" / ".hidden"
        dependency.mkdir(parents=True)
        dependency_file = dependency / "dependency.js"
        dependency_file.write_text("BOUNDED_HIDDEN_TOKEN = true\n")
        local = d / ".project-local"
        local.mkdir()
        local_file = local / "settings.cfg"
        local_file.write_text("BOUNDED_HIDDEN_TOKEN = true\n")
        (d / ".gitignore").write_text("node_modules/\n.project-local/\n")

        # Drive the public search seam while recording the commands that the
        # zero-match probe actually executes. The real rg calls still run. This
        # observes shell command text, so pin the shell lane (native rg runs
        # argv directly and never passes through ``_exec``).
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
        from tools.file_tools import _get_file_ops

        task_id = "t-zm-pruned-hidden"
        ops = _get_file_ops(task_id=task_id)
        commands = []
        real_exec = ops._exec

        def recording_exec(command, *args, **kwargs):
            commands.append(command)
            return real_exec(command, *args, **kwargs)

        monkeypatch.setattr(ops, "_exec", recording_exec)
        r = json.loads(search_tool("BOUNDED_HIDDEN_TOKEN", path=str(d), task_id=task_id))
        warning = r.get("warning", "")

        assert r["total_count"] == 0
        assert "hidden or gitignored" in warning
        assert local_file.name in warning
        assert dependency_file.name not in warning

        hidden_probe_commands = [
            command for command in commands
            if "--hidden" in command and "--no-ignore" in command
        ]
        assert len(hidden_probe_commands) == 1
        hidden_probe = hidden_probe_commands[0]
        assert "--glob" in hidden_probe
        assert "'!node_modules/**'" in hidden_probe
        assert "'!**/node_modules/**'" in hidden_probe

    def test_hidden_probe_prunes_explicit_dependency_root(self, proj, monkeypatch):
        d = proj / "proj"
        dependency = d / "node_modules" / "package" / ".hidden"
        dependency.mkdir(parents=True)
        (dependency / "dependency.js").write_text("EXPLICIT_ROOT_TOKEN = true\n")
        (d / ".gitignore").write_text("node_modules/\n")

        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")  # shell-observer test
        from tools.file_tools import _get_file_ops

        task_id = "t-zm-explicit-pruned-root"
        ops = _get_file_ops(task_id=task_id)
        commands = []
        real_exec = ops._exec

        def recording_exec(command, *args, **kwargs):
            commands.append(command)
            return real_exec(command, *args, **kwargs)

        monkeypatch.setattr(ops, "_exec", recording_exec)
        r = json.loads(search_tool(
            "EXPLICIT_ROOT_TOKEN",
            path=str(d / "node_modules"),
            task_id=task_id,
        ))

        assert r["total_count"] == 0
        assert "warning" not in r
        hidden_probe_commands = [
            command for command in commands
            if "--hidden" in command and "--no-ignore" in command
        ]
        assert len(hidden_probe_commands) == 1
        hidden_probe = hidden_probe_commands[0]
        assert "'!node_modules/**'" in hidden_probe
        assert "'!**/node_modules/**'" in hidden_probe

    def test_probe_path_list_is_capped(self, proj):
        d = proj / "proj"
        for i in range(8):
            (d / f"cap{i}.txt").write_text("capped_case_token = 1\n")
        r = json.loads(search_tool("CAPPED_CASE_TOKEN", path=str(d), task_id="t-zm"))
        w = r.get("warning", "")
        assert "case-insensitive" in w
        assert "+3 more" in w  # 8 files, 5 shown

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


class TestSymlinkedRootOnTheFilesLane:
    """A symlinked root must be searched on the files lane, on every engine (#116270).

    ``find <link> -type f`` tests the link itself, so a symlinked root listed nothing at
    all - ``total_count: 0``, ``error=None``, no warning, indistinguishable from an empty
    directory - while ``rg --files`` followed the same argument. ``find -H`` follows the
    operand (and only the operand), so both engines answer the same thing, and the
    follow happens inside the command, on the host that owns the link.
    """

    @pytest.fixture
    def linked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        target = tmp_path / "target"
        target.mkdir()
        (target / "real.md").write_text("TOKEN\n")
        empty = tmp_path / "empty"
        empty.mkdir()
        (tmp_path / "linkdir").symlink_to(target)
        (tmp_path / "link.md").symlink_to(target / "real.md")
        (tmp_path / "emptylink").symlink_to(empty)
        return tmp_path

    @staticmethod
    def _pin_engine(monkeypatch, ops, keep):
        """Force ``keep`` as the engine so a host that also has rg exercises find."""
        real = ops._has_command

        def only(cmd, _real=real, _keep=keep):
            if cmd in ("rg", "grep"):
                return _real(cmd) if cmd == _keep else False
            return _real(cmd)

        monkeypatch.setattr(ops, "_has_command", only)

    @pytest.mark.parametrize("engine", ["find", "rg"])
    @pytest.mark.parametrize("lane", ["linkdir", "link.md"])
    def test_symlinked_root_lists_the_target_files(self, linked, monkeypatch, engine, lane):
        from tools.file_tools import _get_file_ops

        ops = _get_file_ops(task_id=f"t-symlink-files-{engine}-{lane.replace('.', '-')}")
        if not ops._has_command(engine):
            pytest.skip(f"{engine} not installed")
        self._pin_engine(monkeypatch, ops, engine)
        r = ops.search("*.md", path=str(linked / lane), target="files")
        assert r.error is None, r.error
        assert r.total_count >= 1 and any(f.endswith(".md") for f in r.files), (
            f"a symlinked root ({lane}) listed nothing and said nothing on the {engine} "
            f"lane: total_count={r.total_count} files={r.files!r}")

    def test_plain_and_empty_roots_keep_their_answer(self, linked, monkeypatch):
        """Following the operand must not invent matches or move an ordinary root."""
        from tools.file_tools import _get_file_ops

        ops = _get_file_ops(task_id="t-symlink-files-guards")
        self._pin_engine(monkeypatch, ops, "find")
        plain = ops.search("*.md", path=str(linked / "target"), target="files")
        assert plain.error is None, plain.error
        assert plain.total_count >= 1 and any(f.endswith("real.md") for f in plain.files)
        for root in (str(linked / "empty"), str(linked / "emptylink")):
            nothing = ops.search("*.md", path=root, target="files")
            assert nothing.error is None and nothing.total_count == 0, (
                f"an empty root ({root}) answered {nothing.total_count} file(s): {nothing.files!r}")

    def test_symlinked_content_root_under_a_dot_dir_matches_on_the_grep_lane(self, tmp_path, monkeypatch):
        """The pruned grep lane (root under a dot-dir) must search through a symlinked
        root instead of answering a silent zero (#116270).

        ``find <link> -type f`` returned no files on every platform, so the pipeline was
        byte-identical to "no match"; ``find -H`` follows the operand.
        """
        from tools.file_tools import _get_file_ops

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        hidden = tmp_path / ".dot"
        (hidden / "real").mkdir(parents=True)
        (hidden / "real" / "f.md").write_text("NEEDLE\n")
        (hidden / "link.md").symlink_to(hidden / "real" / "f.md")
        (hidden / "dirlink").symlink_to(hidden / "real")

        ops = _get_file_ops(task_id="t-symlink-content-grep")
        if not ops._has_command("grep"):
            pytest.skip("grep not installed")
        self._pin_engine(monkeypatch, ops, "grep")
        for root in ("link.md", "dirlink"):
            r = ops.search("NEEDLE", path=str(hidden / root), target="content")
            assert r.error is None, r.error
            assert r.total_count == 1, (
                f"symlinked root {root} answered total_count={r.total_count} on the grep lane")
        miss = ops.search("ABSENT_TOKEN", path=str(hidden / "dirlink"), target="content")
        assert miss.error is None and miss.total_count == 0

    def test_a_symlinked_root_pointing_at_home_is_still_refused(self, tmp_path, monkeypatch):
        """The no-rg breadth guard must classify the link's target (#116270)."""
        import tools.file_operations as file_operations
        from tools.environments.local import LocalEnvironment
        from tools.file_operations import ShellFileOperations

        home = tmp_path / "home"
        home.mkdir()
        (tmp_path / "link-to-home").symlink_to(home)
        monkeypatch.setattr(file_operations, "_HOME", str(home))
        ops = ShellFileOperations(LocalEnvironment(str(tmp_path)))

        assert ops._is_broad_local_search_root(str(tmp_path / "link-to-home")) is True
