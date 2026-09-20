"""Tests for #117213/#117214: MCP stdio command lookup must stay inside the child env.

#117214 — ``_which_with_config_pathext`` emulates the config PATHEXT with a direct
directory scan; it never writes ``os.environ["PATHEXT"]`` (the old write-then-restore
window raced every other thread's ``shutil.which``/``subprocess``/``os.environ`` read
in this multi-threaded gateway).

#117213 — an absent child PATH must not silently resolve against the parent's ambient
PATH: ``shutil.which(path=None)`` reads ``os.environ["PATH"]``, but the child spawns
without one. Absent and empty PATH both miss; only the explicit node-fallback dirs can
still rescue bare ``npx``/``npm``/``node``.
"""

import os

import pytest

from tools.mcp_tool_config import _resolve_stdio_command, _which_with_config_pathext


CFG_PATH_ENV = {"PATH": None, "PATHEXT": ".CMD;.EXE"}  # PATH filled per-test


class _NoWriteEnviron:
    """os.environ stand-in that fails the test on any write (#117214 fail-closed probe)."""

    def __init__(self, real):
        self._real = real

    def get(self, key, default=None):
        return self._real.get(key, default)

    def __getitem__(self, key):
        return self._real[key]

    def __contains__(self, key):
        return key in self._real

    def __setitem__(self, key, value):
        raise AssertionError(f"os.environ[{key!r}] was mutated during the lookup (#117214)")


def _make_executable(directory, name):
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n" if not name.lower().endswith((".cmd", ".exe", ".bat"))
                    else "@echo off\r\n", encoding="utf-8")
    path.chmod(0o755)
    return path


class TestWhichWithConfigPathextNeverMutatesParentEnviron:
    def test_resolves_config_pathext_without_writing_parent_environ(self, tmp_path, monkeypatch):
        """The lookup honours the child env's PATHEXT while a write-guard on os.environ
        proves the parent process environment is never touched."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        cmd_path = _make_executable(bin_dir, "demo.CMD")
        monkeypatch.setenv("PATHEXT", ".COM;.BAT")  # parent differs from the config value

        guard = _NoWriteEnviron(os.environ)
        os.environ = guard  # guard lives only inside the lookup window, never across teardown
        try:
            hit = _which_with_config_pathext("demo", str(bin_dir), {"PATHEXT": ".CMD;.EXE"})
        finally:
            os.environ = guard._real

        assert hit == str(cmd_path)

    def test_appends_each_config_extension_for_a_bare_command(self, tmp_path, monkeypatch):
        """A command without one of the configured extensions is looked up as
        ``command + ext`` for each ext, in order — same precedence as shutil.which."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        _make_executable(bin_dir, "first.CMD")
        exe_path = _make_executable(bin_dir, "second.EXE")
        monkeypatch.setenv("PATHEXT", ".COM;.BAT")

        hit = _which_with_config_pathext("second", str(bin_dir), {"PATHEXT": ".CMD;.EXE"})

        assert hit == str(exe_path)

    def test_command_already_carrying_a_config_extension_looks_up_as_is(self, tmp_path, monkeypatch):
        """``demo.CMD`` (extension in the config list) must resolve even when a fake
        ``demo.CMD.EXE`` sits next to it: no further extension is appended."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        cmd_path = _make_executable(bin_dir, "demo.CMD")
        _make_executable(bin_dir, "demo.CMD.EXE")
        monkeypatch.setenv("PATHEXT", ".COM;.BAT")

        hit = _which_with_config_pathext("demo.CMD", str(bin_dir), {"PATHEXT": ".CMD;.EXE"})

        assert hit == str(cmd_path)

    def test_pathext_dot_entry_still_appends_other_extensions(self, tmp_path, monkeypatch):
        """A ``.`` PATHEXT entry means "extensionless match", not "every command matches
        as-is": a bare command must still probe ``command + ext`` for the real extensions."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        cmd_path = _make_executable(bin_dir, "demo.CMD")
        monkeypatch.setenv("PATHEXT", ".COM;.BAT")

        hit = _which_with_config_pathext("demo", str(bin_dir), {"PATHEXT": ".;.CMD"})

        assert hit == str(cmd_path)

    def test_pathext_dot_entry_matches_the_extensionless_command_itself(self, tmp_path, monkeypatch):
        """With a ``.`` entry, a bare executable (no extension at all) resolves — the
        cmd.exe "extensionless match" semantics the rstrip-into-empty-ext emulates."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        bare_path = _make_executable(bin_dir, "demo")
        monkeypatch.setenv("PATHEXT", ".COM;.BAT")

        hit = _which_with_config_pathext("demo", str(bin_dir), {"PATHEXT": ".;.CMD"})

        assert hit == str(bare_path)

    def test_pathext_dot_entry_never_probes_a_trailing_dot_name(self, tmp_path, monkeypatch):
        """The ``.`` entry must not degrade to shutil.which's literal-append reading:
        ``demo.`` (trailing-dot name) is never a candidate — pinning the emulation."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        _make_executable(bin_dir, "demo.")
        monkeypatch.setenv("PATHEXT", ".COM;.BAT")

        assert _which_with_config_pathext("demo", str(bin_dir), {"PATHEXT": ".;.CMD"}) is None

    def test_absent_path_arg_searches_nowhere_even_when_parent_path_has_it(self, tmp_path, monkeypatch):
        """path_arg=None (child env has no PATH) must NOT fall back to the parent's
        ambient PATH — the command stays unresolved even though the parent PATH has it."""
        parent_dir = tmp_path / "parent-bin"
        parent_dir.mkdir()
        _make_executable(parent_dir, "demo.CMD")
        monkeypatch.setenv("PATHEXT", ".COM;.BAT")
        monkeypatch.setenv("PATH", str(parent_dir))

        assert _which_with_config_pathext("demo", None, {"PATHEXT": ".CMD;.EXE"}) is None

    def test_pathext_matching_parent_is_still_short_circuited(self, monkeypatch):
        """Config PATHEXT equal to the parent's returns None (the caller's first
        shutil.which pass already covered it) — no duplicate lookup."""
        monkeypatch.setenv("PATHEXT", ".CMD;.EXE")

        assert _which_with_config_pathext("demo", "/nonexistent", {"PATHEXT": ".CMD;.EXE"}) is None


class TestResolveStdioCommandAbsentPath:
    def test_absent_child_path_never_resolves_against_parent_path(self, tmp_path, monkeypatch):
        """A command that exists on the PARENT's PATH must stay unresolved when the
        server env declares no PATH at all — the child spawns without one."""
        parent_dir = tmp_path / "parent-bin"
        parent_dir.mkdir()
        parent_hit = _make_executable(parent_dir, "custom-srv")
        monkeypatch.setenv("PATH", str(parent_dir))

        command, _env = _resolve_stdio_command("custom-srv", {"PATHEXT": ".CMD;.EXE"})

        assert command == "custom-srv"  # never the parent's ambient hit

    def test_empty_child_path_misses_the_same_way(self, tmp_path, monkeypatch):
        """PATH="" and absent PATH are different inputs but both must miss — no lookup
        happens against the parent's PATH for either."""
        parent_dir = tmp_path / "parent-bin"
        parent_dir.mkdir()
        _make_executable(parent_dir, "custom-srv")
        monkeypatch.setenv("PATH", str(parent_dir))

        command, _env = _resolve_stdio_command("custom-srv", {"PATH": "", "PATHEXT": ".CMD;.EXE"})

        assert command == "custom-srv"

    def test_bare_npx_still_reaches_the_node_fallback_dirs(self, monkeypatch):
        """With no child PATH, bare npx/npm/node keep their explicit well-known-dir
        rescue — the only ambient-independent resolution left."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("tools.mcp_tool_config.shutil.which", lambda *_a, **_k: None)
            mp.setattr("tools.mcp_tool_config._node_fallback", lambda c, **_k: "/opt/fake/npx")
            command, _env = _resolve_stdio_command("npx", {"PATHEXT": ".CMD;.EXE"})
        assert command == "/opt/fake/npx"

    def test_explicit_child_path_still_resolves(self, tmp_path):
        """Regression: a server env that DOES declare a PATH keeps resolving against it
        (this is the original feature — a filtered child PATH)."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        srv = _make_executable(bin_dir, "custom-srv")

        command, env = _resolve_stdio_command("custom-srv", {"PATH": str(bin_dir)})

        assert command == str(srv)
        assert env["PATH"].split(os.pathsep)[0] == str(bin_dir)
