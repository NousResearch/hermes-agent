"""Live integration tests for file operations and terminal tools.

These tests run REAL commands through the LocalEnvironment -- no mocks.
They verify that shell noise is properly filtered, commands actually work,
and the tool outputs are EXACTLY what the agent would see.

Every test with output validates against a known-good value AND
asserts zero contamination from shell noise via _assert_clean().
"""

import pytest


import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


# ── Shared noise detection ───────────────────────────────────────────────
# Known shell noise patterns that should never appear in command output.

_ALL_NOISE_PATTERNS = [
    "bash: cannot set terminal process group",
    "bash: no job control in this shell",
    "no job control in this shell",
    "cannot set terminal process group",
    "tcsetattr: Inappropriate ioctl for device",
    "bash: ",
    "Inappropriate ioctl",
    "Auto-suggestions:",
]


def _assert_clean(text: str, context: str = "output"):
    """Assert text contains zero shell noise contamination."""
    if not text:
        return
    for noise in _ALL_NOISE_PATTERNS:
        assert noise not in text, (
            f"Shell noise leaked into {context}: found {noise!r} in:\n"
            f"{text[:500]}"
        )


# ── Fixtures ─────────────────────────────────────────────────────────────

# Deterministic file content used across tests. Every byte is known,
# so any unexpected text in results is immediately caught.
SIMPLE_CONTENT = "alpha\nbravo\ncharlie\n"
MULTIFILE_A = "def func_alpha():\n    return 42\n"
MULTIFILE_B = "def func_bravo():\n    return 99\n"
MULTIFILE_C = "nothing relevant here\n"


@pytest.fixture
def env(tmp_path):
    """A real LocalEnvironment rooted in a temp directory."""
    return LocalEnvironment(cwd=str(tmp_path), timeout=15)


@pytest.fixture
def ops(env, tmp_path):
    """ShellFileOperations wired to the real local environment."""
    return ShellFileOperations(env, cwd=str(tmp_path))


@pytest.fixture
def populated_dir(tmp_path):
    """A temp directory with known files for search/read tests."""
    (tmp_path / "alpha.py").write_text(MULTIFILE_A)
    (tmp_path / "bravo.py").write_text(MULTIFILE_B)
    (tmp_path / "notes.txt").write_text(MULTIFILE_C)
    (tmp_path / "data.csv").write_text("col1,col2\n1,2\n3,4\n")
    return tmp_path


# ── LocalEnvironment.execute() ───────────────────────────────────────────

class TestLocalEnvironmentExecute:
    def test_echo_exact_output(self, env):
        result = env.execute("echo DETERMINISTIC_OUTPUT_12345")
        assert result["returncode"] == 0
        assert result["output"].strip() == "DETERMINISTIC_OUTPUT_12345"
        _assert_clean(result["output"])

    def test_printf_no_trailing_newline(self, env):
        result = env.execute("printf 'exact'")
        assert result["returncode"] == 0
        assert result["output"] == "exact"
        _assert_clean(result["output"])


    def test_cat_deterministic_content(self, env, tmp_path):
        f = tmp_path / "det.txt"
        f.write_text(SIMPLE_CONTENT)
        result = env.execute(f"cat {f}")
        assert result["returncode"] == 0
        assert result["output"] == SIMPLE_CONTENT
        _assert_clean(result["output"])


# ── _has_command ─────────────────────────────────────────────────────────

class TestHasCommand:
    def test_finds_echo(self, ops):
        assert ops._has_command("echo") is True


    def test_missing_command(self, ops):
        assert ops._has_command("nonexistent_tool_xyz_abc_999") is False


# ── read_file ────────────────────────────────────────────────────────────

class TestReadFile:
    def test_exact_content(self, ops, tmp_path):
        f = tmp_path / "exact.txt"
        f.write_text(SIMPLE_CONTENT)
        result = ops.read_file(str(f))
        assert result.error is None
        # Content has line numbers prepended, check the actual text is there
        assert "alpha" in result.content
        assert "bravo" in result.content
        assert "charlie" in result.content
        assert result.total_lines == 3
        _assert_clean(result.content)


# ── write_file ───────────────────────────────────────────────────────────

class TestWriteFile:
    def test_write_and_verify(self, ops, tmp_path):
        path = str(tmp_path / "written.txt")
        result = ops.write_file(path, SIMPLE_CONTENT)
        assert result.error is None
        assert result.bytes_written == len(SIMPLE_CONTENT.encode())
        assert Path(path).read_text() == SIMPLE_CONTENT


# ── patch_replace ────────────────────────────────────────────────────────

class TestPatchReplace:
    def test_exact_replacement(self, ops, tmp_path):
        path = str(tmp_path / "patch.txt")
        Path(path).write_text("hello world\n")
        result = ops.patch_replace(path, "world", "earth")
        assert result.error is None
        assert Path(path).read_text() == "hello earth\n"


    def test_identical_replacement_explains_no_change(self, ops, tmp_path):
        path = str(tmp_path / "unchanged.txt")
        Path(path).write_text("hello world\n")

        result = ops.patch_replace(path, "world", "world")

        assert result.success is False
        assert result.error is not None
        assert Path(path).read_text() == "hello world\n"

    def test_multiline_patch(self, ops, tmp_path):
        path = str(tmp_path / "multi.txt")
        Path(path).write_text("line1\nline2\nline3\n")
        result = ops.patch_replace(path, "line2", "REPLACED")
        assert result.error is None
        assert Path(path).read_text() == "line1\nREPLACED\nline3\n"


# ── search ───────────────────────────────────────────────────────────────

class TestSearch:
    def test_content_search_finds_exact_match(self, ops, populated_dir):
        result = ops.search("func_alpha", str(populated_dir), target="content")
        assert result.error is None
        assert result.total_count >= 1
        assert any("func_alpha" in m.content for m in result.matches)
        for m in result.matches:
            _assert_clean(m.content)
            _assert_clean(m.path)


# ── _expand_path ─────────────────────────────────────────────────────────

class TestExpandPath:
    def test_tilde_exact(self, ops):
        result = ops._expand_path("~/test.txt")
        expected = f"{Path.home()}/test.txt"
        assert result == expected
        _assert_clean(result)


    def test_tilde_injection_blocked(self, ops):
        """Paths like ~; rm -rf / must NOT execute shell commands."""
        malicious = "~; echo PWNED > /tmp/_hermes_injection_test"
        result = ops._expand_path(malicious)
        # The invalid username (contains ";") should prevent shell expansion.
        # The path should be returned as-is (no expansion).
        assert result == malicious
        # Verify the injected command did NOT execute
        assert not os.path.exists("/tmp/_hermes_injection_test")

    def test_tilde_username_with_subpath(self, ops):
        """~root/file.txt should attempt expansion (valid username)."""
        result = ops._expand_path("~root/file.txt")
        # On most systems ~root expands to /root
        if result != "~root/file.txt":
            assert result.endswith("/file.txt")
            assert "~" not in result


# ── Terminal output cleanliness ──────────────────────────────────────────


# ── _atomic_write failure cleanup ────────────────────────────────────────

class _FailingMvEnv:
    """Terminal-env double: runs the emitted script under real ``bash -c``
    (what _run_bash does) with a PATH-stubbed ``mv`` that fails after the temp
    file exists, so the EXIT trap is what removes it -- or leaks it.
    """

    def __init__(self, stub_dir: Path, cwd: str):
        self.cwd = cwd
        self._path = f"{stub_dir}{os.pathsep}{os.environ['PATH']}"

    def execute(self, command, cwd=None, **kw):
        import subprocess
        proc = subprocess.run(
            ["bash", "-c", command], input=kw.get("stdin_data"),
            text=True, capture_output=True,
            env={**os.environ, "PATH": self._path},
            cwd=cwd or self.cwd)
        return {"output": (proc.stdout or "") + (proc.stderr or ""),
                "returncode": proc.returncode}


class TestAtomicWriteTrapCleanup:
    """A failed atomic write must not leak the .hermes-tmp.* staging file.

    The emitted script registers an EXIT trap for cleanup; a regression in its
    quoting makes the trap's rm target a filename containing literal quote
    characters, so the temp survives every failure path (cat failure, mv
    failure, signal).
    """

    def test_failed_write_removes_staging_file(self, tmp_path):
        stub_dir = tmp_path / "stubbin"
        stub_dir.mkdir()
        stub_mv = stub_dir / "mv"
        stub_mv.write_text("#!/bin/sh\nexit 1\n")
        stub_mv.chmod(0o755)

        ops = ShellFileOperations(
            _FailingMvEnv(stub_dir, str(tmp_path)), cwd=str(tmp_path))
        target = tmp_path / "target.txt"
        result = ops.write_file(str(target), "content that never lands")

        assert result.error is not None
        assert not target.exists()
        leftovers = list(tmp_path.glob(".hermes-tmp.*"))
        assert leftovers == [], (
            f"failed atomic write leaked staging file(s): {leftovers}")

    def test_failed_replace_leaves_original_and_cleans_temp(self, tmp_path):
        # The atomicity contract ("non-zero = original intact") plus cleanup:
        # an mv failure on a REPLACE must leave the pre-existing file's bytes
        # untouched and no staging file behind.
        stub_dir = tmp_path / "stubbin"
        stub_dir.mkdir()
        stub_mv = stub_dir / "mv"
        stub_mv.write_text("#!/bin/sh\nexit 1\n")
        stub_mv.chmod(0o755)

        ops = ShellFileOperations(
            _FailingMvEnv(stub_dir, str(tmp_path)), cwd=str(tmp_path))
        target = tmp_path / "existing.txt"
        target.write_text("original\n")
        result = ops.write_file(str(target), "replacement\n")

        assert result.error is not None
        assert target.read_text() == "original\n"
        assert list(tmp_path.glob(".hermes-tmp.*")) == []

    def test_successful_write_leaves_no_staging_file(self, ops, tmp_path):
        # trap - EXIT must disarm cleanly: a successful write leaves the target
        # and nothing else.
        target = tmp_path / "ok.txt"
        result = ops.write_file(str(target), "lands fine\n")

        assert result.error is None
        assert target.read_text() == "lands fine\n"
        assert list(tmp_path.glob(".hermes-tmp.*")) == []
