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


    def test_roundtrip_read_write(self, ops, tmp_path):
        """Write -> read back -> verify exact match."""
        path = str(tmp_path / "roundtrip.txt")
        ops.write_file(path, SIMPLE_CONTENT)
        result = ops.read_file(path)
        assert result.error is None
        assert "alpha" in result.content
        assert "charlie" in result.content
        _assert_clean(result.content)

    # .py so write_file probes the previous content (lint coverage) for the rewrite check.
    def test_large_rewrite_of_existing_file_warns_with_replaced_line_count(self, ops, tmp_path):
        path = tmp_path / "big.py"
        path.write_text("".join(f"x_{i} = {i}\n" for i in range(40)))
        rewritten = "".join(f"x_{i} = {i + 1000 if i < 30 else i}\n" for i in range(40))
        result = ops.write_file(str(path), rewritten)
        assert result.error is None
        assert result.warning is not None
        assert "30 of 40 previous lines" in result.warning

    def test_small_edit_of_existing_file_has_no_warning(self, ops, tmp_path):
        path = tmp_path / "small.py"
        path.write_text("".join(f"x_{i} = {i}\n" for i in range(40)))
        result = ops.write_file(str(path), path.read_text().replace("x_7 = 7", "x_7 = 70"))
        assert result.error is None
        assert result.warning is None


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

    # Each case first pins which fuzzy strategy the inputs take, so the note assertion
    # is about that strategy rather than about an accidental exact match.
    FUNC = "def total(x):\n    y = x + 1\n    return y\n"

    def test_similarity_match_gets_boundary_note(self, ops, tmp_path):
        from tools.fuzzy_match import fuzzy_find_and_replace
        old = "def total(x):\n    y = x + 2\n    return y"
        assert fuzzy_find_and_replace(self.FUNC, old, "pass", False)[2] == "block_anchor"
        path = tmp_path / "anchor.txt"
        path.write_text(self.FUNC)
        result = ops.patch_replace(str(path), old, "def total(x):\n    return x + 1")
        assert result.error is None
        assert result.note and "block_anchor" in result.note

    def test_line_trimmed_match_gets_no_note(self, ops, tmp_path):
        from tools.fuzzy_match import fuzzy_find_and_replace
        old = "y = x + 1\nreturn y"  # model dropped the base indentation
        assert fuzzy_find_and_replace(self.FUNC, old, "pass", False)[2] == "line_trimmed"
        path = tmp_path / "trimmed.txt"
        path.write_text(self.FUNC)
        result = ops.patch_replace(str(path), old, "y = x + 3\nreturn y")
        assert result.error is None
        assert path.read_text() == "def total(x):\n    y = x + 3\n    return y\n"
        assert result.note is None


class TestPatchV4A:
    """The V4A path runs the same fuzzy chain as patch_replace, so it gets the same boundary note."""
    FUNC = TestPatchReplace.FUNC

    @staticmethod
    def _patch(path, *hunk_lines):
        return "\n".join(["*** Begin Patch", f"*** Update File: {path}", "@@ @@", *hunk_lines, "*** End Patch"])

    def test_similarity_match_gets_boundary_note_naming_the_file(self, ops, tmp_path):
        from tools.fuzzy_match import fuzzy_find_and_replace
        old = "def total(x):\n    y = x + 2\n    return y"
        assert fuzzy_find_and_replace(self.FUNC, old, "pass", False)[2] == "block_anchor"
        path = tmp_path / "anchor.py"
        path.write_text(self.FUNC)
        result = ops.patch_v4a(self._patch(
            path, " def total(x):", "-    y = x + 2", "-    return y", "+    return x + 1"))
        assert result.success, result.error
        assert result.note and "block_anchor" in result.note and str(path) in result.note

    def test_line_trimmed_match_gets_no_note(self, ops, tmp_path):
        from tools.fuzzy_match import fuzzy_find_and_replace
        assert fuzzy_find_and_replace(self.FUNC, "y = x + 1\nreturn y", "pass", False)[2] == "line_trimmed"
        path = tmp_path / "trimmed.py"
        path.write_text(self.FUNC)
        result = ops.patch_v4a(self._patch(path, "-y = x + 1", "+y = x + 3", " return y"))
        assert result.success, result.error
        assert path.read_text() == "def total(x):\n    y = x + 3\n    return y\n"
        assert result.note is None

    def test_exact_match_gets_no_note(self, ops, tmp_path):
        path = tmp_path / "exact.py"
        path.write_text(self.FUNC)
        result = ops.patch_v4a(self._patch(
            path, " def total(x):", "-    y = x + 1", "+    y = x + 3", "     return y"))
        assert result.success, result.error
        assert result.note is None

    def test_note_names_only_the_file_that_matched_approximately(self, ops, tmp_path):
        approx, exact = tmp_path / "approx.py", tmp_path / "exact.py"
        approx.write_text(self.FUNC)
        exact.write_text("a = 1\nb = 2\n")
        patch = "\n".join([
            "*** Begin Patch",
            f"*** Update File: {approx}", "@@ @@",
            " def total(x):", "-    y = x + 2", "-    return y", "+    return x + 1",
            f"*** Update File: {exact}", "@@ @@",
            "-a = 1", "+a = 5",
            "*** End Patch"])
        result = ops.patch_v4a(patch)
        assert result.success, result.error
        assert str(approx) in result.note and str(exact) not in result.note


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
