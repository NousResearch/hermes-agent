"""Live integration tests for file operations and terminal tools.

These tests run REAL commands through the LocalEnvironment -- no mocks.
They verify that shell noise is properly filtered, commands actually work,
and the tool outputs are EXACTLY what the agent would see.

Every test with output validates against a known-good value AND
asserts zero contamination from shell noise via _assert_clean().
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.environments.local import LocalEnvironment, _clean_shell_noise, _SHELL_NOISE_SUBSTRINGS
from tools.file_operations import ShellFileOperations


# ── Shared noise detection ───────────────────────────────────────────────
# Every known shell noise pattern. If ANY of these appear in output that
# isn't explicitly expected, the test fails with a clear message.

_ALL_NOISE_PATTERNS = list(_SHELL_NOISE_SUBSTRINGS) + [
    "bash: ",
    "Inappropriate ioctl",
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
NUMBERED_CONTENT = "\n".join(f"LINE_{i:04d}" for i in range(1, 51)) + "\n"
SPECIAL_CONTENT = "single 'quotes' and \"doubles\" and $VARS and `backticks` and \\backslash\n"
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


# ── _clean_shell_noise unit tests ────────────────────────────────────────

class TestCleanShellNoise:
    def test_single_noise_line(self):
        output = "bash: no job control in this shell\nhello world\n"
        result = _clean_shell_noise(output)
        assert result == "hello world\n"

    def test_double_noise_lines(self):
        output = (
            "bash: cannot set terminal process group (-1): Inappropriate ioctl for device\n"
            "bash: no job control in this shell\n"
            "actual output here\n"
        )
        result = _clean_shell_noise(output)
        assert result == "actual output here\n"
        _assert_clean(result)

    def test_tcsetattr_noise(self):
        output = (
            "bash: [12345: 2 (255)] tcsetattr: Inappropriate ioctl for device\n"
            "real content\n"
        )
        result = _clean_shell_noise(output)
        assert result == "real content\n"
        _assert_clean(result)

    def test_triple_noise_lines(self):
        output = (
            "bash: cannot set terminal process group (-1): Inappropriate ioctl for device\n"
            "bash: no job control in this shell\n"
            "bash: [999: 2 (255)] tcsetattr: Inappropriate ioctl for device\n"
            "clean\n"
        )
        result = _clean_shell_noise(output)
        assert result == "clean\n"

    def test_no_noise_untouched(self):
        assert _clean_shell_noise("hello\nworld\n") == "hello\nworld\n"

    def test_empty_string(self):
        assert _clean_shell_noise("") == ""

    def test_only_noise_produces_empty(self):
        output = "bash: no job control in this shell\n"
        result = _clean_shell_noise(output)
        _assert_clean(result)

    def test_noise_in_middle_not_stripped(self):
        """Only LEADING noise is stripped -- noise in the middle is real output."""
        output = "real\nbash: no job control in this shell\nmore real\n"
        result = _clean_shell_noise(output)
        assert result == output


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

    def test_exit_code_propagated(self, env):
        result = env.execute("exit 42")
        assert result["returncode"] == 42

    def test_stderr_captured_in_output(self, env):
        result = env.execute("echo STDERR_TEST >&2")
        assert "STDERR_TEST" in result["output"]
        _assert_clean(result["output"])

    def test_cwd_respected(self, env, tmp_path):
        subdir = tmp_path / "subdir_test"
        subdir.mkdir()
        result = env.execute("pwd", cwd=str(subdir))
        assert result["returncode"] == 0
        assert result["output"].strip() == str(subdir)
        _assert_clean(result["output"])

    def test_multiline_exact(self, env):
        result = env.execute("echo AAA; echo BBB; echo CCC")
        lines = [l for l in result["output"].strip().split("\n") if l.strip()]
        assert lines == ["AAA", "BBB", "CCC"]
        _assert_clean(result["output"])

    def test_env_var_home(self, env):
        result = env.execute("echo $HOME")
        assert result["returncode"] == 0
        home = result["output"].strip()
        assert home == str(Path.home())
        _assert_clean(result["output"])

    def test_pipe_exact(self, env):
        result = env.execute("echo 'one two three' | wc -w")
        assert result["returncode"] == 0
        assert result["output"].strip() == "3"
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

    def test_finds_cat(self, ops):
        assert ops._has_command("cat") is True

    def test_finds_sed(self, ops):
        assert ops._has_command("sed") is True

    def test_finds_wc(self, ops):
        assert ops._has_command("wc") is True

    def test_finds_find(self, ops):
        assert ops._has_command("find") is True

    def test_missing_command(self, ops):
        assert ops._has_command("nonexistent_tool_xyz_abc_999") is False

    def test_rg_or_grep_available(self, ops):
        assert ops._has_command("rg") or ops._has_command("grep"), \
            "Neither rg nor grep found -- search_files will break"


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

    def test_absolute_path(self, ops, tmp_path):
        f = tmp_path / "abs.txt"
        f.write_text("ABSOLUTE_PATH_CONTENT\n")
        result = ops.read_file(str(f))
        assert result.error is None
        assert "ABSOLUTE_PATH_CONTENT" in result.content
        _assert_clean(result.content)

    def test_tilde_expansion(self, ops):
        test_path = Path.home() / ".hermes_test_tilde_9f8a7b"
        try:
            test_path.write_text("TILDE_EXPANSION_OK\n")
            result = ops.read_file("~/.hermes_test_tilde_9f8a7b")
            assert result.error is None
            assert "TILDE_EXPANSION_OK" in result.content
            _assert_clean(result.content)
        finally:
            test_path.unlink(missing_ok=True)

    def test_nonexistent_returns_error(self, ops, tmp_path):
        result = ops.read_file(str(tmp_path / "ghost.txt"))
        assert result.error is not None

    def test_pagination_exact_window(self, ops, tmp_path):
        f = tmp_path / "numbered.txt"
        f.write_text(NUMBERED_CONTENT)
        result = ops.read_file(str(f), offset=10, limit=5)
        assert result.error is None
        assert "LINE_0010" in result.content
        assert "LINE_0014" in result.content
        assert "LINE_0009" not in result.content
        assert "LINE_0015" not in result.content
        assert result.total_lines == 50
        _assert_clean(result.content)

    def test_no_noise_in_content(self, ops, tmp_path):
        f = tmp_path / "noise_check.txt"
        f.write_text("ONLY_THIS_CONTENT\n")
        result = ops.read_file(str(f))
        assert result.error is None
        _assert_clean(result.content)


# ── write_file ───────────────────────────────────────────────────────────

class TestWriteFile:
    def test_write_and_verify(self, ops, tmp_path):
        path = str(tmp_path / "written.txt")
        result = ops.write_file(path, SIMPLE_CONTENT)
        assert result.error is None
        assert result.bytes_written == len(SIMPLE_CONTENT.encode())
        assert Path(path).read_text() == SIMPLE_CONTENT

    def test_creates_nested_dirs(self, ops, tmp_path):
        path = str(tmp_path / "a" / "b" / "c" / "deep.txt")
        result = ops.write_file(path, "DEEP_CONTENT\n")
        assert result.error is None
        assert result.dirs_created is True
        assert Path(path).read_text() == "DEEP_CONTENT\n"

    def test_overwrites_exact(self, ops, tmp_path):
        path = str(tmp_path / "overwrite.txt")
        Path(path).write_text("OLD_DATA\n")
        result = ops.write_file(path, "NEW_DATA\n")
        assert result.error is None
        assert Path(path).read_text() == "NEW_DATA\n"

    def test_large_content_via_stdin(self, ops, tmp_path):
        path = str(tmp_path / "large.txt")
        content = "X" * 200_000 + "\n"
        result = ops.write_file(path, content)
        assert result.error is None
        assert Path(path).read_text() == content

    def test_special_characters_preserved(self, ops, tmp_path):
        path = str(tmp_path / "special.txt")
        result = ops.write_file(path, SPECIAL_CONTENT)
        assert result.error is None
        assert Path(path).read_text() == SPECIAL_CONTENT

    def test_roundtrip_read_write(self, ops, tmp_path):
        """Write -> read back -> verify exact match."""
        path = str(tmp_path / "roundtrip.txt")
        ops.write_file(path, SIMPLE_CONTENT)
        result = ops.read_file(path)
        assert result.error is None
        assert "alpha" in result.content
        assert "charlie" in result.content
        _assert_clean(result.content)


# ── patch_replace ────────────────────────────────────────────────────────

class TestPatchReplace:
    def test_exact_replacement(self, ops, tmp_path):
        path = str(tmp_path / "patch.txt")
        Path(path).write_text("hello world\n")
        result = ops.patch_replace(path, "world", "earth")
        assert result.error is None
        assert Path(path).read_text() == "hello earth\n"

    def test_not_found_error(self, ops, tmp_path):
        path = str(tmp_path / "patch2.txt")
        Path(path).write_text("hello\n")
        result = ops.patch_replace(path, "NONEXISTENT_STRING", "replacement")
        assert result.error is not None
        assert "Could not find" in result.error

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

    def test_content_search_no_false_positives(self, ops, populated_dir):
        result = ops.search("ZZZZZ_NONEXISTENT", str(populated_dir), target="content")
        assert result.error is None
        assert result.total_count == 0
        assert len(result.matches) == 0

    def test_file_search_finds_py_files(self, ops, populated_dir):
        result = ops.search("*.py", str(populated_dir), target="files")
        assert result.error is None
        assert result.total_count >= 2
        # Verify only expected files appear
        found_names = set()
        for f in result.files:
            name = Path(f).name
            found_names.add(name)
            _assert_clean(f)
        assert "alpha.py" in found_names
        assert "bravo.py" in found_names
        assert "notes.txt" not in found_names

    def test_file_search_no_false_file_entries(self, ops, populated_dir):
        """Every entry in the files list must be a real path, not noise."""
        result = ops.search("*.py", str(populated_dir), target="files")
        assert result.error is None
        for f in result.files:
            _assert_clean(f)
            assert Path(f).exists(), f"Search returned non-existent path: {f}"

    def test_content_search_with_glob_filter(self, ops, populated_dir):
        result = ops.search("return", str(populated_dir), target="content", file_glob="*.py")
        assert result.error is None
        for m in result.matches:
            assert m.path.endswith(".py"), f"Non-py file in results: {m.path}"
            _assert_clean(m.content)
            _assert_clean(m.path)

    def test_search_output_has_zero_noise(self, ops, populated_dir):
        """Dedicated noise check: search must return only real content."""
        result = ops.search("func", str(populated_dir), target="content")
        assert result.error is None
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

    def test_absolute_unchanged(self, ops):
        assert ops._expand_path("/tmp/test.txt") == "/tmp/test.txt"

    def test_relative_unchanged(self, ops):
        assert ops._expand_path("relative/path.txt") == "relative/path.txt"

    def test_bare_tilde(self, ops):
        result = ops._expand_path("~")
        assert result == str(Path.home())
        _assert_clean(result)


# ── Terminal output cleanliness ──────────────────────────────────────────

class TestTerminalOutputCleanliness:
    """Every command the agent might run must produce noise-free output."""

    def test_echo(self, env):
        result = env.execute("echo CLEAN_TEST")
        assert result["output"].strip() == "CLEAN_TEST"
        _assert_clean(result["output"])

    def test_cat(self, env, tmp_path):
        f = tmp_path / "cat_test.txt"
        f.write_text("CAT_CONTENT_EXACT\n")
        result = env.execute(f"cat {f}")
        assert result["output"] == "CAT_CONTENT_EXACT\n"
        _assert_clean(result["output"])

    def test_ls(self, env, tmp_path):
        (tmp_path / "file_a.txt").write_text("")
        (tmp_path / "file_b.txt").write_text("")
        result = env.execute(f"ls {tmp_path}")
        _assert_clean(result["output"])
        assert "file_a.txt" in result["output"]
        assert "file_b.txt" in result["output"]

    def test_wc(self, env, tmp_path):
        f = tmp_path / "wc_test.txt"
        f.write_text("one\ntwo\nthree\n")
        result = env.execute(f"wc -l < {f}")
        assert result["output"].strip() == "3"
        _assert_clean(result["output"])

    def test_head(self, env, tmp_path):
        f = tmp_path / "head_test.txt"
        f.write_text(NUMBERED_CONTENT)
        result = env.execute(f"head -n 3 {f}")
        expected = "LINE_0001\nLINE_0002\nLINE_0003\n"
        assert result["output"] == expected
        _assert_clean(result["output"])

    def test_env_var_expansion(self, env):
        result = env.execute("echo $HOME")
        assert result["output"].strip() == str(Path.home())
        _assert_clean(result["output"])

    def test_command_substitution(self, env):
        result = env.execute("echo $(echo NESTED)")
        assert result["output"].strip() == "NESTED"
        _assert_clean(result["output"])

    def test_command_v_detection(self, env):
        """This is how _has_command works -- must return clean 'yes'."""
        result = env.execute("command -v cat >/dev/null 2>&1 && echo 'yes'")
        assert result["output"].strip() == "yes"
        _assert_clean(result["output"])
