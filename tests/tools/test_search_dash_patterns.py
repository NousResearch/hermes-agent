"""search_files must not feed leading-dash patterns to rg/grep as flags (#93750).

A user pattern like ``-77``, ``--pdf-engine`` or ``->`` used to be passed
as a positional argument after the option block, so ripgrep parsed it as
an option ("unhandled flag") and the search failed. Both engines now get
a ``--`` end-of-flags separator before the pattern.
"""

import shutil

import pytest

from tools.file_operations import ShellFileOperations


# ─────────────────────────────────────────────────────────────────────
# Command construction (mocked executor — engine-independent)
# ─────────────────────────────────────────────────────────────────────


def _capturing_ops():
    from unittest.mock import MagicMock

    captured = {}

    env = MagicMock()
    env.cwd = "/tmp/test"

    def execute(command, **kwargs):
        captured["cmd"] = command
        return {"output": "", "returncode": 0}

    env.execute = execute
    return ShellFileOperations(env), captured


@pytest.mark.parametrize("pattern", ["-77", "--pdf-engine", "-e", "->"])
def test_rg_command_separates_leading_dash_pattern(pattern):
    ops, captured = _capturing_ops()
    result = ops._search_with_rg(
        pattern=pattern, path=".", file_glob=None,
        limit=10, offset=0, output_mode="content", context=0,
    )
    cmd = captured["cmd"]
    assert " -- " in cmd, f"end-of-flags separator missing: {cmd}"
    # The separator must sit immediately before the quoted pattern.
    assert f" -- '{pattern}'" in cmd.replace('"', "'")


@pytest.mark.parametrize("pattern", ["-77", "--pdf-engine", "-e", "->"])
def test_grep_command_separates_leading_dash_pattern(pattern):
    """Mirror of the rg construction set — grep's own flag parser is the
    fallback engine most users hit, so it gets the same coverage. The
    trailing ``--`` is honored by GNU/BSD/BusyBox grep alike (BusyBox
    compatibility is assumed for minimal remote appliances)."""
    ops, captured = _capturing_ops()
    result = ops._search_with_grep(
        pattern=pattern, path=".", file_glob=None,
        limit=10, offset=0, output_mode="content", context=0,
    )
    cmd = captured["cmd"]
    assert " -- " in cmd, f"end-of-flags separator missing: {cmd}"
    assert f" -- '{pattern}'" in cmd.replace('"', "'")


# ─────────────────────────────────────────────────────────────────────
# Real execution: leading-dash patterns actually match now
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(shutil.which("grep") is None, reason="grep not available")
def test_leading_dash_pattern_finds_matches_grep_fallback(tmp_path, monkeypatch):
    """End-to-end through search() with the POSIX-guaranteed grep engine.

    rg is hidden so ``search()`` deterministically takes the grep branch;
    the constructed command must find every leading-dash pattern.
    """
    (tmp_path / "notes.md").write_text(
        "uses --pdf-engine=lualatex here\nscore was -77\narrow -> point\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    from tests.tools.test_file_operations import make_real_subprocess_env

    ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
    # Force the deterministic engine regardless of whether rg is installed.
    ops._has_command = lambda name: name == "grep"

    for pattern in ("-77", "--pdf-engine", "->"):
        result = ops.search(pattern=pattern, path=str(tmp_path), target="content")
        assert not result.error, f"{pattern!r} failed: {result.error}"
        assert result.total_count >= 1, (
            f"leading-dash pattern {pattern!r} returned no matches"
        )


@pytest.mark.skipif(shutil.which("rg") is None, reason="rg not available")
def test_rg_engine_accepts_end_of_flags_pattern(tmp_path):
    """Engine-capability half of the fix: rg matches a literal '-77' when
    given as ``rg -- '-77' <path>`` — the exact argument form Hermes now
    constructs (verified separately by the command-construction tests).
    """
    target = tmp_path / "notes.md"
    target.write_text("score was -77\n", encoding="utf-8")

    import subprocess

    proc = subprocess.run(
        ["rg", "--line-number", "--", "-77", str(target)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"rg rc={proc.returncode}: {proc.stderr}"
    assert "-77" in proc.stdout
