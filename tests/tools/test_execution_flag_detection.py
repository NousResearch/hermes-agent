"""Execution-bearing option detection across interpreters and read-only tools."""

import os
import shlex
import shutil
import subprocess
import time

import pytest

from tools.approval import detect_dangerous_command, detect_hardline_command


@pytest.mark.parametrize(
    ("argv", "stdin", "expected_returncode", "expected_output"),
    [
        (["rg", "--", "--pre"], "ordinary text\n", 1, ""),
        (["sort", "--", "--compress-program"], "", 2, ""),
        (["rg", "--pre-glob", "--pre", "needle"], "needle\n", 0, "needle\n"),
    ],
)
def test_real_read_tool_binaries_confirm_option_ownership(
    argv, stdin, expected_returncode, expected_output
):
    """Pin the CLI grammar that the approval detector models."""
    if shutil.which(argv[0]) is None:
        pytest.skip(f"{argv[0]} is not installed")

    completed = subprocess.run(argv, input=stdin, text=True, capture_output=True)

    assert completed.returncode == expected_returncode
    assert completed.stdout == expected_output


@pytest.mark.parametrize(
    ("tool", "args", "stdin", "needs_tty"),
    [
        ("rg", ["--pre", "-payload-marker", "needle", "{input}"], None, False),
        ("rg", ["--hostname-bin=-payload-marker", "needle", "{input}"], None, False),
        ("sort", ["--buffer-size=1K", "--compress-program", "-payload-marker"], "{bulk}", False),
        ("ag", ["--pager=-payload-marker", "needle", "{input}"], None, True),
        ("man", ["--pager", "-payload-marker", "ls"], None, True),
        ("man", ["-P", "-payload-marker", "ls"], None, True),
    ],
)
def test_real_binaries_execute_leading_dash_program_payload(
    tmp_path, tool, args, stdin, needs_tty
):
    """A PATH marker proves these binaries do not reparse '-program' as an option."""
    if shutil.which(tool) is None or (needs_tty and shutil.which("script") is None):
        pytest.skip(f"{tool} or script is not installed")

    marker = tmp_path / "executed"
    payload = tmp_path / "-payload-marker"
    payload.write_text("#!/bin/sh\nprintf executed > \"$MARKER\"\ncat\n")
    payload.chmod(0o755)
    input_file = tmp_path / "input.txt"
    input_file.write_text("needle\n")
    resolved_args = [arg.format(input=str(input_file)) for arg in args]
    input_text = (
        "\n".join(str(number) for number in range(10_000, 0, -1)) + "\n"
        if stdin == "{bulk}"
        else stdin
    )
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "MARKER": str(marker),
        "TERM": "xterm",
    }
    argv = [tool, *resolved_args]
    if needs_tty:
        argv = ["script", "-qec", shlex.join(argv), "/dev/null"]

    subprocess.run(argv, input=input_text, text=True, capture_output=True, env=env, timeout=20)

    assert marker.read_text() == "executed"


@pytest.mark.parametrize(
    "command",
    [
        "rg -- --pre sh",
        "sort -- --compress-program sh",
        "rg --pre-glob --pre needle",
        "sort --output --compress-program names.txt",
        "man --config-file --pager printf",
        "ag --ignore --pager needle",
        "rg -g --pre needle",
        "sort -o --compress-program names.txt",
        "man -C --pager printf",
        "ag -G --pager needle",
    ],
)
def test_read_tool_exec_like_operands_owned_by_other_syntax_are_not_flagged(command):
    assert detect_dangerous_command(command) == (False, None, None)
    assert detect_hardline_command(command) == (False, None)


@pytest.mark.parametrize(
    "command",
    [
        "python3 -W ignore -c 'print(1)'",
        "python3.11 -c 'print(1)'",
        "node --no-warnings --eval=\"require('fs')\"",
        "node -p '1+1'",
        "perl -wne 'print' file.txt",
        "ruby3.2 -e 'puts 1'",
        "php -r 'echo 1;'",
        "powershell -ExecutionPolicy Bypass -File helper.ps1",
        "pwsh -Command 'Get-Process'",
        "python3.11 << 'PY'\nprint(1)\nPY",
    ],
)
def test_interpreter_execution_mechanisms_require_approval(command):
    dangerous, _, _ = detect_dangerous_command(command)
    assert dangerous is True


@pytest.mark.parametrize(
    "command",
    [
        "sort --compress-program=sh names.txt",
        "rg --pre sh -e . names.txt",
        "rg --hostname-bin=sh pattern",
        "ag --pager sh foo",
        "man -Psh ls",
        "man --pager=sh ls",
        "man -H sh ls",
        "man --html=sh ls",
    ],
)
def test_read_only_tool_exec_flags_require_approval(command):
    dangerous, _, description = detect_dangerous_command(command)
    assert dangerous is True
    assert "execution" in description


def test_ag_pager_less_is_an_executable_option_and_requires_approval():
    assert detect_dangerous_command("ag --pager=less needle src/") == (
        True,
        "arbitrary program execution via ag --pager",
        "arbitrary program execution via ag --pager",
    )


@pytest.mark.parametrize(
    ("command", "description"),
    [
        ("rg --pre -payload-marker needle", "arbitrary program execution via rg --pre"),
        ("rg --hostname-bin=-payload-marker needle", "arbitrary program execution via rg --hostname-bin"),
        ("sort --compress-program -payload-marker names", "arbitrary program execution via sort --compress-program"),
        ("ag --pager=-payload-marker needle", "arbitrary program execution via ag --pager"),
        ("man --pager -payload-marker ls", "arbitrary program execution via man --pager"),
        ("man -P -payload-marker ls", "arbitrary program execution via man -P"),
        ("man -H-payload-marker ls", "arbitrary program execution via man -H"),
    ],
)
def test_leading_dash_program_payloads_require_approval(command, description):
    """Program options own the next argv even when its spelling starts with '-'."""
    assert detect_dangerous_command(command) == (True, description, description)


@pytest.mark.parametrize(
    "command",
    [
        "rg --pre '-payload; rm -rf --no-preserve-root /' needle",
        "sort --compress-program='-payload; rm -rf --no-preserve-root /' names",
        "ag --pager='-payload; rm -rf --no-preserve-root /' needle",
        "man --pager '-payload; rm -rf --no-preserve-root /' ls",
        "man -P '-payload; rm -rf --no-preserve-root /' ls",
        "man -H'-payload; rm -rf --no-preserve-root /' ls",
    ],
)
def test_leading_dash_program_payloads_reach_hardline_floor(command):
    assert detect_hardline_command(command) == (
        True,
        "recursive delete of root filesystem",
    )


@pytest.mark.parametrize(
    "command",
    [
        "sort --compress-program='rm -rf --no-preserve-root /' names.txt",
        "rg --pre 'rm -rf --no-preserve-root /' -e . x",
        "ag --pager 'rm -rf --no-preserve-root /' foo",
        "man -P 'rm -rf --no-preserve-root /' ls",
    ],
)
def test_exec_flag_payload_reaches_hardline_floor(command):
    hardline, description = detect_hardline_command(command)
    assert hardline is True
    assert description == "recursive delete of root filesystem"


@pytest.mark.parametrize(
    "command",
    [
        "node -c script.js",
        "node --check script.js",
        "ruby -c script.rb",
        "python3 -m http.server",
        "python3 --version",
        "sort names.txt",
        "rg --pretty pattern src/",
        "pip install --pre somepackage",
        "man -k pager",
        "man -p e ls",
    ],
)
def test_non_executing_flags_are_not_flagged(command):
    hardline, _ = detect_hardline_command(command)
    dangerous, _, _ = detect_dangerous_command(command)
    assert hardline is False
    assert dangerous is False


@pytest.mark.parametrize(
    "command",
    [
        "pip install --pre 'rm -rf --no-preserve-root /'",
        "grep -P 'rm -rf --no-preserve-root /' file.txt",
        "printf '%s' 'man -P rm -rf --no-preserve-root /'",
    ],
)
def test_unrelated_options_do_not_promote_payload_text_to_hardline(command):
    hardline, _ = detect_hardline_command(command)
    assert hardline is False


def test_grep_pcre_pattern_with_grouped_root_delete_text_stays_safe():
    """Regex syntax is grep data, even when it contains a hardline command."""
    command = "grep -P '(?:safe|rm -rf --no-preserve-root /)' audit.log"
    assert detect_hardline_command(command) == (False, None)


def test_interpreter_heredoc_keeps_legacy_approval_key_compatibility():
    from tools.approval import _approval_key_aliases

    aliases = _approval_key_aliases("script execution via heredoc")
    assert r"(python[23]?|perl|ruby|node)\s+<<" in aliases


@pytest.mark.parametrize(
    "flag",
    ["-c", "-lc", "-ic", "-lic", "-cl", "-cil", "-lci", "-ilc", "-cli", "-abc"],
)
def test_shell_exact_short_exec_flags_require_approval(flag):
    dangerous, _, description = detect_dangerous_command(f"bash {flag} 'printf safe'")
    assert dangerous is True
    assert description == "shell command via -c/-lc flag"


@pytest.mark.parametrize(
    "command",
    [
        "sort --compress-program=\"sh -c 'unterminated names",
        "rg --pre=\"bash -lc 'unterminated pattern files",
        "man --pager=\"sh -c 'unterminated ls",
    ],
)
def test_malformed_quoted_executable_payloads_fail_closed(command):
    dangerous, _, description = detect_dangerous_command(command)
    assert dangerous is True
    assert description == "command parser limit or malformed executable payload"


def _time_benign_segments(count):
    command = ";".join(f"printf segment-{index}" for index in range(count))
    started = time.perf_counter()
    result = detect_dangerous_command(command)
    return time.perf_counter() - started, result


def test_benign_segment_scaling_benchmark():
    """Retain real metrics without making correctness depend on wall-clock ratios."""
    small, small_result = _time_benign_segments(2_000)
    large, large_result = _time_benign_segments(4_000)

    assert small_result == (False, None, None)
    assert large_result == (False, None, None)
    print(f"benign segment benchmark: 2k={small:.3f}s, 4k={large:.3f}s")


def test_max_accepted_separator_free_input_is_fast():
    from tools.approval import _MAX_SEPARATOR_FREE_COMMAND_CHARS

    command = "x" * _MAX_SEPARATOR_FREE_COMMAND_CHARS
    started = time.perf_counter()
    result = detect_dangerous_command(command)
    elapsed = time.perf_counter() - started

    assert result == (False, None, None)
    # Loose bound: catches the O(n^2)/backtracking regression class without
    # flaking on CI scheduler stalls (see comment above).
    assert elapsed < 2.0, f"max accepted token took {elapsed:.3f}s"
