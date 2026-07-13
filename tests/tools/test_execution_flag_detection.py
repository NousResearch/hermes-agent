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
        "rg --pre-glob '*.gz' --pre sh needle",
        "sort --output result --compress-program sh names.txt",
        "man --config-file man.conf --pager sh ls",
        "ag --ignore vendor --pager sh needle",
    ],
)
def test_read_tool_non_exec_option_arguments_do_not_hide_later_exec_flags(command):
    assert detect_dangerous_command(command)[0] is True


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


@pytest.mark.parametrize(
    "command",
    [
        "grep --color=auto -n -P '(?:safe|rm -rf --no-preserve-root /)' audit.log",
        "grep -A 3 --binary-files=without-match --perl-regexp '(safe|reboot)' audit.log",
        "grep -P -e '(safe|shutdown -h now)' audit.log",
        "grep -P --regexp='(safe|rm -rf --no-preserve-root /)' audit.log",
        "grep -P -- '(safe|rm -rf --no-preserve-root /)' audit.log",
        "env LC_ALL=C grep -P '(safe|rm -rf --no-preserve-root /)' audit.log",
    ],
)
def test_grep_pattern_operands_are_structurally_scoped_data(command):
    assert detect_hardline_command(command) == (False, None)


@pytest.mark.parametrize(
    "command",
    [
        "grep -P '(safe|printf x)' audit.log; rm -rf --no-preserve-root /",
        "grep -P '(safe|printf x)' audit.log && reboot",
        "grep -P '(safe|printf x)' audit.log | shutdown -h now",
        "grep -P -e '(safe|printf x)' audit.log\nrm -rf --no-preserve-root /",
    ],
)
def test_grep_pattern_operand_never_masks_a_later_command(command):
    assert detect_hardline_command(command)[0] is True


@pytest.mark.parametrize(
    "command",
    [
        "grep -P '(safe|rm -rf --no-preserve-root /) audit.log",
        'grep -P "(safe|reboot) audit.log',
        "grep -P -e",
        "grep -P --",
    ],
)
def test_ambiguous_or_malformed_grep_syntax_is_never_hidden(command):
    assert detect_hardline_command(command)[0] is True


def test_execution_detection_handles_wrappers_and_compound_commands():
    dangerous, _, _ = detect_dangerous_command(
        "echo ready && env DEBUG=1 python3 -W ignore -c 'print(1)'"
    )
    assert dangerous is True


def test_hardline_payload_after_wrapper_still_reaches_floor():
    hardline, _ = detect_hardline_command(
        "sudo -u nobody sort --compress-program='rm -rf --no-preserve-root /' names"
    )
    assert hardline is True


def test_malformed_quoted_command_does_not_crash():
    detect_dangerous_command("python3 -c 'unterminated")
    detect_hardline_command("sort --compress-program='unterminated")


@pytest.mark.parametrize(
    "command",
    [
        "python3 -Wonce script.py",
        "ruby -rrubygems script.rb",
        "powershell -ConfigurationName Microsoft.PowerShell",
        "powershell -ExecutionPolicy RemoteSigned -NoProfile",
    ],
)
def test_option_values_and_long_options_are_not_treated_as_combined_exec_flags(command):
    dangerous, _, _ = detect_dangerous_command(command)
    assert dangerous is False


def test_valid_exec_flag_before_later_malformed_quote_is_still_detected():
    dangerous, _, _ = detect_dangerous_command(
        "python3 -c 'print(1)' ; printf 'unterminated"
    )
    assert dangerous is True


@pytest.mark.parametrize(
    "command",
    [
        "sort --compress-program=\"sh -c 'rm -rf --no-preserve-root /'\" names",
        "rg --pre \"bash -c 'rm -rf --no-preserve-root /'\" -e . names",
        "man --pager \"sh -c 'rm -rf --no-preserve-root /'\" ls",
    ],
)
def test_wrapped_exec_flag_payload_reaches_hardline_floor(command):
    hardline, description = detect_hardline_command(command)
    assert hardline is True
    assert description == "recursive delete of root filesystem"


def test_interpreter_heredoc_keeps_legacy_approval_key_compatibility():
    from tools.approval import _approval_key_aliases

    aliases = _approval_key_aliases("script execution via heredoc")
    assert r"(python[23]?|perl|ruby|node)\s+<<" in aliases


@pytest.mark.parametrize(
    "command",
    [
        "bash --norc script.sh",
        "bash --rcfile ./bashrc script.sh",
        "bash --restricted script.sh",
        "bash --noediting script.sh",
        "zsh --rcs script.zsh",
    ],
)
def test_shell_long_options_containing_c_are_not_exec_flags(command):
    assert detect_dangerous_command(command) == (False, None, None)


@pytest.mark.parametrize("flag", ["-Wc"])
def test_shell_invalid_short_bundles_are_not_exec_flags(flag):
    assert detect_dangerous_command(f"bash {flag} harmless.sh") == (False, None, None)


def test_shell_double_dash_stops_exec_flag_parsing():
    assert detect_dangerous_command("bash -- -c harmless.sh") == (False, None, None)


@pytest.mark.parametrize(
    "flag",
    ["-c", "-lc", "-ic", "-lic", "-cl", "-cil", "-lci", "-ilc", "-cli", "-abc"],
)
def test_shell_valid_exec_bundle_requires_a_payload(flag):
    assert detect_dangerous_command(f"bash {flag}")[0] is True


@pytest.mark.parametrize(
    "flag",
    ["-c", "-lc", "-ic", "-lic", "-cl", "-cil", "-lci", "-ilc", "-cli", "-abc"],
)
def test_shell_exact_short_exec_flags_require_approval(flag):
    dangerous, _, description = detect_dangerous_command(f"bash {flag} 'printf safe'")
    assert dangerous is True
    assert description == "shell command via -c/-lc flag"


@pytest.mark.parametrize(
    "option_args",
    [
        "-O extglob",
        "+O extglob",
        "-o posix",
        "+o posix",
        "--rcfile /dev/null",
        "--init-file /dev/null",
        "-lO extglob",
        "+lO extglob",
        "-lo posix",
        "+lo posix",
    ],
)
def test_bash_options_consuming_arguments_do_not_hide_later_exec_flag(option_args):
    command = f"bash {option_args} -lc 'rm -rf --no-preserve-root /'"

    assert detect_hardline_command(command) == (
        True,
        "recursive delete of root filesystem",
    )


@pytest.mark.parametrize(
    "command",
    [
        "bash -O -c harmless.sh",
        "bash +O -c harmless.sh",
        "bash -o -c harmless.sh",
        "bash +o -c harmless.sh",
        "bash --rcfile -c harmless.sh",
        "bash --init-file -c harmless.sh",
    ],
)
def test_bash_option_arguments_that_look_like_exec_flags_are_not_promoted(command):
    assert detect_dangerous_command(command) == (False, None, None)


@pytest.mark.parametrize(
    "command",
    [
        'grep -P "$(rm -rf --no-preserve-root /)" audit.log',
        'grep -P "`rm -rf --no-preserve-root /`" audit.log',
        "grep -P $(rm -rf --no-preserve-root /) audit.log",
        "grep -P `rm -rf --no-preserve-root /` audit.log",
    ],
)
def test_grep_patterns_with_executable_substitutions_reach_hardline(command):
    assert detect_hardline_command(command) == (
        True,
        "recursive delete of root filesystem",
    )


def test_single_quoted_grep_substitution_syntax_is_inert_data():
    command = "grep -P '$(printf \"rm -rf --no-preserve-root /\")' audit.log"
    assert detect_hardline_command(command) == (False, None)


@pytest.mark.parametrize(
    "command",
    [
        'sort --compress-program="sh -c \'rm -rf --no-preserve-root /\'" names',
        'sort --compress-program="bash -lc \'rm -rf --no-preserve-root /\'" names',
        'rg --pre="env X=1 sh -c \'rm -rf --no-preserve-root /\'" pattern files',
    ],
)
def test_nested_quoted_executable_payloads_reach_hardline(command):
    assert detect_hardline_command(command) == (
        True,
        "recursive delete of root filesystem",
    )


def test_depth_ten_wrapped_executable_payload_hits_early_size_cap():
    payload = "rm -rf --no-preserve-root /"
    for _ in range(10):
        payload = f"sh -c {shlex.quote(payload)}"
    command = f"man --pager {shlex.quote(payload)} ls"

    assert detect_hardline_command(command) == (
        True,
        "command parser limit exceeded",
    )


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


def test_command_start_reconstruction_copies_each_input_span_once(monkeypatch):
    import tools.approval as approval

    class SliceCountingString(str):
        sliced_characters = 0
        slices = 0

        def __getitem__(self, key):
            value = super().__getitem__(key)
            if isinstance(key, slice):
                type(self).slices += 1
                type(self).sliced_characters += len(value)
            return value

    segment_count = 4_000
    command = SliceCountingString(";".join(["true"] * segment_count))
    monkeypatch.setattr(
        approval,
        "_iter_shell_command_starts",
        lambda _command: range(5, len(command), 5),
    )

    marked = approval._mark_command_starts(command)

    assert marked.count("\n") == segment_count - 1
    assert command.slices == segment_count
    assert command.sliced_characters == len(command)


def test_benign_segment_scaling_benchmark():
    """Retain real metrics without making correctness depend on wall-clock ratios."""
    small, small_result = _time_benign_segments(2_000)
    large, large_result = _time_benign_segments(4_000)

    assert small_result == (False, None, None)
    assert large_result == (False, None, None)
    print(f"benign segment benchmark: 2k={small:.3f}s, 4k={large:.3f}s")


def test_payload_beyond_segment_scan_cap_fails_closed():
    command = ";".join(["true"] * 25_001 + ["rm -rf /"])
    hardline, description = detect_hardline_command(command)
    assert hardline is True
    assert description == "command parser limit exceeded"


@pytest.mark.parametrize("size", [200_000, 500_000])
def test_long_separator_free_token_hits_early_cap_before_regexes(size):
    command = "x" * size
    started = time.perf_counter()
    result = detect_dangerous_command(command)
    elapsed = time.perf_counter() - started

    assert result == (
        True,
        "command parser limit exceeded",
        "command parser limit exceeded",
    )
    assert elapsed < 0.15, f"{size} byte token took {elapsed:.3f}s"


def test_max_accepted_separator_free_input_is_fast():
    from tools.approval import _MAX_SEPARATOR_FREE_COMMAND_CHARS

    command = "x" * _MAX_SEPARATOR_FREE_COMMAND_CHARS
    started = time.perf_counter()
    result = detect_dangerous_command(command)
    elapsed = time.perf_counter() - started

    assert result == (False, None, None)
    assert elapsed < 0.15, f"max accepted token took {elapsed:.3f}s"
