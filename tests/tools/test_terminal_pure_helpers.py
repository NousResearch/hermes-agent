"""Coverage for pure helper functions in tools/terminal_tool.py.

These are all side-effect-free (no subprocess/network/Docker), so they are
exercised directly against the live module. Assertions are written as behavior
contracts (invariants/relationships) rather than change-detector snapshots
wherever the exact prose is incidental (e.g. guidance strings), and as exact
values where the return is an enum-like decision.
"""

import pytest

import tools.terminal_tool as terminal_tool


# ---------------------------------------------------------------------------
# _safe_command_preview
# ---------------------------------------------------------------------------

def test_safe_command_preview_none():
    assert terminal_tool._safe_command_preview(None) == "<None>"


def test_safe_command_preview_str_under_limit():
    assert terminal_tool._safe_command_preview("hello") == "hello"


def test_safe_command_preview_str_truncated():
    long_str = "x" * 300
    result = terminal_tool._safe_command_preview(long_str, limit=200)
    assert result == "x" * 200
    assert len(result) == 200


def test_safe_command_preview_str_respects_custom_limit():
    assert terminal_tool._safe_command_preview("abcde", limit=3) == "abc"


def test_safe_command_preview_non_str_uses_repr():
    # repr(12345) == "12345"; limit 3 truncates
    assert terminal_tool._safe_command_preview(12345, limit=3) == "123"
    assert terminal_tool._safe_command_preview([1, 2, 3]) == "[1, 2, 3]"


def test_safe_command_preview_repr_raises_falls_back_to_type():
    class _BadRepr:
        def __repr__(self):
            raise ValueError("boom")

    assert terminal_tool._safe_command_preview(_BadRepr()) == "<_BadRepr>"


# ---------------------------------------------------------------------------
# _looks_like_env_assignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "token, expected",
    [
        ("FOO=bar", True),
        ("A1_B2=x", True),
        ("_FOO=1", True),
        ("FOO=", True),
        ("=val", False),
        ("no-eq", False),
        ("FOO", False),
        ("1FOO=x", False),   # name can't start with a digit
        ("A B=x", False),    # whitespace not allowed in name
    ],
)
def test_looks_like_env_assignment(token, expected):
    assert terminal_tool._looks_like_env_assignment(token) is expected


# ---------------------------------------------------------------------------
# _read_shell_token
# ---------------------------------------------------------------------------

def test_read_shell_token_single_quoted():
    token, i = terminal_tool._read_shell_token("'single' word", 0)
    assert token == "'single'"
    assert i == 8


def test_read_shell_token_double_quoted_keeps_inner_whitespace():
    token, i = terminal_tool._read_shell_token('"a b" rest', 0)
    assert token == '"a b"'
    assert i == 5


def test_read_shell_token_double_quoted_with_escaped_quote():
    token, i = terminal_tool._read_shell_token('"a\\"b" rest', 0)
    assert token == '"a\\"b"'
    assert i == 6


def test_read_shell_token_backslash_escape():
    # After a backslash the next char is consumed literally (escaped space kept)
    token, i = terminal_tool._read_shell_token(r"\a\ b c", 0)
    assert token == r"\a\ b"
    assert i == 5


@pytest.mark.parametrize("sep", [";", "&", "|", "("])
def test_read_shell_token_stops_at_operator_separators(sep):
    token, i = terminal_tool._read_shell_token(f"x{sep}y", 0)
    assert token == "x"
    assert i == 1


def test_read_shell_token_stops_at_whitespace():
    assert terminal_tool._read_shell_token("foo bar", 0) == ("foo", 3)


# ---------------------------------------------------------------------------
# _strip_quotes
# ---------------------------------------------------------------------------

def test_strip_quotes_single():
    assert terminal_tool._strip_quotes("cmd 'nohup' arg") == "cmd '' arg"


def test_strip_quotes_double():
    assert terminal_tool._strip_quotes("cmd 'setsid' arg") == "cmd '' arg"


def test_strip_quotes_backtick():
    assert terminal_tool._strip_quotes("cmd `foo` arg") == "cmd `` arg"


def test_strip_quotes_preserves_unquoted_text():
    assert terminal_tool._strip_quotes("echo 'a' 'b' \"c\" x") == "echo '' '' \"\" x"


def test_strip_quotes_handles_escaped_double_quote():
    assert terminal_tool._strip_quotes('echo "a\\"b" c') == 'echo "" c'


# ---------------------------------------------------------------------------
# _looks_like_help_or_version_command
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "command",
    [
        "cmd --help",
        "cmd -h",
        "cmd --version",
        "cmd -V",       # case-insensitive (lowered before matching)
        "cmd -v",
        "git --help sub",
    ],
)
def test_looks_like_help_or_version_true(command):
    assert terminal_tool._looks_like_help_or_version_command(command) is True


def test_looks_like_help_or_version_plain_command():
    assert terminal_tool._looks_like_help_or_version_command("cmd run") is False


def test_looks_like_help_or_version_embedded_flag_is_not_version():
    # '--version' anywhere counts, but a bare '-h'/'--help' only counts as a
    # trailing/space-delimited informational flag.
    assert terminal_tool._looks_like_help_or_version_command("cmd --version run") is True


# ---------------------------------------------------------------------------
# _command_requires_pipe_stdin
# ---------------------------------------------------------------------------

def test_command_requires_pipe_stdin_gh_with_token():
    assert terminal_tool._command_requires_pipe_stdin("gh auth login --with-token") is True


def test_command_requires_pipe_stdin_case_insensitive():
    assert terminal_tool._command_requires_pipe_stdin("GH AUTH LOGIN --WITH-TOKEN") is True


def test_command_requires_pipe_stdin_missing_token():
    assert terminal_tool._command_requires_pipe_stdin("gh auth login") is False


def test_command_requires_pipe_stdin_non_gh_with_token():
    assert terminal_tool._command_requires_pipe_stdin("gh auth logout --with-token") is False


def test_command_requires_pipe_stdin_pipe_operator_is_not_signal():
    # A literal pipe is NOT the trigger for this helper — only `gh auth login
    # --with-token` is.
    assert terminal_tool._command_requires_pipe_stdin("echo hi | cat") is False


# ---------------------------------------------------------------------------
# _foreground_background_guidance
# ---------------------------------------------------------------------------

def _assert_guidance_mentions_background(result):
    assert result is not None
    assert "background" in result


def test_guidance_none_for_help_version():
    assert terminal_tool._foreground_background_guidance("cmd --help") is None
    assert terminal_tool._foreground_background_guidance("cmd -h") is None
    assert terminal_tool._foreground_background_guidance("cmd --version") is None


@pytest.mark.parametrize("command", ["nohup python s.py", "disown", "setsid cmd", "a && setsid b", "nohup python server.py &"])
def test_guidance_shell_level_background_wrappers(command):
    _assert_guidance_mentions_background(
        terminal_tool._foreground_background_guidance(command)
    )


@pytest.mark.parametrize("command", ["python server.py &", "python server.py & # bg", "cmd & other"])
def test_guidance_inline_or_trailing_amp(command):
    _assert_guidance_mentions_background(
        terminal_tool._foreground_background_guidance(command)
    )


@pytest.mark.parametrize(
    "command",
    [
        "python -m http.server 8000",
        "npm run dev",
        "docker compose up",
        "uvicorn app:app",
        "nodemon",
    ],
)
def test_guidance_long_lived_server_patterns(command):
    _assert_guidance_mentions_background(
        terminal_tool._foreground_background_guidance(command)
    )


def test_guidance_normal_command():
    assert terminal_tool._foreground_background_guidance("ls -la") is None


def test_guidance_ignores_keywords_inside_strings():
    # 'setsid' inside a quoted string must not trigger the wrapper guidance.
    assert terminal_tool._foreground_background_guidance("git commit -m 'setsid foo'") is None


# ---------------------------------------------------------------------------
# _resolve_notification_flag_conflict
# ---------------------------------------------------------------------------

def test_notification_flag_conflict_drops_watch_patterns():
    watch_patterns, note = terminal_tool._resolve_notification_flag_conflict(
        notify_on_complete=True, watch_patterns=["foo"], background=True
    )
    assert watch_patterns is None
    assert note != ""
    assert "duplicate" in note


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(notify_on_complete=True, watch_patterns=["x"], background=False),
        dict(notify_on_complete=False, watch_patterns=[], background=True),
        dict(notify_on_complete=False, watch_patterns=["x"], background=False),
        dict(notify_on_complete=True, watch_patterns=[], background=True),
    ],
)
def test_notification_flag_no_conflict_preserves_watch_patterns(kwargs):
    watch_patterns, note = terminal_tool._resolve_notification_flag_conflict(**kwargs)
    assert watch_patterns == kwargs["watch_patterns"]
    assert note == ""


# ---------------------------------------------------------------------------
# _is_safe_workdir_char
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ch", list("abcXYZ0129_ /"))
def test_is_safe_workdir_char_allowed(ch):
    assert terminal_tool._is_safe_workdir_char(ch) is True


@pytest.mark.parametrize(
    "ch",
    [
        "\x00",     # NUL
        "\n",       # control
        "\t",       # control
        "\x1f",     # control
        "\x7f",     # DEL
        "$",
        ";",
        "|",
        "",
    ],
)
def test_is_safe_workdir_char_rejected(ch):
    assert terminal_tool._is_safe_workdir_char(ch) is False


def test_is_safe_workdir_char_unicode_letter():
    assert terminal_tool._is_safe_workdir_char("用") is True


# ---------------------------------------------------------------------------
# _validate_workdir
# ---------------------------------------------------------------------------

def test_validate_workdir_empty_safe():
    assert terminal_tool._validate_workdir("") is None


@pytest.mark.parametrize("workdir", ["/safe/path", "/tmp/  x", "/用户/文档", "relative/path"])
def test_validate_workdir_safe(workdir):
    assert terminal_tool._validate_workdir(workdir) is None


@pytest.mark.parametrize("bad_char", ["$", ";", "|", "\n"])
def test_validate_workdir_rejects_metachar(bad_char):
    result = terminal_tool._validate_workdir(f"/tmp/x{bad_char}y")
    assert isinstance(result, str)
    assert "Blocked: workdir contains disallowed character" in result


# ---------------------------------------------------------------------------
# _docker_volume_uses_host_path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spec",
    [
        "/host:/container",
        "~/data:/data",
        "./rel:/abs",
        "../up:/abs",
        "C:\\win",
    ],
)
def test_docker_volume_uses_host_path_true(spec):
    assert terminal_tool._docker_volume_uses_host_path(spec) is True


@pytest.mark.parametrize("spec", ["named_volume", "", "   ", 123, None, ["/h:/c"]])
def test_docker_volume_uses_host_path_false(spec):
    assert terminal_tool._docker_volume_uses_host_path(spec) is False


# ---------------------------------------------------------------------------
# _docker_has_host_access
# ---------------------------------------------------------------------------

def test_docker_has_host_access_requires_docker_env():
    assert terminal_tool._docker_has_host_access({"env_type": "local"}) is False


def test_docker_has_host_access_via_mount_cwd():
    config = {
        "env_type": "docker",
        "host_cwd": "/home/me",
        "docker_mount_cwd_to_workspace": True,
    }
    assert terminal_tool._docker_has_host_access(config) is True


def test_docker_has_host_access_via_volume():
    config = {"env_type": "docker", "docker_volumes": ["/host:/container"]}
    assert terminal_tool._docker_has_host_access(config) is True


def test_docker_has_host_access_via_mount_cwd_requires_both():
    # host_cwd set but docker_mount_cwd_to_workspace falsy -> no host access
    config = {"env_type": "docker", "host_cwd": "/home/me", "docker_mount_cwd_to_workspace": False}
    assert terminal_tool._docker_has_host_access(config) is False


def test_docker_has_host_access_no_sources():
    assert terminal_tool._docker_has_host_access({"env_type": "docker"}) is False


def test_docker_has_host_access_named_volume_not_host():
    config = {"env_type": "docker", "docker_volumes": ["named_volume"]}
    assert terminal_tool._docker_has_host_access(config) is False
