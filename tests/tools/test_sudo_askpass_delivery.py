"""Sudo password delivery through an askpass helper.

Where the backend's sudo supports ``-A`` (per-environment capability probe), the command gets
the askpass prologue and the same password lines on stdin as the stdin form; the prologue takes
them off stdin once its helper is set up, so the rest of stdin reaches the command untouched and
each sudo gets the password through its helper. When setup fails at run time the lines stay on
stdin and each sudo reads one, as in the stdin form. Everywhere else the command and stdin stay
byte-for-byte the stdin form.
"""

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.waiting import wait_until
from tools import terminal_tool, terminal_tool_sudo
from tools import terminal_tool_sudo_askpass as askpass
from tools.environments.base import BaseEnvironment

PW = "synthetic-pw-7"
PRE = askpass.SUDO_ASKPASS_PREFIX
OPT = askpass.SUDO_ASKPASS_SUFFIX.strip()
TOKEN = askpass.SUDO_ASKPASS_TIMEOUT_TOKEN
NOPASSWD_PROBE = "sudo -n true"


def setup_function():
    terminal_tool_sudo._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool_sudo._reset_cached_sudo_passwords()


class _RecordingEnv(BaseEnvironment):
    """Records every ``_run_bash`` call; the probes answer with the configured exit codes."""

    _passwordless_sudo_probe_supported = True

    def __init__(self, probe_rc=0, nopasswd_rc=1, timeout=10):
        super().__init__(cwd="/", timeout=timeout)
        self.calls = []
        self.probe_rc = probe_rc
        self.nopasswd_rc = nopasswd_rc

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        self.calls.append((cmd_string, stdin_data, timeout))
        return cmd_string

    def _wait_for_process(self, proc, timeout=120, **_kwargs):
        rc = {askpass.CAPABILITY_PROBE: self.probe_rc, NOPASSWD_PROBE: self.nopasswd_rc}.get(proc, 0)
        return {"output": "", "returncode": rc}

    def cleanup(self):
        pass

    def probe_runs(self):
        return [call for call in self.calls if call[0] == askpass.CAPABILITY_PROBE]

    def command_runs(self):
        return [call for call in self.calls if call[0] not in (askpass.CAPABILITY_PROBE, NOPASSWD_PROBE)]


@pytest.fixture
def configured_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", PW)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool.set_sudo_password_callback(None)


# --- 1. askpass form through _prepare_command / execute -----------------------------------------


@pytest.mark.parametrize(("command", "count", "words"), [
    ("sudo id -u", 1, ("sudo",)),
    ("sudo id -u && /usr/bin/sudo whoami; env FOO=1 sudo true", 3, ("sudo", "/usr/bin/sudo")),
])
def test_askpass_form_prologue_plus_rewrite_and_one_password_line(configured_password, command, count, words):
    env = _RecordingEnv()
    exec_command, sudo_stdin = env._prepare_command(command)

    rewritten = command.replace("sudo", f"{PRE}sudo {OPT}")
    rewritten = rewritten.replace(f"/usr/bin/{PRE}sudo {OPT}", f"{PRE}/usr/bin/sudo {OPT}")
    assert exec_command == askpass.build_prologue(count, TOKEN, False, sudo_words=words) + rewritten
    assert f"__hermes_sudo_n={count}\n" in exec_command
    assert PW not in exec_command
    assert sudo_stdin == (PW + "\n") * count


@pytest.mark.parametrize("command", ["sudo id -u", "sudo a && sudo b && sudo c"])
def test_execute_hands_backend_one_password_line_then_caller_stdin(configured_password, command):
    env = _RecordingEnv()
    env.execute(command, stdin_data="caller-1\ncaller-2\n")

    (wrapped, stdin_data, _timeout), = env.command_runs()
    assert stdin_data == (PW + "\n") * command.count("sudo") + "caller-1\ncaller-2\n"
    assert PW not in wrapped
    assert f"{PRE}sudo {OPT}" in wrapped


# --- 2. timeout token ------------------------------------------------------------------------------


@pytest.mark.parametrize(("timeout", "expected"), [(None, "40"), (17, "47"), (300, "330")])
def test_execute_fills_timeout_token_with_effective_timeout_plus_30(configured_password, timeout, expected):
    env = _RecordingEnv(timeout=10)
    env.execute("sudo true && sudo true", timeout=timeout)

    (wrapped, _stdin, run_timeout), = env.command_runs()
    assert TOKEN not in wrapped
    assert f"__hermes_sudo_t={expected};" in wrapped
    assert run_timeout == (timeout or 10)


def test_no_executed_command_contains_the_timeout_token(configured_password):
    env = _RecordingEnv()
    for command in ("sudo id -u", "sudo a; sudo b", "echo hi", "sudo -S id", "sudo true & wait"):
        env.execute(command, stdin_data="x\n")
    env._stdin_mode = "heredoc"
    env.execute("sudo id -u", stdin_data="x\n")
    assert env.command_runs()
    for cmd_string, _stdin, _timeout in env.calls:
        assert TOKEN not in cmd_string


def test_fill_timeout_token_leaves_other_text_byte_for_byte():
    text = f"echo {TOKEN}"
    assert askpass.fill_timeout_token(text, 10) == text
    prologue = askpass.build_prologue(1, TOKEN) + f"echo {TOKEN}"
    filled = askpass.fill_timeout_token(prologue, 10)
    assert filled == askpass.build_prologue(1, "40") + f"echo {TOKEN}"
    assert askpass.fill_timeout_token(askpass.build_prologue(1, TOKEN), "bad") == askpass.build_prologue(1, "150")


# --- 3. fallbacks keep the stdin form byte for byte --------------------------------------------------


def test_fallback_without_askpass_keyword(configured_password):
    assert terminal_tool_sudo._transform_sudo_command("sudo a && sudo b") == (
        "sudo -S -p '' a && sudo -S -p '' b", f"{PW}\n{PW}\n")


def test_fallback_when_askpass_supported_returns_false(configured_password):
    assert terminal_tool_sudo._transform_sudo_command("sudo a && sudo b", askpass_supported=lambda: False) == (
        "sudo -S -p '' a && sudo -S -p '' b", f"{PW}\n{PW}\n")


def test_fallback_when_capability_probe_fails(configured_password):
    env = _RecordingEnv(probe_rc=1)
    assert env._prepare_command("sudo a && sudo b") == ("sudo -S -p '' a && sudo -S -p '' b", f"{PW}\n{PW}\n")
    env.execute("sudo id -u", stdin_data="rest\n")
    (wrapped, stdin_data, _timeout), = env.command_runs()
    assert stdin_data == f"{PW}\nrest\n"
    assert "sudo -S -p '\\'''\\'' id -u" in wrapped  # the wrapper's eval quoting of ''
    assert "__hermes_sudo" not in wrapped


@pytest.mark.parametrize(("command", "expected"), [
    ("printf x | sudo -S id -u", "printf x | sudo -S -p '' -S id -u"),
    ("sudo -nS true; sudo id", "sudo -S -p '' -nS true; sudo -S -p '' id"),
    ("sudo id; sudo -A id", "sudo -S -p '' id; sudo -S -p '' -A id"),
    ("sudo --stdin id", "sudo -S -p '' --stdin id"),
    ("sudo --askpass id", "sudo -S -p '' --askpass id"),
    ("sudo -u root -S id", "sudo -S -p '' -u root -S id"),
    ("sudo -E -p 'Pw?' -kA id", "sudo -S -p '' -E -p 'Pw?' -kA id"),
    ("env FOO=1 sudo -iS id", "env FOO=1 sudo -S -p '' -iS id"),
])
def test_fallback_when_command_carries_own_stdin_or_askpass(configured_password, command, expected):
    env = _RecordingEnv()
    count = expected.count("-S -p ''")
    assert env._prepare_command(command) == (expected, f"{PW}\n" * count)


def test_fallback_for_backend_without_in_backend_probes(configured_password):
    env = _RecordingEnv()
    env._passwordless_sudo_probe_supported = False
    assert env._prepare_command("sudo a && sudo b") == ("sudo -S -p '' a && sudo -S -p '' b", f"{PW}\n{PW}\n")
    assert env.calls == []


def test_windows_local_backend_gets_false_without_probing(configured_password, monkeypatch):
    env = _RecordingEnv()
    env.is_local = True
    monkeypatch.setattr(askpass.platform, "system", lambda: "Windows")
    assert env._prepare_command("sudo true") == ("sudo -S -p '' true", f"{PW}\n")
    assert env.calls == []


def test_probe_error_counts_as_unsupported(configured_password):
    env = _RecordingEnv()
    attempts = []

    def _broken_run_bash(cmd_string, **_kwargs):
        attempts.append(cmd_string)
        raise OSError("backend gone")

    env._run_bash = _broken_run_bash
    assert env._prepare_command("sudo true") == ("sudo -S -p '' true", f"{PW}\n")
    assert env._prepare_command("sudo true") == ("sudo -S -p '' true", f"{PW}\n")
    assert len(attempts) <= 1


# --- 4. probe cached ------------------------------------------------------------------------------


def test_capability_probe_runs_once_per_environment(configured_password):
    env = _RecordingEnv()
    env.execute("sudo id -u")
    env.execute("sudo a && sudo b")
    assert len(env.probe_runs()) == 1
    assert env.probe_runs()[0][2] == env._SUDO_PROBE_TIMEOUT_S
    assert all("__hermes_sudo_dir" in call[0] for call in env.command_runs())


# --- 5. no password / NOPASSWD -----------------------------------------------------------------------


def test_no_password_leaves_command_and_skips_probe(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool.set_sudo_password_callback(None)
    env = _RecordingEnv()
    assert env._prepare_command("sudo id -u") == ("sudo id -u", None)
    assert env.calls == []

    def _fail():
        raise AssertionError("askpass probe must not run without a password")

    assert terminal_tool_sudo._transform_sudo_command("sudo id -u", askpass_supported=_fail) == ("sudo id -u", None)


def test_nopasswd_leaves_command_and_skips_askpass_probe(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("no prompt under NOPASSWD")

    monkeypatch.setattr(terminal_tool_sudo, "_prompt_for_sudo_password", _fail_prompt)
    env = _RecordingEnv(nopasswd_rc=0)
    assert env._prepare_command("sudo id -u") == ("sudo id -u", None)
    assert [call[0] for call in env.calls] == [NOPASSWD_PROBE]


# --- 6. empty configured password --------------------------------------------------------------------


def test_empty_configured_password_gets_one_empty_line_in_askpass_form(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    env = _RecordingEnv()
    exec_command, sudo_stdin = env._prepare_command("sudo a && sudo b")
    assert exec_command == askpass.build_prologue(2, TOKEN, False) + f"{PRE}sudo {OPT} a && {PRE}sudo {OPT} b"
    assert sudo_stdin == "\n\n"


# --- 7. has_own_stdin_or_askpass table ----------------------------------------------------------------


@pytest.mark.parametrize(("command", "expected"), [
    ("sudo id", False),
    ("sudo -S id", True),
    ("sudo -A id", True),
    ("sudo --stdin id", True),
    ("sudo --askpass id", True),
    ("sudo -nS true", True),
    ("sudo -kA id", True),
    ("sudo -u root -S id", True),
    ("sudo -uroot -S id", True),
    ("sudo --user root --stdin id", True),
    ("sudo --user=root -A id", True),
    ("sudo -u S id", False),
    ("sudo -uS id", False),
    ("sudo -pS id", False),
    ("sudo -p 'Pw?' id", False),
    ("sudo -g A id", False),
    ("sudo id -S", False),
    ("sudo -- -S", False),
    ("sudo 2>/dev/null -S id", True),
    ("sudo > out -S true", True),
    ("sudo 2> out -S true", True),
    ("sudo >> out -A true", True),
    ("sudo < in -S true", True),
    ("sudo 2>&1 -S true", True),
    ("sudo >&2 -S true", True),
    ("sudo &> out -S true", True),
    ("sudo &>out -S true", True),
    ("sudo > out id -S", False),
    ("sudo 2>&1 id", False),
    ("sudo id 2>&1 && sudo -S id", True),
    ("sudo id 2>&1 | tail -S", False),
    ("sudo id > -S", False),
    ("printf x | sudo '-S' id", True),
    ("sudo $FLAGS id", True),
    ("echo sudo -S", False),
    ("grep -S sudo file", False),
    ("sudo id; sudo -S id", True),
    ("sudo id\n-S", False),
    ("/usr/bin/sudo -A id", True),
    ("env FOO=1 sudo -A x", True),
    ("bash -c 'sudo -S id'", False),
    ("", False),
])
def test_has_own_stdin_or_askpass_table(command, expected):
    assert askpass.has_own_stdin_or_askpass(command) is expected


def test_default_rewrite_is_unchanged_and_askpass_rewrite_prefixes_each_sudo():
    command = "sudo a && '/opt/my tools/sudo' -u root b; env -i sudo c"
    assert terminal_tool_sudo._rewrite_real_sudo_invocations(command) == (
        "sudo -S -p '' a && '/opt/my tools/sudo' -S -p '' -u root b; env -i sudo -S -p '' c", 3)
    assert terminal_tool_sudo._rewrite_real_sudo_invocations(command, askpass=True) == (
        f"{PRE}sudo {OPT} a && {PRE}'/opt/my tools/sudo' {OPT} -u root b; env -i {PRE}sudo {OPT} c", 3)


# --- output consumers: wrong-password text from both forms ------------------------------------------

# sudo 1.9.15 stderr for a wrong password, captured live in ubuntu:24.04 for each form.
_WRONG_PW_STDIN_FORM = "Sorry, try again.\n\nsudo: no password was provided\nsudo: 1 incorrect password attempt\n"
_WRONG_PW_ASKPASS_FORM = "Sorry, try again.\nsudo: no password was provided\nsudo: 1 incorrect password attempt\n"


def test_wrong_password_text_is_judged_the_same_for_both_forms(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    for output in (_WRONG_PW_STDIN_FORM, _WRONG_PW_ASKPASS_FORM, "sudo: 3 incorrect password attempts\n"):
        terminal_tool_sudo._set_cached_sudo_password(PW)
        first = terminal_tool_sudo._sudo_wrong_password_failure(output)
        cleared = terminal_tool_sudo._invalidate_cached_sudo_on_auth_failure("sudo id -u", output)
        assert cleared is first
        assert (terminal_tool_sudo._get_cached_sudo_password() == "") is first
    assert terminal_tool_sudo._sudo_wrong_password_failure(_WRONG_PW_STDIN_FORM) == (
        terminal_tool_sudo._sudo_wrong_password_failure(_WRONG_PW_ASKPASS_FORM))


# --- live: real bash, fake sudo --------------------------------------------------------------------

_LIVE_TOOLS = ("bash", "mkfifo", "mktemp")


def _needs_posix_shell(test):
    """Linux and macOS (the macOS lane imports this file through the marker), with bash, mkfifo, mktemp."""
    test = pytest.mark.skipif(not all(shutil.which(tool) for tool in _LIVE_TOOLS),
                              reason="needs bash with mkfifo and mktemp")(test)
    return pytest.mark.platforms("posix")(test)


# Emulates sudo's password handling: ``-A`` asks $SUDO_ASKPASS, ``-S`` reads one stdin line, and
# a ticket file stands in for sudo's timestamp (a later sudo does not ask again).
_FAKE_SUDO = """#!/bin/sh
ticket="$(dirname "$0")/ticket"
mode=none
case "$1" in
  -A) mode=askpass; shift ;;
  -S) mode=stdin; shift; if [ "$1" = "-p" ]; then shift 2; fi ;;
esac
# Like sudo: a repeated -S is accepted, -S together with -A is refused.
while [ "$1" = "-S" ]; do
  if [ "$mode" = askpass ]; then echo "sudo: the -A and -S options may not be used together" >&2; exit 90; fi
  shift
done
if [ "$1" = "-V" ]; then echo "fake sudo"; exit 0; fi
if [ ! -e "$ticket" ]; then
  case $mode in
    askpass) ans=$("$SUDO_ASKPASS" "Password: ") ;;
    stdin) IFS= read -r ans ;;
    *) echo "sudo: a terminal is required to read the password" >&2; exit 1 ;;
  esac
  if [ "$ans" != "@PW@" ]; then echo "sudo: 1 incorrect password attempt" >&2; exit 1; fi
  : > "$ticket"
fi
exec "$@"
"""


def _write_fake_sudo(bin_dir: Path, answer: str = PW, accepts_askpass: bool = True) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake = bin_dir / "sudo"
    text = _FAKE_SUDO.replace("@PW@", answer)
    if not accepts_askpass:
        text = text.replace('if [ "$1" = "-V" ]', 'if [ "$mode" = askpass ]; then exit 1; fi\nif [ "$1" = "-V" ]')
    fake.write_text(text, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake


def _prologue_processes(*needles: str) -> list[str]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if any(needle in cmdline for needle in needles):
            found.append(f"{entry.name}: {cmdline[:120]}")
    return found


@pytest.fixture
def live_env(tmp_path, monkeypatch):
    """Real LocalEnvironment (real bash, real wrapper and snapshot). Commands call the fake sudo
    by absolute path, so no host sudo can ever be reached; the cached password is the source."""
    if sys.platform == "win32" or not all(shutil.which(tool) for tool in _LIVE_TOOLS):
        pytest.skip("needs a POSIX bash with mkfifo and mktemp")
    from tools.environments.local import LocalEnvironment

    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool.set_sudo_password_callback(None)
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    fake = _write_fake_sudo(tmp_path / "bin")
    monkeypatch.setenv("TERMINAL_TEMP_DIR", str(temp_dir))
    env = LocalEnvironment(cwd=str(tmp_path), timeout=10, env={"TMPDIR": str(temp_dir)})
    # The prologue checks ``sudo -A -V`` on PATH at run time: the fake answers it, not a host sudo.
    env.execute(f"export PATH={fake.parent}:$PATH")
    terminal_tool_sudo._set_cached_sudo_password(PW)
    # The probe itself is covered by test_capability_probe_live_*; pin the answer here so the
    # result does not depend on which sudo the host has.
    env._sudo_askpass_ok = True
    try:
        yield env, str(fake), temp_dir
    finally:
        env.cleanup()


def _run(env, command, stdin=None):
    result = env.execute(command, stdin_data=stdin)
    return result["returncode"], result["output"]


@_needs_posix_shell
def test_live_earlier_stdin_reader_gets_caller_stdin_not_password(live_env):
    env, sudo, _ = live_env
    rc, output = _run(env, f'read -r p; echo "got:$p"; {sudo} true', "line1\n")
    assert rc == 0, output
    assert "got:line1" in output
    assert PW not in output


@_needs_posix_shell
@pytest.mark.parametrize("form", [
    "{sudo} > {out} -S true",
    "{sudo} 2>&1 -S true",
    "{sudo} &> {out} -S true",
    "{sudo} -S true",
])
def test_live_own_stdin_option_after_a_redirection_keeps_the_stdin_form(live_env, form):
    env, sudo, temp_dir = live_env
    rc, output = _run(env, form.format(sudo=sudo, out=temp_dir / "out"))
    assert rc == 0, output
    assert "may not be used together" not in output


@_needs_posix_shell
def test_live_leftover_password_line_reaches_later_reader_only_in_stdin_form(live_env):
    env, sudo, _ = live_env
    command = f"{sudo} true && {sudo} true; cat"
    rc, output = _run(env, command, "rest\n")
    assert rc == 0, output
    assert output.strip() == "rest"

    # Same command with askpass support forced off: the unused second line is read by cat.
    Path(sudo).with_name("ticket").unlink()
    env._sudo_askpass_ok = False
    rc, output = _run(env, command, "rest\n")
    assert rc == 0, output
    assert output.split() == [PW, "rest"]


@_needs_posix_shell
def test_live_xtrace_never_shows_password(live_env):
    env, sudo, _ = live_env
    rc, output = _run(env, f"set -x; {sudo} true; echo after")
    assert rc == 0, output
    assert "after" in output and "-A true" in output
    assert PW not in output


@_needs_posix_shell
def test_live_no_temp_dir_or_process_left_after_return(live_env):
    env, sudo, temp_dir = live_env
    for command in (f"{sudo} true", f"false && {sudo} true; echo done", f"{sudo} true && {sudo} true"):
        rc, output = _run(env, command)
        assert rc == 0, output
        deadline = time.monotonic() + 3
        left, procs = [Path("unpolled")], ["unpolled"]
        while time.monotonic() < deadline:
            left = list(temp_dir.glob("hermes-sudo.*"))
            procs = _prologue_processes(str(temp_dir), env._session_id) if Path("/proc").is_dir() else []
            if not left and not procs:
                break
            time.sleep(0.05)
        assert left == [] and procs == [], (left, procs)


@_needs_posix_shell
def test_live_timeout_kill_leaves_no_temp_dir(live_env):
    env, sudo, temp_dir = live_env
    result = env.execute(f"sleep 20; {sudo} true", timeout=2)
    assert result["returncode"] == 124, result
    wait_until(lambda: not list(temp_dir.glob("hermes-sudo.*")), timeout=5, what="askpass dir removed")


@_needs_posix_shell
def test_live_background_sudo_and_wait_return_promptly(live_env):
    env, sudo, _ = live_env
    started = time.monotonic()
    rc, output = _run(env, f"{sudo} true & wait; echo done")
    elapsed = time.monotonic() - started
    assert rc == 0, output
    assert output.strip().endswith("done")
    assert elapsed < 8, elapsed  # the watchdog's own limit here is 40 s


@_needs_posix_shell
def test_live_snapshot_holds_neither_password_nor_prologue_names(live_env):
    env, sudo, _ = live_env
    assert env._snapshot_ready
    rc, output = _run(env, f"{sudo} true; export AFTER_SUDO=1", "x\n")
    assert rc == 0, output
    snapshot = Path(env._snapshot_path).read_text(encoding="utf-8")
    assert "AFTER_SUDO" in snapshot  # the snapshot really was re-dumped after this command
    assert PW not in snapshot
    assert "__hermes_sudo" not in snapshot and "SUDO_ASKPASS" not in snapshot


# --- live: prologue hardening and the capability probe ------------------------------------------------


def _fake_path(sudo: Path) -> str:
    """PATH with the fake sudo first, so the prologue's ``sudo -A -V`` check asks the fake."""
    return f"{sudo.parent}{os.pathsep}{os.environ.get('PATH', '')}"


def _bash(script, stdin="", env_vars=None, timeout=20):
    return subprocess.run(["bash", "-c", script], input=stdin, capture_output=True, text=True, check=False,
                          env={**os.environ, **(env_vars or {})}, timeout=timeout)


@_needs_posix_shell
def test_live_prologue_ignores_aliases_and_functions_for_its_commands(tmp_path):
    sudo = _write_fake_sudo(tmp_path / "bin")
    command, count = terminal_tool_sudo._rewrite_real_sudo_invocations(f"{sudo} id -u", askpass=True)
    body = askpass.build_prologue(count, "20") + command
    shadow = ("shopt -s expand_aliases\n"
              "alias command=false builtin=false exec=false test=false printf=false mktemp=false\n"
              "mkfifo() { return 1; }\nmktemp() { return 1; }\nchmod() { return 1; }\n")
    escaped = body.replace("'", "'\\''")
    proc = _bash(shadow + f"eval '{escaped}'", stdin=PW + "\n", env_vars={"TMPDIR": str(tmp_path), "PATH": _fake_path(sudo)})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(os.getuid())


@_needs_posix_shell
@pytest.mark.parametrize("special", ['"', "\\", "$", "`"])
def test_live_temp_path_with_quote_special_char_is_setup_failure(tmp_path, special):
    """Setup refuses the path, and the sudo then reads its password line from stdin as the stdin form does."""
    odd = tmp_path / f"t{special}x"
    odd.mkdir()
    sudo = _write_fake_sudo(tmp_path / "bin")
    command, count = terminal_tool_sudo._rewrite_real_sudo_invocations(f"{sudo} id -u", askpass=True)
    proc = _bash(askpass.build_prologue(count, "20") + command, stdin=PW + "\n", env_vars={"TMPDIR": str(odd), "PATH": _fake_path(sudo)})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(os.getuid())
    assert list(odd.iterdir()) == []
    assert _bash(askpass.CAPABILITY_PROBE, env_vars={"TMPDIR": str(odd), "PATH": _fake_path(sudo)}).returncode == 1


@_needs_posix_shell
def test_live_sudo_after_watchdog_timeout_is_served_while_the_command_runs(tmp_path):
    """A command yielded to the background outlives its timeout; its later sudo still authenticates."""
    sudo = _write_fake_sudo(tmp_path / "bin")
    command, count = terminal_tool_sudo._rewrite_real_sudo_invocations(f"sleep 3; {sudo} id -u", askpass=True)
    proc = _bash(askpass.build_prologue(count, "1") + command, stdin=PW + "\n", env_vars={"TMPDIR": str(tmp_path), "PATH": _fake_path(sudo)})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(os.getuid())
    wait_until(lambda: not list(tmp_path.glob("hermes-sudo.*")), timeout=5, what="askpass dir removed")


@_needs_posix_shell
def test_live_every_counted_sudo_that_asks_is_served(tmp_path):
    """Three sudos that each ask (the ticket is dropped between them): the FIFO swap between servings works."""
    sudo = _write_fake_sudo(tmp_path / "bin")
    drop = f"rm -f {sudo.parent / 'ticket'}"
    command, count = terminal_tool_sudo._rewrite_real_sudo_invocations(
        f"{sudo} id -u; {drop}; {sudo} id -u; {drop}; {sudo} id -u", askpass=True)
    assert count == 3
    proc = _bash(askpass.build_prologue(count, "20") + command, stdin=PW + "\n", env_vars={"TMPDIR": str(tmp_path), "PATH": _fake_path(sudo)})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == [str(os.getuid())] * 3
    wait_until(lambda: not list(tmp_path.glob("hermes-sudo.*")), timeout=5, what="askpass dir removed")


@_needs_posix_shell
def test_live_sudos_that_ask_at_the_same_time_are_each_served(tmp_path):
    """Four background sudos with no shared ticket all ask at once; each helper gets its own line."""
    sudo = _write_fake_sudo(tmp_path / "bin")
    sudo.write_text(sudo.read_text(encoding="utf-8").replace('if [ ! -e "$ticket" ]; then', "if true; then"),
                    encoding="utf-8")
    body = " ".join(f"({sudo} true && echo OK{i} || echo FAIL{i}) &" for i in range(4)) + " wait"
    command, count = terminal_tool_sudo._rewrite_real_sudo_invocations(body, askpass=True)
    assert count == 4
    for _ in range(5):
        proc = _bash(askpass.build_prologue(count, "20") + command, stdin=PW + "\n",
                     env_vars={"TMPDIR": str(tmp_path), "PATH": _fake_path(sudo)})
        assert sorted(proc.stdout.split()) == ["OK0", "OK1", "OK2", "OK3"], proc.stderr
    wait_until(lambda: not list(tmp_path.glob("hermes-sudo.*")), timeout=5, what="askpass dir removed")


@_needs_posix_shell
def test_live_setup_failure_at_run_time_falls_back_to_the_stdin_form(live_env):
    """The temp dir goes away after the capability probe passed: every sudo still authenticates from stdin."""
    env, sudo, _ = live_env
    env.execute("export TMPDIR=/nonexistent-hermes-sudo-dir")
    drop = f"rm -f {Path(sudo).parent / 'ticket'}"
    rc, output = _run(env, f"{sudo} id -u; {drop}; {sudo} id -u; cat", "rest\n")
    assert rc == 0, output
    assert output.split() == [str(os.getuid())] * 2 + ["rest"]


@_needs_posix_shell
def test_live_unused_serving_is_gone_once_the_command_returns(live_env):
    """A counted sudo that never asks, plus a background child that outlives the command: nothing is left to read."""
    env, sudo, temp_dir = live_env
    rc, output = _run(env, f"{sudo} true && {sudo} id -u; sleep 15 >/dev/null 2>&1 &")
    assert rc == 0, output
    wait_until(lambda: not list(temp_dir.glob("hermes-sudo.*")), timeout=5, what="askpass dir removed")


@_needs_posix_shell
def test_live_backgrounded_sudo_is_served_after_the_shell_exits(tmp_path):
    sudo = _write_fake_sudo(tmp_path / "bin")
    out = tmp_path / "out"
    body = f"( sleep 3; {sudo} id -u > {out} 2>&1 ) &"
    command, count = terminal_tool_sudo._rewrite_real_sudo_invocations(body, askpass=True)
    assert askpass.has_background_sudo(body)
    proc = _bash(askpass.build_prologue(count, "20", True) + command, stdin=PW + "\n", env_vars={"TMPDIR": str(tmp_path), "PATH": _fake_path(sudo)})
    assert proc.returncode == 0, proc.stderr
    wait_until(lambda: out.exists() and out.read_text().strip() == str(os.getuid()), timeout=10,
               what="backgrounded sudo authenticated")
    wait_until(lambda: not list(tmp_path.glob("hermes-sudo.*")), timeout=8, what="askpass dir removed")


@_needs_posix_shell
@pytest.mark.parametrize("body", [
    "case x in x) sleep 3; {sudo} id -u > {out} 2>&1;; esac &",
    "coproc {{ sleep 3; {sudo} id -u > {out} 2>&1; }}",
])
def test_live_backgrounded_sudo_in_case_or_coproc_is_served_after_the_shell_exits(live_env, body):
    env, sudo, temp_dir = live_env
    if body.startswith("coproc"):
        from tools.environments.local import _find_bash
        if subprocess.run([_find_bash(), "-c", "coproc true"], capture_output=True, check=False).returncode:
            pytest.skip("this bash has no coproc (bash 3.2)")
    out = temp_dir.parent / "out"
    rc, output = _run(env, body.format(sudo=sudo, out=out))
    assert rc == 0, output
    wait_until(lambda: out.exists() and out.read_text().strip() == str(os.getuid()), timeout=10,
               what="backgrounded sudo authenticated")
    wait_until(lambda: not list(temp_dir.glob("hermes-sudo.*")), timeout=8, what="askpass dir removed")


@pytest.mark.parametrize(("command", "expected"), [
    ("sudo a", False),
    ("sudo a && sudo b", False),
    ("sudo a && sudo b; nohup sleep 20 >/dev/null 2>&1 &", False),
    ("sudo a 2>&1; sleep 5 &", False),
    ("sudo a &> out; sleep 5 &", False),
    ("sleep 5 & sudo a", False),
    ("sudo a\nsleep 5 &", False),
    ("sudo a &", True),
    ("sudo a & wait", True),
    ("sudo a && sudo b &", True),
    ("sudo a | tee log &", True),
    ("( sleep 3; sudo a ) &", True),
    ("{ sleep 3; sudo a; } &", True),
    ("f() { :; sudo a; }; f &", True),
    ("( sudo a; sleep 1 & )", True),
    ("echo hi &", False),
    ("case x in x) sleep 3; sudo a;; esac &", True),
    ("coproc { sleep 3; sudo a; }", True),
])
def test_has_background_sudo_table(command, expected):
    assert askpass.has_background_sudo(command) is expected


@_needs_posix_shell
def test_live_orphan_holding_fd_19_does_not_keep_the_writer_past_the_timeout(tmp_path):
    sudo = _write_fake_sudo(tmp_path / "bin")
    command, count = terminal_tool_sudo._rewrite_real_sudo_invocations(
        f"sleep 8 >/dev/null 2>&1 & false && {sudo} true", askpass=True)
    proc = _bash(askpass.build_prologue(count, "1") + command, stdin=PW + "\n", env_vars={"TMPDIR": str(tmp_path), "PATH": _fake_path(sudo)})
    assert proc.returncode == 1, proc.stderr
    wait_until(lambda: not list(tmp_path.glob("hermes-sudo.*")), timeout=5, what="askpass dir removed")


def _probe_env(tmp_path, bin_dir):
    class _ShellEnv(BaseEnvironment):
        """Smallest real backend: spawns bash with a fixed PATH."""

        _passwordless_sudo_probe_supported = True

        def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
            path = f"{bin_dir}:/usr/bin:/bin"
            return subprocess.Popen(["bash", "-c", cmd_string], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, text=True,
                                    env={"PATH": path, "TMPDIR": str(tmp_path), "HOME": str(tmp_path)})

        def cleanup(self):
            pass

    return _ShellEnv(cwd=str(tmp_path), timeout=10)


@_needs_posix_shell
def test_capability_probe_live_passes_with_askpass_capable_sudo(tmp_path, configured_password):
    _write_fake_sudo(tmp_path / "bin")
    env = _probe_env(tmp_path, tmp_path / "bin")
    exec_command, sudo_stdin = env._prepare_command("sudo id -u")
    assert exec_command == askpass.build_prologue(1, TOKEN, False) + f"{PRE}sudo {OPT} id -u"
    assert sudo_stdin == PW + "\n"
    assert list(tmp_path.glob("hermes-sudo.*")) == []


@_needs_posix_shell
def test_capability_probe_live_fails_when_sudo_rejects_askpass(tmp_path, configured_password):
    _write_fake_sudo(tmp_path / "bin", accepts_askpass=False)
    env = _probe_env(tmp_path, tmp_path / "bin")
    assert env._prepare_command("sudo id -u") == ("sudo -S -p '' id -u", PW + "\n")
    assert list(tmp_path.glob("hermes-sudo.*")) == []


@_needs_posix_shell
def test_capability_probe_live_fails_without_sudo(tmp_path, configured_password):
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    env = _probe_env(tmp_path, empty)
    if shutil.which("sudo", path=f"{empty}:/usr/bin:/bin"):
        pytest.skip("a sudo binary is on the fixed PATH")
    assert env._prepare_command("sudo id -u") == ("sudo -S -p '' id -u", PW + "\n")


# --- noclobber: the session's ``set -C`` must not turn the askpass form into the stdin form ------


@_needs_posix_shell
def test_live_noclobber_session_keeps_the_askpass_form(configured_password, tmp_path):
    fake = _write_fake_sudo(tmp_path / "bin")
    command, stdin = _RecordingEnv()._prepare_command('read -r p; echo "got:$p"; sudo id -u')
    command = askpass.fill_timeout_token(command, 10)
    done = subprocess.run(["bash", "-C", "-c", command], input=stdin + "rest\n", text=True,
                          capture_output=True, timeout=15, check=False,
                          env={"PATH": f"{fake.parent}:/usr/bin:/bin", "TMPDIR": str(tmp_path)})
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert done.stdout.split() == ["got:rest", str(os.getuid())], (done.stdout, done.stderr)
    assert PW not in done.stdout + done.stderr


@_needs_posix_shell
def test_live_local_backend_noclobber_startup_keeps_caller_input(configured_password, tmp_path, monkeypatch):
    from tools.environments.local import LocalEnvironment

    fake = _write_fake_sudo(tmp_path / "bin")
    startup = tmp_path / "bash-env"
    startup.write_text("set -C\n")
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setenv("TERMINAL_TEMP_DIR", str(temp))
    env = LocalEnvironment(cwd=str(tmp_path), timeout=10, env={"TMPDIR": str(temp), "BASH_ENV": str(startup)})
    try:
        assert env.execute(f"export PATH={fake.parent}:$PATH")["returncode"] == 0
        result = env.execute('read -r p; echo "got:$p"; sudo true', stdin_data="rest\n")
    finally:
        env.cleanup()
    assert result["returncode"] == 0, result
    assert result["output"].split() == ["got:rest"], result
