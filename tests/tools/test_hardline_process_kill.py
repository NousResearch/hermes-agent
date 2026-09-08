"""Process-destruction spellings on the unconditional hardline floor.

The floor below yolo is only worth the spellings it recognizes, and the two
process rules each recognized exactly one:

* the fork bomb rule matched a function literally NAMED ``:``, so renaming it
  (``bomb(){ bomb|bomb& };bomb``) — same bomb, same fork storm — ran under
  ``--yolo`` / ``approvals.mode: off`` with no prompt at all, and so did the
  canonical bomb written with a ``;`` before the closing brace;
* the kill rule read every flag as a single token, so a signal handed over as
  a SEPARATE token (``kill -s KILL -1``) walked straight past it, and
  ``killall5`` — whose entire job is signalling every process on the host —
  was not covered at all.

The same kill rule fired on ``-1`` wherever it appeared, so ``kill -1 1234``
(signal 1 is SIGHUP: the ordinary "tell the daemon to reload" idiom) was
hardline-blocked. The floor cannot be approved, so that command could not be
run at all — while ``kill -HUP 1234``, the exact same syscall, was fine.
"""

import pytest

from tools.approval import (
    check_all_command_guards,
    check_dangerous_command,
    detect_dangerous_command,
    detect_hardline_command,
    disable_session_yolo,
)
from tools.approval_context import reset_current_session_key, set_current_session_key


# (command, expected description) — every one of these must hit the floor.
_PROCESS_KILL_BLOCK = [
    # ---- fork bombs. The NAME is decoration; the self-reference is the bomb.
    (":(){ :|:& };:", "fork bomb"),
    (":() { : | : & }; :", "fork bomb"),
    (":(){ :|:&; };:", "fork bomb"),            # `;` before the closing brace
    ("bomb(){ bomb|bomb& };bomb", "fork bomb"),
    ("b() { b | b & }; b", "fork bomb"),
    ("function bomb { bomb | bomb & }; bomb", "fork bomb"),
    ("fork(){ fork|fork& }; fork", "fork bomb"),
    ("bash -c 'bomb(){ bomb|bomb& };bomb'", "fork bomb"),
    ("true && bomb(){ bomb|bomb& };bomb", "fork bomb"),
    # ---- kill aimed at pid -1 (every process), signal given as a separate token
    ("kill -s KILL -1", "kill all processes"),
    ("kill -s 9 -1", "kill all processes"),
    ("kill -s SIGKILL -1", "kill all processes"),
    ("kill -s TERM -1", "kill all processes"),
    ("kill -n 9 -1", "kill all processes"),
    ("kill --signal KILL -1", "kill all processes"),
    ("kill --signal=KILL -1", "kill all processes"),
    ("sudo kill -s KILL -1", "kill all processes"),
    ("sh -c 'kill -s KILL -1'", "kill all processes"),
    # -1 is the target once a signal or `--` came first, whatever follows it.
    ("kill -- -1", "kill all processes"),
    ("kill -9 -- -1", "kill all processes"),
    ("kill -1 -1", "kill all processes"),
    ("kill -9 -1 1234", "kill all processes"),
    ("kill -s KILL -1 1234", "kill all processes"),
    # ---- -1 after other pids is still the pid -1 operand (bash and /bin/kill both read it so)
    ("kill 1234 -1", "kill all processes"),
    ("kill -9 1234 -1", "kill all processes"),
    ("kill -s KILL 1234 -1", "kill all processes"),
    ("sudo kill -TERM 1234 5678 -1", "kill all processes"),
    # ---- killall5: "send a signal to all processes" is the whole program
    ("killall5", "kill all processes"),
    ("killall5 -9", "kill all processes"),
    ("killall5 -15", "kill all processes"),
    ("$(killall5 -9)", "kill all processes"),
    ("nohup killall5", "kill all processes"),
    ("sudo /sbin/killall5", "kill all processes"),
]

# Targeted process commands that must stay runnable: the floor has no approval
# path, so a false positive here bans the command outright.
_PROCESS_KILL_ALLOW = [
    # SIGHUP-to-a-pid reload idiom — signal 1 is HUP, the same thing `-HUP` spells.
    "kill -1 1234",
    "kill -1 $(cat /var/run/nginx.pid)",
    "kill -1 -- 1234",
    "kill -HUP 1234",
    "kill -s HUP 1234",
    "kill -s KILL 1234",
    "kill -l",
    "kill 1234 1235",
    # Whole-token rule: -19 is SIGSTOP, -1234 is a process group — neither is pid -1.
    "kill -19 1234",
    "kill -9 -- -1234",
    # killall/pkill by name are the softer dangerous tier, never the floor.
    "killall firefox",
    "killall -9 node",
    "pkill -HUP nginx",
    # killall5 named as DATA, not run.
    'echo "killall5 -9"',
    "grep killall5 script.sh",
    # A fork bomb quoted as prose is an argument, not a command (#93392).
    'echo "bomb(){ bomb|bomb& };bomb"',
    'git commit -m ":(){ :|:& };:"',
    # Ordinary shell functions: no self-pipe, or defined and never invoked.
    "f(){ echo hi; }; f",
    'retry(){ retry_impl "$@" & }; retry',
    "a(){ a|b& };a",
    "g(){ g_helper | g_other & }; g",
    "x(){ x|x& }",
]


@pytest.mark.parametrize("command,expected", _PROCESS_KILL_BLOCK)
def test_process_destruction_spellings_are_hardline(command, expected):
    """Renaming the function or splitting the signal off is not a bypass."""
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"process-kill spelling leaked past the hardline floor: {command!r}"
    assert desc == expected, f"unexpected description {desc!r} for {command!r}"


@pytest.mark.parametrize("command", _PROCESS_KILL_ALLOW)
def test_targeted_process_commands_are_not_hardline(command):
    """A targeted signal (or a mention of one) must stay runnable."""
    is_hl, desc = detect_hardline_command(command)
    assert not is_hl, (
        f"targeted process command false-positived the hardline floor: "
        f"{command!r} (got: {desc})"
    )
    assert desc is None


@pytest.fixture
def clean_session(monkeypatch):
    """Reset session-scoped approval state around each test."""
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    token = set_current_session_key("hardline_process_test")
    try:
        disable_session_yolo("hardline_process_test")
        yield
    finally:
        disable_session_yolo("hardline_process_test")
        reset_current_session_key(token)


def test_yolo_cannot_bypass_renamed_process_kills(clean_session, monkeypatch):
    """HERMES_YOLO_MODE=1 must not bypass the newly recognized spellings."""
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    for cmd in ["bomb(){ bomb|bomb& };bomb", "kill -s KILL -1", "killall5 -9"]:
        r1 = check_dangerous_command(cmd, "local")
        assert r1["approved"] is False, f"yolo leaked hardline on {cmd!r} (check_dangerous_command)"
        assert r1.get("hardline") is True

        r2 = check_all_command_guards(cmd, "local")
        assert r2["approved"] is False, f"yolo leaked hardline on {cmd!r} (check_all_command_guards)"
        assert r2.get("hardline") is True


def test_dangerous_tier_also_flags_renamed_fork_bomb():
    """The dangerous-tier duplicate of the rule tracks the same spellings."""
    for cmd in ["bomb(){ bomb|bomb& };bomb", ":(){ :|:& };:"]:
        is_dangerous, _, desc = detect_dangerous_command(cmd)
        assert is_dangerous, f"fork bomb not flagged on the dangerous tier: {cmd!r}"
        assert desc == "fork bomb"
