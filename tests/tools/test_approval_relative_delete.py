"""Relative-path, glob and `..` deletes must need the same approval as absolute ones (#151542).

The rm rules used to key on the FIRST CHARACTER of the argument, so the gate depended on how a
command was spelled rather than on what it deletes: `rm /srv/app/x/*.log.gz` prompted while
`cd /srv/app/x && rm logs/archive/*.log.gz` — the same files — ran with no prompt. These tests
pin the effect-based rule and, just as importantly, pin the shapes that must stay quiet so the
approval prompt does not become noise (#18083 is the over-flagging half of the same defect).
"""

import pytest

from tools.approval_detection import detect_dangerous_command


def _flags(command: str) -> bool:
    return detect_dangerous_command(command)[0]


def _key(command: str) -> str | None:
    return detect_dangerous_command(command)[1]


# ── the issue's table: an unrunnable path spelling must not decide the gate ────────────

@pytest.mark.parametrize("command", [
    # absolute path — already flagged before this fix
    "rm /srv/app/checkout-api/logs/archive/*.log.gz",
    # the same deletion after a `cd`, spelled relative: previously (False, None, None)
    "cd /srv/app/checkout-api && rm logs/archive/*.log.gz",
    "rm -f logs/archive/*",
    # traversal out of the working directory: previously (False, None, None)
    "rm ../../etc/passwd",
    "rm -rf ../../etc",
    "rm -f *.log",
    "rm logs/*.gz",
])
def test_relative_and_glob_deletes_are_gated(command):
    assert _flags(command), f"{command!r} deletes without an approval prompt"


@pytest.mark.parametrize("command", [
    "rm a.txt b.txt",
    "rm -rf foo bar baz",
    "rm -f one.log two.log three.log",
])
def test_multiple_targets_are_gated(command):
    assert _flags(command), f"{command!r} removes several paths without an approval prompt"


def test_the_two_spellings_of_one_deletion_agree():
    """The reported asymmetry: the same files must not need approval only when absolute."""
    absolute = "rm /srv/app/checkout-api/logs/archive/*.log.gz"
    relative = "cd /srv/app/checkout-api && rm logs/archive/*.log.gz"
    assert _flags(absolute) == _flags(relative) is True


# ── noise guards: the rule must not fire on prose, redirections, or single files ────────

@pytest.mark.parametrize("command", [
    # a single plain relative file: quiet, matching #18083's ask that the gate stop keying on spelling
    "rm notes.txt",
    "rm -f build/output.o",
    # one target plus a redirection is not two targets
    "rm a.txt 2>/dev/null",
    # the shape carried as DATA in quotes — a quoted glob is not even expanded by the shell
    "grep -r 'rm *.log' .",
    'echo "rm a b"',
    "git commit -m 'rm small big'",
    # `--` ends options; the operand is a literal filename
    "rm -- -weird.txt",
    # the operand run must not be attributed across a command separator
    "rm foo && echo done",
    "rm foo; ls",
    "chmod +x rm",
    "ls -la",
    "rmdir emptydir",
])
def test_quiet_cases_stay_quiet(command):
    assert not _flags(command), f"{command!r} would prompt for no deletion"


# ── the keys themselves ───────────────────────────────────────────────────────────────

def test_new_rules_have_their_own_approval_keys():
    """Approvals are stored per key, so a new shape needs a key that is not an old one."""
    keys = {
        _key("rm -f logs/archive/*"),
        _key("rm ../../etc/passwd"),
        _key("rm a.txt b.txt"),
    }
    assert keys == {
        "delete with a glob",
        "delete outside the working directory through ..",
        "delete of multiple paths",
    }


def test_existing_rm_rules_keep_their_keys():
    """The older rm rules must still win where they already applied — stored approvals depend on it."""
    assert _key("rm /srv/app/x/*.log.gz") == "delete in root path"
    assert _key("cd /srv/app && rm -r checkout-api/logs/archive") == "recursive delete"
    assert _key("find logs -delete") == "find -delete"
    assert _key("ls logs/archive | xargs rm") == "xargs with rm"
