"""Hardline floor: a leading redirection, an abbreviated sudo long option or Git Bash's ``rm.exe``
does not move the verb off the command position (#155258 review, cases from #82830)."""
import pytest

from tools.approval_detection import detect_hardline_command

_BLOCKED = [
    # The shell applies a redirection before the first word and runs the next one.
    "2>/dev/null reboot",
    "</dev/null shutdown -h now",
    "&>/dev/null /sbin/reboot",
    "2>&1 reboot",
    ">>/tmp/log reboot",
    "ls && >/dev/null reboot",
    "X=1 2>/dev/null reboot",
    # ... also between a wrapper and its program, and between wrapper options.
    "nice 2>/dev/null reboot",
    "sudo 2>/dev/null reboot",
    "2>/dev/null nice -n 5 reboot",
    # sudo parses long options with getopt_long: any unambiguous prefix names the option.
    "sudo --use root /sbin/shutdown",
    "sudo --chd /tmp /sbin/shutdown",
    "sudo --us root rm -rf /etc",
    "sudo --auth=x reboot",
    # sudo's other valued short options.
    "sudo -a type /sbin/shutdown",
    "sudo -r role -t type reboot",
    # Git Bash runs `rm.exe` as `rm`.
    "rm.exe -rf /",
    "C:/Git/usr/bin/rm.exe -rf /",
    "./rm.exe -rf /etc",
]

_ALLOWED = [
    # A redirection after the command name is an argument position, not the verb.
    "echo 2>/dev/null reboot",
    "2>/dev/null echo reboot",
    "2>&1 grep reboot log",
    "ls 2>reboot.log",
    ">reboot",
    "echo hi >reboot",
    "<<<reboot cat",
    "cat <(echo reboot)",
    # `--ch` is ambiguous (--chdir / --chroot), so sudo refuses it.
    "sudo --ch /tmp reboot",
    "rm.exe -rf ./build",
    "echo rm.exe -rf /",
]


@pytest.mark.parametrize("command", _BLOCKED)
def test_verb_behind_redirect_or_prefix_is_hardline(command):
    assert detect_hardline_command(command)[0], command


@pytest.mark.parametrize("command", _ALLOWED)
def test_redirect_targets_and_arguments_stay_off_the_floor(command):
    assert detect_hardline_command(command) == (False, None)
