"""Hardline floor: command-position projections beyond the bare verb (#58642, #71995, #65391, #58736).

The floor must see a verb the shell will run even when it sits behind an exec wrapper, is spelled by
path, or is a literal command string handed to env -S / su -c / runuser -c. Commands that only NAME the
verb (lookup modes, echoed strings, argv data) must stay off the floor.
"""

import pytest

from tools.approval_detection import detect_hardline_command

_R = "r" + "m"

_BLOCKED = [
    # Literal command strings run by a carrier (#58736).
    "env --split-string='reboot'",
    "env -Sreboot",
    "env -S 'nice reboot'",
    "su -c 'reboot'",
    "su root -c reboot",
    "su -l -c 'nice reboot'",
    "su --command=reboot",
    "runuser -c 'reboot'",
    "runuser -u root -c 'systemctl poweroff'",
    "sudo env -S reboot",
    f"su -c '{_R} -rf /'",
    # Wrappers inside an already-scanned shell -c payload are peeled too.
    "sh -c 'sudo reboot'",
    "bash -c 'nice -n 5 reboot'",
    "sh -c \"su -c 'reboot'\"",
    # Quote/escape-spelled wrappers and leading assignments do not hide the verb.
    "n\\ice reboot",
    "'nice' reboot",
    "X=1 nice reboot",
    "xargs reboot",
    "xargs -n1 reboot",
    # A quoted verb behind a wrapper or separator is dequoted like the bare `"reboot"`.
    'nice "reboot"',
    "timeout 5 'reboot'",
    "nice 're'boot",
    'command "shutdown" -h now',
    f'nice "{_R}" -rf /',
    "su -c 'nice \"reboot\"'",
    "sh -c 'nice \"reboot\"'",
    'ls; "reboot"',
    'ls && sudo "reboot"',
    # The busybox/toybox applet (argv[1]) is a command word too.
    'busybox "reboot"',
    "toybox 'reboot'",
    '/bin/busybox "reboot"',
    'nice busybox "reboot"',
    "su -c 'busybox \"reboot\"'",
    # Payloads carried behind a peeled wrapper (xargs sh -c) are scanned.
    "xargs sh -c 'nice reboot'",
    "xargs -n1 sh -c 'nice reboot'",
    "xargs sh -c '\"reboot\"'",
    "xargs su -c 'nice reboot'",
]

_ALLOWED = [
    "su -c 'echo reboot'",
    "su -s /bin/sh -c 'echo hi'",
    "runuser -u deploy -- ls",
    "env -S 'echo reboot'",
    "command -v reboot",
    "command -V shutdown",
    "command -pv reboot",
    "ionice -p 1 reboot",
    "sh -c 'echo nice reboot'",
    "nice -n 5 -- echo reboot",
    'nice "echo" reboot',
    'command -v "reboot"',
    'git commit -m "x; reboot"',
    "su -c 'nice \"echo\" reboot'",
    "busybox --help reboot",
    "busybox --install",
    "xargs sh -c 'echo nice reboot'",
]


@pytest.mark.parametrize("command", _BLOCKED)
def test_carried_and_wrapped_verbs_hit_the_floor(command):
    assert detect_hardline_command(command)[0], command


@pytest.mark.parametrize("command", _ALLOWED)
def test_named_but_not_run_verbs_stay_off_the_floor(command):
    assert not detect_hardline_command(command)[0], command


@pytest.mark.parametrize("command", ["nice reboot", "/sbin/shutdown -h now", "su -c reboot"])
def test_wrapped_verb_is_blocked_end_to_end_under_yolo(command, monkeypatch):
    from tools.approval import check_all_command_guards, disable_session_yolo
    from tools.approval_context import reset_current_session_key, set_current_session_key

    for name in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    token = set_current_session_key("hardline_carrier_test")
    try:
        result = check_all_command_guards(command, "local")
    finally:
        disable_session_yolo("hardline_carrier_test")
        reset_current_session_key(token)
    assert result["approved"] is False
    assert result.get("hardline") is True
