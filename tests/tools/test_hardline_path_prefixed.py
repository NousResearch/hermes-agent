"""Path-prefixed command names must hit the hardline floor like their bare spellings.

`/sbin/reboot`, `/usr/sbin/shutdown -h now`, `sudo -n /sbin/poweroff`, `/bin/systemctl reboot` and
`env /sbin/halt` used to slip past every _CMDPOS-anchored command-name rule because the rules
required the bare name right after the command-start position. The same gap existed for mkfs, dd,
rm -rf /, kill -1, init 0/6 and telinit 0/6.
"""

import time

import pytest

from tools.approval import detect_dangerous_command, detect_hardline_command


_PATH_PREFIXED_BLOCK = [
    # power state
    ("/sbin/reboot", "system shutdown/reboot"),
    ("/usr/sbin/shutdown -h now", "system shutdown/reboot"),
    ("sudo -n /sbin/poweroff", "system shutdown/reboot"),
    ("sudo /sbin/halt -p", "system shutdown/reboot"),
    ("env /sbin/halt", "system shutdown/reboot"),
    ("env FOO=1 /sbin/reboot", "system shutdown/reboot"),
    ("nohup /sbin/reboot", "system shutdown/reboot"),
    ("exec /usr/sbin/reboot", "system shutdown/reboot"),
    ("command /sbin/reboot", "system shutdown/reboot"),
    ("command -p reboot", "system shutdown/reboot"),
    ("./reboot", "system shutdown/reboot"),
    ("ls && /sbin/reboot", "system shutdown/reboot"),
    ("echo hi; /sbin/shutdown now", "system shutdown/reboot"),
    ("$(/sbin/reboot)", "system shutdown/reboot"),
    ("`/sbin/reboot`", "system shutdown/reboot"),
    ("(/sbin/reboot)", "system shutdown/reboot"),
    ("{ /sbin/reboot; }", "system shutdown/reboot"),
    ("/bin/systemctl reboot", "systemctl poweroff/reboot"),
    ("sudo /usr/bin/systemctl poweroff", "systemctl poweroff/reboot"),
    ("/sbin/init 0", "init 0/6 (shutdown/reboot)"),
    ("/sbin/telinit 6", "telinit 0/6 (shutdown/reboot)"),
    # other command-name hardline rules with the same gap
    ("/sbin/mkfs.ext4 /dev/sda1", "format filesystem (mkfs)"),
    ("/usr/sbin/mkfs -t ext4 /dev/sda1", "format filesystem (mkfs)"),
    ("/bin/dd if=/dev/zero of=/dev/sda", "dd to raw block device"),
    ("/bin/rm -rf /", None),
    ("sudo /bin/rm -rf /etc", None),
    ("/bin/kill -9 -1", "kill all processes"),
]


@pytest.mark.parametrize("command,description", _PATH_PREFIXED_BLOCK)
def test_path_prefixed_command_hits_hardline(command, description):
    blocked, desc = detect_hardline_command(command)
    assert blocked, f"expected hardline block for {command!r}"
    if description is not None:
        assert desc == description


_PATH_PREFIXED_ALLOW = [
    "cat /var/log/reboot.log",
    "grep shutdown file",
    "ls /sbin/reboot",
    "man reboot",
    "echo reboot",
    "echo /sbin/reboot",
    "git commit -m '/sbin/reboot is now path-aware'",
    "which /sbin/reboot",
    "stat /usr/sbin/shutdown",
    "file /bin/systemctl",
    "command -v reboot",
    "cat /sbin/reboot.d/notes",
    # path-prefixed lookalikes are different binaries, not the hardline command
    "/opt/reboot-helper.sh",
    "/usr/bin/ddrescue in.img out.img",
    "/usr/bin/mkfsfoo",
    "/bin/systemctl status",
    "/bin/rm -rf ./build",
]


@pytest.mark.parametrize("command", _PATH_PREFIXED_ALLOW)
def test_path_mentions_do_not_hit_hardline(command):
    blocked, desc = detect_hardline_command(command)
    assert not blocked, f"false hardline block for {command!r}: {desc}"


def test_path_prefixed_mkfs_is_dangerous_too():
    is_dangerous, _key, desc = detect_dangerous_command("/sbin/mkfs /dev/sdb1")
    assert is_dangerous
    assert desc == "format filesystem"


@pytest.mark.parametrize("command", ["/" * 5000 + "x", "sudo " + "/" * 5000, "a/" * 5000 + " ", "/" * 20000])
def test_path_prefix_matcher_does_not_backtrack(command):
    start = time.monotonic()
    detect_hardline_command(command)
    detect_dangerous_command(command)
    assert time.monotonic() - start < 5
