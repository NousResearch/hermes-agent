import pytest
from tools.approval_detection import detect_hardline_command

_WRAPPER_PATH_BYPASS = [
    "nice -n -10 reboot",
    # shutdown / reboot family behind a wrapper or a path
    "nice reboot",
    "command reboot",
    "command shutdown -h now",
    "command systemctl poweroff",
    "ionice poweroff",
    "nice init 0",
    "command telinit 6",
    "timeout 5 reboot",
    "nice -n 10 reboot",
    "nice -10 reboot",
    "timeout -k 5s 10 reboot",
    "/bin/reboot",
    "/sbin/shutdown -h now",
    "/usr/bin/systemctl poweroff",
    "./reboot",
    # root / protected / home wipe behind a wrapper or a path
    "command rm -rf /",
    "nice rm -rf /",
    "ionice rm -rf /",
    "stdbuf -oL rm -rf /",
    "timeout 5 rm -rf /",
    "/bin/rm -rf /",
    "/usr/bin/rm -rf /",
    "./rm -rf /",
    "command rm -rf /etc",
    "nice rm -rf ~",
    'nice rm -rf "/"',
    # Wrapper flags whose operand is non-numeric command metadata (a signal
    # name, an ionice class, a stdbuf mode) must be consumed too, so the verb
    # after them still anchors. Consuming only flags and digit-leading operands
    # left these outside the floor.
    "timeout -s KILL 5 reboot",
    "timeout --signal KILL 5 reboot",
    "ionice -c best-effort rm -rf /",
    "stdbuf -o L rm -rf /",
    "env FOO=1 timeout --signal KILL 5 rm -rf /",
    # sudo/env/doas are wrappers too, and their OPTIONS (not just env
    # assignments and sudo flags-without-operands) must be consumed so the verb
    # after them still anchors. `env -i reboot` and `sudo -u root reboot` were
    # left outside the floor by the narrower per-wrapper handling.
    "env -i reboot",
    "env --ignore-environment reboot",
    "env -u FOO reboot",
    "env -i systemctl poweroff",
    "env --unset=FOO rm -rf /",
    "sudo -u root reboot",
    "sudo -u root rm -rf /",
    # Operand-taking flags whose operand is case-distinct from a no-operand
    # sibling: `sudo -h host` (lowercase) vs `sudo -H` (uppercase, no operand).
    # Matching options case-insensitively would drop one side of this pair.
    "sudo -h host reboot",
    "sudo --host host reboot",
    "doas reboot",
    "doas -u root rm -rf /",
    # A path to the WRAPPER executable (not just to the verb) must be consumed
    # too, so `/usr/bin/nice reboot` anchors like `nice reboot` does. The path
    # prefix previously applied only in front of the final verb.
    "/usr/bin/nice reboot",
    "/usr/bin/timeout 5 reboot",
    "/usr/bin/env -i reboot",
    "/usr/bin/sudo -u root reboot",
    "/usr/bin/env --unset=FOO rm -rf /",
    "/bin/nice rm -rf /",
    "./env -i reboot",
    "../bin/env -i reboot",
    # A slash-containing RELATIVE path names the same executable as its absolute
    # form (identical after `cd /`), so a relative path whose first segment is a
    # bare name (no leading `/`, `./`, or `../`) must be consumed too. These
    # slipped the anchor while the shell still ran the system binary.
    "usr/bin/env -i reboot",
    "usr/bin/nice reboot",
    "bin/reboot",
    "bin/rm -rf /",
    "sbin/shutdown -h now",
    "usr/bin/systemctl poweroff",
    "cd / && usr/bin/env -i reboot",
    "cd / && bin/rm -rf /",
]

_WRAPPER_PREFIX_NOT_A_HARDLINE = [
    "sudo -u reboot /bin/true",
    "nice grep reboot log.txt",
    "timeout 5 echo reboot",
    "nice -n 10 grep reboot file",
    "command grep reboot",
    "command ls",
    "nice make -j4",
    "timeout 30 pytest tests/",
    "ionice -c3 cp big.iso /mnt/backup/",
    "echo /bin/reboot",
    "nice rm -rf ./build",
    "timeout 5 rm -rf /tmp/scratch",
    # A flag operand must not be mistaken for the command: the benign command
    # after an operand-taking flag stays runnable.
    "ionice -c best-effort tar -cf out.tar /etc",
    "timeout -s KILL 5 echo reboot",
    "nice -n 10 grep -c reboot file",
    # sudo/env/doas options and assignments in front of a benign command stay
    # runnable, and a non-destructive systemctl verb after a wrapper option is
    # not on the floor.
    "sudo apt update",
    "env EDITOR=vim git commit",
    "env -i make",
    "sudo -u deploy systemctl status nginx",
    "doas -u deploy ls",
    # A path to a NON-wrapper program is not a wrapper prefix, and a path that
    # merely contains a trigger word is not the verb at command position.
    "/usr/bin/grep reboot file",
    "/usr/local/bin/my-reboot-log --tail",
    "/home/user/nice-tool reboot-helper",
    # A slash-containing relative path to a NON-wrapper program, or a relative
    # path that merely contains a trigger word, is not the verb at command
    # position and stays runnable.
    "usr/bin/grep reboot file",
    "usr/local/bin/my-reboot-log --tail",
    "git log origin/reboot-branch",
    # No-operand wrapper flags must never swallow the real program as their
    # "operand" and re-anchor its arguments as a command. These all blocked
    # when the anchor regex treated every wrapper option as maybe-consuming an
    # operand. `sudo -H` is a no-operand flag (set HOME) while `sudo -h` takes
    # a host operand, so option matching has to be case-sensitive.
    "env -i echo reboot",
    "sudo -n echo reboot",
    "doas -n echo reboot",
    "timeout --foreground echo reboot",
    "env -i echo rm -rf /",
    "nice -n echo reboot",
    "sudo -E echo rm -rf /",
    "sudo -H echo reboot",
    "timeout --preserve-status echo reboot",
]

@pytest.mark.parametrize("command", _WRAPPER_PATH_BYPASS)
def test_wrapper_commands_blocked(command):
    assert detect_hardline_command(command)[0]

@pytest.mark.parametrize("command", _WRAPPER_PREFIX_NOT_A_HARDLINE)
def test_wrapper_data_allowed(command):
    assert not detect_hardline_command(command)[0]
