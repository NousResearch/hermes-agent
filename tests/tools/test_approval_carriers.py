"""Command-carrier behavior contracts for #58742."""
import pytest
from tools.approval_detection import detect_dangerous_command, detect_hardline_command

_LONG_CARRIER_PREFIX = "env " + " ".join(f"A{i}=x" for i in range(21))

_CARRIED_HARDLINE_BYPASS = [
    "sudo -u sh su -c reboot",
    "sh -c 'reboot'",
    "bash -c 'reboot'",
    "dash -c 'reboot'",
    "zsh -c 'reboot'",
    "ksh -c 'reboot'",
    "sh -c 'systemctl poweroff'",
    "bash -c 'shutdown -h now'",
    "sh -c 'rm -rf /'",
    "bash -c 'rm -rf ~'",
    "su -c 'reboot'",
    "runuser -c 'reboot'",
    "su root -c 'reboot'",
    "su --command 'reboot' root",
    "su --command='reboot' root",
    "su --session-command 'reboot' root",
    "su --session-command='reboot' root",
    "runuser --command 'reboot' root",
    "runuser --command='reboot' root",
    "runuser --session-command 'reboot' root",
    "runuser --session-command='reboot' root",
    "su --command=re'boot' root",
    "runuser --session-command re'boot' root",
    "su --command 'echo safe' --command reboot root",
    "runuser --session-command 'echo safe' --session-command reboot root",
    "su -c 'echo safe' --command reboot root",
    "runuser --command 'echo safe' -c reboot root",
    "su -c reboot -w --command root",
    "runuser -c reboot --whitelist-environment --command root",
    "su -c reboot -g --command root",
    "runuser -c reboot --shell --command root",
    "su -c reboot root # --command 'echo safe'",
    "runuser -c reboot root # --session-command 'echo safe'",
    "su -c reboot root\nprintf safe",
    "runuser -c reboot root\nrunuser --command 'echo safe' root",
    "su --comm reboot root",
    "runuser --sess reboot root",
    'env --split-string="reboot"',
    "env --split-string='rm -rf /'",
    "env --split-string reboot",
    "env --s='sh -c reboot'",
    "env --split 'sh -c reboot'",
    "env --split-strin='sh -c reboot'",
    "env --s''='sh -c reboot'",
    "env --sp''lit='sh -c reboot'",
    "env --s$''='sh -c reboot'",
    "env --$'split'='sh -c reboot'",
    r"env --$'\x73'='sh -c reboot'",
    r"env --s$'\0'='sh -c reboot'",
    r"env --s$'\c@'='sh -c reboot'",
    r"su --command re$'\0'boot root",
    r"env --s=sh\ -c\ reboot",
    r'''env -S "sh\_-c\_reboot"''',
    r'''env --s="sh\_-c\_reboot"''',
    "env --u sh -- sh -c reboot",
    "env --c /tmp -- sh -c reboot",
    "env --a sh -- sh -c reboot",
    "env --u FOO --s reboot",
    "env --c /tmp --s reboot",
    "env --a custom --s reboot",
    "env --b --s reboot",
    "env -Sreboot",
    "/bin/sh -c 'reboot'",
    "sudo sh -c 'reboot'",
    "env FOO=1 sh -c 'reboot'",
    "env FOO=1 BAR=2 bash -c 'rm -rf /'",
    "env A-B=1 sh -c reboot",
    "env -S 'A-B=1 sh -c reboot'",
    "sh -c 'sh -c reboot'",           # carrier nested in carrier
    "ls; sh -c 'reboot'",             # after a separator
    "sh -ec 'reboot'",                # clustered short options, -c is last
    "bash -ec 'reboot'",
    "dash -ec 'reboot'",
    "sh -xc 'rm -rf /'",
    "bash -exc 'rm -rf /'",
    "sh -lc 'reboot'",
    "sh -cx 'reboot'",                # letters after c are still shell options
    "sh -ce 'reboot'",
    "bash -cx 'reboot'",
    "bash -ce 'reboot'",
    # Shells keep parsing invocation options after seeing c; the first word
    # after the complete option prefix is still the command string.
    "sh -c -x 'reboot'",
    "bash -c -e 'rm -rf /'",
    "sh -ce -- 'reboot'",
    "bash -c -o nounset 'reboot'",
    "su -lc 'reboot'",
    # GNU env permits S after no-argument options in a short-option bundle and
    # appends operands after the split string to the resulting argv.
    "env -iS 'reboot'",
    "env -viS 'reboot'",
    "env -iS 'rm -rf /'",
    "env -iSreboot",
    "env -viS/bin/sh -c reboot",
    "env -vS/bin/sh -c 'rm -rf /'",
    r"env -S '/bin/sh\_-c\_reboot'",
    "env -S '-i sh -c reboot'",
    "env -iC /tmp sh -c reboot",
    "env -a custom -S 'sh -c reboot'",
    "env --argv0 custom -S 'sh -c reboot'",
    # A carrier reached behind wrapper OPTIONS, not just wrapper words and
    # NAME=VALUE assignments. The whole option prefix must be skipped.
    "env -i sh -c 'reboot'",
    "env --ignore-environment sh -c 'reboot'",
    "env -u FOO sh -c 'reboot'",
    "env --unset=FOO sh -c 'reboot'",
    "sudo -u root sh -c 'reboot'",
    "sudo --user root sh -c 'reboot'",
    "sudo -E sh -c 'reboot'",
    "sudo -n sh -c 'reboot'",
    "sudo -u root -E sh -c 'reboot'",
    "sudo -u root sh -c 'rm -rf /'",
    # Wrappers that carry no command of their own but still hide a carrier.
    "nice sh -c 'reboot'",
    "nice -n 10 sh -c 'reboot'",
    "ionice sh -c 'reboot'",
    "stdbuf -oL sh -c 'reboot'",
    "timeout 5 sh -c 'reboot'",
    "timeout -s KILL 5 sh -c 'reboot'",
    "timeout 1.5s sh -c 'reboot'",
    "timeout 5 bash -ec 'reboot'",
    "doas sh -c 'reboot'",
    "doas -u root sh -c 'reboot'",
    # Wrappers nested and path-prefixed with options in between.
    "sudo -u root env -i sh -c 'reboot'",
    "env -i sudo sh -c 'reboot'",
    "/usr/bin/env -i /bin/sh -c 'reboot'",
    "/bin/sudo -u root sh -c 'reboot'",
    # Options that really do take an operand still reach the carrier after it.
    "ionice -c 2 sh -c 'reboot'",
    "ionice -c best-effort sh -c 'reboot'",
    "stdbuf -o L sh -c 'reboot'",
    "timeout -s KILL 5 sh -c 'reboot'",
]

_CARRIED_NOT_A_COMMAND = [
    "env -S 'sh -c echo'",
    "sh -c 'echo reboot'",
    "bash -c 'git commit -m reboot'",
    "sh -c 'systemctl status nginx'",
    "sh -c 'grep -r reboot /etc'",
    "bash -c 'ls -la'",
    "env --split-string='echo reboot'",
    "env --s='/bin/echo reboot'",
    "env --split '/bin/echo reboot'",
    "env --n --s reboot",
    "env --nu --s reboot",
    "env --nul --s reboot",
    "env FOO=1 --s reboot",
    "env FOO=1 --debug sh -c reboot",
    '''env --s='$(printf "reboot")' ''',
    '''env --s=A=1 '$(printf sh)' -c reboot''',
    r"env --unset=$'\U00110000' printf OK",
    "su --command 'echo reboot' root",
    "su --session-command='printf reboot' root",
    "runuser --command 'echo reboot' root",
    "runuser --session-command='printf reboot' root",
    "su --command reboot --command 'echo safe' root",
    "runuser --session-command reboot -c 'echo safe' root",
    "su -- root --command reboot",
    "runuser -- root --session-command reboot",
    "su --command 'echo safe' --command",
    "runuser --session-command 'echo safe' -c",
    "su -wFOO,creboot root",
    "runuser --whitelist-environment=FOO,creboot root",
    "su -s/bin/sh root",
    "runuser -u root --command reboot",
    "su -m -w FOO --command reboot root",
    "su --preserve-environment --whitelist-environment FOO --command reboot root",
    "runuser -p -w FOO --session-command reboot root",
    # Non-carrier programs whose own `-c` means something else entirely.
    "gcc -c main.c",
    "grep -c reboot access.log",
    "env EDITOR=vim git commit",
    "find . -name reboot.service",
    "sh -ec 'echo reboot'",           # clustered options, benign payload
    "sh -cx 'echo reboot'",           # options after c do not become its payload
    "sh -- -c reboot",                # option parsing already ended
    "sh /dev/null -c reboot",         # script path already selected
    "bash -lc 'ls -la'",
    "env -iS 'echo reboot'",
    "env -Svi reboot",                # vi is the split-string program
    "env -uS /bin/echo reboot",       # S is the operand of -u, not an option
    "env -S/bin/echo reboot",
    r"env -S '/bin/echo\_reboot'",
    "env -- /bin/echo -iS reboot",
    "env /bin/echo -viS reboot",
    "env -0S reboot",                 # --null is incompatible with a command
    # Wrapped commands with options, but no carrier or a benign payload, must
    # not be swept up by the option-skipping prefix walk.
    "sudo -u root ls",
    "sudo -u root sh",                # interactive shell, no -c payload
    "timeout 5 curl https://example.com",
    "nice -n 10 make -j4",
    "env -i printenv",
    "env -u FOO make",
    "stdbuf -oL grep reboot app.log",
    "sudo -E git push",
    "doas -u root ls",
    "timeout 5 sh -c 'echo reboot'",
    "sudo -u root sh -c 'echo reboot'",
    # Interpreter carriers run another language rather than a shell command
    # string, so they are the runtime-computed / arbitrary-code class and are
    # deliberately left to the softer guards, not the shell-carrier extractor.
    "python3 -c 'import os; os.system(\"reboot\")'",
    "perl -e 'system(\"reboot\")'",
    # A no-operand wrapper flag must not consume the wrapper's real program, so
    # its arguments (which merely look like a carrier) are not rescanned. Here
    # `echo` is the program and `sh -c reboot` are the words it prints.
    "sudo -E echo sh -c reboot",
    "sudo -n echo sh -c reboot",
    "timeout --foreground echo sh -c reboot",
    "timeout --preserve-status echo sh -c reboot",
    "sudo -H echo sh -c reboot",
    "env -v echo sh -c reboot",
    "stdbuf -oL echo sh -c reboot",
    # `nice -n` takes a numeric operand, so a non-numeric next token is the
    # program, not the niceness value.
    "nice -n echo sh -c reboot",
    # `env` must only parse its own option/assignment prefix for `-S` /
    # `--split-string`. After `--` or after the program word, later tokens are
    # data for that program, not an env-carried command string.
    "env -- echo --split-string reboot",
    "env echo --split-string reboot",
    "env -i echo --split-string reboot",
    "env FOO=1 echo --split-string reboot",
]


@pytest.mark.parametrize("command", _CARRIED_HARDLINE_BYPASS)
def test_carried_commands_are_blocked(command):
    assert detect_hardline_command(command)[0] is True


@pytest.mark.parametrize("command", _CARRIED_NOT_A_COMMAND)
def test_carried_data_stays_allowed(command):
    assert detect_hardline_command(command)[0] is False
    if command == "env -S 'sh -c echo'":
        assert detect_dangerous_command(command)[0] is True
