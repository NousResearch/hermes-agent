"""Wrapper option grammar behind the hardline floor: what the shell runs is what gets checked."""

import pytest

from tools.approval_detection import detect_hardline_command

_RUNS_THE_VERB = [
    # Command starts discovered out of offset order (an `if` body after a later command).
    "if nice true; nice reboot; then true; fi",
    "if nice true; then nice true; fi; nice reboot",
    # GNU timeout takes strtod durations.
    "timeout .5 reboot",
    "timeout infinity reboot",
    "timeout 1e3 reboot",
    "timeout -s KILL .5s reboot",
    "timeout +5 reboot",
    "timeout +.5 reboot",
    # A carrier exposed only once its payload is wrapper-peeled.
    "su -c \"xargs su -c 'nice reboot'\"",
    "xargs env -S 'nice reboot'",
    "xargs env -S reboot",
    # env -S inside a short-option bundle still carries the command (`-iS` = -i plus -S).
    "env -iS 'nice' reboot",
    "env -vS 'nice' reboot",
    "env -iS reboot",
    "env -iSnice reboot",
    "env -iS '-S reboot'",
    "env -S '-S reboot'",
    "env -S '-S -S reboot'",
    # In `sudo -ul`, `l` is -u's value (the user), not --list.
    "sudo -ul reboot",
]

_PRINTS_THE_VERB = [
    # The -I replacement string is attached: `fooI` is its value, `echo` is the program.
    "xargs -IfooI echo reboot",
    "xargs -I{} echo reboot",
    # The env -S split string is the program; following words are its arguments.
    "env -S 'echo' reboot",
    "env -S echo reboot",
    "env -S'echo' reboot",
    "env --split-string='echo' reboot",
    "env -iS echo reboot",
    "env -S '-S echo' reboot",
]


@pytest.mark.parametrize("command", _RUNS_THE_VERB)
def test_verb_behind_wrapper_grammar_blocked(command):
    assert detect_hardline_command(command)[0]


@pytest.mark.parametrize("command", _PRINTS_THE_VERB)
def test_wrapper_argument_data_not_blocked(command):
    assert not detect_hardline_command(command)[0]
