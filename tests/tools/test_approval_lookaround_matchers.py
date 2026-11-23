"""Matchers that stand in for approval rules whose stdlib spelling holds lookarounds match exactly where the rule does.

Each table entry keeps the stdlib pattern; DANGEROUS_PATTERNS_COMPILED swaps in a matcher (RE2 has no lookaround).
The rule compiled with stdlib ``re`` is the reference on a corpus built around each lookaround.
"""
import itertools
import random
import re

import pytest

from tools.approval_detection import _CMDPOS, DANGEROUS_PATTERNS, DANGEROUS_PATTERNS_COMPILED
from tools.approval_hermes_store import HomeCreateTargetMatcher


def _rule(description):
    original = re.compile(next(p for p, d in DANGEROUS_PATTERNS if d == description), re.IGNORECASE | re.DOTALL)
    return original, next(p for p, d in DANGEROUS_PATTERNS_COMPILED if d == description)


def _sample(parts, size, seed):
    cases = ["".join(c) for c in itertools.product(*parts)]
    return random.Random(seed).sample(cases, min(size, len(cases)))


_BLANKS = [" ", "  ", "\t", "\n", " \n "]

_CREATE = "create a file in a sensitive credential/SSH/shell-rc path"
_CREATE_WORDS = ["", "-r ", "--reference ", "-R ", "-- ", "-- -r ", "--  -r ", "x -- --reference ", "-x ", "a ",
                 "--referencex ", "-rr ", "x-r ", '"-r" ', "a\n-r ", "-r  ", "-r\t", "-p -r ", "-- -- -r ", "a\n"]
_CREATE_TARGETS = ["~/.ssh/x", "'~/.bashrc'", "$home/.netrc", "~/.bashrc.bak", "~/.ssh", "${home}/.hermes/.env", "x"]


def _create_cases():
    return _sample([["", "sudo ", "x; ", "echo `", "$(", "\n  ", "a\n", "touch a;", "touch a\n", "touch a\nb "], ["touch", "mkdir", "touchx"],
                    _BLANKS, _CREATE_WORDS, _CREATE_WORDS, _CREATE_TARGETS, ["", ";", " | y", "\nz", ")", " x"]],
                   40_000, 1)


_RM = "recursive delete (flags after operands)"
_RM_WORDS = ["a", "build/", "--", "-f", "-rf", "-R", "--recursive", "--recursivex", "-x-r", '"q"', "a;b", "a|", "&",
             "--x", "-", "x--", "rm"]


def _rm_cases():
    return _sample([["", "x ", "xrm ", "sudo ", "a;"], ["rm", "RM", "rm2"], _BLANKS, _RM_WORDS, _BLANKS, _RM_WORDS,
                    _BLANKS, _RM_WORDS, ["", " ", "\n", "x"]], 40_000, 2)


_LAUNCHCTL = "stop/restart hermes launchd service (kills running agents)"
_LAUNCHCTL_WORDS = ["launchctl stop", "launchctl  bootout", "launchctl\nkill", "launchctl stopx", "xlaunchctl stop",
                    "launchctl list", "hermes", "ai.hermes", "hermesx", "_hermes", "gui/501/ai.hermes.gateway", "x"]


def _launchctl_cases():
    return _sample([_LAUNCHCTL_WORDS, ["", " ", "\n", "/"], _LAUNCHCTL_WORDS, [" ", ";"], _LAUNCHCTL_WORDS], 5_000, 3)


@pytest.mark.parametrize(("description", "cases"), [
    (_CREATE, _create_cases), (_RM, _rm_cases), (_LAUNCHCTL, _launchctl_cases)])
def test_matcher_matches_exactly_where_the_stdlib_rule_does(description, cases):
    original, matcher = _rule(description)
    corpus = cases()
    differ = [c for c in corpus if bool(original.search(c)) != bool(matcher.search(c))]
    assert differ == []
    assert any(original.search(c) for c in corpus) and not all(original.search(c) for c in corpus)


@pytest.mark.parametrize(("description", "command", "expected"), [
    (_CREATE, "touch -r ~/.bashrc /tmp/x", False),
    (_CREATE, "touch -- -r ~/.bashrc", True),
    (_CREATE, "touch -r a ~/.ssh/x", True),
    (_CREATE, "touch a;touch -r ~/.ssh/x", False),
    (_RM, "rm build/ -rf", True),
    (_RM, "rm -- build -rf", False),
    (_RM, "rm build -- -rf", False),
    (_RM, "rm build | grep -r x", False),
    (_LAUNCHCTL, 'for l in ai.hermes.gateway; do launchctl bootout "$l"; done', True),
    (_LAUNCHCTL, "launchctl list | grep hermes", False),
])
def test_known_commands(description, command, expected):
    original, matcher = _rule(description)
    assert bool(original.search(command)) is expected
    assert bool(matcher.search(command)) is expected


def test_create_target_search_resumes_inside_a_dropped_command():
    # A prefix whose longest parse reaches a later command word: the search must go on one character after the
    # dropped start, where the earlier command word and its target are reached, as the stdlib rule does.
    prefix = r"(?:a[^\n]*)?"
    rule = prefix + next(p for p, d in DANGEROUS_PATTERNS if d == _CREATE).removeprefix(_CMDPOS)
    command = "atouch ~/.ssh/x touch -r b"
    assert re.search(rule, command, re.IGNORECASE | re.DOTALL)
    assert HomeCreateTargetMatcher(prefix).search(command)
