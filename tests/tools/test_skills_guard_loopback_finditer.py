"""The env_exfil_* loopback matcher's ``finditer`` against its lookahead original, on lines holding the command
twice: a match the exemption drops must not hide a later command on the same line, and every match keeps the
original's span (the plugin-install context checks read the spans).
"""
import itertools
import random
import re

import pytest

from tests.tools.test_skills_guard_loopback_exemption import _COMMANDS, _ORIGINAL, _TOKENS
from tools.skills_guard import _COMPILED_THREAT_PATTERNS


def _pieces():
    pieces = [*_TOKENS, *("".join(pair) for pair in itertools.product(_TOKENS, repeat=2))]
    rng = random.Random(20261124)
    return _TOKENS, rng.sample(pieces, 120)


@pytest.mark.parametrize("pid", sorted(_ORIGINAL))
def test_loopback_matcher_finditer_matches_its_lookahead_original(pid):
    original = re.compile(_ORIGINAL[pid], re.IGNORECASE)
    rewritten = next(compiled for compiled, rule, *_ in _COMPILED_THREAT_PATTERNS if rule == pid)
    command = _COMMANDS[pid]
    singles, mixed = _pieces()
    differ = []
    for first, second, third in itertools.chain(itertools.product(singles, singles, [""]),
                                                itertools.product(mixed, mixed, ["", " $KEY"])):
        text = command + first + command + second + third
        was = [m.span() for m in original.finditer(text)]
        now = [m.span() for m in rewritten.finditer(text)]
        if was != now:
            differ.append((text, was, now))
    assert differ == [], f"{pid} finditer spans differ from the original: {differ[:5]!r}"
