"""Nested ``$(`` must not make command detection super-linear or recurse per level.

``_scan_dollar_paren_end`` rescanned to the end of the string from every ``$(``,
``_replace_simple_command_substitutions`` re-tokenised the whole balanced body at
every level, and ``_iter_shell_command_starts`` recursed once per nesting level:
depth 200 (~600 B) took 22 s per detection, depth 400 > 40 s, and ``"$(e" * 1000``
raised RecursionError. These tests bound that, and pin that the faster code returns
exactly what the original did.
"""
import random
import subprocess
import sys
from pathlib import Path

import pytest

from tools import approval_detection as det


def _reference_scan_dollar_paren_end(command: str, start: int):
    """The original, unmemoised balanced ``$(...)`` scanner."""
    depth = 1
    for kind, i, _, quote in det._scan_shell(command, start + 2):
        if kind == "char" and not quote:
            depth += command.startswith("$(", i) - (command[i] == ")")
            if depth == 0:
                return i + 1
    return None


def _reference_replace_simple_command_substitutions(word: str) -> str:
    """The original whole-body substitution rewrite."""
    chars = []
    i = 0
    while i < len(word):
        opener = 2 if word.startswith("$(", i) else 1 if word[i] == "`" else 0
        scan = _reference_scan_dollar_paren_end if opener == 2 else det._scan_backtick_end
        end = scan(word, i) if opener else None
        replacement = det._literal_command_substitution_output(word[i + opener:end - 1]) if end is not None else None
        if replacement is None:
            replacement, end = word[i], i + 1
        chars.append(replacement)
        i = end
    return "".join(chars)


def _reference_iter_shell_command_starts(command: str):
    """The original recursive command-start scan."""
    starts = [0]

    def scan(start, end):
        skip = -1
        for kind, i, j, quote in det._scan_shell(command, start, end, subst="uq", stop_unterminated=True,
                                                comments=True):
            if kind == "subst":
                inner = i + (1 if command[i] == "`" else 2)
                starts.append(inner)
                scan(inner, end if j is None else j - 1)
            elif kind == "char" and quote is None and i != skip:
                if command[i] in "(;\n" or (command[i] == "{" and (i == 0 or command[i - 1].isspace()
                                                                   or command[i - 1] in "(;&|)")):
                    starts.append(i + 1)
                elif command[i] in "&|":
                    repeated = i + 1 < end and command[i + 1] == command[i]
                    skip = i + 1 if repeated else skip
                    starts.append(i + 1 + repeated)

    scan(0, len(command))
    seen = set()
    for start in starts:
        start = det._skip_shell_whitespace(command, start)
        if start >= len(command) or start in seen or det._is_shell_comment_start(command, start):
            continue
        seen.add(start)
        yield start
        _, end, word = det._read_shell_word(command, start)
        if word in det._SHELL_COMMAND_TRANSITIONS:
            starts.append(end)


_ALPHABET = list("$()`'\"\\ x;|&{}#\n\t") + [
    "$(", "$(e", "))", "')'", '")"', "\\)", "\\$(", "\\`", "&&", "||",
    "echo ", "printf ", "printf %s ", "echo -n ", "rm", "$(echo rm)", "`echo rm`", "sudo ", "${IFS}",
]


def _random_words(seed: int, count: int, max_len: int):
    rng = random.Random(seed)
    for _ in range(count):
        yield "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(1, max_len)))


def test_memoised_scanner_matches_reference_in_any_call_order():
    rng = random.Random(1)
    for word in _random_words(seed=2, count=20_000, max_len=30):
        starts = [i for i in range(len(word)) if word.startswith("$(", i)]
        rng.shuffle(starts)  # the memo is shared across calls: order must not matter
        for start in starts:
            assert det._scan_dollar_paren_end(word, start) == _reference_scan_dollar_paren_end(word, start), (word, start)


def test_substitution_rewrite_matches_reference():
    for word in _random_words(seed=3, count=20_000, max_len=28):
        assert det._replace_simple_command_substitutions(word) == _reference_replace_simple_command_substitutions(word), word


def test_iterative_command_starts_match_reference():
    for command in _random_words(seed=4, count=10_000, max_len=30):
        assert list(det._iter_shell_command_starts(command)) == list(_reference_iter_shell_command_starts(command)), command


@pytest.mark.parametrize(
    "command",
    [
        "$(echo rm) -rf /tmp/x",
        "`echo rm` -rf /tmp/x",
        "$( printf rm ) -rf /tmp/x",
        "$(printf %s rm) -rf /tmp/x",
        "$(\\echo rm) -rf /tmp/x",
        "$('echo' rm) -rf /tmp/x",
        "$(echo 'r'm) -rf /tmp/x",
    ],
)
def test_literal_substitution_deobfuscation_still_detects(command):
    assert det.detect_dangerous_command(command)[0] is True


@pytest.mark.parametrize(
    "expr",
    [
        '"echo " + "$(" * 400 + "x" + ")" * 400',
        '"$(e" * 400 + "x" + ")" * 400',
        '"$(e" * 400',
        '"$(" * 400',
        '"$(echo " * 400 + "x" + ")" * 400',
        '"$(e" * 1000',  # RecursionError before
        '"\\"$(" * 1000',
    ],
)
def test_nested_substitution_detection_is_bounded(expr):
    # Subprocess + timeout so a regression cannot hang pytest on the GIL; the in-process bound is
    # generous for slow CI hosts (~1.5 s worst case locally, 22 s+ before).
    code = f"""
import time
from tools.approval import _match_user_deny_rule, detect_dangerous_command
command = {expr}
start = time.perf_counter()
detect_dangerous_command(command)
_match_user_deny_rule(command)
assert time.perf_counter() - start < 6.0, time.perf_counter() - start
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=40, cwd=Path(__file__).resolve().parents[2])
