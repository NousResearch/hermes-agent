"""Rules whose stdlib original ended in negative lookaheads, checked for RE2 against those originals.

On RE2 each of these rules is a ``tools.skills_guard._TrailingExclusionMatcher``: the table entry is the original
without its trailing lookaheads, and the matcher evaluates them in Python at the ends the original tries. The
stdlib originals below are the reference: on every generated line ``finditer`` must give the same spans (a
security rule never matches less, so an exclusion never covers more, and no new false positives), which also
covers a dropped candidate followed by a real one on the same line.
"""
import itertools
import random
import re

import pytest

from tools.skills_guard import _COMPILED_THREAT_PATTERNS, _TrailingExclusionMatcher, _only_end

_CRED_FILE = (r'(?:\.ssh[/\\]id_(?:rsa|ed25519|ecdsa|dsa)(?!\.pub)|\.env\b|credentials\b|\.netrc\b|\.pgpass\b'
              r'|\.npmrc\b|\.pypirc\b)')
_PY_CRED_FILE_ARG = r'(?:os\.path\.expanduser\s*\(\s*)?["\'][^"\'\n]*' + _CRED_FILE + r'[^"\'\n]*["\']\s*\)?'
# The patterns as they were before the RE2 rewrite (stdlib lookahead spelling), compiled as the table is.
_ORIGINAL = {
    "py_read_secrets_file": (
        r'\bopen\s*\(\s*' + _PY_CRED_FILE_ARG
        + r'(?!\s*\)?\s*,\s*["\'][^"\']*[wax][^"\']*["\'])(?![^\n]*\bmode\s*=\s*["\'][^"\']*[wax])'
        + r'|\bPath\s*\(\s*' + _PY_CRED_FILE_ARG + r'\s*\)\.(?:read_text|read_bytes|readlines|readline)\s*\('),
    "deception_hide": (
        r'do\s+not\s+(?:\w+\s+)*tell\s+(?:\w+\s+)*the\s+user(?!\s+to\s+["\'\u201c\u2018])'
        r'(?!.*\b(?:unless|except|until|confirm|diagnos\w*|verify|check)\b)'),
    "exec_string": (
        r'(?<![.\w])exec\s*\(\s*["\'](?!(?-i:PRAGMA|BEGIN|COMMIT|ROLLBACK|CREATE|INSERT|SELECT|UPDATE|DELETE|DROP|ALTER'
        r'|VACUUM|SAVEPOINT|RELEASE|ANALYZE|REINDEX|ATTACH|DETACH|REPLACE)\b)'),
}
# Per rule: line heads, and the tokens the rest of the line is built from (every sequence of up to three, plus a
# seeded sample of longer ones, some holding a second head).
_OPEN_HEADS = ['open("~/.env"', "open ( '.ssh/id_rsa'", 'xopen("x/credentials"', 'open("id_rsa.pub"',
               "open(os.path.expanduser('~/.netrc'", 'Path("~/.env"', 'open(".envrc"']
_HEADS = {
    "py_read_secrets_file": _OPEN_HEADS,
    "deception_hide": ["do not tell the user", "Do  not tell x the user", "do not x the user", "do not tell"],
    "exec_string": ['exec("', "exec ( '", 'x.exec("', 'xexec("', "\u00e9exec('", 'exec(  "'],
}
_TOKENS = {
    "py_read_secrets_file": [")", " ", ",", '"w"', "'rb'", '"a+"', "'x'", '"r"', "mode=", "mode = ", "xmode=",
                             "_mode=", "MODE=", "\u00e9mode=", 'encoding="utf-8"', '"', "'", "\n", ".read_text(",
                             "wax", " as f:", "\t", "))", ", 'w'"],
    "deception_hide": [" ", "x ", " the user", " the  user", "the user", " tell", " to \"", " to \u201c", " to x",
                       " unless", "unless", " checks", " check", " Diagnosis", "\u00e9", "\n", ", ", ".", " Verify",
                       "\u017f", "users", " TO '"],
    "exec_string": [" ", '"', "'", "PRAGMA", "pragma", "SELECT", "X", "_", "\u00e9", "\u017f", " x", "exec(", "\n",
                    "DROP", "DROPS", ")", 'exec("'],
}


def _corpus(pid):
    tokens, heads = _TOKENS[pid], _HEADS[pid]
    tails = ["".join(seq) for n in range(4) for seq in itertools.product(tokens, repeat=n)]
    lines = [head + tail for head in heads for tail in tails]
    rng = random.Random(20261125)
    for _ in range(4000):
        parts = [rng.choice(heads)] + rng.choices(tokens + heads, k=rng.randint(3, 8))
        lines.append("".join(parts))
    return lines


@pytest.mark.parametrize("pid", sorted(_ORIGINAL))
def test_trailing_exclusion_matches_its_lookahead_original(pid):
    original = re.compile(_ORIGINAL[pid], re.IGNORECASE)
    rewritten = next(compiled for compiled, rule, *_ in _COMPILED_THREAT_PATTERNS if rule == pid)
    less, more, moved = [], [], []
    for text in _corpus(pid):
        was = [m.span() for m in original.finditer(text)]
        now = [m.span() for m in rewritten.finditer(text)]
        if was == now:
            continue
        (less if len(now) < len(was) else more if len(now) > len(was) else moved).append((text, was, now))
    assert less == [], f"{pid} misses matches the original finds: {less[:5]!r}"
    assert more == [], f"{pid} finds matches the original did not: {more[:5]!r}"
    assert moved == [], f"{pid} match spans differ from the original: {moved[:5]!r}"


def test_a_dropped_candidate_does_not_hide_a_later_start_inside_it():
    """None of the three rules can start a match inside a candidate that every end excluded, so their corpus
    cannot show the resume step; a two-digit rule can: `123` drops `12` and the original then finds `23`."""
    original = re.compile(r'\d\d(?!\d)')
    digit = re.compile(r'\d')
    matcher = _TrailingExclusionMatcher(r'\d\d', _only_end, lambda text, m, end: digit.match(text, end) is not None)
    for text in ("123", "1234", "12 345", "a123b", "12"):
        assert [m.span() for m in matcher.finditer(text)] == [m.span() for m in original.finditer(text)], text
