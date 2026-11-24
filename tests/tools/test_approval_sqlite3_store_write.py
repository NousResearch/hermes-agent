"""The sqlite3 store-write matcher never matches less than the stdlib lookahead rule it replaces."""
import itertools
import re

import pytest

from tools.approval_detection import DANGEROUS_PATTERNS, DANGEROUS_PATTERNS_COMPILED

_DESCRIPTION = "modify Hermes session store via sqlite3"
_ORIGINAL = re.compile(next(p for p, d in DANGEROUS_PATTERNS if d == _DESCRIPTION), re.IGNORECASE | re.DOTALL)
_MATCHER = next(p for p, d in DANGEROUS_PATTERNS_COMPILED if d == _DESCRIPTION)

_PREFIXES = ["", "sudo -x/", "env a=sqlite3/ ", "sqlite3.restore/", "x/sqlite3<<", "echo hi;\n", "`", "$("]
_COMMANDS = ["sqlite3", "sqlite3.restore", "sqlite3<<", "sqlite3)", "sqlite3x"]
_WORDS = ["~/.hermes/state.db", "'$hermes_home/sessions/x'", "~/.hermes/state.db.bak", "delete\nfrom t",
          "vacuum", "vacuum main into x", "vacuum\ninto x", "vacuum int", ".restore x", "<<eof", "<<<x", "select 1"]
_SEPARATORS = [" ", "\n", "/", "\n sqlite3 "]


def _cases():
    for prefix, command, sep, first, second in itertools.product(_PREFIXES, _COMMANDS, _SEPARATORS, _WORDS, _WORDS):
        yield f"{prefix}{command}{sep}{first} {second}"


@pytest.mark.parametrize("command", [
    "sqlite3 ~/.hermes/state.db 'delete from sessions'",
    "sqlite3 ~/.hermes/state.db vacuum",
    "sqlite3 ~/.hermes/state.db <<eof",
    "sqlite3.restore/sqlite3 ~/.hermes/state.db",
    "sudo -x/sqlite3\nsqlite3 ~/.hermes/state.db .restore b",
])
def test_store_writes_match(command):
    assert _ORIGINAL.search(command)
    assert _MATCHER.search(command)


@pytest.mark.parametrize("command", [
    "sqlite3 ~/.hermes/state.db .tables",
    "sqlite3 ~/.hermes/state.db 'vacuum into /tmp/b.db'",
    "sqlite3 ~/.hermes/state.db <<<'select 1'",
    "sqlite3 ~/.hermes/state.db\ndelete from t",
])
def test_reads_and_copies_out_stay_clean(command):
    assert not _ORIGINAL.search(command)
    assert not _MATCHER.search(command)


def test_never_matches_less_than_the_lookahead_rule():
    missed = [c for c in _cases() if _ORIGINAL.search(c) and not _MATCHER.search(c)]
    assert missed == []
