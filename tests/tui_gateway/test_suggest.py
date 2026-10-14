"""Deterministic composer ghost-suggestion extractor: suggests only when the reply asks something,
never from fenced commands the user was told to RUN, most recent path first, capped at three."""

import pytest

from tui_gateway.suggest import extract_suggestions

_OPTIONS = "Which approach?\n1. Fast path with a flag\n2. Safe path in a worktree\n3. Skip it"


@pytest.mark.parametrize(("text", "expected"), [
    ("", []),
    ("Deployed the fix and all tests pass.", []),
    ("Want me to proceed?\n" + "\n".join(f"line {i}" for i in range(40)), []),  # ask far above tail
    ("Changes made:\n1. Fixed the plist\n2. Added the marker", []),  # list without a question
    ("I compared `/srv/old.csv` against `~/Work/new.csv`.\nWhich file should I keep?",
     [("~/Work/new.csv", "path"), ("/srv/old.csv", "path")]),
    ("Run this:\n```bash\ncat /etc/hosts\n```\nDoes the output look right?", [("Yes, go ahead", "confirm")]),
    (_OPTIONS, [("Fast path with a flag", "option"), ("Safe path in a worktree", "option")]),
    ("What should the timeout be?", []),
    ("Wrote `/a/1`, `/a/2`, `/a/3`, `/a/4`, again `/a/4`.\nWant me to keep going?",
     [("/a/4", "path"), ("/a/3", "path"), ("/a/2", "path")]),
])
def test_extractor_contract(text, expected):
    assert [(c.text, c.kind) for c in extract_suggestions(text)] == expected
