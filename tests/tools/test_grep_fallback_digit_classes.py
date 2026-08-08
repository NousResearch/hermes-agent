"""The grep fallback must answer the same question ripgrep does.

`_search_with_grep` runs `grep -rnHE`. POSIX ERE has no `\\d`/`\\D`; GNU grep
reads the backslash as a literal, so `\\d+` matches runs of the letter ``d``
instead of digits — false hits *and* missed hits, with no error. The engine is
picked by whether ripgrep is installed, so the same search silently answers
differently inside a container that ships grep but not rg.

GNU grep does support `\\s`, `\\S`, `\\w`, `\\W`, `\\b` as extensions; only the
digit classes need translating.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tools.file_operations import ShellFileOperations

BS = chr(92)  # a literal backslash, spelled without escape ambiguity


@pytest.mark.parametrize(
    "pattern,expected",
    [
        (BS + "d+", "[[:digit:]]+"),
        (BS + "D", "[^[:digit:]]"),
        ("v" + BS + "d" + BS + "." + BS + "d", "v[[:digit:]]" + BS + ".[[:digit:]]"),
        # An escaped backslash is a literal backslash — the ``d`` after it is
        # not a class.
        (BS + BS + "d", BS + BS + "d"),
        # Inside a bracket expression ``[\d]`` already means the set {\, d}.
        ("[" + BS + "d]", "[" + BS + "d]"),
        # ']' as the first member belongs to the set, so the set is still open.
        ("[]" + BS + "d]", "[]" + BS + "d]"),
        # GNU grep understands these; leave them alone.
        (BS + "s" + BS + "w" + BS + "b", BS + "s" + BS + "w" + BS + "b"),
        ("plain", "plain"),
        ("", ""),
    ],
)
def test_translation_covers_only_the_digit_classes(pattern, expected):
    from tools.file_operations import _ere_translate_digit_classes

    assert _ere_translate_digit_classes(pattern) == expected


def _fops_capturing(commands):
    env = MagicMock()
    env.cwd = "/work"

    def _execute(cmd, *args, **kwargs):
        commands.append(cmd if isinstance(cmd, str) else " ".join(map(str, cmd)))
        return {"output": "", "returncode": 1}

    env.execute = _execute
    fops = ShellFileOperations(env)
    fops._has_command = lambda cmd: cmd == "grep"
    return fops


def test_grep_engine_receives_a_digit_class_not_a_bare_escape():
    commands: list[str] = []
    fops = _fops_capturing(commands)

    fops._search_content(BS + "d+", "/work", None, 20, 0, "content", 0)

    grep_cmds = [c for c in commands if "grep -rnHE" in c]
    assert grep_cmds, f"no grep command was issued (commands={commands!r})"
    assert "[[:digit:]]" in grep_cmds[0], (
        f"grep got a pattern it cannot interpret as digits: {grep_cmds[0]!r}"
    )


def test_ripgrep_engine_keeps_the_pattern_verbatim():
    """rg speaks the Perl shorthand natively — do not rewrite it there."""
    commands: list[str] = []
    fops = _fops_capturing(commands)
    fops._has_command = lambda cmd: cmd == "rg"

    fops._search_content(BS + "d+", "/work", None, 20, 0, "content", 0)

    rg_cmds = [c for c in commands if c.startswith("rg ") or " rg " in c]
    assert rg_cmds, f"no rg command was issued (commands={commands!r})"
    assert "[[:digit:]]" not in rg_cmds[0], (
        f"the rg pattern was rewritten unnecessarily: {rg_cmds[0]!r}"
    )
