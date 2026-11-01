"""Does an inline awk program hand text to /bin/sh? (detection only; nothing is executed).

``system()``, ``print | "cmd"``, ``"cmd" | getline`` and gawk's ``|&`` coprocess all run a shell
command. Those spellings are only execution syntax in awk CODE: inside a string (``print "a|b"``),
a regex literal (``/system\\(/``, ``print /a|b/``) or a ``#`` comment they are data, and treating
them as execution made read-only one-liners prompt.
"""

import re

AWK_NAMES = frozenset({"awk", "gawk", "mawk", "nawk"})
AWK_EXEC_DESCRIPTION = "awk program runs a shell command (system()/pipe)"
_AWK_OPTIONS_WITH_ARG = {"-F", "--field-separator", "-v", "--assign", "-f", "--file", "-e", "--source",
                         "-i", "--include", "-l", "--load", "-W"}
_AWK_COMMAND_EXEC_RE = re.compile(r'\bsystem\s*\(|\|&|\|\s*getline\b|\bprintf?\b[^;}\n]*(?<!\|)\|(?![|&])')
# After one of these keywords a `/` opens a regex (`print /re/`); after any other word it divides.
# Words that can themselves be a value (`getline`, a variable) are left out on purpose: reading a
# division as a regex opener would hide the code up to the next `/` from the execution check.
_REGEX_AFTER_KEYWORDS = frozenset({"print", "printf", "return", "case"})
_WORD = re.compile(r"[A-Za-z0-9_.]+")


def awk_program_texts(args: list[str]) -> list[str]:
    """Return the inline awk program text(s) in *args* (``-f`` files are not inspectable)."""
    texts, index, from_file = [], 0, False
    while index < len(args):
        token = args[index]
        if token == "--":
            index += 1
            break
        if token == "-" or not token.startswith("-"):
            break
        option, equals, value = token.partition("=")
        attached = token[2:] if not token.startswith("--") and len(token) > 2 else None
        takes_arg = option in _AWK_OPTIONS_WITH_ARG or token[:2] in _AWK_OPTIONS_WITH_ARG
        if takes_arg and not equals and attached is None:
            value = args[index + 1] if index + 1 < len(args) else ""
            index += 2
        else:
            value = value if equals else (attached or "")
            index += 1
        name = option if option.startswith("--") else token[:2]
        if name in ("-e", "--source"):
            texts.append(value)
        from_file = from_file or name in ("-f", "--file")
    if not texts and not from_file and index < len(args):
        texts.append(args[index])
    return texts


def _scan_delimited(program: str, i: int, closer: str, brackets: bool) -> int | None:
    """Index just past the literal opened at *i* (``"..."`` or ``/.../``), or None when it is not
    closed on its line. A regex ``[...]`` class may hold an unescaped ``/``."""
    j, n, in_class = i + 1, len(program), False
    while j < n and program[j] != "\n":
        ch = program[j]
        if ch == "\\":
            j += 2
            continue
        if brackets and ch == "[" and not in_class:
            # A `]` right after `[` or `[^` is a literal member, not the class closer.
            in_class, j = True, j + 1
            j += program.startswith("^", j)
            j += program.startswith("]", j)
            continue
        if in_class:
            in_class = ch != "]"
        elif ch == closer:
            return j + 1
        j += 1
    return None


def awk_code_only(program: str) -> str:
    """*program* with string literals reduced to ``""``, regex literals to ``//`` and comments
    dropped, so only awk code remains.

    ``operand`` records whether the last token ends a value (word, number, string, regex, ``)``,
    ``]``, postfix ``++``/``--``): a ``/`` there is division, anywhere else it opens a regex. When
    unsure the ``/`` is kept as division, which leaves the following code visible to the check.
    """
    code: list[str] = []
    i, n, operand = 0, len(program), False
    while i < n:
        ch = program[i]
        if ch == '"':
            end = _scan_delimited(program, i, '"', brackets=False)
            # Unterminated: keep scanning what follows as code rather than hide it.
            i = i + 1 if end is None else end
            code.append('""')
            operand = True
        elif ch == "/" and not operand and (end := _scan_delimited(program, i, "/", brackets=True)):
            i = end
            code.append("//")
            operand = True
        elif ch == "#":
            end = program.find("\n", i)
            i = n if end < 0 else end
        elif ch == "\\" and i + 1 < n:
            code.append(program[i:i + 2])  # escaped char or line continuation: state unchanged
            i += 2
        elif program.startswith(("++", "--"), i):
            code.append(program[i:i + 2])  # postfix keeps a value, prefix still awaits one
            i += 2
        elif (word := _WORD.match(program, i)):
            code.append(word.group())
            i = word.end()
            operand = word.group() not in _REGEX_AFTER_KEYWORDS
        else:
            code.append(ch)
            i += 1
            if ch not in " \t":
                operand = ch in ")]"
    return "".join(code)


def awk_program_runs_shell(args: list[str]) -> bool:
    return any(_AWK_COMMAND_EXEC_RE.search(awk_code_only(text)) for text in awk_program_texts(args))
