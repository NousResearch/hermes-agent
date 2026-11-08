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
# gawk options whose value is attached only (`-oF` writes to file F): they end a bundle and never
# take the next word.
_AWK_ATTACHED_ONLY_OPTIONS = frozenset("opdDL")
_AWK_LONG_OPTIONS = tuple(option for option in _AWK_OPTIONS_WITH_ARG if option.startswith("--"))
# `@name(` is gawk's indirect function call (`f = "system"; @f("id")`).
_AWK_COMMAND_EXEC_RE = re.compile(r'\bsystem\s*\(|\|&|\|\s*getline\b|@[A-Za-z_]\w*\s*\(')
_AWK_PRINT_RE = re.compile(r'\bprintf?\b')
_AWK_LONE_PIPE_RE = re.compile(r'(?<!\|)\|(?![|&])')
# A backslash-newline or a newline after a comma continues the `print a, b | "cmd"` statement.
_AWK_CONTINUATION_RE = re.compile(r'\\\n|(?:,|&&|\|\|)[ \t]*\n\s*')
_AWK_STATEMENT_END_RE = re.compile(r'[;}\n]')
# After one of these keywords a `/` opens a regex (`print /re/`); after any other word it divides.
# Words that can themselves be a value (`getline`, a variable) are left out on purpose: reading a
# division as a regex opener would hide the code up to the next `/` from the execution check.
_REGEX_AFTER_KEYWORDS = frozenset({"print", "printf", "return", "case"})
_WORD = re.compile(r"[A-Za-z0-9_.]+")


def _awk_option(args: list[str], index: int) -> tuple[str | None, str, int]:
    """Parse the option at ``args[index]`` -> (canonical name, its value, index of the next token).
    Handles ``--long[=value]`` with getopt_long's unique prefixes (``--sou=``) and short-option
    bundles, where the first option that takes an operand ends the bundle (``-bv x=1``)."""
    token = args[index]
    if token.startswith("--"):
        option, equals, value = token.partition("=")
        matches = [name for name in _AWK_LONG_OPTIONS if name.startswith(option)]
        option = matches[0] if option not in _AWK_LONG_OPTIONS and len(matches) == 1 else option
        if option in _AWK_OPTIONS_WITH_ARG and not equals:
            return option, args[index + 1] if index + 1 < len(args) else "", index + 2
        return option, value, index + 1
    for position, flag in enumerate(token[1:], 1):
        if flag in _AWK_ATTACHED_ONLY_OPTIONS:
            return None, "", index + 1
        if "-" + flag in _AWK_OPTIONS_WITH_ARG:
            attached = token[position + 1:]
            if attached:
                return "-" + flag, attached, index + 1
            return "-" + flag, args[index + 1] if index + 1 < len(args) else "", index + 2
    return None, "", index + 1


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
        name, value, index = _awk_option(args, index)
        if name in ("-e", "--source"):
            texts.append(value)
        elif name == "-W":  # gawk -W source=PROGRAM (any unique prefix of `source`)
            key, equals, program = value.partition("=")
            if equals and len(key) > 1 and "source".startswith(key):
                texts.append(program)
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


def _prints_to_pipe(code: str) -> bool:
    """Whether a statement of *code* is ``print ... | "cmd"``. One linear pass: the first ``print`` of a
    statement and a lone ``|`` after it decide, however many more ``print`` words follow."""
    code = _AWK_CONTINUATION_RE.sub(" ", code)
    for statement in _AWK_STATEMENT_END_RE.split(code):
        keyword = _AWK_PRINT_RE.search(statement)
        if keyword and _AWK_LONE_PIPE_RE.search(statement, keyword.end()):
            return True
    return False


def awk_program_runs_shell(args: list[str]) -> bool:
    for text in awk_program_texts(args):
        code = awk_code_only(text)
        if _AWK_COMMAND_EXEC_RE.search(code) or _prints_to_pipe(code):
            return True
    return False
