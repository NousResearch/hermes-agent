"""Which shell names one simple command may overwrite with a value its text does not show.

The indirection resolver (``approval_detection_assignments``) treats ``X=echo; ...; $X`` as running
``echo`` only while nothing in between can change X. A builtin that stores input, output or a
computed value under a NAME breaks that proof: ``read X``, ``printf -vX``, ``mapfile``, ``wait -p X``.
So does anything that can set every name (``eval``, ``source``, a ``trap`` handler, a
``declare -n`` nameref). This module finds those writes for one simple command, however the
command is spelled:

* the prefix is skipped: ``IFS= read X``, ``! read X``, ``time -p read X``, ``builtin read X``,
  ``command -p read X`` all run the builtin in this shell;
* implicit targets count: a bare ``read`` sets REPLY, a bare ``mapfile`` sets MAPFILE, and
  ``getopts`` also sets OPTARG/OPTIND;
* attached option targets count: ``printf -vX``, ``read -raX``, ``wait -pX``;
* a target the text cannot name (``read "$N"``, ``declare "$N=v"``) means every name;
* redirections are removed wherever they sit (``< in read X``, ``read X 2>/dev/null``) before the
  command word is found (``approval_detection_assignments._simple_command_spans``);
* shell code a builtin runs later counts as LATE: a ``mapfile -C`` callback, a ``trap`` handler, an
  alias whose text writes a name;
* a bash builtin with no handler here and not in _NON_WRITING_BUILTINS (``enable``, ``fc``,
  ``bind``, ``coproc``) means every name, so an unmodeled builtin fails closed.

A command word that is itself an expansion (``$R X`` with R=read) cannot be classified here. It is
returned as DYNAMIC, and the resolver decides once it knows what the word may hold. Anything it
cannot resolve counts as every name. Over-reporting only turns a later ``$X`` into "cannot be
read", which refuses it; it never approves one.
"""

import re

# Every name, at this point (``source``/``.``, or a target the text does not show). A later
# assignment that dominates a use proves its value again: the sourced code is unreadable anyway,
# and could just as well run the destructive command itself.
ALL = None
# Every name, at ANY later point: a trap handler, a nameref alias, an eval payload (which can
# install either). No later assignment in the text proves a value over it.
LATE = "late"
# The command word is an expansion; the caller must resolve it and ask again.
DYNAMIC = "dynamic"

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# NAME=, NAME+=, NAME[sub]= as a prefix word (bash never quotes the name part).
_ASSIGNMENT_PREFIX_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?\+?=", re.DOTALL)
# Words that run what follows in this shell. Options after `time`/`builtin`/`command` are skipped.
_PREFIX_WORDS = frozenset({"!", "time", "builtin", "command"})
# `let` operands: NAME op= ..., NAME++ / NAME--, ++NAME / --NAME.
_LET_TARGET_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*(?:[-+*/%&|^]|<<|>>)?=(?!=)"
    r"|([A-Za-z_][A-Za-z0-9_]*)\s*(?:\+\+|--)|(?:\+\+|--)\s*([A-Za-z_][A-Za-z0-9_]*)")


def is_dynamic(raw: str) -> bool:
    """Does the shell word *raw* expand anything (``$``/backtick outside single quotes)?"""
    return bool(re.search(r"[$`]", re.sub(r"'[^']*'", "", raw)))


_EXPANSION_SPAN_RE = re.compile(r"\$\{[^}]*\}|\$\([^)]*\)|`[^`]*`")


def _is_path(raw: str) -> bool:
    """A literal ``/`` outside every expansion makes the word a path, and bash never looks a path
    up as a builtin (``"$VIRTUAL_ENV/bin/python"``, ``$HOME/bin/read``)."""
    return "/" in _EXPANSION_SPAN_RE.sub("", re.sub(r"'[^']*'", "q", raw))


def _plain(raw: str) -> str:
    from tools.approval_detection import _deobfuscate_shell_word_for_detection
    return _deobfuscate_shell_word_for_detection(raw)


def _targets(raws) -> frozenset[str] | None:
    """Variable names written as the words *raws*; ALL if any is not a literal name."""
    names = set()
    for raw in raws:
        if is_dynamic(raw):
            return ALL
        name = re.sub(r"\[.*\]$", "", _plain(raw), flags=re.DOTALL)
        if not _NAME_RE.fullmatch(name):
            return ALL      # not a valid identifier: bash errors, but do not guess
        names.add(name)
    return frozenset(names)


def _options(args, with_value: str, target_opts: str = ""):
    """Split getopt-style *args*: (option targets, operands, unreadable). Options in *with_value*
    take a value (attached ``-vX`` or the next word); those also in *target_opts* name a variable.
    A dynamic word where an option may appear is unreadable."""
    targets, pos = [], 0
    while pos < len(args):
        raw = args[pos]
        if is_dynamic(raw) and raw.lstrip("\"'").startswith(("$", "`")):
            return targets, args[pos:], True
        word = _plain(raw)
        if word == "--":
            return targets, args[pos + 1:], False
        if not word.startswith("-") or word == "-":
            break
        for k in range(1, len(word)):
            if word[k] in with_value:
                value = word[k + 1:]
                if not value:
                    pos += 1
                    value = args[pos] if pos < len(args) else ""
                if word[k] in target_opts:
                    targets.append(value)
                break
        pos += 1
    return targets, args[pos:], False


def _read(args):
    targets, operands, unreadable = _options(args, "adinNptu", "a")
    if unreadable:
        return ALL
    return _targets(targets + operands if (targets or operands) else ["REPLY"])


def _mapfile(args):
    callbacks, operands, unreadable = _options(args, "dnOsuCc", "C")
    if unreadable:
        return ALL
    if callbacks:
        # `-C code` runs shell code for every chunk read: it can write any name, install a trap
        # or a nameref (so it reaches past later assignments, like eval).
        return LATE
    return _targets(operands[:1] or ["MAPFILE"])


def _getopts(args):
    names = _targets(args[1:2])
    return None if names is None else names | {"OPTARG", "OPTIND"}


def _output_var(option: str):
    def handler(args):
        targets, _, unreadable = _options(args, option, option)
        return ALL if unreadable else _targets(targets)
    return handler


def _unset(args):
    _, operands, unreadable = _options(args, "")
    return ALL if unreadable else _targets(operands)


def _declaration(nameref: bool):
    def handler(args, include_assignments=False):
        names = set()
        for raw in args:
            word = _plain(raw)
            if is_dynamic(raw) and not _ASSIGNMENT_PREFIX_RE.match(raw):
                return ALL          # `declare "$N=v"`: which name is set is not in the text
            if nameref and re.fullmatch(r"[-+][A-Za-z]*n[A-Za-z]*", word):
                return LATE         # a nameref aliases another name from here on
            match = _ASSIGNMENT_PREFIX_RE.match(raw)
            if include_assignments and match:
                names.add(re.split(r"[\[+=]", raw, maxsplit=1)[0])
        # Literal NAME=value operands are bindings with a value; the assignment collector owns
        # them unless the declaration was reached through an expansion (*include_assignments*).
        return frozenset(names)
    return handler


def _let(args):
    names = set()
    for raw in args:
        if is_dynamic(raw):
            return ALL
        for groups in _LET_TARGET_RE.findall(_plain(raw)):
            names.update(g for g in groups if g)
    return frozenset(names)


def _every_name(args):
    return ALL


def _eval(args):
    # eval's payload may install a trap or a nameref, which reach past later assignments too.
    return LATE


def _trap(args):
    _, operands, unreadable = _options(args, "")
    # `trap -p`/`trap -l` only print; any handler runs later, anywhere, and can set any name.
    return LATE if unreadable or operands else frozenset()


# Bash builtins that never write a shell variable the text does not show (``cd`` only rewrites
# the shell-managed PWD/OLDPWD, which are never proven anyway). Every OTHER builtin must have a
# handler in _HANDLERS; a builtin on neither list counts as writing every name. External programs
# cannot write this shell's variables at all.
_NON_WRITING_BUILTINS = frozenset({
    ":", "[", "test", "true", "false", "echo", "cd", "pwd", "pushd", "popd", "dirs", "exit",
    "return", "break", "continue", "shift", "kill", "jobs", "bg", "fg", "disown", "hash", "type",
    "help", "history", "logout", "times", "ulimit", "umask", "unalias", "set", "shopt",
    "complete", "compopt", "caller", "suspend", "exec",
})
# Every bash 5.x builtin (``compgen -b``) plus the ``coproc`` keyword, which names an array.
_BASH_BUILTINS = _NON_WRITING_BUILTINS | frozenset({
    ".", "alias", "bind", "builtin", "command", "compgen", "declare", "enable", "eval", "export", "fc",
    "getopts", "let", "local", "mapfile", "printf", "read", "readarray", "readonly", "source",
    "trap", "typeset", "unset", "wait", "coproc",
})


def _alias(args):
    """``alias echo=read`` makes every later ``echo X`` a reader: any name, at any later point. An
    alias whose own text writes no name (``alias ll='ls -l'``) cannot.

    The replacement text is shell syntax, not one word list: ``alias r=':; read X'`` runs ``read X``
    after ``:``. Every simple command in it is classified; an assignment (``alias r='X=v'``) or any
    compound syntax this does not walk (``( )``, ``{ }``, a substitution) counts as a writer."""
    for raw in args:
        if is_dynamic(raw):
            return LATE
        word = _plain(raw)
        if "=" in word and _alias_value_writes(word.split("=", 1)[1]):
            return LATE
    return frozenset()


_ALIAS_COMPOUND_RE = re.compile(r"[(){}`]")


# Stands for the words that follow an alias at its invocation (unknown here).
_INVOCATION_OPERANDS = '"$@"'


def _alias_value_writes(value: str) -> bool:
    """Can running *value* as an alias write a name? The words after the alias at its invocation
    are appended to its LAST simple command (``alias a='builtin '``; ``a read X`` runs
    ``builtin read X``; ``alias p=printf``; ``p -v X``), so that command is classified with
    unknown operands added. A value ending in a separator starts a new command with them
    (``alias a='true;'``; ``a read X``)."""
    from tools.approval_detection import _iter_shell_command_starts
    from tools.approval_detection_assignments import _simple_command_words
    if is_dynamic(value) or _ALIAS_COMPOUND_RE.search(value):
        return True
    if not value.strip() or value.rstrip()[-1] in ";&|\n\\":
        return True
    starts = sorted(_iter_shell_command_starts(value))
    for pos in starts:
        words = _simple_command_words(value, pos)
        if pos == starts[-1]:
            words = words + [_INVOCATION_OPERANDS]
        if words and _ASSIGNMENT_PREFIX_RE.match(words[0]):
            return True
        if command_clobbers(words) != frozenset():
            return True
    return not starts


_HANDLERS = {
    "alias": _alias,
    "compgen": _output_var("V"),    # bash 5.3: `compgen -V NAME` stores the matches in NAME
    "read": _read,
    "mapfile": _mapfile,
    "readarray": _mapfile,
    "getopts": _getopts,
    "printf": _output_var("v"),
    "wait": _output_var("p"),
    "unset": _unset,
    "declare": _declaration(True),
    "typeset": _declaration(True),
    "local": _declaration(True),
    "export": _declaration(False),
    "readonly": _declaration(False),
    "let": _let,
    "eval": _eval,
    "source": _every_name,
    ".": _every_name,
    "trap": _trap,
}


_DECLARATIONS = frozenset({"declare", "typeset", "local", "export", "readonly"})


def command_clobbers(words: list[str], *, include_assignments: bool = False):
    """What the simple command made of the raw shell *words* may overwrite: a frozenset of names,
    ALL, LATE, or ``(DYNAMIC, index)`` when the command word at *index* is an expansion. With
    *include_assignments*, a declaration's literal ``NAME=value`` operands count as well (used when
    the declaration was reached through an expansion, so no binding records their values)."""
    pos = 0
    while pos < len(words) and _ASSIGNMENT_PREFIX_RE.match(words[pos]):
        pos += 1
    while pos < len(words) and not is_dynamic(words[pos]) and _plain(words[pos]) in _PREFIX_WORDS:
        wrapper = _plain(words[pos])
        pos += 1
        while wrapper != "!" and pos < len(words) and _plain(words[pos]).startswith("-"):
            if wrapper == "command" and _plain(words[pos]) in ("-v", "-V"):
                return frozenset()      # `command -v read` only looks the name up
            pos += 1
    if pos >= len(words):
        return frozenset()
    if is_dynamic(words[pos]):
        return frozenset() if _is_path(words[pos]) else (DYNAMIC, pos)
    leader = _plain(words[pos])
    handler = _HANDLERS.get(leader)
    if handler is None:
        # enable (loads new builtins), fc (re-runs history), bind -x, coproc NAME, and any
        # builtin added later: not modeled, so every name.
        return ALL if leader in _BASH_BUILTINS and leader not in _NON_WRITING_BUILTINS else frozenset()
    if leader in _DECLARATIONS:
        return handler(words[pos + 1:], include_assignments)
    return handler(words[pos + 1:])
