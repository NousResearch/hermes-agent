"""Shell forms the ``approvals.deny`` projection descends into beyond ``-c`` payloads.

``eval`` code, ``env -S`` strings holding unresolved ``${NAME}`` expansions, and a
command word that can expand to nothing. Used by
:func:`tools.approval_detection._deny_command_variants`.
"""

import re

_ENV_SPLIT_EXPANSION_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
# A bare expansion that can expand to no word: $NAME (zsh also $=NAME, $~NAME, $^NAME), a
# positional parameter, $@, $* or $!. $?, $$, $#, $- and $0 always expand to something.
_VANISHING_EXPANSION_RE = re.compile(r"\$(?:[=~^]*[A-Za-z_][A-Za-z0-9_]*|[1-9@*!])")
_BRACE_DEFAULT_RE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*|[0-9@*!]):?[-=](?P<default>.*)", re.DOTALL)


def _scan_dollar_brace_end(word: str, start: int) -> int | None:
    """Return the offset after a ``${...}`` expansion, counting nested ``${``."""
    from tools.approval_detection import _scan_shell

    depth = 1
    for kind, i, _, quote in _scan_shell(word, start + 2):
        if kind == "char" and quote != "'":
            depth += word.startswith("${", i) - (word[i] == "}")
            if depth == 0:
                return i + 1
    return None


def _brace_expansion_may_vanish(body: str) -> bool:
    if body.startswith("#"):
        return False  # ${#NAME} is a length: never empty
    default = _BRACE_DEFAULT_RE.fullmatch(body)
    if default is None:
        return True
    text = "".join(default.group("default").split())
    # A literal or quoted default always leaves a word behind (`${X:-tool}`, `${X:-""}`).
    return not text or "$" in text or "`" in text


def _command_word_may_vanish(word: str) -> bool:
    """True when *word* is made only of unquoted expansions that can expand to nothing.

    Such a command word disappears and the shell runs its argument tail instead:
    ``$TOOL cmd``, ``${TOOL} cmd``, ``$(true) cmd`` and the backtick form all run
    ``cmd`` in bash, sh and zsh. A quoted, escaped or literal-prefixed word leaves a
    (possibly empty) command name behind and its tail stays an argument list.
    ``$((...))`` counts as a substitution: the shells fall back to one when the text
    is not arithmetic (``$((true);(true)) cmd`` runs ``cmd``).
    """
    from tools.approval_detection import _scan_backtick_end, _scan_dollar_paren_end

    index = 0
    while index < len(word):
        if word[index] == "`":
            end = _scan_backtick_end(word, index)
        elif word.startswith("$(", index):
            end = _scan_dollar_paren_end(word, index)
        elif word.startswith("${", index):
            end = _scan_dollar_brace_end(word, index)
            if end is not None and not _brace_expansion_may_vanish(word[index + 2:end - 1]):
                end = None
        else:
            bare = _VANISHING_EXPANSION_RE.match(word, index)
            end = bare.end() if bare else None
        if end is None:
            return False
        index = end
    return bool(word)


def _split_env_string_unresolved(payload: str) -> list[str] | None:
    """Split an env -S string that holds ${NAME} expansions, as an argv.

    Keep expansions in argument fields as data. In the executable field, also
    consider empty expansions: a literal suffix or surrounding letters can
    name the command. An all-expansion field disappears only when unquoted.
    Anything env rejects (a bare ``$``, an unclosed quote) yields None.
    """
    from tools.approval_detection import _split_env_string

    out, names, removable = [], [], set()
    quote, index = None, 0
    while index < len(payload):
        char = payload[index]
        if char == "\\":
            out.append(payload[index:index + 2])
            index += 2
            continue
        if char in ("'", '"') and quote in (None, char):
            quote = None if quote else char
        elif char == "$" and quote != "'":
            match = _ENV_SPLIT_EXPANSION_RE.match(payload, index)
            if not match:
                return None
            if quote is None:
                removable.add(len(names))
            out.append(f"\0{len(names)}\0")
            names.append(match.group())
            index = match.end()
            continue
        out.append(char)
        index += 1
    args = _split_env_string("".join(out))
    if args is None:
        return None
    marker = re.compile("\0(\\d+)\0")
    while args:
        matches = list(marker.finditer(args[0]))
        if not matches:
            break
        literal = marker.sub("", args[0])
        if literal:
            args[0] = literal
            break
        if not all(int(found.group(1)) in removable for found in matches):
            break
        args.pop(0)
    return [marker.sub(lambda found: names[int(found.group(1))], arg) for arg in args]


def _eval_payload(args: list[str]) -> str | None:
    """Return the shell code eval would run: its arguments joined by spaces."""
    # eval takes no options: only an explicit ``--`` is consumed. A first word such
    # as ``-x`` is code (zsh reports it missing and runs the rest of the list).
    if args and args[0] == "--":
        args = args[1:]
    return " ".join(args) or None
