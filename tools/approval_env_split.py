"""GNU ``env -S`` / ``--split-string`` projection for the approval detectors.

Moved out of :mod:`tools.approval_detection` (size cap); import from here.
"""

import re
import shlex

from tools import approval_parser_budget as _parser_budget

_ENV_SPLIT_EXPANSION_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_ENV_SPLIT_MARKER_RE = re.compile("\0(\\d+|Q)\0")  # \0N\0: the Nth expansion; \0Q\0: a quote opened here


def _split_env_string(payload: str) -> list[str] | None:
    r"""Project GNU env -S literal argv, not POSIX shell words.

    Dynamic ${NAME} expansion is deliberately not evaluated: the execution
    backend's environment need not be this process's environment.
    """
    escapes = {"f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v",
               "#": "#", "$": "$", "\"": "\"", "'": "'", "\\": "\\"}
    args, word = [], []
    quote, started, index = None, False, 0
    while index < len(payload):
        char = payload[index]
        index += 1
        if char == "\\":
            if index == len(payload):
                return None
            escaped = payload[index]
            if quote == "'" and escaped not in ("'", "\\"):
                word.append(char)
                started = True
                continue
            index += 1
            if escaped == "c":
                if quote:
                    return None
                break
            if escaped == "_" and quote is None:
                if started:
                    args.append("".join(word))
                word, started = [], False
                continue
            if escaped not in escapes and escaped != "_":
                return None
            word.append(" " if escaped == "_" else escapes[escaped])
            started = True
            continue
        if char in ("'", '"') and (quote is None or char == quote):
            quote = char if quote is None else None
            started = True
            continue
        if quote is None and char in " \t\n\r\v\f":
            if started:
                args.append("".join(word))
            word, started = [], False
            continue
        if quote is None and char == "#" and not started:
            break
        if char == "$" and quote != "'":
            return None
        word.append(char)
        started = True
    if quote:
        return None
    if started:
        args.append("".join(word))
    return args


def _split_env_string_unresolved(payload: str) -> list[str] | None:
    """Split an env -S string that holds ${NAME} expansions, as an argv.

    Keep expansions in argument fields as data. In the executable field, also
    consider empty expansions: a literal suffix or surrounding letters can
    name the command. A field vanishes only when it is made of nothing but
    unquoted expansions: any quote in it (even ``""``) keeps an empty field.
    Anything env rejects (a bare ``$``, an unclosed quote) yields None. Linear in *payload*, and
    charged to the caller's parser budget (a no-op outside one).
    """
    _parser_budget.charge(len(payload))
    out, names = [], []
    quote, index, word_start = None, 0, True  # word_start: no character of the current field yet
    while index < len(payload):
        char = payload[index]
        if quote is None and (char == "#" and word_start or payload.startswith("\\c", index)):
            # env ignores the rest after a comment word or `\c`: a `$` there is not an error.
            out.append(payload[index:index + 2] if char == "\\" else char)
            break
        if char == "\\":
            out.append(payload[index:index + 2])
            word_start = quote is None and payload.startswith("\\_", index)
            index += 2
            continue
        word_start = quote is None and char in " \t\n\r\v\f"
        if char in ("'", '"') and quote in (None, char):
            if quote is None:
                out.append(f"{char}\0Q\0")  # the marker survives quote removal, so the field stays non-empty
                quote, index = char, index + 1
                continue
            quote = None
        elif char == "$" and quote != "'":
            match = _ENV_SPLIT_EXPANSION_RE.match(payload, index)
            if not match:
                return None
            out.append(f"\0{len(names)}\0")
            names.append(match.group())
            index = match.end()
            continue
        out.append(char)
        index += 1
    marked = "".join(out)
    _parser_budget.charge(len(marked))  # the literal split and the marker passes below read it once each
    args = _split_env_string(marked)
    if args is None:
        return None
    first = 0  # leading fields that vanish; dropped with one slice
    while first < len(args):
        matches = list(_ENV_SPLIT_MARKER_RE.finditer(args[first]))
        if not matches:
            break
        literal = _ENV_SPLIT_MARKER_RE.sub("", args[first])
        if literal:
            args[first] = literal
            break
        if any(found.group(1) == "Q" for found in matches):
            break
        first += 1
    def _expand(found: re.Match) -> str:
        if found.group(1) == "Q":
            return ""
        index = int(found.group(1))
        return names[index] if index < len(names) else ""  # a literal NUL-digit-NUL in the payload is data, not a marker

    return [_ENV_SPLIT_MARKER_RE.sub(_expand, arg) for arg in args[first:]]


def _env_split_payload(tokens: list[str], _depth: int = 0, *, unresolved: bool = False,
                       capped: list[str] | None = None) -> str | None:
    """The argv an env -S / --split-string string runs, joined with literal separators protected.

    *unresolved* (the ``approvals.deny`` projection only) also splits a string holding ``${NAME}``
    expansions instead of giving up on it; the detection floors keep the literal-only projection.
    A split that still starts with options past the nesting limit is appended to *capped* (when
    given) and yields None: the deny caller refuses it rather than reading an option as the command.
    """
    from tools.approval_detection import _COMMAND_WRAPPER_OPTIONS_WITH_ARG
    from tools.approval_wrappers import _short_bundle

    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--" or not token.startswith("-"):
            return None
        option, equals, value = token.partition("=")
        if option == "--split-string":
            attached = value if equals else None
        elif token.startswith("--"):
            index += 2 if not equals and option in _COMMAND_WRAPPER_OPTIONS_WITH_ARG["env"] else 1
            continue
        else:
            # getopt bundle: the first valued option owns the rest of the word (`-iS'nice'`, `-uS`).
            _, owner, rest = _short_bundle("env", token)
            if owner != "-S":
                index += 2 if owner and not rest else 1
                continue
            attached = rest or None
        if attached is None:
            index += 1
        payload = attached if attached is not None else (tokens[index] if index < len(tokens) else "")
        args = _split_env_string(payload)
        if args is None and unresolved:
            # env still splits an argv (separators stay data), including the executable
            # that remains when its unresolved expansions are empty.
            args = _split_env_string_unresolved(payload)
        if args is None:
            return None
        rebuilt = args + tokens[index + 1:]
        # env re-parses a split string that starts with options (`-S '-S reboot'`).
        if rebuilt and rebuilt[0].startswith("-") and _depth < 8:
            nested = _env_split_payload(["env", *rebuilt], _depth + 1, unresolved=unresolved, capped=capped)
            if nested is not None:
                return nested
        elif rebuilt and rebuilt[0].startswith("-") and capped is not None:
            capped.append(shlex.join(rebuilt))
            return None
        # Protect literal separators: only a real shell -c carrier may turn these argv bytes into code.
        return shlex.join(rebuilt)
    return None
