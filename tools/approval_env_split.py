"""GNU ``env -S`` / ``--split-string`` projection for the approval detectors.

Moved out of :mod:`tools.approval_detection` (size cap); that module re-exports both names.
"""

import shlex


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


def _env_split_payload(tokens: list[str], _depth: int = 0) -> str | None:
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
        if args is None:
            return None
        rebuilt = args + tokens[index + 1:]
        # env re-parses a split string that starts with options (`-S '-S reboot'`).
        if rebuilt and rebuilt[0].startswith("-") and _depth < 8:
            nested = _env_split_payload(["env", *rebuilt], _depth + 1)
            if nested is not None:
                return nested
        # Protect literal separators: only a real shell -c carrier may turn these argv bytes into code.
        return shlex.join(rebuilt)
    return None
